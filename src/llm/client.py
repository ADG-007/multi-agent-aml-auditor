"""The one call path to a local model (Ollama's HTTP API).

    client.call(role, prompt, schema, prompt_version, images=...) -> CallResult

Every call: cache lookup, model lock (one LLM on the GPU at a time), JSON-schema output,
one retry that shows the model its validation error, an audit line, and a saved answer.
A failure raises LLMStageFailure; the caller decides how its agent degrades.
"""

from __future__ import annotations

import base64
import hashlib
import json
import platform
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from src.llm import config
from src.llm.cache import LLMMode, ResponseCache, fingerprint
from src.llm.config import ModelSpec, Role
from src.llm.errors import LLMFailureReason, LLMStageFailure


TSchema = TypeVar("TSchema", bound=BaseModel)

#: (api path, payload) -> the response JSON. Tests pass a fake.
Transport = Callable[[str, dict], dict]

AUDIT_PATH = Path("outputs") / "audit" / "llm_calls.jsonl"
MAX_ATTEMPTS = 2

_GPU_LOCK = threading.Lock()


@dataclass(frozen=True)
class CallResult:
    value: Any
    model: str
    seconds: float
    cached: bool
    attempts: int


class LLMClient:
    def __init__(
        self,
        mode: LLMMode = LLMMode.OFF,
        cache_root: Path | None = None,
        audit_path: Path | None = AUDIT_PATH,
        transport: Transport | None = None,
        models: dict[Role, ModelSpec] | None = None,
    ) -> None:
        self.mode = mode
        self.cache = ResponseCache(cache_root)
        self.audit_path = audit_path
        self.models = models or config.MODELS
        self._transport = transport or _http
        self._loaded: str | None = None
        #: Every call of this run, for the per-agent timing report.
        self.calls: list[dict] = []

    @property
    def off(self) -> bool:
        return self.mode == LLMMode.OFF

    def model_for(self, role: Role) -> str:
        return self.models[role].model

    # ----------------------------------------------------------------------------------
    # Structured chat
    # ----------------------------------------------------------------------------------

    def call(
        self,
        role: Role,
        prompt: str,
        schema: type[TSchema],
        prompt_version: str,
        images: Sequence[Path] = (),
        system: str | None = None,
        agent: str | None = None,
    ) -> CallResult:
        """A validated `schema` object from the role's model, or LLMStageFailure."""
        spec = self.models[role]
        agent = agent or role.value
        if self.mode == LLMMode.OFF:
            raise LLMStageFailure(LLMFailureReason.MODEL_OFF, agent, "models are off for this run")

        image_hashes = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in images]
        key = fingerprint(
            spec.model, prompt_version, prompt, [*image_hashes, system or "", schema.__name__]
        )
        saved = self.cache.get(agent, key)
        if saved is not None:
            value = schema.model_validate(saved["response"])
            self._audit(agent, spec.model, prompt_version, saved.get("seconds", 0.0), True, 1)
            return CallResult(value, spec.model, saved.get("seconds", 0.0), True, 1)
        if self.mode == LLMMode.CACHED:
            self._audit(agent, spec.model, prompt_version, 0.0, True, 0, error="cache_miss")
            raise LLMStageFailure(
                LLMFailureReason.CACHE_MISS,
                agent,
                "no saved answer for these inputs (cached mode)",
                spec.model,
            )

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        user: dict = {"role": "user", "content": prompt}
        if images:
            user["images"] = [base64.b64encode(Path(p).read_bytes()).decode() for p in images]
        messages.append(user)

        started = time.perf_counter()
        error = ""
        stats: dict = {}
        for attempt in range(1, MAX_ATTEMPTS + 1):
            body = self._chat(spec, messages, schema, agent)
            stats = body
            text = (body.get("message") or {}).get("content", "")
            try:
                value = schema.model_validate_json(_json_text(text))
            except ValidationError as exc:
                error = _short(exc)
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": "That answer was not valid: "
                        f"{error}. Answer again with valid JSON only.",
                    },
                ]
                continue
            seconds = round(time.perf_counter() - started, 2)
            self.cache.put(
                agent,
                key,
                {
                    "model": spec.model,
                    "prompt_version": prompt_version,
                    "seconds": seconds,
                    "attempts": attempt,
                    "machine": platform.node(),
                    "made_at": datetime.now().isoformat(timespec="seconds"),
                    "response": value.model_dump(mode="json"),
                },
            )
            self._audit(agent, spec.model, prompt_version, seconds, False, attempt, stats=stats)
            return CallResult(value, spec.model, seconds, False, attempt)

        seconds = round(time.perf_counter() - started, 2)
        self._audit(agent, spec.model, prompt_version, seconds, False, MAX_ATTEMPTS, error="schema")
        raise LLMStageFailure(
            LLMFailureReason.SCHEMA_VALIDATION_FAILED,
            agent,
            f"no valid {schema.__name__} after {MAX_ATTEMPTS} attempts: {error}",
            spec.model,
        )

    def _chat(self, spec: ModelSpec, messages: list[dict], schema: type, agent: str) -> dict:
        options: dict = {
            "temperature": config.TEMPERATURE,
            "seed": config.SEED,
            "num_ctx": spec.num_ctx,
            "num_predict": spec.num_predict,
        }
        if spec.num_gpu is not None:
            options["num_gpu"] = spec.num_gpu
        payload: dict = {
            "model": spec.model,
            "messages": messages,
            "format": schema.model_json_schema(),
            "stream": False,
            "keep_alive": config.KEEP_ALIVE,
            "options": options,
        }
        if spec.think is not None:
            payload["think"] = spec.think
        with _GPU_LOCK:
            self._swap_to(spec.model)
            return self._send("/api/chat", payload, agent, spec.model)

    def _swap_to(self, model: str) -> None:
        """Unload the LLM that is loaded now before a different one is called."""
        if self._loaded and self._loaded != model:
            try:
                self._transport("/api/generate", {"model": self._loaded, "keep_alive": 0})
            except Exception:  # noqa: BLE001 - an unload that fails must not fail the call
                pass
        self._loaded = model

    def unload(self) -> None:
        """Free the GPU at the end of a phase."""
        with _GPU_LOCK:
            self._swap_to("")
            self._loaded = None

    def _send(self, path: str, payload: dict, agent: str, model: str) -> dict:
        try:
            return self._transport(path, payload)
        except httpx.TimeoutException as exc:
            raise LLMStageFailure(LLMFailureReason.TIMED_OUT, agent, str(exc), model) from exc
        except (httpx.HTTPError, OSError) as exc:
            raise LLMStageFailure(
                LLMFailureReason.PROVIDER_UNAVAILABLE, agent, _short(exc), model
            ) from exc

    # ----------------------------------------------------------------------------------
    # Embeddings (CPU)
    # ----------------------------------------------------------------------------------

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per text from the embedding model. Never cached here; callers cache
        what they derive from it."""
        spec = self.models[Role.EMBED]
        if self.mode != LLMMode.LIVE:
            raise LLMStageFailure(
                LLMFailureReason.MODEL_OFF, Role.EMBED.value, "embedding needs live mode"
            )
        options = {"num_gpu": spec.num_gpu} if spec.num_gpu is not None else {}
        payload = {
            "model": spec.model,
            "input": list(texts),
            "keep_alive": config.KEEP_ALIVE,
            "options": options,
        }
        started = time.perf_counter()
        body = self._send("/api/embed", payload, Role.EMBED.value, spec.model)
        self._audit(
            Role.EMBED.value, spec.model, "embed", round(time.perf_counter() - started, 2), False, 1
        )
        return body["embeddings"]

    # ----------------------------------------------------------------------------------
    # Audit log
    # ----------------------------------------------------------------------------------

    def _audit(
        self,
        agent: str,
        model: str,
        prompt_version: str,
        seconds: float,
        cached: bool,
        attempts: int,
        error: str = "",
        stats: dict | None = None,
    ) -> None:
        line = {
            "at": datetime.now().isoformat(timespec="seconds"),
            "agent": agent,
            "model": model,
            "prompt_version": prompt_version,
            "seconds": seconds,
            "cached": cached,
            "attempts": attempts,
            "ok": not error,
            "error": error,
            "prompt_tokens": (stats or {}).get("prompt_eval_count"),
            "output_tokens": (stats or {}).get("eval_count"),
        }
        self.calls.append(line)
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")


def _http(path: str, payload: dict) -> dict:
    response = httpx.post(config.OLLAMA_URL + path, json=payload, timeout=config.TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _json_text(text: str) -> str:
    """The JSON object inside the answer: code fences and stray words around it are dropped."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else text


def _short(exc: Exception) -> str:
    return " ".join(str(exc).split())[:300]
