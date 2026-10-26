"""Saved model answers (data/llm_cache/<stage>/), keyed by model + prompt version + prompt.

The files live in git, so a fresh copy replays the 50 with no model installed.

Modes (one per run):
    off     no model call, no cache. LLM stages give their neutral stand-in (tests).
    cached  a saved answer or nothing: a miss is a FAILED stage, never a silent 0.
    live    a saved answer if there is one, else call the model and save the answer.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path


CACHE_ROOT = Path("data") / "llm_cache"


class LLMMode(str, Enum):
    OFF = "off"
    CACHED = "cached"
    LIVE = "live"


def fingerprint(model_id: str, prompt_version: str, prompt: str) -> str:
    raw = json.dumps([model_id, prompt_version, prompt], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, stage: str, root: Path | None = None) -> None:
        self.folder = (root or CACHE_ROOT) / stage

    def path(self, name: str) -> Path:
        return self.folder / f"{name}.json"

    def load(self, name: str, key: str) -> dict | None:
        """The saved record, only if it was made from exactly these inputs."""
        p = self.path(name)
        if not p.exists():
            return None
        record = json.loads(p.read_text(encoding="utf-8"))
        return record if record.get("key") == key else None

    def count(self, prompt_version: str) -> int:
        """Saved answers made with this prompt version."""
        if not self.folder.exists():
            return 0
        return sum(
            json.loads(p.read_text(encoding="utf-8")).get("prompt_version") == prompt_version
            for p in self.folder.glob("*.json")
        )

    def save(self, name: str, record: dict) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path(name).write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
