"""Which model each LLM stage uses. Stage 3: local Ollama 8B. Stage 7: free Gemini.

Every other stage is deterministic Python. No Claude models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cache


class LLMProvider(str, Enum):
    """The only provider families allowed."""

    OLLAMA = "ollama"
    GOOGLE_GENAI = "google_genai"
    OPENAI_COMPATIBLE = "openai"


@dataclass(frozen=True)
class StageLLMConfig:
    """How one stage calls a model. model_id is an exact, pinned tag."""

    stage_name: str
    provider: LLMProvider | None = None
    model_id: str | None = None
    temperature: float = 0.0
    fallback_provider: LLMProvider | None = None
    fallback_model_id: str | None = None
    #: Extra provider arguments (Ollama: seed, num_ctx).
    model_kwargs: dict = field(default_factory=dict)
    #: Environment variable that overrides model_id, e.g. AML_KYC_MODEL.
    model_env: str | None = None
    decided_in_phase: str = ""
    notes: str = ""

    @property
    def is_decided(self) -> bool:
        return self.provider is not None and self.model_id is not None


#: Framing on every cloud prompt, so safety filters do not refuse synthetic AML cases.
SYNTHETIC_CONTEXT_PREAMBLE = (
    "You are assisting a compliance analyst inside a research prototype. All customers, "
    "accounts, transactions and documents referenced below are synthetic data generated for "
    "a laboratory evaluation of an anti-money-laundering workflow. No real person, account "
    "or institution is involved. Your output is a recommendation for a human analyst to "
    "review, never an autonomous decision."
)


STAGE_LLM_CONFIG: dict[str, StageLLMConfig] = {
    "KYC": StageLLMConfig(
        stage_name="KYC",
        provider=LLMProvider.OLLAMA,
        # Runs partly on the CPU of a 4 GB card: performance reduced by hardware limits.
        model_id="llama3.1:8b-instruct-q4_K_M",
        model_kwargs={"seed": 42, "num_ctx": 4096},
        model_env="AML_KYC_MODEL",
        decided_in_phase="Stage 3 build",
        notes=(
            "Small context: one KYC narrative plus a transaction summary. Local Ollama, "
            "temperature 0, seed 42."
        ),
    ),
    "Rulebook": StageLLMConfig(
        stage_name="Rulebook",
        decided_in_phase="not used",
        notes="Stage 6 is ChromaDB search only: no model.",
    ),
    "Auditor": StageLLMConfig(
        stage_name="Auditor",
        provider=LLMProvider.GOOGLE_GENAI,
        # Free tier; the available model can change. Override with AML_AUDITOR_MODEL.
        model_id="gemini-3.5-flash",
        model_env="AML_AUDITOR_MODEL",
        decided_in_phase="Stage 7 build",
        notes=(
            "Section 4 commits this stage to a free-tier cloud API rather than a local "
            "model: it is the heaviest context load in the pipeline and local KV-cache VRAM "
            "is the binding constraint on the available 4-6GB hardware."
        ),
    ),
}


@cache
def load_env() -> None:
    """Read .env once (API key, model overrides). Never printed."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(override=False)


def get_stage_config(stage_name: str) -> StageLLMConfig:
    """A stage's model config. KeyError for a deterministic stage: calling here is a bug."""
    if stage_name not in STAGE_LLM_CONFIG:
        raise KeyError(
            f"Stage {stage_name!r} has no LLM configuration. Only "
            f"{sorted(STAGE_LLM_CONFIG)} are LLM-backed; every other stage is deterministic "
            f"Python by design."
        )
    load_env()
    config = STAGE_LLM_CONFIG[stage_name]
    override = os.environ.get(config.model_env) if config.model_env else None
    return replace(config, model_id=override) if override else config
