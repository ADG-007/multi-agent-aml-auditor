"""Per-stage LLM provider/model configuration.

MASTER_PLAN.md Section 4: the pipeline's LLM strategy is a hybrid of locally-run Ollama
models and free-tier cloud APIs,
chosen per stage based on that stage's real reasoning demands.

Only three stages ever call an LLM: Stage 3 (KYC plausibility), Stage 6 (rulebook clause
wording) and Stage 7 (auditor synthesis). Stages 0, 1, 2, 4 and 5 are pure deterministic
Python and must stay that way.

The concrete model for each stage is deliberately NOT decided here. Section 4 commits only
Stage 7 (free-tier cloud, because it assembles seven reasoning trails into the pipeline's
heaviest single context load and a quantized local model's usable context on a 4-6GB card
is far below its advertised maximum). Stage 3 and Stage 6 stay open until their own phase
kickoffs, when real local-model benchmarking is possible - matching the project's
decide-with-real-information approach rather than guessing now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class LLMProvider(str, Enum):
    """Provider families this project is allowed to route to.

    Only these families are permitted. Adding any other provider here would silently violate
    MASTER_PLAN.md Section 4.
    """

    OLLAMA = "ollama"
    GOOGLE_GENAI = "google_genai"
    OPENAI_COMPATIBLE = "openai"


@dataclass(frozen=True)
class StageLLMConfig:
    """How one stage calls a model.

    model_id is pinned to an exact tag once chosen, for the same reproducibility reason the
    data generation pins a fixed seed - an Ollama tag like 'llama3.1:8b-instruct-q4_K_M'
    rather than a floating 'llama3.1'.
    """

    stage_name: str
    provider: LLMProvider | None = None
    model_id: str | None = None
    temperature: float = 0.0
    fallback_provider: LLMProvider | None = None
    fallback_model_id: str | None = None
    decided_in_phase: str = ""
    notes: str = ""

    @property
    def is_decided(self) -> bool:
        return self.provider is not None and self.model_id is not None


#: The synthetic/research framing every cloud prompt carries.
#:
#: Step 1 of the six-step content-policy chain. Free-tier consumer-facing APIs run safety
#: classifiers that can refuse a request discussing financial crime even when the scenario
#: is entirely fabricated. Stating the context up front reduces the false-trigger rate; it
#: is not a jailbreak and does not ask the model to relax any policy.
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
        decided_in_phase="Phase 4",
        notes=(
            "Small context: one KYC narrative plus a transaction summary. Local Ollama is "
            "plausible but not committed - benchmark before choosing."
        ),
    ),
    "Rulebook": StageLLMConfig(
        stage_name="Rulebook",
        decided_in_phase="Phase 8",
        notes=(
            "Wording only. Retrieval and citation validation are deterministic, so this "
            "stage's model never selects which clauses exist - it phrases findings against "
            "an already-retrieved set."
        ),
    ),
    "Auditor": StageLLMConfig(
        stage_name="Auditor",
        decided_in_phase="Phase 9",
        notes=(
            "Section 4 commits this stage to a free-tier cloud API rather than a local "
            "model: it is the heaviest context load in the pipeline and local KV-cache VRAM "
            "is the binding constraint on the available 4-6GB hardware."
        ),
    ),
}


def get_stage_config(stage_name: str) -> StageLLMConfig:
    """Look up a stage's model configuration.

    Raises KeyError for a stage that is not supposed to call an LLM at all, which is the
    intended behaviour: a deterministic stage reaching for this module is a bug, not a
    configuration gap.
    """
    if stage_name not in STAGE_LLM_CONFIG:
        raise KeyError(
            f"Stage {stage_name!r} has no LLM configuration. Only "
            f"{sorted(STAGE_LLM_CONFIG)} are LLM-backed; every other stage is deterministic "
            f"Python by design (MASTER_PLAN.md Section 2.1)."
        )
    return STAGE_LLM_CONFIG[stage_name]
