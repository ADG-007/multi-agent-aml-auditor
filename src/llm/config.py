"""Which local model serves which role. Every model runs on Ollama on this laptop.

Two models can change a verdict input (both Qwen): the Vision agent's KYC claims and the Web
Research agent's same-person call. The Gemma roles label, quote and write; they never score.
The 9B backup takes queued jobs only. The embedding model stays on the CPU, so it never
competes with an LLM for the 4 GB card.

The tags and settings were fixed by the model gate (src/evaluation/model_gate.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    VISION = "vision"
    WEB_RESEARCH = "web_research"
    GRAPH = "graph"
    RAG = "rag"
    AUDITOR = "auditor"
    BACKUP = "backup"
    EMBED = "embed"


@dataclass(frozen=True)
class ModelSpec:
    """One Ollama model and how it is called."""

    model: str
    #: True / False switches thinking on models that have it; None sends no switch.
    think: bool | None = None
    num_ctx: int = 4096
    num_predict: int = 800
    #: 0 keeps the model on the CPU; None lets Ollama place it.
    num_gpu: int | None = None


#: Ollama's own split leaves half of the 4B on the CPU (8 tokens/s). With every layer forced
#: onto the GPU and a 2k context it fits the 4 GB card (46 tokens/s, measured in the gate).
QWEN_4B = ModelSpec("qwen3.5:4b-q4_K_M", think=False, num_ctx=2048, num_gpu=99)
GEMMA = ModelSpec("gemma4:e4b-it-qat")
QWEN_9B = ModelSpec("qwen3.5:9b-q4_K_M", think=True, num_predict=2000)
EMBEDDING = ModelSpec("qwen3-embedding:0.6b", num_gpu=0)

MODELS: dict[Role, ModelSpec] = {
    Role.VISION: QWEN_4B,
    Role.WEB_RESEARCH: QWEN_4B,
    Role.GRAPH: GEMMA,
    Role.RAG: GEMMA,
    Role.AUDITOR: GEMMA,
    Role.BACKUP: QWEN_9B,
    Role.EMBED: EMBEDDING,
}

OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

#: A model stays loaded this long after a call, so one phase pays one load.
KEEP_ALIVE: str = "30m"

#: A stuck call is cut off after this many seconds and counts as TIMED_OUT.
TIMEOUT_SECONDS: float = 600.0

TEMPERATURE: float = 0.0
SEED: int = 42
