"""Shared structured-LLM-calling abstraction used by every LLM-backed stage."""

from src.llm.client import get_structured_response, looks_like_refusal, soften_prompt
from src.llm.config import STAGE_LLM_CONFIG, LLMProvider, StageLLMConfig, get_stage_config
from src.llm.errors import LLMFailureReason, LLMStageFailure

__all__ = [
    "STAGE_LLM_CONFIG",
    "LLMFailureReason",
    "LLMProvider",
    "LLMStageFailure",
    "StageLLMConfig",
    "get_stage_config",
    "get_structured_response",
    "looks_like_refusal",
    "soften_prompt",
]
