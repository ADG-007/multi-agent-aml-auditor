"""Typed LLM failures. Orchestration maps them to a FAILED stage; nothing crashes the run."""

from __future__ import annotations

from enum import Enum


class LLMFailureReason(str, Enum):
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    TIMED_OUT = "timed_out"
    #: Cached mode and no saved answer for these exact inputs.
    CACHE_MISS = "cache_miss"
    #: The run has its models switched off.
    MODEL_OFF = "model_off"


class LLMStageFailure(Exception):
    """A model call could not produce a validated answer."""

    def __init__(
        self,
        reason: LLMFailureReason,
        role: str,
        message: str,
        model_id: str | None = None,
    ) -> None:
        self.reason = reason
        self.role = role
        self.model_id = model_id
        super().__init__(f"[{role}/{reason.value}] {message}")
