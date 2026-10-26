"""Typed LLM failures. The orchestrator turns them into a FAILED stage, never a crash.

The reason keeps refusals apart from bad reasoning and rate limits.
"""

from __future__ import annotations

from enum import Enum


class LLMFailureReason(str, Enum):
    CONTENT_POLICY_REFUSAL = "content_policy_refusal"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    RATE_LIMITED = "rate_limited"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    NOT_CONFIGURED = "not_configured"
    TIMED_OUT = "timed_out"


class LLMStageFailure(Exception):
    """A stage's LLM call gave no validated result. Carries the reason, provider and model."""

    def __init__(
        self,
        reason: LLMFailureReason,
        stage_name: str,
        message: str,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self.reason = reason
        self.stage_name = stage_name
        self.provider = provider
        self.model_id = model_id
        super().__init__(f"[{stage_name}/{reason.value}] {message}")
