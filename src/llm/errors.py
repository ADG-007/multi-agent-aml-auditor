"""Typed LLM failures.

Phase 10's orchestration layer catches these and maps them to stage_status = FAILED with a
distinguishable reason, rather than letting an exception crash the pipeline (invariant 8:
failures degrade, they do not crash).

The reason code matters for Phase 11's reporting: a spike in content-policy refusals must
never silently read as poor model reasoning, so refusals are counted separately from
genuine reasoning errors and from rate-limit exhaustion.
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
    """A stage's LLM call could not produce a validated result.

    Carries enough context for Phase 11 to report refusal rate per provider separately from
    reasoning accuracy.
    """

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
