"""One output model per stage. Each carries a reasoning_trail.

Confidence is risk size on 0-1 (0 = nothing found), not certainty. Doubt goes in the trail.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from src.schemas.common import (
    ReasoningStep,
    SeverityTier,
    StageName,
    TriageEventKind,
    TriageRule,
)
from src.schemas.domain import Transaction


class TriageEvent(BaseModel):
    """What Stage 0 evaluates: a new transaction, a KYC submission or a periodic review."""

    kind: TriageEventKind
    account_id: str
    transaction: Transaction | None = Field(
        default=None,
        description="Present for TRANSACTION events only.",
    )

    @model_validator(mode="after")
    def _transaction_matches_kind(self) -> TriageEvent:
        if (self.kind == TriageEventKind.TRANSACTION) != (self.transaction is not None):
            raise ValueError("a TRANSACTION event needs a transaction; other kinds take none")
        if self.transaction is not None and self.transaction.account_id != self.account_id:
            raise ValueError("the transaction belongs to a different account")
        return self


class FiredRule(BaseModel):
    """One routing-table row that fired, in machine-readable form."""

    rule: TriageRule
    transaction_id: str | None = Field(
        default=None,
        description="The transaction that fired it; None for whole-account rules.",
    )
    clause_id: str | None = Field(
        default=None,
        description="The rulebook clause whose metadata the rule read, if any.",
    )
    detail: str


class Stage0TriageOutput(BaseModel):
    """Which stages this case needs, and why. trigger_reasons and fired_rules run in parallel."""

    routed_stages: list[StageName] = Field(default_factory=list)
    trigger_reasons: list[str] = Field(default_factory=list)
    fired_rules: list[FiredRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def _reasons_parallel_rules(self) -> Stage0TriageOutput:
        if len(self.trigger_reasons) != len(self.fired_rules):
            raise ValueError("trigger_reasons and fired_rules must have one entry per rule")
        return self


class Stage1VisionOutput(BaseModel):
    """ID check. Never scored: at >= 0.90 the case pauses for an admin instead."""

    document_type: str
    extracted_fields: dict[str, str] = Field(
        default_factory=dict,
        description="OCR output, including id_expiration_date.",
    )
    field_discrepancies: list[str] = Field(default_factory=list)
    forgery_confidence: float = Field(ge=0.0, le=1.0)
    reasoning_trail: list[ReasoningStep] = Field(default_factory=list)


class Stage2TransactionPatternOutput(BaseModel):
    """Structuring, pass-through and velocity. Weight 0.25. Confidence = the stronger sub-score."""

    structuring_indicators: list[str] = Field(default_factory=list)
    pass_through_indicators: list[str] = Field(default_factory=list)
    velocity_summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_trail: list[ReasoningStep] = Field(default_factory=list)


class Stage3KYCOutput(BaseModel):
    """Whether the stated profile is behaviourally plausible. LLM-backed. Weight 0.20."""

    profile_behavior_mismatches: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_trail: list[ReasoningStep] = Field(default_factory=list)


class Stage4AdverseMediaOutput(BaseModel):
    """Sanctions / adverse-media screening. Veto tier at >= 0.90. Weight 0.25."""

    matched_entries: list[str] = Field(default_factory=list)
    severity_tier: SeverityTier | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_trail: list[ReasoningStep] = Field(default_factory=list)


class Stage5GraphOutput(BaseModel):
    """Shared-identifier links. Veto at >= 0.90. Weight 0.30.

    connected_flagged_accounts comes from this run's flagged accounts, so batch order is fixed.
    """

    shared_identifiers: list[str] = Field(default_factory=list)
    connected_flagged_accounts: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_trail: list[ReasoningStep] = Field(default_factory=list)


class Stage6RulebookOutput(BaseModel):
    """Clause search only. Never routes, never scores. Stage 7 may cite only these ids."""

    retrieved_clause_ids: list[str] = Field(default_factory=list)
    policy_violations: list[str] = Field(default_factory=list)
    reasoning_trail: list[ReasoningStep] = Field(default_factory=list)
