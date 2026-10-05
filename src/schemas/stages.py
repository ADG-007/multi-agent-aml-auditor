"""Per-stage output contracts.

One model per stage, each produced by that stage and consumed by Stage 7's synthesis.
Every model carries a reasoning_trail (invariant 5). Confidence is risk magnitude on
0-1, not epistemic certainty: 0 means no adverse signal was found, 1 means certain of an
adverse finding. A stage that is genuinely unsure says so in the trail, never by shading
confidence toward the middle.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.schemas.common import ReasoningStep, SeverityTier


class Stage0TriageOutput(BaseModel):
    """Which stages this case needs. Stage 0 is itself the flagging step."""

    routed_stages: list[str] = Field(default_factory=list)
    trigger_reason: str


class Stage1VisionOutput(BaseModel):
    """Document integrity. Never enters the scoring formula (invariant 2).

    forgery_confidence is a separate compliance gate: at or above 0.90 the pipeline pauses
    for a human admin decision, rather than the system guessing about document authenticity
    and folding that guess into a risk score.
    """

    document_type: str
    extracted_fields: dict[str, str] = Field(
        default_factory=dict,
        description="OCR output, including id_expiration_date.",
    )
    field_discrepancies: list[str] = Field(default_factory=list)
    forgery_confidence: float = Field(ge=0.0, le=1.0)
    reasoning_trail: list[ReasoningStep] = Field(default_factory=list)


class Stage2TransactionPatternOutput(BaseModel):
    """Structuring, pass-through and velocity analysis over the account's rolling history.

    Weight 0.25. Planned design, confirmed at Phase 3's kickoff: structuring and pass-through
    are separate sub-scores and confidence reports the stronger of the two, so adding the
    pass-through measure leaves structuring untouched.
    """

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
    """Shared-identifier topology analysis. Veto tier at >= 0.90. Weight 0.30.

    connected_flagged_accounts is computed live from accounts already elevated in this run,
    never read from a ground-truth field - which is why batch evaluation must fix its
    processing order (Phase 11).
    """

    shared_identifiers: list[str] = Field(default_factory=list)
    connected_flagged_accounts: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_trail: list[ReasoningStep] = Field(default_factory=list)


class Stage6RulebookOutput(BaseModel):
    """Post-hoc clause citation only - never routes, never scores.

    retrieved_clause_ids is the authoritative set for citation validation: Stage 7 may cite
    only from this list, and a deterministic check rejects anything else (invariant 4).
    """

    retrieved_clause_ids: list[str] = Field(default_factory=list)
    policy_violations: list[str] = Field(default_factory=list)
    reasoning_trail: list[ReasoningStep] = Field(default_factory=list)
