"""InvestigationReport - the pipeline's single deliverable.

Assembled by Stage 7 from the already-computed deterministic disposition and confidence
(score-then-narrate order), plus every upstream stage's findings.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field

from src.schemas.common import Disposition, PipelineStatus, StageStatus
from src.schemas.stages import (
    Stage0TriageOutput,
    Stage1VisionOutput,
    Stage2TransactionPatternOutput,
    Stage3KYCOutput,
    Stage4AdverseMediaOutput,
    Stage5GraphOutput,
    Stage6RulebookOutput,
)


class InvestigationReport(BaseModel):
    """One case, one verdict.

    A case is triggered by a single transaction but Stages 2-5 evaluate the account's full
    rolling history. Re-investigating an account creates a new case_id rather than
    overwriting the prior one.
    """

    case_id: str = Field(description="CASE-{account_id}-{sequence}.")
    triggering_transaction_id: str
    target_account: str

    pipeline_status: PipelineStatus
    stage_status: dict[str, StageStatus] = Field(default_factory=dict)

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Deterministically computed. Never LLM arithmetic (invariant 1).",
    )
    disposition: Disposition

    degraded_stages: list[str] = Field(
        default_factory=list,
        description="Weighted stages that FAILED or TIMED_OUT, flooring the case at MONITOR.",
    )
    evidence_incomplete: bool = False
    single_signal_capped: bool = Field(
        default=False,
        description="Score cleared 0.6 but the corroboration gate held it at MONITOR.",
    )
    policy_gap_identified: bool = Field(
        default=False,
        description="Real weighted evidence, but no retrieved clause met the citation bar.",
    )
    forgery_notice: str | None = Field(
        default=None,
        description="Non-blocking notice for forgery confidence in [0.60, 0.90).",
    )

    executive_summary: str = ""
    investigation_narrative: str = ""
    str_narrative: str | None = Field(
        default=None,
        description="Suspicious Transaction Report draft. Populated only for SUSPICIOUS-ESCALATE.",
    )
    compliance_conclusion: str = ""

    triage_findings: Stage0TriageOutput | None = None
    vision_findings: Stage1VisionOutput | None = None
    transaction_pattern_findings: Stage2TransactionPatternOutput | None = None
    kyc_findings: Stage3KYCOutput | None = None
    adverse_media_findings: Stage4AdverseMediaOutput | None = None
    graph_findings: Stage5GraphOutput | None = None
    rulebook_findings: Stage6RulebookOutput | None = None

    @computed_field
    @property
    def risk_score_display(self) -> float:
        """UI only. Must never feed logic - confidence is the value everything reads."""
        return round(self.confidence * 100, 1)
