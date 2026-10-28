"""InvestigationReport: the pipeline's one output. Scored first, then narrated."""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field

from src.schemas.common import Disposition, PipelineStatus, ReasoningStep, StageStatus
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
    """One case, one verdict. A re-investigation gets a new case_id."""

    case_id: str = Field(description="CASE-{account_id}-{sequence}.")
    triggering_transaction_id: str | None = Field(
        default=None,
        description="Earliest transaction that fired a rule; None for a review nothing alerted on.",
    )
    target_account: str

    pipeline_status: PipelineStatus
    stage_status: dict[str, StageStatus] = Field(default_factory=dict)

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Deterministically computed. Never LLM arithmetic.",
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
    judge_trail: list[ReasoningStep] = Field(
        default_factory=list,
        description="The judge's own trail: veto, weighted average, thresholds, gate, floors.",
    )
    auditor_trail: list[ReasoningStep] = Field(
        default_factory=list,
        description="Stage 7's own trail: how the text was made and which citations were removed.",
    )

    @computed_field
    @property
    def risk_score_display(self) -> float:
        """UI only. Must never feed logic - confidence is the value everything reads."""
        return round(self.confidence * 100, 1)
