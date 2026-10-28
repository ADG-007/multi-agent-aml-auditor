"""Stand-in stage outputs for tests. A StubPlan sets each one's status and confidence.

Stubs never read the answer key.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from src.schemas import (
    Disposition,
    ReasoningStep,
    Stage1VisionOutput,
    Stage2TransactionPatternOutput,
    Stage3KYCOutput,
    Stage4AdverseMediaOutput,
    Stage5GraphOutput,
    Stage6RulebookOutput,
    StageName,
    StageStatus,
)
from src.scoring import StageResult


#: The module of the real stage each stub stands in for.
REPLACED_IN: dict[StageName, str] = {
    StageName.VISION: "stage1_vision",
    StageName.TRANSACTION_PATTERN: "stage2_transaction_pattern",
    StageName.KYC: "stage3_kyc",
    StageName.ADVERSE_MEDIA: "stage4_adverse_media",
    StageName.GRAPH: "stage5_graph",
    StageName.RULEBOOK: "stage6_rulebook",
    StageName.AUDITOR: "stage7_auditor",
}

_DEFAULT = StageResult(StageStatus.RAN, 0.0)


@dataclass(frozen=True)
class StubPlan:
    """What each stub reports. Unlisted stages RAN at 0.0. Vision's value is its forgery
    confidence. Rulebook and Auditor use only the status."""

    outcomes: dict[StageName, StageResult] = field(default_factory=dict)

    def outcome(self, stage: StageName) -> StageResult:
        return self.outcomes.get(stage, _DEFAULT)


class AuditorStubOutput(BaseModel):
    """The stub writer's report text."""

    executive_summary: str
    investigation_narrative: str
    compliance_conclusion: str
    str_narrative: str | None = None


@dataclass(frozen=True)
class StubOutcome:
    status: StageStatus
    output: BaseModel | None


def run_stub(stage: StageName, plan: StubPlan) -> StubOutcome:
    """Run one evidence stage's stub (Vision, TransactionPattern, KYC, AdverseMedia, Graph)."""
    result = plan.outcome(stage)
    if result.status != StageStatus.RAN:
        return StubOutcome(result.status, None)
    c = result.confidence
    trail = [_stub_step(stage)]
    builders = {
        StageName.VISION: lambda: Stage1VisionOutput(
            document_type="STUB", forgery_confidence=c, reasoning_trail=trail
        ),
        StageName.TRANSACTION_PATTERN: lambda: Stage2TransactionPatternOutput(
            velocity_summary="STUB", confidence=c, reasoning_trail=trail
        ),
        StageName.KYC: lambda: Stage3KYCOutput(confidence=c, reasoning_trail=trail),
        StageName.ADVERSE_MEDIA: lambda: Stage4AdverseMediaOutput(
            confidence=c, reasoning_trail=trail
        ),
        StageName.GRAPH: lambda: Stage5GraphOutput(confidence=c, reasoning_trail=trail),
    }
    if stage not in builders:
        raise ValueError(f"{stage.value} is not an evidence stage; use its own stub")
    return StubOutcome(StageStatus.RAN, builders[stage]())


def run_rulebook_stub(plan: StubPlan) -> StubOutcome:
    result = plan.outcome(StageName.RULEBOOK)
    if result.status != StageStatus.RAN:
        return StubOutcome(result.status, None)
    return StubOutcome(
        StageStatus.RAN, Stage6RulebookOutput(reasoning_trail=[_stub_step(StageName.RULEBOOK)])
    )


def run_auditor_stub(disposition: Disposition, confidence: float, plan: StubPlan) -> StubOutcome:
    """Placeholder synthesis. It restates the already-computed verdict and adds nothing."""
    result = plan.outcome(StageName.AUDITOR)
    if result.status != StageStatus.RAN:
        return StubOutcome(result.status, None)
    verdict = f"{disposition.value} at confidence {confidence:g}"
    return StubOutcome(
        StageStatus.RAN,
        AuditorStubOutput(
            executive_summary=f"STUB: {verdict}.",
            investigation_narrative=(
                f"STUB - narrative written in {REPLACED_IN[StageName.AUDITOR]}."
            ),
            compliance_conclusion=f"STUB: {verdict}.",
            str_narrative=(
                "STUB - STR draft." if disposition == Disposition.SUSPICIOUS_ESCALATE else None
            ),
        ),
    )


def _stub_step(stage: StageName) -> ReasoningStep:
    return ReasoningStep(
        step=f"{stage.value} stub",
        finding="No analysis performed",
        impact=f"STUB - replaced in {REPLACED_IN[stage]}.",
    )
