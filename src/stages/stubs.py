"""Placeholder Stages 1-7, so the pipeline runs end to end before any agent exists.

Each stub returns a schema-valid output for its stage with whatever status and confidence the
StubPlan gives it (default: RAN at 0.0), and its reasoning trail names the phase that replaces
it. Stubs never look at ground truth - a stub that "knew the answer" would make every
downstream number meaningless (phase_2_execution.md D9).
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


#: The phase that replaces each stub with the real stage.
REPLACED_IN: dict[StageName, str] = {
    StageName.VISION: "Phase 5",
    StageName.TRANSACTION_PATTERN: "Phase 3",
    StageName.KYC: "Phase 4",
    StageName.ADVERSE_MEDIA: "Phase 6",
    StageName.GRAPH: "Phase 7",
    StageName.RULEBOOK: "Phase 8",
    StageName.AUDITOR: "Phase 9",
}

_DEFAULT = StageResult(StageStatus.RAN, 0.0)


@dataclass(frozen=True)
class StubPlan:
    """What each stub reports. Unlisted stages RAN at 0.0. For Vision the confidence is the
    forgery confidence; Rulebook and Auditor have none, so only their status is used."""

    outcomes: dict[StageName, StageResult] = field(default_factory=dict)

    def outcome(self, stage: StageName) -> StageResult:
        return self.outcomes.get(stage, _DEFAULT)


class AuditorStubOutput(BaseModel):
    """The report text Stage 7 will write in Phase 9."""

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
                "STUB - STR draft written in Phase 9."
                if disposition == Disposition.SUSPICIOUS_ESCALATE
                else None
            ),
        ),
    )


def _stub_step(stage: StageName) -> ReasoningStep:
    return ReasoningStep(
        step=f"{stage.value} stub",
        finding="No analysis performed",
        impact=f"STUB - replaced in {REPLACED_IN[stage]}.",
    )
