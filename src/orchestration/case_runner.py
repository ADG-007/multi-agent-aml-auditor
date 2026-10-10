"""A plain sequential case runner - TEMPORARY, replaced by the LangGraph pipeline in Phase 10.

Stage 0 -> the routed stage stubs -> the scoring engine -> the Auditor stub -> the report,
then the case is recorded in RunState (which is where an escalation feeds the known-bad list).

What it deliberately does not do yet (Phase 10): the Vision pause/notice gate, concurrency,
timeouts and retries. Vision runs as a stub here and its forgery confidence never touches the
score, exactly as it never will.
"""

from __future__ import annotations

from src.orchestration.run_state import RunState
from src.schemas import (
    STAGE_WEIGHTS,
    InvestigationReport,
    PipelineStatus,
    StageName,
    StageStatus,
    TriageEvent,
    TriageEventKind,
)
from src.scoring import StageResult, score_case
from src.stages.stage0_triage import TriageContext, first_alert, triage
from src.stages.stubs import StubPlan, run_auditor_stub, run_rulebook_stub, run_stub


EVIDENCE_STAGES = (
    StageName.VISION,
    StageName.TRANSACTION_PATTERN,
    StageName.KYC,
    StageName.ADVERSE_MEDIA,
    StageName.GRAPH,
)

FINDINGS_FIELD = {
    StageName.VISION: "vision_findings",
    StageName.TRANSACTION_PATTERN: "transaction_pattern_findings",
    StageName.KYC: "kyc_findings",
    StageName.ADVERSE_MEDIA: "adverse_media_findings",
    StageName.GRAPH: "graph_findings",
    StageName.RULEBOOK: "rulebook_findings",
}


def run_case(
    event: TriageEvent,
    ctx: TriageContext,
    state: RunState,
    plan: StubPlan | None = None,
) -> InvestigationReport | None:
    """Investigate one event. None when a quiet transaction lands on a flagged account."""
    plan = plan or StubPlan()
    account_id = event.account_id
    triage_out = triage(event, ctx, state)

    if (
        event.kind == TriageEventKind.TRANSACTION
        and not triage_out.fired_rules
        and state.is_flagged(account_id)
    ):
        return None  # joins the account's history; its standing disposition is unchanged

    routed = set(triage_out.routed_stages)
    stage_status: dict[str, StageStatus] = {StageName.TRIAGE.value: StageStatus.RAN}
    findings: dict[str, object] = {"triage_findings": triage_out}
    weighted: dict[StageName, StageResult] = {}

    for stage in EVIDENCE_STAGES:
        if stage not in routed:
            stage_status[stage.value] = StageStatus.NOT_ROUTED
            if stage in STAGE_WEIGHTS:
                weighted[stage] = StageResult(StageStatus.NOT_ROUTED)
            continue
        outcome = run_stub(stage, plan)
        stage_status[stage.value] = outcome.status
        findings[FINDINGS_FIELD[stage]] = outcome.output
        if stage in STAGE_WEIGHTS:
            confidence = outcome.output.confidence if outcome.output else None
            weighted[stage] = StageResult(outcome.status, confidence)

    score = score_case(weighted)

    if StageName.RULEBOOK in routed:
        rulebook = run_rulebook_stub(plan)
        stage_status[StageName.RULEBOOK.value] = rulebook.status
        findings[FINDINGS_FIELD[StageName.RULEBOOK]] = rulebook.output
    else:
        stage_status[StageName.RULEBOOK.value] = StageStatus.NOT_ROUTED

    auditor = run_auditor_stub(score.disposition, score.confidence, plan)
    stage_status[StageName.AUDITOR.value] = auditor.status
    text = auditor.output
    scoring_explained = " ".join(
        f"{s.step}: {s.finding} -> {s.impact}" for s in score.reasoning_trail
    )

    report = InvestigationReport(
        case_id=state.next_case_id(account_id),
        triggering_transaction_id=_trigger_id(event, triage_out, ctx, account_id),
        target_account=account_id,
        pipeline_status=PipelineStatus.COMPLETED,
        stage_status=stage_status,
        confidence=score.confidence,
        disposition=score.disposition,
        degraded_stages=[s.value for s in score.degraded_stages],
        evidence_incomplete=score.evidence_incomplete,
        single_signal_capped=score.single_signal_capped,
        executive_summary=text.executive_summary if text else "",
        investigation_narrative=(
            f"{text.investigation_narrative} Scoring - {scoring_explained}" if text else ""
        ),
        str_narrative=text.str_narrative if text else None,
        compliance_conclusion=text.compliance_conclusion if text else "",
        **findings,
    )
    state.record_case(report, ctx.bank)
    return report


def _trigger_id(event, triage_out, ctx, account_id) -> str | None:
    if event.kind == TriageEventKind.TRANSACTION:
        return event.transaction.transaction_id
    alert = first_alert(triage_out, ctx.bank, account_id)
    return alert.transaction_id if alert else None
