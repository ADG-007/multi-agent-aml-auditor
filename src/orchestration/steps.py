"""The case steps both runners share (pipeline.py, case_runner.py).

    triage -> Stage 1 -> 2 -> 3 -> 4 -> 5 -> judge -> admin gate -> Stage 6 -> Stage 7

- A stage that raises is FAILED, never a crash. The judge floors it at MONITOR.
- Stage 1 never reaches the judge. At >= 0.90 the case waits for the admin.
- Stage 7 gets the finished verdict and may cite only Stage 6's clauses.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel

from src.llm.cache import ResponseCache
from src.orchestration.run_state import RunState
from src.rulebook import RulebookLookup
from src.schemas import (
    FORGERY_PAUSE_THRESHOLD,
    STAGE_WEIGHTS,
    Disposition,
    InvestigationReport,
    PipelineStatus,
    Stage0TriageOutput,
    StageName,
    StageStatus,
    Transaction,
    TriageEvent,
    TriageEventKind,
)
from src.scoring import ScoreResult, StageResult, score_case
from src.stages import (
    stage1_vision,
    stage2_transaction_pattern,
    stage3_kyc,
    stage4_adverse_media,
    stage5_graph,
    stage6_rulebook,
    stage7_auditor,
)
from src.stages.stage0_triage import TriageContext, first_alert, triage
from src.stages.stage6_rulebook.agent import RulebookResult, build_queries
from src.stages.stage7_auditor import AuditorResult, CaseBrief
from src.stages.stubs import StubPlan, run_auditor_stub, run_rulebook_stub, run_stub


log = logging.getLogger(__name__)

EVIDENCE_STAGES = (
    StageName.VISION,
    StageName.TRANSACTION_PATTERN,
    StageName.KYC,
    StageName.ADVERSE_MEDIA,
    StageName.GRAPH,
)

StageFn = Callable[[str, TriageContext, datetime | None, Transaction | None, RunState], BaseModel]


def _plain(fn) -> StageFn:
    return lambda account_id, ctx, as_of, txn, state: fn(account_id, ctx, as_of, txn)


#: The real evidence stages. A StubPlan entry still forces a stub.
REAL_STAGES: dict[StageName, StageFn] = {
    StageName.VISION: _plain(stage1_vision.run),
    StageName.TRANSACTION_PATTERN: _plain(stage2_transaction_pattern.run),
    StageName.KYC: _plain(stage3_kyc.run),
    StageName.ADVERSE_MEDIA: _plain(stage4_adverse_media.run),
    StageName.GRAPH: lambda a, c, t, x, s: stage5_graph.run(a, c, t, x, state=s),
}

FINDINGS_FIELD = {
    StageName.VISION: "vision_findings",
    StageName.TRANSACTION_PATTERN: "transaction_pattern_findings",
    StageName.KYC: "kyc_findings",
    StageName.ADVERSE_MEDIA: "adverse_media_findings",
    StageName.GRAPH: "graph_findings",
    StageName.RULEBOOK: "rulebook_findings",
}


@dataclass
class CaseWork:
    """One case in progress. Holds results only; the bank, the run state and the plan are
    passed beside it, so a paused case stays small."""

    event: TriageEvent
    triage_out: Stage0TriageOutput | None = None
    stop: bool = False  # a quiet transaction on a flagged account opens no case
    stage_status: dict[str, StageStatus] = field(default_factory=dict)
    findings: dict[str, BaseModel] = field(default_factory=dict)
    weighted: dict[StageName, StageResult] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    score: ScoreResult | None = None
    admin_decision: str | None = None  # "approve" or "block"
    rulebook: RulebookResult | None = None
    auditor: AuditorResult | None = None

    @property
    def routed(self) -> set[StageName]:
        return set(self.triage_out.routed_stages) if self.triage_out else set()

    @property
    def forgery_confidence(self) -> float | None:
        vision = self.findings.get(FINDINGS_FIELD[StageName.VISION])
        return vision.forgery_confidence if vision is not None else None

    @property
    def needs_admin(self) -> bool:
        c = self.forgery_confidence
        return c is not None and c >= FORGERY_PAUSE_THRESHOLD


# --------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------


def step_triage(work: CaseWork, ctx: TriageContext, state: RunState) -> None:
    work.triage_out = triage(work.event, ctx, state)
    work.stage_status[StageName.TRIAGE.value] = StageStatus.RAN
    work.stop = (
        work.event.kind == TriageEventKind.TRANSACTION
        and not work.triage_out.fired_rules
        and state.is_flagged(work.event.account_id)
    )


def step_evidence(
    work: CaseWork, stage: StageName, ctx: TriageContext, state: RunState, plan: StubPlan
) -> None:
    if stage not in work.routed:
        work.stage_status[stage.value] = StageStatus.NOT_ROUTED
        if stage in STAGE_WEIGHTS:
            work.weighted[stage] = StageResult(StageStatus.NOT_ROUTED)
        return

    if stage in plan.outcomes:
        outcome = run_stub(stage, plan)
        status, output = outcome.status, outcome.output
    else:
        event = work.event
        as_of = (
            event.transaction.timestamp
            if event.kind == TriageEventKind.TRANSACTION
            else ctx.review_date
        )
        try:
            output = REAL_STAGES[stage](event.account_id, ctx, as_of, event.transaction, state)
            status = StageStatus.RAN
        except Exception as exc:  # noqa: BLE001 - failures degrade, they never crash
            log.warning("stage %s failed on %s: %s", stage.value, event.account_id, exc)
            work.errors[stage.value] = f"{type(exc).__name__}: {str(exc)[:200]}"
            status, output = StageStatus.FAILED, None

    work.stage_status[stage.value] = status
    if output is not None:
        work.findings[FINDINGS_FIELD[stage]] = output
    if stage in STAGE_WEIGHTS:
        work.weighted[stage] = StageResult(status, output.confidence if output else None)


def step_judge(work: CaseWork) -> None:
    work.score = score_case(work.weighted)


def step_admin(work: CaseWork, decision: str | None) -> None:
    """Record the admin's decision on a paused case: "approve" or "block"."""
    if decision not in ("approve", "block"):
        raise ValueError(f"admin decision must be 'approve' or 'block', got {decision!r}")
    work.admin_decision = decision


def step_rulebook(work: CaseWork, plan: StubPlan) -> None:
    if StageName.RULEBOOK not in work.routed:
        work.stage_status[StageName.RULEBOOK.value] = StageStatus.NOT_ROUTED
        return
    if StageName.RULEBOOK in plan.outcomes:
        outcome = run_rulebook_stub(plan)
        work.stage_status[StageName.RULEBOOK.value] = outcome.status
        if outcome.output is not None:
            work.findings[FINDINGS_FIELD[StageName.RULEBOOK]] = outcome.output
            work.rulebook = RulebookResult(outcome.output, policy_gap=False)
        return
    f = work.findings
    queries = build_queries(
        vision=f.get("vision_findings"),
        transaction_pattern=f.get("transaction_pattern_findings"),
        kyc=f.get("kyc_findings"),
        adverse_media=f.get("adverse_media_findings"),
        graph=f.get("graph_findings"),
    )
    try:
        work.rulebook = stage6_rulebook.run(queries)
        work.stage_status[StageName.RULEBOOK.value] = StageStatus.RAN
        work.findings[FINDINGS_FIELD[StageName.RULEBOOK]] = work.rulebook.output
    except Exception as exc:  # noqa: BLE001
        log.warning("Rulebook failed on %s: %s", work.event.account_id, exc)
        work.errors[StageName.RULEBOOK.value] = f"{type(exc).__name__}: {str(exc)[:200]}"
        work.stage_status[StageName.RULEBOOK.value] = StageStatus.FAILED


def step_writer(work: CaseWork, ctx: TriageContext, state: RunState, plan: StubPlan) -> None:
    disposition, confidence = final_verdict(work)
    if StageName.AUDITOR in plan.outcomes:
        outcome = run_auditor_stub(disposition, confidence, plan)
        work.stage_status[StageName.AUDITOR.value] = outcome.status
        if outcome.output is not None:
            o = outcome.output
            work.auditor = AuditorResult(
                draft=stage7_auditor.agent.AuditorDraft(
                    executive_summary=o.executive_summary,
                    investigation_narrative=o.investigation_narrative,
                    compliance_conclusion=o.compliance_conclusion,
                    str_narrative=o.str_narrative,
                ),
                status=outcome.status,
                trail=[],
                removed_citations=[],
            )
        return
    brief = case_brief(work, ctx, state, disposition, confidence)
    work.auditor = stage7_auditor.run(
        brief, ctx.writer_mode, cache=ResponseCache("auditor", root=ctx.cache_root)
    )
    work.stage_status[StageName.AUDITOR.value] = work.auditor.status


def final_verdict(work: CaseWork) -> tuple[Disposition, float]:
    if work.admin_decision == "block":
        return Disposition.BLOCKED_BY_ADMIN_REVIEW, work.score.confidence
    return work.score.disposition, work.score.confidence


def case_brief(
    work: CaseWork,
    ctx: TriageContext,
    state: RunState,
    disposition: Disposition,
    confidence: float,
) -> CaseBrief:
    account_id = work.event.account_id
    rb = work.rulebook
    clause_titles = (
        {cid: _clause_title(ctx.rulebook, cid) for cid in rb.output.retrieved_clause_ids}
        if rb
        else {}
    )
    return CaseBrief(
        case_id=state.next_case_id(account_id),
        account_id=account_id,
        disposition=disposition,
        confidence=confidence,
        scoring_steps=[f"{s.step}: {s.finding} -> {s.impact}" for s in work.score.reasoning_trail],
        stage_status={k: v.value for k, v in work.stage_status.items()},
        stage_findings=stage_lines(work),
        clauses=clause_titles,
        policy_gap=bool(rb and rb.policy_gap),
        admin_note=_admin_note(work),
        stage_errors=dict(work.errors),
    )


def stage_lines(work: CaseWork) -> dict[str, list[str]]:
    """A few short lines per evidence stage that ran, for the writer and the template."""
    keep = stage7_auditor.agent.config.TRAIL_STEPS_PER_STAGE
    out: dict[str, list[str]] = {}
    for stage in EVIDENCE_STAGES:
        o = work.findings.get(FINDINGS_FIELD[stage])
        if o is None:
            if stage.value in work.errors:
                out[stage.value] = [f"FAILED: {work.errors[stage.value]}"]
            continue
        if stage == StageName.VISION:
            head = [f"forgery confidence {o.forgery_confidence:g} (never scored)"]
            head += o.field_discrepancies
        else:
            head = [f"confidence {o.confidence:g}"]
            for name in (
                "structuring_indicators",
                "pass_through_indicators",
                "profile_behavior_mismatches",
                "matched_entries",
                "connected_flagged_accounts",
            ):
                values = getattr(o, name, None)
                if values:
                    head.append(f"{name.replace('_', ' ')}: " + "; ".join(map(str, values[:3])))
        trail = [f"{s.step}: {s.finding}" for s in o.reasoning_trail[:keep]]
        out[stage.value] = head + trail
    return out


def build_report(work: CaseWork, ctx: TriageContext, state: RunState) -> InvestigationReport:
    account_id = work.event.account_id
    disposition, confidence = final_verdict(work)
    blocked = work.admin_decision == "block"
    draft = work.auditor.draft if work.auditor else None
    scoring = " ".join(f"{s.step}: {s.finding} -> {s.impact}" for s in work.score.reasoning_trail)
    errors = " ".join(f"{k} failed ({v})." for k, v in work.errors.items())
    narrative = ""
    if draft:
        narrative = f"{draft.investigation_narrative} Scoring - {scoring}"
        if errors:
            narrative += f" Degraded - {errors}"
    report = InvestigationReport(
        case_id=state.next_case_id(account_id),
        triggering_transaction_id=_trigger_id(work, ctx),
        target_account=account_id,
        pipeline_status=PipelineStatus.BLOCKED_BY_ADMIN if blocked else PipelineStatus.COMPLETED,
        stage_status={k: v for k, v in work.stage_status.items()},
        confidence=confidence,
        disposition=disposition,
        degraded_stages=[s.value for s in work.score.degraded_stages],
        evidence_incomplete=work.score.evidence_incomplete,
        single_signal_capped=work.score.single_signal_capped,
        policy_gap_identified=bool(work.rulebook and work.rulebook.policy_gap),
        forgery_notice=_admin_note(work),
        executive_summary=draft.executive_summary if draft else "",
        investigation_narrative=narrative,
        str_narrative=draft.str_narrative if draft else None,
        compliance_conclusion=draft.compliance_conclusion if draft else "",
        triage_findings=work.triage_out,
        judge_trail=work.score.reasoning_trail,
        auditor_trail=work.auditor.trail if work.auditor else [],
        **work.findings,
    )
    return report


def finalise(work: CaseWork, ctx: TriageContext, state: RunState) -> InvestigationReport:
    report = build_report(work, ctx, state)
    state.record_case(report, ctx.bank)
    return report


def _admin_note(work: CaseWork) -> str | None:
    vision = work.findings.get(FINDINGS_FIELD[StageName.VISION])
    if not work.needs_admin:
        return None
    issues = "; ".join(vision.field_discrepancies) or "document fields disagree"
    if work.admin_decision is None:
        return f"ID check paused the case for the admin: {issues}."
    return f"ID check: {issues}. Admin decision: {work.admin_decision.upper()}."


def _trigger_id(work: CaseWork, ctx: TriageContext) -> str | None:
    event = work.event
    if event.kind == TriageEventKind.TRANSACTION:
        return event.transaction.transaction_id
    alert = first_alert(work.triage_out, ctx.bank, event.account_id)
    return alert.transaction_id if alert else None


def _clause_title(rulebook: RulebookLookup, clause_id: str) -> str:
    try:
        return rulebook.clause(clause_id).title
    except KeyError:
        return clause_id
