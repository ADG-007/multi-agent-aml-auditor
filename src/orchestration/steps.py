"""The case steps every runner shares (the live graph, the plain runner, the batch).

    triage -> Vision + Web Research (Qwen) -> pattern and link scores -> judge -> admin gate
           -> Graph label -> RAG -> Auditor (Gemma) -> backup re-check (9B, at most once)

- An agent that raises is FAILED, never a crash. The judge floors the case at MONITOR.
- The ID check never reaches the judge. At 0.90 or more the case waits for the admin.
- Explained cash (the Vision agent's rules) sets Stage 2's monthly sub-score aside before the
  judge runs. The single-deposit and pass-through sub-scores stay.
- The Auditor gets the finished verdict and may cite only the RAG agent's clauses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from src.agents.auditor.agent import AuditorAgent, AuditorResult, CaseBrief
from src.agents.auditor.agent import config as auditor_config
from src.agents.base import AgentRun, QueuedJob
from src.agents.graph.agent import GraphAgent, GraphScores
from src.agents.rag.agent import Evidence, RagAgent, RagResult, Searcher
from src.agents.vision.agent import BACKUP_KIND as KYC_BACKUP
from src.agents.vision.agent import VisionAgent, VisionResult
from src.agents.web_research.agent import BACKUP_KIND as WEB_BACKUP
from src.agents.web_research.agent import WebResearchAgent, WebResult
from src.llm import LLMClient, LLMMode, Role
from src.orchestration.run_state import RunState
from src.schemas import (
    FORGERY_NOTICE_THRESHOLD,
    FORGERY_PAUSE_THRESHOLD,
    SCORE_DECIMALS,
    STAGE_WEIGHTS,
    Disposition,
    InvestigationReport,
    PipelineStatus,
    ReasoningStep,
    Stage0TriageOutput,
    Stage6RulebookOutput,
    StageName,
    StageStatus,
    TriageEvent,
    TriageEventKind,
)
from src.scoring import ScoreResult, StageResult, score_case
from src.stages.stage0_triage import TriageContext, first_alert, triage
from src.stages.stubs import StubPlan, run_stub


log = logging.getLogger(__name__)
S = StageName

EVIDENCE_STAGES = (S.VISION, S.TRANSACTION_PATTERN, S.KYC, S.ADVERSE_MEDIA, S.GRAPH)

FINDINGS_FIELD = {
    S.VISION: "vision_findings",
    S.TRANSACTION_PATTERN: "transaction_pattern_findings",
    S.KYC: "kyc_findings",
    S.ADVERSE_MEDIA: "adverse_media_findings",
    S.GRAPH: "graph_findings",
    S.RULEBOOK: "rulebook_findings",
}

RECHECK_KYC, RECHECK_WEB = "KYC_CLAIMS", "SAME_PERSON"
EXPLAINED_CASH = "Explained cash"


@dataclass
class Agents:
    """The five agents of one run, sharing one model client."""

    client: LLMClient
    vision: VisionAgent
    web: WebResearchAgent
    graph: GraphAgent
    rag: RagAgent
    auditor: AuditorAgent
    documents_dir: Path | None = None
    #: Tests pass a fake rulebook search; the pipeline uses the index.
    searcher: Searcher | None = None

    @classmethod
    def build(
        cls,
        client: LLMClient | None = None,
        documents_dir: Path | None = None,
        searcher: Searcher | None = None,
    ) -> Agents:
        client = client or LLMClient(LLMMode.OFF, audit_path=None)
        return cls(
            client=client,
            vision=VisionAgent(client),
            web=WebResearchAgent(client),
            graph=GraphAgent(client),
            rag=RagAgent(client),
            auditor=AuditorAgent(client),
            documents_dir=documents_dir,
            searcher=searcher,
        )


@dataclass
class CaseWork:
    """One case in progress. Holds results only; the bank, the run state and the agents are
    passed beside it."""

    event: TriageEvent
    case_id: str | None = None
    triage_out: Stage0TriageOutput | None = None
    stop: bool = False  # a quiet transaction on a flagged account opens no case
    stage_status: dict[str, StageStatus] = field(default_factory=dict)
    findings: dict[str, BaseModel] = field(default_factory=dict)
    weighted: dict[StageName, StageResult] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    vision: VisionResult | None = None
    web: WebResult | None = None
    graph: GraphScores | None = None
    rag: RagResult | None = None
    auditor: AuditorResult | None = None
    score: ScoreResult | None = None
    judge_inputs: dict[str, float] = field(default_factory=dict)
    judge_notes: list[ReasoningStep] = field(default_factory=list)
    admin_decision: str | None = None  # "approve" or "block"
    rechecked: bool = False

    @property
    def account_id(self) -> str:
        return self.event.account_id

    @property
    def routed(self) -> set[StageName]:
        return set(self.triage_out.routed_stages) if self.triage_out else set()

    @property
    def forgery_confidence(self) -> float | None:
        vision = self.findings.get(FINDINGS_FIELD[S.VISION])
        return vision.forgery_confidence if vision is not None else None

    @property
    def needs_admin(self) -> bool:
        c = self.forgery_confidence
        return c is not None and c >= FORGERY_PAUSE_THRESHOLD

    @property
    def runs(self) -> dict[str, AgentRun]:
        found = {
            "vision": self.vision.run if self.vision else None,
            "web_research": self.web.run if self.web else None,
            "graph": self.graph.run if self.graph else None,
            "rag": self.rag.run if self.rag else None,
            "auditor": self.auditor.run if self.auditor else None,
        }
        return {name: run for name, run in found.items() if run is not None}

    @property
    def queued(self) -> list[QueuedJob]:
        return [job for run in self.runs.values() for job in run.queued]

    def as_of(self, ctx: TriageContext) -> datetime | None:
        if self.event.kind == TriageEventKind.TRANSACTION:
            return self.event.transaction.timestamp
        return ctx.review_date


# --------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------


def step_triage(work: CaseWork, ctx: TriageContext, state: RunState) -> None:
    work.triage_out = triage(work.event, ctx, state)
    work.stage_status[S.TRIAGE.value] = StageStatus.RAN
    work.stop = (
        work.event.kind == TriageEventKind.TRANSACTION
        and not work.triage_out.fired_rules
        and state.is_flagged(work.account_id)
    )


def run_vision(work: CaseWork, ctx: TriageContext, agents: Agents) -> None:
    """The Vision agent's model work (ID check and KYC claims). Order-free: one account only."""
    try:
        work.vision = agents.vision.run(
            work.account_id,
            ctx,
            work.as_of(ctx),
            work.event.transaction,
            documents_dir=agents.documents_dir,
        )
    except Exception as exc:  # noqa: BLE001 - failures degrade, they never crash
        log.warning("Vision agent failed on %s: %s", work.account_id, exc)
        work.errors["VisionAgent"] = f"{type(exc).__name__}: {str(exc)[:200]}"


def run_web(work: CaseWork, ctx: TriageContext, agents: Agents) -> None:
    """The Web Research agent. Order-free: one account only."""
    try:
        work.web = agents.web.run(work.account_id, ctx, work.as_of(ctx), work.event.transaction)
    except Exception as exc:  # noqa: BLE001
        log.warning("Web Research agent failed on %s: %s", work.account_id, exc)
        work.errors["WebResearchAgent"] = f"{type(exc).__name__}: {str(exc)[:200]}"


def step_vision(work: CaseWork, ctx: TriageContext, agents: Agents, plan: StubPlan) -> None:
    """Record the Vision agent's two results (ID check, KYC) for the routed stages."""
    need = {S.VISION, S.KYC} & work.routed
    forced = {s for s in need if s in plan.outcomes}
    if need - forced and work.vision is None and "VisionAgent" not in work.errors:
        run_vision(work, ctx, agents)
    v = work.vision
    for stage in (S.VISION, S.KYC):
        if stage not in work.routed:
            _set(work, stage, StageStatus.NOT_ROUTED, None)
        elif stage in forced:
            _stub(work, stage, plan)
        elif v is None:
            _fail(work, stage, work.errors.get("VisionAgent", "the Vision agent did not run"))
        elif stage == S.VISION:
            if v.vision is None:
                _fail(work, stage, v.id_error or "ID check failed")
            else:
                _set(work, stage, StageStatus.RAN, v.vision)
        elif v.kyc is None:
            _fail(work, stage, v.kyc_error or "KYC failed")
        elif v.claims.unread:
            # No claim could be read: the Python rules ran, but the stage is not complete.
            work.findings[FINDINGS_FIELD[stage]] = v.kyc
            _fail(work, stage, "both claim reads failed", keep_findings=True)
        else:
            _set(work, stage, StageStatus.RAN, v.kyc)


def step_web(work: CaseWork, ctx: TriageContext, agents: Agents, plan: StubPlan) -> None:
    stage = S.ADVERSE_MEDIA
    if stage not in work.routed:
        _set(work, stage, StageStatus.NOT_ROUTED, None)
    elif stage in plan.outcomes:
        _stub(work, stage, plan)
    else:
        if work.web is None and "WebResearchAgent" not in work.errors:
            run_web(work, ctx, agents)
        if work.web is None:
            _fail(work, stage, work.errors.get("WebResearchAgent", "the agent did not run"))
        else:
            _set(work, stage, StageStatus.RAN, work.web.output)


def step_scores(
    work: CaseWork, ctx: TriageContext, state: RunState, agents: Agents, plan: StubPlan
) -> None:
    """The Graph agent's Python tools: Transaction Pattern and Graph confidences."""
    need = {S.TRANSACTION_PATTERN, S.GRAPH} & work.routed
    forced = {s for s in need if s in plan.outcomes}
    error = ""
    if need - forced:
        try:
            work.graph = agents.graph.score(
                work.account_id, ctx, work.as_of(ctx), work.event.transaction, state
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Graph agent failed on %s: %s", work.account_id, exc)
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
    for stage in (S.TRANSACTION_PATTERN, S.GRAPH):
        if stage not in work.routed:
            _set(work, stage, StageStatus.NOT_ROUTED, None)
        elif stage in forced:
            _stub(work, stage, plan)
        elif work.graph is None:
            _fail(work, stage, error or "the Graph agent did not run")
        elif stage == S.TRANSACTION_PATTERN:
            _set(work, stage, StageStatus.RAN, work.graph.transaction_pattern)
        else:
            _set(work, stage, StageStatus.RAN, work.graph.graph)


def step_judge(work: CaseWork) -> None:
    """The judge. Explained cash sets the monthly sub-score aside first (plan D13, D14)."""
    work.judge_notes = []
    inputs = dict(work.weighted)
    tp = inputs.get(S.TRANSACTION_PATTERN)
    explained = bool(work.vision and work.vision.cash_explained and work.vision.kyc is not None)
    kyc_ran = work.stage_status.get(S.KYC.value) == StageStatus.RAN
    if (
        tp is not None
        and tp.status == StageStatus.RAN
        and explained
        and kyc_ran
        and work.graph is not None
        and work.graph.sub.monthly > 0
    ):
        sub = work.graph.sub
        kept = round(max(sub.structuring, sub.pass_through), SCORE_DECIMALS)
        if kept < tp.confidence:
            inputs[S.TRANSACTION_PATTERN] = StageResult(StageStatus.RAN, kept)
            work.judge_notes.append(
                ReasoningStep(
                    step=EXPLAINED_CASH,
                    finding=f"{work.vision.verdict.cash_note} Monthly-cash sub-score "
                    f"{sub.monthly:g} set aside; single deposits {sub.structuring:g} and "
                    f"pass-through {sub.pass_through:g} stay.",
                    impact=f"Transaction Pattern enters the judge at {kept:g}, not "
                    f"{tp.confidence:g}.",
                )
            )
    work.judge_inputs = {
        stage.value: result.confidence
        for stage, result in inputs.items()
        if result.status == StageStatus.RAN
    }
    work.score = score_case(inputs)


def step_admin(work: CaseWork, decision: str | None) -> None:
    """Record the admin's decision on a paused case: "approve" or "block"."""
    if decision not in ("approve", "block"):
        raise ValueError(f"admin decision must be 'approve' or 'block', got {decision!r}")
    work.admin_decision = decision


def step_label(
    work: CaseWork,
    ctx: TriageContext,
    state: RunState,
    agents: Agents,
    role: Role | None = None,
) -> None:
    """The Graph agent's label. Never scored; a failure leaves the Python label."""
    if work.graph is None:
        return
    try:
        set_aside = any(note.step == EXPLAINED_CASH for note in work.judge_notes)
        agents.graph.label(work.graph, ctx, state, role=role, cash_explained=set_aside)
    except Exception as exc:  # noqa: BLE001
        log.warning("Graph label failed on %s: %s", work.account_id, exc)
        work.errors["GraphLabel"] = f"{type(exc).__name__}: {str(exc)[:200]}"


def step_rag(work: CaseWork, agents: Agents, plan: StubPlan) -> None:
    stage = S.RULEBOOK
    if stage not in work.routed:
        work.stage_status[stage.value] = StageStatus.NOT_ROUTED
        return
    if stage in plan.outcomes:
        status = plan.outcome(stage).status
        work.stage_status[stage.value] = status
        if status == StageStatus.RAN:
            work.findings[FINDINGS_FIELD[stage]] = Stage6RulebookOutput()
        return
    f = work.findings
    evidence = Evidence(
        vision=f.get("vision_findings"),
        transaction_pattern=f.get("transaction_pattern_findings"),
        kyc=f.get("kyc_findings"),
        adverse_media=f.get("adverse_media_findings"),
        graph=f.get("graph_findings"),
        graph_label=work.graph.label if work.graph else None,
        judge_inputs=dict(work.judge_inputs),
        id_flagged=work.needs_admin,
    )
    try:
        work.rag = agents.rag.run(work.account_id, evidence, agents.searcher)
        work.stage_status[stage.value] = StageStatus.RAN
        work.findings[FINDINGS_FIELD[stage]] = work.rag.output
    except Exception as exc:  # noqa: BLE001
        log.warning("RAG agent failed on %s: %s", work.account_id, exc)
        work.errors[stage.value] = f"{type(exc).__name__}: {str(exc)[:200]}"
        work.stage_status[stage.value] = StageStatus.FAILED


def step_auditor(
    work: CaseWork, ctx: TriageContext, state: RunState, agents: Agents, plan: StubPlan
) -> None:
    stage = S.AUDITOR
    if stage in plan.outcomes and plan.outcome(stage).status != StageStatus.RAN:
        work.stage_status[stage.value] = plan.outcome(stage).status
        return
    brief = case_brief(work, ctx, state)
    try:
        work.auditor = agents.auditor.run(brief, allow_recheck=not work.rechecked)
        work.stage_status[stage.value] = work.auditor.status
    except Exception as exc:  # noqa: BLE001
        log.warning("Auditor failed on %s: %s", work.account_id, exc)
        work.errors[stage.value] = f"{type(exc).__name__}: {str(exc)[:200]}"
        work.stage_status[stage.value] = StageStatus.FAILED


def step_backup(work: CaseWork, ctx: TriageContext, state: RunState, agents: Agents) -> bool:
    """Run the case's queued jobs on the backup model, once. True when a judge input changed."""
    if work.rechecked or not work.queued or agents.client.off:
        return False
    work.rechecked = True
    kinds = {job.kind for job in work.queued}
    changed = False
    if work.vision is not None and work.vision.kyc is not None:
        if kinds & {KYC_BACKUP, f"Re-check {RECHECK_KYC}", "Claim read 1", "Claim read 2"}:
            before = (work.vision.kyc.confidence, work.vision.cash_explained)
            try:
                agents.vision.recheck(work.vision, ctx, work.as_of(ctx))
            except Exception as exc:  # noqa: BLE001
                work.errors["Backup"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            if before != (work.vision.kyc.confidence, work.vision.cash_explained):
                changed = True
            work.findings[FINDINGS_FIELD[S.KYC]] = work.vision.kyc
            if work.stage_status.get(S.KYC.value) in (StageStatus.RAN, StageStatus.FAILED):
                if not work.vision.claims.unread or work.vision.run.model_calls:
                    work.stage_status[S.KYC.value] = StageStatus.RAN
                    work.weighted[S.KYC] = StageResult(StageStatus.RAN, work.vision.kyc.confidence)
    if work.web is not None and any(k == WEB_BACKUP or RECHECK_WEB in k for k in kinds):
        before = work.web.output.confidence
        try:
            again = agents.web.run(
                work.account_id, ctx, work.as_of(ctx), work.event.transaction, role=Role.BACKUP
            )
            again.run.trail = work.web.run.trail + again.run.trail
            work.web = again
        except Exception as exc:  # noqa: BLE001
            work.errors["Backup"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        if work.stage_status.get(S.ADVERSE_MEDIA.value) == StageStatus.RAN:
            work.findings[FINDINGS_FIELD[S.ADVERSE_MEDIA]] = work.web.output
            work.weighted[S.ADVERSE_MEDIA] = StageResult(
                StageStatus.RAN, work.web.output.confidence
            )
            changed = changed or before != work.web.output.confidence
    if "Pattern label" in kinds:
        step_label(work, ctx, state, agents, role=Role.BACKUP)
    return changed


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def final_verdict(work: CaseWork) -> tuple[Disposition, float]:
    if work.admin_decision == "block":
        return Disposition.BLOCKED_BY_ADMIN_REVIEW, work.score.confidence
    return work.score.disposition, work.score.confidence


def judge_trail(work: CaseWork) -> list[ReasoningStep]:
    return [*work.judge_notes, *work.score.reasoning_trail]


def case_brief(work: CaseWork, ctx: TriageContext, state: RunState) -> CaseBrief:
    disposition, confidence = final_verdict(work)
    clauses: dict[str, tuple[str, str]] = {}
    if work.rag is not None:
        for applied in work.rag.applied:
            clauses[applied.clause_id] = (applied.title, applied.quote)
    else:
        rb = work.findings.get(FINDINGS_FIELD[S.RULEBOOK])
        for cid in rb.retrieved_clause_ids if rb else []:
            clauses[cid] = (_clause_title(ctx, cid), "")
    possible = []
    if work.vision is not None and work.vision.kyc is not None and work.vision.run.model_calls:
        possible.append(RECHECK_KYC)
    if work.web is not None and work.web.run.model_calls:
        possible.append(RECHECK_WEB)
    return CaseBrief(
        case_id=work.case_id or state.next_case_id(work.account_id),
        account_id=work.account_id,
        disposition=disposition,
        confidence=confidence,
        scoring_steps=[f"{s.step}: {s.finding} -> {s.impact}" for s in judge_trail(work)],
        stage_status={k: v.value for k, v in work.stage_status.items()},
        findings=stage_lines(work),
        clauses=clauses,
        policy_gap=bool(work.rag and work.rag.policy_gap),
        admin_note=admin_note(work),
        graph_label=work.graph.label if work.graph else None,
        rechecks_possible=tuple(possible),
    )


def stage_lines(work: CaseWork) -> dict[str, list[str]]:
    """A few short lines per stage with something to say, for the Auditor and the template."""
    keep = auditor_config.LINES_PER_AGENT
    out: dict[str, list[str]] = {}
    for stage in EVIDENCE_STAGES:
        o = work.findings.get(FINDINGS_FIELD[stage])
        if o is None:
            if stage.value in work.errors:
                out[stage.value] = [f"FAILED: {work.errors[stage.value]}"]
            continue
        if stage == S.VISION:
            if o.forgery_confidence <= 0:
                continue
            lines = [f"ID check {o.forgery_confidence:g} (never scored)", *o.field_discrepancies]
        else:
            if o.confidence <= 0 and not (
                stage == S.KYC and work.vision and work.vision.verdict.cash_note
            ):
                continue
            lines = [f"confidence {o.confidence:g}"]
            for name in (
                "structuring_indicators",
                "pass_through_indicators",
                "profile_behavior_mismatches",
                "matched_entries",
            ):
                lines += [str(v) for v in getattr(o, name, [])[:3]]
            if stage == S.KYC and work.vision and work.vision.verdict.cash_note:
                lines.append(work.vision.verdict.cash_note)
            if stage == S.GRAPH:
                lines.append(o.reasoning_trail[-1].finding)
                lines += o.shared_identifiers[:2]
        out[stage.value] = lines[:keep]
    if work.graph is not None and work.graph.label != "NO_PATTERN":
        out["Pattern"] = [f"label {work.graph.label} (never scored)"]
    return out


def build_report(work: CaseWork, ctx: TriageContext, state: RunState) -> InvestigationReport:
    disposition, confidence = final_verdict(work)
    blocked = work.admin_decision == "block"
    draft = work.auditor.draft if work.auditor else None
    errors = " ".join(f"{k} failed ({v})." for k, v in work.errors.items())
    narrative = draft.investigation_narrative if draft else ""
    if errors:
        narrative = f"{narrative} Degraded - {errors}".strip()
    c = work.forgery_confidence
    notice = admin_note(work)
    if notice is None and c is not None and c >= FORGERY_NOTICE_THRESHOLD:
        vision = work.findings[FINDINGS_FIELD[S.VISION]]
        notice = "ID check notice (no pause): " + "; ".join(vision.field_discrepancies)
    return InvestigationReport(
        case_id=work.case_id or state.next_case_id(work.account_id),
        triggering_transaction_id=_trigger_id(work, ctx),
        target_account=work.account_id,
        pipeline_status=PipelineStatus.BLOCKED_BY_ADMIN if blocked else PipelineStatus.COMPLETED,
        stage_status=dict(work.stage_status),
        confidence=confidence,
        disposition=disposition,
        degraded_stages=[s.value for s in work.score.degraded_stages],
        evidence_incomplete=work.score.evidence_incomplete,
        single_signal_capped=work.score.single_signal_capped,
        policy_gap_identified=bool(work.rag and work.rag.policy_gap),
        forgery_notice=notice,
        executive_summary=draft.executive_summary if draft else "",
        investigation_narrative=narrative,
        str_narrative=draft.str_narrative if draft else None,
        compliance_conclusion=draft.compliance_conclusion if draft else "",
        triage_findings=work.triage_out,
        judge_inputs=dict(work.judge_inputs),
        judge_trail=judge_trail(work),
        graph_label=work.graph.label if work.graph else None,
        agent_trails={name: list(run.trail) for name, run in work.runs.items()},
        report_outline=list(work.auditor.outline) if work.auditor else [],
        **work.findings,
    )


def finalise(work: CaseWork, ctx: TriageContext, state: RunState) -> InvestigationReport:
    report = build_report(work, ctx, state)
    state.record_case(report, ctx.bank)
    work.case_id = report.case_id
    return report


def admin_note(work: CaseWork) -> str | None:
    if not work.needs_admin:
        return None
    vision = work.findings.get(FINDINGS_FIELD[S.VISION])
    issues = "; ".join(vision.field_discrepancies) or "document fields disagree"
    if work.admin_decision is None:
        return f"ID check paused the case for the admin: {issues}."
    return f"ID check: {issues}. Admin decision: {work.admin_decision.upper()}."


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _set(work: CaseWork, stage: StageName, status: StageStatus, output: BaseModel | None) -> None:
    work.stage_status[stage.value] = status
    if output is not None:
        work.findings[FINDINGS_FIELD[stage]] = output
    if stage in STAGE_WEIGHTS:
        confidence = output.confidence if output is not None else None
        work.weighted[stage] = StageResult(status, confidence)


def _fail(work: CaseWork, stage: StageName, why: str, keep_findings: bool = False) -> None:
    work.errors.setdefault(stage.value, why)
    work.stage_status[stage.value] = StageStatus.FAILED
    if not keep_findings:
        work.findings.pop(FINDINGS_FIELD[stage], None)
    if stage in STAGE_WEIGHTS:
        work.weighted[stage] = StageResult(StageStatus.FAILED)


def _stub(work: CaseWork, stage: StageName, plan: StubPlan) -> None:
    outcome = run_stub(stage, plan)
    work.stage_status[stage.value] = outcome.status
    if outcome.output is not None:
        work.findings[FINDINGS_FIELD[stage]] = outcome.output
    if stage in STAGE_WEIGHTS:
        confidence = outcome.output.confidence if outcome.output else None
        work.weighted[stage] = StageResult(outcome.status, confidence)


def _trigger_id(work: CaseWork, ctx: TriageContext) -> str | None:
    if work.event.kind == TriageEventKind.TRANSACTION:
        return work.event.transaction.transaction_id
    alert = first_alert(work.triage_out, ctx.bank, work.account_id)
    return alert.transaction_id if alert else None


def _clause_title(ctx: TriageContext, clause_id: str) -> str:
    try:
        return ctx.rulebook.clause(clause_id).title
    except KeyError:
        return clause_id
