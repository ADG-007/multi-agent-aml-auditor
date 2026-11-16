"""Batch mode: many periodic reviews with the models grouped into phases.

    1. Vision + Web Research for every account            Qwen
    2. Judge, pattern and link scores, re-screening rounds Python
    3. Admin queue (ID check at 0.90 or more)              the admin, or the default decision
    4. Graph label, RAG, Auditor                           Gemma
    5. Queued jobs (unsure, schema failed twice, re-checks) the 9B backup
    6. Judge again; new label, clauses and text for the changed cases only

Phases 1 and 4-5 read one account at a time, so their order does not matter and every answer
is cached. Phase 2 is ordered: earliest alerting transaction first, quiet accounts last, then
re-screening rounds until no account's links or known-bad hits change (at most MAX_ROUNDS).
An account's verdict is its latest case. Batch reports are drafts until the analyst signs off.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.agents.graph import links
from src.orchestration.run_state import RunState
from src.orchestration.steps import (
    Agents,
    CaseWork,
    build_report,
    run_vision,
    run_web,
    step_admin,
    step_auditor,
    step_backup,
    step_judge,
    step_label,
    step_rag,
    step_scores,
    step_triage,
    step_vision,
    step_web,
)
from src.schemas import Disposition, InvestigationReport, TriageEvent, TriageEventKind, TriageRule
from src.stages.stage0_triage import TriageContext, first_alert, triage
from src.stages.stubs import StubPlan


MAX_ROUNDS = 6
BACKUP_SHARE_WARNING = 0.10

AdminPolicy = str | Callable[[CaseWork], str]
Progress = Callable[[str], None]


@dataclass
class BatchResult:
    order: list[str]
    latest: dict[str, InvestigationReport] = field(default_factory=dict)
    cases: list[InvestigationReport] = field(default_factory=list)
    rounds: list[list[str]] = field(default_factory=list)  # accounts judged in each round
    admin_queue: list[dict] = field(default_factory=list)
    backup_accounts: list[str] = field(default_factory=list)
    changed_by_backup: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    phase_seconds: dict[str, float] = field(default_factory=dict)
    state: RunState = field(default_factory=RunState)

    @property
    def backup_share(self) -> float:
        return len(self.backup_accounts) / len(self.order) if self.order else 0.0


def review(account_id: str) -> TriageEvent:
    return TriageEvent(kind=TriageEventKind.PERIODIC_REVIEW, account_id=account_id)


def review_order(ctx: TriageContext, account_ids: Iterable[str]) -> list[str]:
    """Earliest alerting transaction first, ties by id, accounts nothing alerted on last."""
    probe = RunState()
    keyed: list[tuple[int, datetime, str]] = []
    for account_id in account_ids:
        alert = first_alert(triage(review(account_id), ctx, probe), ctx.bank, account_id)
        keyed.append((0, alert.timestamp, account_id) if alert else (1, datetime.min, account_id))
    return [a for *_, a in sorted(keyed)]


def run_batch(
    ctx: TriageContext,
    account_ids: Iterable[str],
    agents: Agents,
    admin: AdminPolicy = "approve",
    progress: Progress | None = None,
    plan: StubPlan | None = None,
    judge_only: bool = False,
) -> BatchResult:
    """`judge_only` stops after phase 3: verdicts without labels, clauses or report text."""
    plan = plan or StubPlan()
    say = progress or (lambda line: None)
    result = BatchResult(order=review_order(ctx, account_ids))
    clock = _Clock(result)

    # Phase 1: the Qwen agents, one account at a time.
    pre: dict[str, CaseWork] = {}
    for n, account_id in enumerate(result.order, start=1):
        work = CaseWork(event=review(account_id))
        run_vision(work, ctx, agents)
        run_web(work, ctx, agents)
        pre[account_id] = work
        say(f"phase 1 {n}/{len(result.order)} {account_id}")
    agents.client.unload()
    clock.mark("1 Vision + Web Research")

    # Phase 2: the ordered judge pass.
    works = _judge_pass(ctx, result, pre, agents, plan)
    clock.mark("2 Judge")

    # Phase 3: the admin queue.
    latest = _latest(works)
    for account_id, work in latest.items():
        if work.needs_admin:
            step_admin(work, admin(work) if callable(admin) else admin)
            vision = work.findings["vision_findings"]
            result.admin_queue.append(
                {
                    "account_id": account_id,
                    "case_id": work.case_id,
                    "forgery_confidence": vision.forgery_confidence,
                    "field_discrepancies": vision.field_discrepancies,
                    "disposition_if_approved": work.score.disposition.value,
                    "decision": work.admin_decision,
                }
            )
    clock.mark("3 Admin queue")
    if judge_only:
        for work in works:
            report = build_report(work, ctx, result.state).model_copy(update={"is_draft": True})
            result.cases.append(report)
            result.latest[work.account_id] = report
        return result

    # Phase 4: the Gemma agents on each account's latest case.
    for n, (account_id, work) in enumerate(latest.items(), start=1):
        _write(work, ctx, result.state, agents, plan)
        say(f"phase 4 {n}/{len(latest)} {account_id}")
    agents.client.unload()
    clock.mark("4 Graph label + RAG + Auditor")

    # Phase 5: queued jobs on the backup model.
    changed: list[str] = []
    for account_id, work in latest.items():
        if work.queued and not work.rechecked and not agents.client.off:
            result.backup_accounts.append(account_id)
            if step_backup(work, ctx, result.state, agents):
                changed.append(account_id)
                pre[account_id].vision, pre[account_id].web = work.vision, work.web
            say(f"phase 5 backup {account_id}")
    agents.client.unload()
    clock.mark("5 Backup queue")
    if result.backup_share > BACKUP_SHARE_WARNING:
        result.warnings.append(
            f"{len(result.backup_accounts)} of {len(result.order)} cases "
            f"({result.backup_share:.0%}) went to the backup model (limit "
            f"{BACKUP_SHARE_WARNING:.0%})."
        )

    # Phase 6: judge again; write again only where the verdict's inputs moved.
    if changed:
        result.changed_by_backup = changed
        before = {a: _fingerprint(w) for a, w in latest.items()}
        decisions = {a: w.admin_decision for a, w in latest.items()}
        written = {a: w for a, w in latest.items()}
        result.state = RunState()
        result.rounds = []
        works = _judge_pass(ctx, result, pre, agents, plan)
        latest = _latest(works)
        for account_id, work in latest.items():
            work.rechecked = True
            if work.needs_admin and decisions.get(account_id):
                step_admin(work, decisions[account_id])
            if _fingerprint(work) == before.get(account_id):
                old = written[account_id]
                work.rag, work.auditor = old.rag, old.auditor
                work.graph.label, work.graph.label_by = old.graph.label, old.graph.label_by
                for key in ("Rulebook", "Auditor"):
                    if key in old.stage_status:
                        work.stage_status[key] = old.stage_status[key]
                if old.rag is not None:
                    work.findings["rulebook_findings"] = old.rag.output
            else:
                _write(work, ctx, result.state, agents, plan)
                say(f"phase 6 rewrite {account_id}")
        agents.client.unload()
    clock.mark("6 Judge again")

    for work in works:
        report = build_report(work, ctx, result.state).model_copy(update={"is_draft": True})
        result.cases.append(report)
        result.latest[work.account_id] = report
    return result


def _write(
    work: CaseWork, ctx: TriageContext, state: RunState, agents: Agents, plan: StubPlan
) -> None:
    step_label(work, ctx, state, agents)
    step_rag(work, agents, plan)
    step_auditor(work, ctx, state, agents, plan)


def _judge_pass(
    ctx: TriageContext,
    result: BatchResult,
    pre: dict[str, CaseWork],
    agents: Agents,
    plan: StubPlan,
) -> list[CaseWork]:
    """Round 1 judges every account in order; later rounds re-screen until quiet."""
    works: list[CaseWork] = []
    latest: dict[str, CaseWork] = {}
    before: dict[str, RunState] = {}
    state = result.state

    def one(account_id: str) -> None:
        work = CaseWork(event=review(account_id))
        work.vision, work.web = pre[account_id].vision, pre[account_id].web
        work.errors.update(pre[account_id].errors)
        snapshot = state.copy()
        step_triage(work, ctx, state)
        step_vision(work, ctx, agents, plan)
        step_web(work, ctx, agents, plan)
        step_scores(work, ctx, state, agents, plan)
        step_judge(work)
        report = build_report(work, ctx, state)
        state.record_case(report, ctx.bank)
        work.case_id = report.case_id
        before[report.case_id] = snapshot
        works.append(work)
        latest[account_id] = work

    for account_id in result.order:
        one(account_id)
    result.rounds.append(list(result.order))

    graph = links.build_graph(ctx.bank)  # a review adds no transaction: one graph serves the run
    for _ in range(MAX_ROUNDS):
        escalated = {
            a for a, w in latest.items() if w.score.disposition == Disposition.SUSPICIOUS_ESCALATE
        }
        rerun = [
            a
            for a in result.order
            if a not in escalated
            and (
                _new_hit(a, ctx, state, latest[a])
                or _new_link(a, graph, state, before[latest[a].case_id])
            )
        ]
        if not rerun:
            break
        for account_id in rerun:
            one(account_id)
        result.rounds.append(rerun)
    return works


def _latest(works: list[CaseWork]) -> dict[str, CaseWork]:
    return {work.account_id: work for work in works}


def _fingerprint(work: CaseWork) -> tuple:
    """What the written report depends on: the verdict and what the judge was given."""
    return (
        work.score.disposition,
        work.score.confidence,
        tuple(sorted(work.judge_inputs.items())),
        tuple(sorted((k, v.value) for k, v in work.stage_status.items() if k != "Auditor")),
    )


def _new_hit(account_id: str, ctx: TriageContext, state: RunState, work: CaseWork) -> bool:
    """A known-bad identifier fires now that did not fire on the account's latest case."""

    def known_bad(fired) -> set[str]:
        return {f.detail for f in fired if f.rule == TriageRule.KNOWN_BAD_IDENTIFIER}

    now = known_bad(triage(review(account_id), ctx, state).fired_rules)
    return bool(now - known_bad(work.triage_out.fired_rules))


def _new_link(account_id: str, graph, state: RunState, then: RunState) -> bool:
    """A linked account is flagged on its own now but was not when the latest case ran."""
    return any(
        state.is_flagged_on_own(a) and not then.is_flagged_on_own(a)
        for a in links.links(graph, account_id)
    )


class _Clock:
    def __init__(self, result: BatchResult) -> None:
        self.result = result
        self.last = datetime.now()

    def mark(self, phase: str) -> None:
        now = datetime.now()
        self.result.phase_seconds[phase] = round((now - self.last).total_seconds(), 1)
        self.last = now


def save(result: BatchResult, out_dir: Path, client_calls: list[dict] | None = None) -> None:
    """reports.jsonl (latest case per account), cases.jsonl, admin_queue.json, summary.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "reports.jsonl").open("w", encoding="utf-8") as f:
        for account_id in sorted(result.latest):
            f.write(result.latest[account_id].model_dump_json() + "\n")
    with (out_dir / "cases.jsonl").open("w", encoding="utf-8") as f:
        for report in result.cases:
            f.write(report.model_dump_json() + "\n")
    (out_dir / "admin_queue.json").write_text(
        json.dumps(result.admin_queue, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = {
        "accounts": len(result.order),
        "cases": len(result.cases),
        "rounds": [len(r) for r in result.rounds],
        "dispositions": _count(r.disposition.value for r in result.latest.values()),
        "admin_queue": len(result.admin_queue),
        "backup_accounts": result.backup_accounts,
        "backup_share": round(result.backup_share, 4),
        "changed_by_backup": result.changed_by_backup,
        "warnings": result.warnings,
        "phase_seconds": result.phase_seconds,
        "model_calls": call_stats(client_calls or []),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def load_cases(folder: Path) -> list[InvestigationReport]:
    """Every case of a saved run, in the order the run judged them."""
    with (folder / "cases.jsonl").open(encoding="utf-8") as f:
        return [InvestigationReport.model_validate_json(line) for line in f if line.strip()]


def call_stats(calls: list[dict]) -> dict[str, dict]:
    """Per agent and model: calls, failures, and seconds per call (as measured when made)."""
    stats: dict[str, dict] = {}
    for c in calls:
        key = f"{c['agent']} ({c['model']})"
        row = stats.setdefault(
            key, {"calls": 0, "from_cache": 0, "failed": 0, "seconds": 0.0, "retried": 0}
        )
        row["calls"] += 1
        row["from_cache"] += bool(c["cached"])
        row["failed"] += not c["ok"]
        row["retried"] += c["attempts"] > 1
        row["seconds"] += c["seconds"] if c["ok"] else 0.0
    for row in stats.values():
        done = row["calls"] - row["failed"]
        row["seconds"] = round(row["seconds"], 1)
        row["seconds_per_call"] = round(row["seconds"] / done, 2) if done else None
    return stats


def _count(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))
