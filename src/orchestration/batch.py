"""A batch of periodic reviews in a fixed order, then re-screening rounds.

1. Order: earliest alerting transaction first, ties by id, quiet accounts last.
2. Round 1 reviews every account.
3. Re-screen accounts not escalated: a new case when a new known-bad identifier fires, or a
   linked account became flagged on its own evidence. Repeat until quiet (<= MAX_ROUNDS).

An account's verdict is its latest case. `before` keeps the state before each case, so the
page can replay any final case exactly.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime

import networkx as nx

from src.llm.cache import LLMMode
from src.orchestration.case_runner import AdminPolicy, run_case
from src.orchestration.run_state import RunState
from src.schemas import Disposition, InvestigationReport, TriageEvent, TriageEventKind, TriageRule
from src.stages.stage0_triage import TriageContext, first_alert, triage
from src.stages.stage5_graph.agent import build_graph, links


MAX_ROUNDS = 6


@dataclass
class BatchResult:
    order: list[str]
    latest: dict[str, InvestigationReport] = field(default_factory=dict)
    cases: list[InvestigationReport] = field(default_factory=list)
    rounds: list[list[str]] = field(default_factory=list)  # accounts run in each round
    state: RunState = field(default_factory=RunState)
    before: dict[str, RunState] = field(default_factory=dict)  # case id -> state before it


def review(account_id: str) -> TriageEvent:
    return TriageEvent(kind=TriageEventKind.PERIODIC_REVIEW, account_id=account_id)


def review_order(ctx: TriageContext, account_ids: Iterable[str]) -> list[str]:
    probe = RunState()
    keyed: list[tuple[int, datetime, str]] = []
    for account_id in account_ids:
        alert = first_alert(triage(review(account_id), ctx, probe), ctx.bank, account_id)
        keyed.append((0, alert.timestamp, account_id) if alert else (1, datetime.min, account_id))
    return [a for *_, a in sorted(keyed)]


def run_batch(
    ctx: TriageContext,
    account_ids: Iterable[str],
    admin: AdminPolicy = "approve",
    writer_for: Callable[[str], LLMMode] | None = None,
    progress: Callable[[str], None] | None = None,
) -> BatchResult:
    """`writer_for` picks Stage 7's mode per account (default: the context's own)."""
    result = BatchResult(order=review_order(ctx, account_ids))

    def one(account_id: str) -> None:
        run_ctx = replace(ctx, writer_mode=writer_for(account_id)) if writer_for else ctx
        snapshot = copy.deepcopy(result.state)
        report = run_case(review(account_id), run_ctx, result.state, admin=admin)
        result.before[report.case_id] = snapshot
        result.cases.append(report)
        result.latest[account_id] = report
        if progress:
            progress(f"{report.case_id}: {report.disposition.value} {report.confidence:g}")

    for account_id in result.order:
        one(account_id)
    result.rounds.append(list(result.order))

    graph = build_graph(ctx.bank)  # a review adds no transaction, so one graph serves the run
    for _ in range(MAX_ROUNDS):
        escalated = _escalated(result)
        rerun = [
            a
            for a in result.order
            if a not in escalated and (_new_hit(a, ctx, result) or _new_link(a, graph, result))
        ]
        if not rerun:
            break
        for account_id in rerun:
            one(account_id)
        result.rounds.append(rerun)
    return result


def _escalated(result: BatchResult) -> set[str]:
    return {a for a, r in result.latest.items() if r.disposition == Disposition.SUSPICIOUS_ESCALATE}


def _new_hit(account_id: str, ctx: TriageContext, result: BatchResult) -> bool:
    """A known-bad identifier fires now that did not fire on the account's latest case."""

    def known_bad(fired) -> set[str]:
        return {f.detail for f in fired if f.rule == TriageRule.KNOWN_BAD_IDENTIFIER}

    now = known_bad(triage(review(account_id), ctx, result.state).fired_rules)
    before = known_bad(result.latest[account_id].triage_findings.fired_rules)
    return bool(now - before)


def _new_link(account_id: str, graph: nx.Graph, result: BatchResult) -> bool:
    """A linked account is flagged on its own now but was not when the latest case ran.

    Compared with the run state saved before that case, so a case where Stage 5 failed or did
    not run is not re-run every round.
    """
    then = result.before[result.latest[account_id].case_id]
    return any(
        result.state.is_flagged_on_own(a) and not then.is_flagged_on_own(a)
        for a in links(graph, account_id)
    )
