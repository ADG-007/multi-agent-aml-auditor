"""The plain case runner beside the LangGraph pipeline. Same steps, so they always agree.

No checkpoint: `admin` decides a pause on the spot. The batch, the CLI and tests use it.
"""

from __future__ import annotations

from collections.abc import Callable

from src.orchestration.run_state import RunState
from src.orchestration.steps import (
    EVIDENCE_STAGES,
    FINDINGS_FIELD,
    REAL_STAGES,
    CaseWork,
    finalise,
    step_admin,
    step_evidence,
    step_judge,
    step_rulebook,
    step_triage,
    step_writer,
)
from src.schemas import InvestigationReport, TriageEvent
from src.stages.stage0_triage import TriageContext
from src.stages.stubs import StubPlan


__all__ = ["EVIDENCE_STAGES", "FINDINGS_FIELD", "REAL_STAGES", "run_case"]

AdminPolicy = str | Callable[[CaseWork], str]


def run_case(
    event: TriageEvent,
    ctx: TriageContext,
    state: RunState,
    plan: StubPlan | None = None,
    admin: AdminPolicy = "approve",
) -> InvestigationReport | None:
    """Investigate one event. None when a quiet transaction lands on a flagged account."""
    plan = plan or StubPlan()
    work = CaseWork(event=event)
    step_triage(work, ctx, state)
    if work.stop:
        return None  # joins the account's history; its standing disposition is unchanged
    for stage in EVIDENCE_STAGES:
        step_evidence(work, stage, ctx, state, plan)
    step_judge(work)
    if work.needs_admin:
        step_admin(work, admin(work) if callable(admin) else admin)
    step_rulebook(work, plan)
    step_writer(work, ctx, state, plan)
    return finalise(work, ctx, state)
