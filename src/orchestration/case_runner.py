"""The plain case runner beside the LangGraph pipeline. Same steps, so they always agree.

No checkpoint: `admin` decides a pause on the spot. Tests and the Python-only dry run use it.
With no agents given, the models are off and every agent gives its Python-only result.
"""

from __future__ import annotations

from collections.abc import Callable

from src.orchestration.run_state import RunState
from src.orchestration.steps import (
    EVIDENCE_STAGES,
    FINDINGS_FIELD,
    Agents,
    CaseWork,
    finalise,
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
from src.schemas import InvestigationReport, TriageEvent
from src.stages.stage0_triage import TriageContext
from src.stages.stubs import StubPlan


__all__ = ["EVIDENCE_STAGES", "FINDINGS_FIELD", "run_case"]

AdminPolicy = str | Callable[[CaseWork], str]


def run_case(
    event: TriageEvent,
    ctx: TriageContext,
    state: RunState,
    plan: StubPlan | None = None,
    admin: AdminPolicy = "approve",
    agents: Agents | None = None,
) -> InvestigationReport | None:
    """Investigate one event. None when a quiet transaction lands on a flagged account."""
    plan = plan or StubPlan()
    agents = agents or Agents.build()
    work = CaseWork(event=event)
    step_triage(work, ctx, state)
    if work.stop:
        return None  # joins the account's history; its standing disposition is unchanged
    step_vision(work, ctx, agents, plan)
    step_web(work, ctx, agents, plan)
    step_scores(work, ctx, state, agents, plan)
    step_judge(work)
    if work.needs_admin:
        step_admin(work, admin(work) if callable(admin) else admin)
    step_label(work, ctx, state, agents)
    step_rag(work, agents, plan)
    step_auditor(work, ctx, state, agents, plan)
    if step_backup(work, ctx, state, agents):
        # The backup model changed a judge input: judge again, then write again.
        step_judge(work)
        step_rag(work, agents, plan)
        step_auditor(work, ctx, state, agents, plan)
    return finalise(work, ctx, state)
