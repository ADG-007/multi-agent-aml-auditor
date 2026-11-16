"""Live mode: one case through a LangGraph state graph, with the admin pause.

    triage -> vision -> web_research -> judge -> admin_gate -> graph_label -> rag -> auditor
           -> backup (the 9B, at most one re-check) -> report

Nodes call the shared steps (steps.py). The admin gate uses interrupt(): an ID check at 0.90
or more waits for resume("approve" / "block"). The checkpoint is in memory and holds only the
case's thread id; the case itself, the bank, the run state and the agents stay on the
Pipeline object, so a paused case is small and nothing needs to be serialised.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from src.orchestration.run_state import RunState
from src.orchestration.steps import (
    Agents,
    CaseWork,
    build_report,
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
from src.schemas import InvestigationReport, PipelineStatus, TriageEvent
from src.stages.stage0_triage import TriageContext
from src.stages.stubs import StubPlan


NODES = (
    "triage",
    "vision",
    "web_research",
    "judge",
    "admin_gate",
    "graph_label",
    "rag",
    "auditor",
    "backup",
    "report",
)


class CaseState(TypedDict, total=False):
    thread: str
    done: list[str]


@dataclass(frozen=True)
class Paused:
    """A case waiting for the admin. `evidence` is what the admin sees."""

    thread_id: str
    account_id: str
    evidence: dict[str, Any]


#: Called after each node finishes: (node name, the case so far, seconds the node took).
OnNode = Callable[[str, CaseWork, float], None]


class Pipeline:
    """One run: a bank context, a run state, the agents, and the paused cases."""

    def __init__(
        self,
        ctx: TriageContext,
        agents: Agents | None = None,
        state: RunState | None = None,
        plan: StubPlan | None = None,
    ) -> None:
        self.ctx = ctx
        self.agents = agents or Agents.build()
        self.state = state or RunState()
        self.plan = plan or StubPlan()
        self._works: dict[str, CaseWork] = {}
        self._reports: dict[str, InvestigationReport] = {}
        self._graph = self._build()
        self._count = 0

    # ----------------------------------------------------------------------------------
    # The graph
    # ----------------------------------------------------------------------------------

    def _build(self):
        g = StateGraph(CaseState)

        def node(name: str, fn: Callable[[CaseWork], None]):
            def run(s: CaseState) -> CaseState:
                fn(self._works[s["thread"]])
                return {"done": [*s.get("done", []), name]}

            g.add_node(name, run)

        ctx, agents, plan = self.ctx, self.agents, self.plan
        node("triage", lambda w: step_triage(w, ctx, self.state))
        node("vision", lambda w: step_vision(w, ctx, agents, plan))
        node("web_research", lambda w: step_web(w, ctx, agents, plan))
        node("judge", self._judge)
        g.add_node("admin_gate", self._admin_gate)
        node("graph_label", lambda w: step_label(w, ctx, self.state, agents))
        node("rag", lambda w: step_rag(w, agents, plan))
        node("auditor", lambda w: step_auditor(w, ctx, self.state, agents, plan))
        node("backup", self._backup)
        g.add_node("report", self._report)

        g.add_edge(START, "triage")
        g.add_conditional_edges(
            "triage", lambda s: END if self._works[s["thread"]].stop else "vision"
        )
        chain = list(NODES[1:])
        for a, b in zip(chain, chain[1:], strict=False):
            g.add_edge(a, b)
        g.add_edge("report", END)
        return g.compile(checkpointer=InMemorySaver())

    def _judge(self, work: CaseWork) -> None:
        step_scores(work, self.ctx, self.state, self.agents, self.plan)
        step_judge(work)

    def _admin_gate(self, s: CaseState) -> CaseState:
        work = self._works[s["thread"]]
        if work.needs_admin and work.admin_decision is None:
            vision = work.findings["vision_findings"]
            decision = interrupt(
                {
                    "account_id": work.account_id,
                    "forgery_confidence": vision.forgery_confidence,
                    "field_discrepancies": vision.field_discrepancies,
                    "stage_status": {k: v.value for k, v in work.stage_status.items()},
                    "score_if_approved": work.score.confidence,
                    "disposition_if_approved": work.score.disposition.value,
                }
            )
            step_admin(work, decision)
        return {"done": [*s.get("done", []), "admin_gate"]}

    def _backup(self, work: CaseWork) -> None:
        if step_backup(work, self.ctx, self.state, self.agents):
            step_judge(work)
            step_rag(work, self.agents, self.plan)
            step_auditor(work, self.ctx, self.state, self.agents, self.plan)

    def _report(self, s: CaseState) -> CaseState:
        thread = s["thread"]
        self._reports[thread] = finalise(self._works[thread], self.ctx, self.state)
        return {"done": [*s.get("done", []), "report"]}

    # ----------------------------------------------------------------------------------
    # Running
    # ----------------------------------------------------------------------------------

    def run(
        self, event: TriageEvent, on_node: OnNode | None = None
    ) -> InvestigationReport | Paused | None:
        """A report, a Paused case awaiting resume(), or None (quiet transaction)."""
        self._count += 1
        thread = f"{event.account_id}-{self._count}"
        self._works[thread] = CaseWork(event=event)
        return self._drive({"thread": thread, "done": []}, thread, on_node)

    def resume(
        self, paused: Paused, decision: str, on_node: OnNode | None = None
    ) -> InvestigationReport:
        result = self._drive(Command(resume=decision), paused.thread_id, on_node)
        if not isinstance(result, InvestigationReport):  # pragma: no cover - defensive
            raise RuntimeError(f"case {paused.thread_id} did not finish after resume")
        return result

    def preview(self, paused: Paused) -> InvestigationReport:
        """The paused case as a report: every check and the judge's verdict, no text yet."""
        report = build_report(self._works[paused.thread_id], self.ctx, self.state)
        return report.model_copy(
            update={"pipeline_status": PipelineStatus.PAUSED_PENDING_ADMIN_REVIEW}
        )

    def work(self, thread_id: str) -> CaseWork:
        return self._works[thread_id]

    @staticmethod
    def sign_off(report: InvestigationReport) -> InvestigationReport:
        """The analyst's sign-off: the report stops being a draft."""
        return report.model_copy(update={"is_draft": False})

    def _drive(self, payload, thread: str, on_node: OnNode | None):
        """Stream the graph node by node; report each finished node to `on_node`."""
        config = {"configurable": {"thread_id": thread}}
        work = self._works[thread]
        paused = None
        last = time.perf_counter()
        for chunk in self._graph.stream(payload, config, stream_mode="updates"):
            now = time.perf_counter()
            for name, update in chunk.items():
                if name == "__interrupt__":
                    paused = Paused(thread, work.account_id, update[0].value)
                elif on_node:
                    on_node(name, work, now - last)
            last = now
        if paused is not None:
            return paused
        return self._reports.get(thread)
