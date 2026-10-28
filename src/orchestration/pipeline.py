"""The LangGraph pipeline: the eight stages as one state graph, one after another.

    triage -> vision -> transaction_pattern -> kyc -> adverse_media -> graph
           -> judge -> admin_gate -> rulebook -> writer -> report

Nodes call the shared steps (steps.py). The admin gate uses interrupt(): an ID check >= 0.90
waits for resume("approve" / "block"). The checkpoint is in memory. The bank and run state go
through the graph config, not the checkpoint, so a pause stays small.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from src.orchestration.run_state import RunState
from src.orchestration.steps import (
    EVIDENCE_STAGES,
    CaseWork,
    build_report,
    finalise,
    step_admin,
    step_evidence,
    step_judge,
    step_rulebook,
    step_triage,
    step_writer,
)
from src.schemas import InvestigationReport, PipelineStatus, StageName, TriageEvent
from src.stages.stage0_triage import TriageContext
from src.stages.stubs import StubPlan


class CaseState(TypedDict, total=False):
    work: CaseWork
    report: InvestigationReport | None


@dataclass(frozen=True)
class Paused:
    """A case waiting for the admin. `evidence` is what the admin sees."""

    thread_id: str
    account_id: str
    evidence: dict[str, Any]


#: Called after each node finishes: (node name, the case so far, seconds the node took).
OnNode = Callable[[str, CaseWork, float], None]

NODE = {
    StageName.VISION: "vision",
    StageName.TRANSACTION_PATTERN: "transaction_pattern",
    StageName.KYC: "kyc",
    StageName.ADVERSE_MEDIA: "adverse_media",
    StageName.GRAPH: "graph",
}


def _run(config) -> tuple[TriageContext, RunState, StubPlan]:
    c = config["configurable"]
    return c["ctx"], c["state"], c["plan"]


def _triage(s: CaseState, config) -> CaseState:
    ctx, state, _ = _run(config)
    step_triage(s["work"], ctx, state)
    return {"work": s["work"]}


def _evidence(stage: StageName):
    def node(s: CaseState, config) -> CaseState:
        ctx, state, plan = _run(config)
        step_evidence(s["work"], stage, ctx, state, plan)
        return {"work": s["work"]}

    return node


def _judge(s: CaseState) -> CaseState:
    step_judge(s["work"])
    return {"work": s["work"]}


def _admin_gate(s: CaseState) -> CaseState:
    work = s["work"]
    if work.needs_admin and work.admin_decision is None:
        vision = work.findings["vision_findings"]
        decision = interrupt(
            {
                "account_id": work.event.account_id,
                "forgery_confidence": vision.forgery_confidence,
                "field_discrepancies": vision.field_discrepancies,
                "stage_status": {k: v.value for k, v in work.stage_status.items()},
                "score_if_approved": work.score.confidence,
                "disposition_if_approved": work.score.disposition.value,
            }
        )
        step_admin(work, decision)
    return {"work": work}


def _rulebook(s: CaseState, config) -> CaseState:
    _, _, plan = _run(config)
    step_rulebook(s["work"], plan)
    return {"work": s["work"]}


def _writer(s: CaseState, config) -> CaseState:
    ctx, state, plan = _run(config)
    step_writer(s["work"], ctx, state, plan)
    return {"work": s["work"]}


def _report(s: CaseState, config) -> CaseState:
    ctx, state, _ = _run(config)
    return {"report": finalise(s["work"], ctx, state)}


def build_graph(checkpointer=None):
    g = StateGraph(CaseState)
    g.add_node("triage", _triage)
    for stage in EVIDENCE_STAGES:
        g.add_node(NODE[stage], _evidence(stage))
    g.add_node("judge", _judge)
    g.add_node("admin_gate", _admin_gate)
    g.add_node("rulebook", _rulebook)
    g.add_node("writer", _writer)
    g.add_node("report", _report)

    g.add_edge(START, "triage")
    g.add_conditional_edges("triage", lambda s: END if s["work"].stop else NODE[EVIDENCE_STAGES[0]])
    chain = [NODE[s] for s in EVIDENCE_STAGES] + ["judge", "admin_gate", "rulebook", "writer"]
    for a, b in zip(chain, chain[1:] + ["report"], strict=True):
        g.add_edge(a, b)
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)


def checkpoint_types() -> list[type]:
    """Every type a paused case holds: the schemas, plus the step and stage result types.
    Listed explicitly so the checkpoint loads nothing else."""
    import src.schemas as schemas  # noqa: PLC0415
    from src.scoring import ScoreResult, StageResult  # noqa: PLC0415
    from src.stages.stage6_rulebook import RulebookResult  # noqa: PLC0415
    from src.stages.stage7_auditor import AuditorResult  # noqa: PLC0415
    from src.stages.stage7_auditor.agent import AuditorDraft  # noqa: PLC0415

    found = [
        v
        for v in vars(schemas).values()
        if isinstance(v, type) and issubclass(v, (BaseModel, Enum))
    ]
    return [*found, CaseWork, StageResult, ScoreResult, RulebookResult, AuditorResult, AuditorDraft]


class Pipeline:
    """One run: a bank context, a run state and an in-memory checkpoint for paused cases."""

    def __init__(
        self, ctx: TriageContext, state: RunState | None = None, plan: StubPlan | None = None
    ) -> None:
        self.ctx = ctx
        self.state = state or RunState()
        self.plan = plan or StubPlan()
        serde = JsonPlusSerializer(allowed_msgpack_modules=None).with_msgpack_allowlist(
            checkpoint_types()
        )
        self._graph = build_graph(InMemorySaver(serde=serde))
        self._count = 0

    def _config(self, thread_id: str) -> dict:
        return {
            "configurable": {
                "thread_id": thread_id,
                "ctx": self.ctx,
                "state": self.state,
                "plan": self.plan,
            }
        }

    def run(
        self, event: TriageEvent, on_node: OnNode | None = None
    ) -> InvestigationReport | Paused | None:
        """A report, a Paused case awaiting resume(), or None (quiet transaction)."""
        self._count += 1
        thread_id = f"{event.account_id}-{self._count}"
        return self._drive({"work": CaseWork(event=event)}, thread_id, event.account_id, on_node)

    def resume(
        self, paused: Paused, decision: str, on_node: OnNode | None = None
    ) -> InvestigationReport:
        result = self._drive(Command(resume=decision), paused.thread_id, paused.account_id, on_node)
        if not isinstance(result, InvestigationReport):  # pragma: no cover - defensive
            raise RuntimeError(f"case {paused.thread_id} did not finish after resume")
        return result

    def preview(self, paused: Paused) -> InvestigationReport:
        """The paused case as a report: every check and the judge's verdict, no writer text."""
        work = self._graph.get_state(self._config(paused.thread_id)).values["work"]
        report = build_report(work, self.ctx, self.state)
        return report.model_copy(
            update={"pipeline_status": PipelineStatus.PAUSED_PENDING_ADMIN_REVIEW}
        )

    def _drive(self, payload, thread_id: str, account_id: str, on_node: OnNode | None):
        """Stream the graph node by node; report each finished node to `on_node`."""
        paused, report = None, None
        last = time.perf_counter()
        for chunk in self._graph.stream(payload, self._config(thread_id), stream_mode="updates"):
            now = time.perf_counter()
            for node, update in chunk.items():
                if node == "__interrupt__":
                    paused = Paused(thread_id, account_id, update[0].value)
                elif update and "report" in update:
                    report = update["report"]
                elif on_node and update and "work" in update:
                    on_node(node, update["work"], now - last)
            last = now
        return paused or report
