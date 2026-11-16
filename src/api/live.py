"""One case through the live graph, as a stream of events the site can draw.

run_case() is the one place that turns a Pipeline run into events. LiveJobs puts runs in a
queue served by one worker thread (this laptop runs one model at a time): one case running,
two waiting, the next is refused. A case the ID check pauses waits for the admin's decision
and is given up after a time limit, so a forgotten dialog cannot block the queue.

Two modes. Replay uses saved answers only and reports the seconds each answer took when it
was first made. Live calls the local models and reports measured seconds.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from src.llm import LLMClient, LLMMode
from src.orchestration import batch
from src.orchestration.pipeline import NODES, Paused, Pipeline
from src.orchestration.run_state import RunState
from src.orchestration.steps import Agents, CaseWork
from src.schemas import InvestigationReport
from src.stages.stage0_triage import TriageContext


MAX_WAITING = 2
PAUSE_TIMEOUT_SECONDS = 15 * 60
KEEP_ALIVE_SECONDS = 15
KEPT_JOBS = 20
FINISHED = ("done", "error", "cancelled", "abandoned")
#: Nodes that call no model when nothing sends them to one.
OPTIONAL_NODES = ("admin_gate", "backup")

Emit = Callable[..., None]


class QueueFull(Exception):
    pass


class NotAllowed(Exception):
    pass


class UnknownJob(KeyError):
    pass


class _Stop(Exception):
    """The job was cancelled or abandoned; `args[0]` is its final status."""


def run_case(
    pipeline: Pipeline,
    client: LLMClient,
    account_id: str,
    emit: Emit,
    decide: Callable[[Paused], str],
    cancelled: Callable[[], bool] = lambda: False,
) -> InvestigationReport | None:
    """Run one periodic review, calling emit(type=..., ...) for every step of it."""
    seen = len(client.calls)
    decision: str | None = None

    def on_node(name: str, work: CaseWork, seconds: float) -> None:
        nonlocal seen
        calls = client.calls[seen:]
        seen = len(client.calls)
        if calls and all(c["cached"] for c in calls):
            took, source = sum(c["seconds"] for c in calls), "recorded"
        else:
            took, source = seconds, "live" if calls else "python"
        note = None
        if name == "judge" and work.score is not None:
            note = f"{work.score.disposition.value} {work.score.confidence:.2f}"
        elif name == "admin_gate" and decision:
            note = decision
        elif name in OPTIONAL_NODES and not calls:
            note = "skipped"
        emit(
            type="node_done",
            node=name,
            seconds=round(took, 2),
            source=source,
            note=note,
            calls=len(calls),
            detail=_detail(name, work),
        )
        if cancelled():
            raise _Stop("cancelled")

    emit(type="started", note=account_id)
    result = pipeline.run(batch.review(account_id), on_node=on_node)
    if isinstance(result, Paused):
        emit(type="paused", node="admin_gate", evidence=result.evidence)
        decision = decide(result)
        emit(type="resumed", node="admin_gate", note=decision)
        result = pipeline.resume(result, decision, on_node=on_node)
    case = None
    if result is not None:
        case = {
            "account_id": result.target_account,
            "case_id": result.case_id,
            "disposition": result.disposition.value,
            "confidence": result.confidence,
        }
    emit(
        type="done",
        case=case,
        report=result.model_dump(mode="json") if result is not None else None,
    )
    return result


def _detail(name: str, work: CaseWork) -> str | None:
    """One plain line about what the node found, for the step list."""
    if name == "triage" and work.triage_out is not None:
        if work.stop:
            return "Nothing new on this account: no case is opened."
        reasons = work.triage_out.trigger_reasons
        return f"{len(work.routed)} checks routed. {reasons[0] if reasons else ''}".strip()
    if name == "vision":
        vision = work.findings.get("vision_findings")
        if vision is not None:
            return (
                f"ID check {vision.forgery_confidence:.2f}; "
                f"{len(vision.field_discrepancies)} field(s) disagree with the record."
            )
    if name == "web_research":
        media = work.findings.get("adverse_media_findings")
        if media is not None:
            return f"{len(media.matched_entries)} sanctions or news match(es)."
    if name == "judge" and work.judge_inputs:
        return "Inputs: " + ", ".join(f"{k} {v:.2f}" for k, v in work.judge_inputs.items())
    if name == "graph_label" and work.graph is not None:
        return f"Pattern label: {work.graph.label} (never scored)."
    if name == "rag":
        rules = work.findings.get("rulebook_findings")
        if rules is not None:
            return "Clauses: " + (", ".join(rules.retrieved_clause_ids) or "none")
    return None


# --------------------------------------------------------------------------------------
# The queue
# --------------------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    account_id: str
    mode: str  # "replay" or "live"
    run_folder: Path | None
    status: str = "queued"
    events: list[dict] = field(default_factory=list)
    changed: threading.Condition = field(default_factory=threading.Condition)
    decision: str | None = None
    cancel: bool = False

    def emit(self, **event) -> None:
        with self.changed:
            self.events.append({"seq": len(self.events) + 1, **event})
            self.changed.notify_all()

    def finish(self, status: str, **event) -> None:
        with self.changed:
            self.status = status
            if event:
                self.events.append({"seq": len(self.events) + 1, **event})
            self.changed.notify_all()


class LiveJobs:
    def __init__(
        self,
        ctx: TriageContext,
        *,
        audit_path: Path | None = None,
        client_factory: Callable[[LLMMode], LLMClient] | None = None,
        agents_factory: Callable[[LLMClient], Agents] = Agents.build,
        pause_timeout: float = PAUSE_TIMEOUT_SECONDS,
    ) -> None:
        self.ctx = ctx
        self._client = client_factory or (lambda mode: LLMClient(mode, audit_path=audit_path))
        self._agents = agents_factory
        self.pause_timeout = pause_timeout
        self._jobs: dict[str, Job] = {}
        self._waiting: deque[Job] = deque()
        self._running: Job | None = None
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._worker: threading.Thread | None = None

    def submit(self, account_id: str, mode: str, run_folder: Path | None = None) -> tuple[Job, int]:
        """Queue a case. Returns the job and how many jobs are ahead of it."""
        self.ctx.bank.account(account_id)
        with self._wake:
            ahead = len(self._waiting) + (self._running is not None)
            if len(self._waiting) >= MAX_WAITING:
                raise QueueFull(ahead)
            job = Job(uuid.uuid4().hex[:12], account_id, mode, run_folder)
            self._jobs[job.id] = job
            for old in list(self._jobs)[:-KEPT_JOBS]:
                if self._jobs[old].status in FINISHED:
                    del self._jobs[old]
            job.emit(type="queued", position=ahead)
            self._waiting.append(job)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, daemon=True)
                self._worker.start()
            self._wake.notify_all()
        return job, ahead

    def get(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise UnknownJob(job_id) from None

    def busy(self) -> int:
        with self._lock:
            return len(self._waiting) + (self._running is not None)

    def events(
        self, job_id: str, after: int = 0, keep_alive: float = KEEP_ALIVE_SECONDS
    ) -> Iterator[dict | None]:
        """Every event after `after`, as it happens; None is a keep-alive tick."""
        job = self.get(job_id)
        sent = after
        while True:
            with job.changed:
                if len(job.events) <= sent and job.status not in FINISHED:
                    job.changed.wait(keep_alive)
                fresh = job.events[sent:]
                over = job.status in FINISHED
            if not fresh and not over:
                yield None
            yield from fresh
            sent += len(fresh)
            if over and not fresh:
                return

    def decide(self, job_id: str, choice: str) -> None:
        if choice not in ("approve", "block"):
            raise ValueError(choice)
        job = self.get(job_id)
        with job.changed:
            if job.status != "paused":
                raise NotAllowed("This case is not waiting for a decision.")
            job.decision = choice
            job.changed.notify_all()

    def cancel(self, job_id: str) -> None:
        job = self.get(job_id)
        with self._wake:
            if job in self._waiting:
                self._waiting.remove(job)
                job.finish("cancelled", type="error", message="cancelled")
                return
        with job.changed:
            job.cancel = True
            job.changed.notify_all()

    # ----------------------------------------------------------------------------------
    # The worker
    # ----------------------------------------------------------------------------------

    def _work(self) -> None:
        while True:
            with self._wake:
                while not self._waiting:
                    if not self._wake.wait(timeout=30):
                        self._worker = None
                        return
                job = self._running = self._waiting.popleft()
            client = None
            try:
                client = self._client(LLMMode.LIVE if job.mode == "live" else LLMMode.CACHED)
                self._run(job, client)
            except _Stop as stop:
                job.finish(stop.args[0], type="error", message=stop.args[0])
            except Exception as exc:  # a failed job must not stop the queue
                job.finish("error", type="error", message=f"{type(exc).__name__}: {exc}")
            finally:
                with self._wake:
                    self._running = None
                    idle = not self._waiting
                if idle and client is not None and job.mode == "live":
                    try:
                        client.unload()
                    except Exception:  # the models may already be gone; nothing to free
                        pass

    def _run(self, job: Job, client: LLMClient) -> None:
        state = None
        if job.run_folder is not None:
            cases = batch.load_cases(job.run_folder)
            state = RunState.from_cases(cases, self.ctx.bank, skip=job.account_id)
        pipeline = Pipeline(self.ctx, self._agents(client), state=state)
        job.status = "running"

        def decide(paused: Paused) -> str:
            deadline = time.monotonic() + self.pause_timeout
            with job.changed:
                job.status = "paused"
                while job.decision is None:
                    left = deadline - time.monotonic()
                    if job.cancel or left <= 0:
                        raise _Stop("cancelled" if job.cancel else "abandoned")
                    job.changed.wait(left)
                job.status = "running"
                return job.decision

        run_case(pipeline, client, job.account_id, job.emit, decide, lambda: job.cancel)
        job.finish("done")


def node_weights(summary: dict) -> dict[str, float]:
    """Typical seconds per node, from a saved run's measured seconds per model call."""
    agent_node = {
        "vision": "vision",
        "web_research": "web_research",
        "graph": "graph_label",
        "rag": "rag",
        "auditor": "auditor",
    }
    weights = dict.fromkeys(NODES, 0.3)
    for key, row in (summary.get("model_calls") or {}).items():
        node = agent_node.get(key.split(" ")[0])
        if node and row.get("seconds_per_call"):
            weights[node] = max(weights[node], float(row["seconds_per_call"]))
    return weights
