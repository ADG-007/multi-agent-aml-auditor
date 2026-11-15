"""What every agent is built from.

One agent = one pattern:

1. Python tools turn bank data into facts with ids (F1, F2 ...). Required tools always run.
2. The model reads the facts and returns a small Pydantic contract that cites fact ids. It
   may ask for up to MAX_OPTIONAL_TOOLS optional tools by name; unknown names are dropped.
3. A Python guard drops unknown ids and quotes that are not in the source text. Python, never
   the model, turns the result into a score.
4. Every step lands in the reasoning trail.

A failed model call never raises out of an agent: `ask` returns None, the trail says why, and
the agent falls back to its Python-only result or queues the job for the backup model.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from src.llm import LLMClient, LLMFailureReason, LLMStageFailure, Role
from src.schemas import ReasoningStep


TSchema = TypeVar("TSchema", bound=BaseModel)

MAX_OPTIONAL_TOOLS = 2

#: Failures the backup model can retry later. A switched-off or replay-only run cannot.
QUEUEABLE = frozenset({LLMFailureReason.SCHEMA_VALIDATION_FAILED, LLMFailureReason.TIMED_OUT})


@dataclass(frozen=True)
class Fact:
    fact_id: str
    topic: str
    text: str
    #: The number behind the text, when a Python rule needs it.
    value: float | None = None


class FactSheet:
    """The facts of one agent run, numbered in the order the tools made them."""

    def __init__(self) -> None:
        self._facts: list[Fact] = []

    def add(self, topic: str, text: str, value: float | None = None) -> Fact:
        fact = Fact(f"F{len(self._facts) + 1}", topic, text, value)
        self._facts.append(fact)
        return fact

    def __iter__(self):
        return iter(self._facts)

    def __len__(self) -> int:
        return len(self._facts)

    @property
    def ids(self) -> set[str]:
        return {f.fact_id for f in self._facts}

    def get(self, fact_id: str) -> Fact | None:
        wanted = fact_id.strip().upper()
        return next((f for f in self._facts if f.fact_id == wanted), None)

    def topic(self, topic: str) -> list[Fact]:
        return [f for f in self._facts if f.topic == topic]

    def render(self) -> str:
        return "\n".join(f"{f.fact_id}: {f.text}" for f in self._facts)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    role: Role
    prompt_version: str
    required_tools: tuple[str, ...]
    optional_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueuedJob:
    """A job for the backup model: an unsure answer, a schema that failed twice, a re-check."""

    agent: str
    account_id: str
    kind: str
    reason: str


@dataclass
class AgentRun:
    """What one agent did on one case."""

    spec: AgentSpec
    account_id: str
    facts: FactSheet = field(default_factory=FactSheet)
    trail: list[ReasoningStep] = field(default_factory=list)
    queued: list[QueuedJob] = field(default_factory=list)
    tools_run: list[str] = field(default_factory=list)
    model_calls: int = 0
    model_seconds: float = 0.0
    model_failed: bool = False

    def step(self, step: str, finding: str, impact: str) -> None:
        self.trail.append(ReasoningStep(step=step, finding=finding, impact=impact))

    def queue(self, kind: str, reason: str) -> None:
        self.queued.append(QueuedJob(self.spec.name, self.account_id, kind, reason))


class Agent:
    """Base class. A subclass sets `spec` and writes its own `run`."""

    spec: AgentSpec

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def start(self, account_id: str) -> AgentRun:
        return AgentRun(spec=self.spec, account_id=account_id)

    def tool(self, run: AgentRun, name: str) -> None:
        """Record that a tool ran. A tool the spec does not list is a bug."""
        if name not in (*self.spec.required_tools, *self.spec.optional_tools):
            raise ValueError(f"{self.spec.name} has no tool {name!r}")
        run.tools_run.append(name)

    def optional(
        self, run: AgentRun, requested: Iterable[str], tools: dict[str, Callable[[], None]]
    ) -> list[str]:
        """Run the optional tools the model asked for: known names only, at most two."""
        allowed = [n for n in dict.fromkeys(requested) if n in self.spec.optional_tools]
        allowed = [n for n in allowed if n in tools and n not in run.tools_run]
        chosen = allowed[:MAX_OPTIONAL_TOOLS]
        dropped = [n for n in dict.fromkeys(requested) if n not in chosen]
        for name in chosen:
            tools[name]()
            self.tool(run, name)
        if chosen or dropped:
            run.step(
                "Optional tools",
                f"Ran: {', '.join(chosen) or 'none'}. Dropped: {', '.join(dropped) or 'none'}.",
                f"The model may add at most {MAX_OPTIONAL_TOOLS} listed tools.",
            )
        return chosen

    def missing_required(self, run: AgentRun) -> list[str]:
        return [t for t in self.spec.required_tools if t not in run.tools_run]

    def ask(
        self,
        run: AgentRun,
        prompt: str,
        schema: type[TSchema],
        what: str,
        images: Sequence[Path] = (),
        role: Role | None = None,
        queue_on_failure: bool = True,
    ) -> TSchema | None:
        """One model call. None on any failure, with the reason in the trail."""
        if self.client.off:
            run.step(what, "Models are off for this run.", "Python-only result.")
            return None
        try:
            result = self.client.call(
                role or self.spec.role,
                prompt,
                schema,
                self.spec.prompt_version,
                images=images,
                agent=self.spec.name,
            )
        except LLMStageFailure as failure:
            run.model_failed = True
            queued = queue_on_failure and failure.reason in QUEUEABLE
            if queued:
                run.queue(what, failure.reason.value)
            run.step(
                what,
                f"Model call failed: {failure.reason.value}.",
                "Queued for the backup model." if queued else "Python-only result.",
            )
            return None
        run.model_calls += 1
        run.model_seconds += result.seconds
        how = "saved answer" if result.cached else "new answer"
        run.step(
            what,
            f"{result.model}, {self.spec.prompt_version}: {how}, {result.seconds:.1f} s"
            + (f", {result.attempts} attempts" if result.attempts > 1 else "")
            + ".",
            "The model reads and cites; Python scores.",
        )
        return result.value


# --------------------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------------------


def known_ids(cited: Iterable[str], sheet: FactSheet) -> tuple[list[str], list[str]]:
    """(cited ids that exist, cited ids that do not), in the order cited, without repeats."""
    kept: list[str] = []
    dropped: list[str] = []
    for raw in cited:
        fact = sheet.get(raw)
        target = kept if fact else dropped
        value = fact.fact_id if fact else raw.strip()
        if value not in target:
            target.append(value)
    return kept, dropped


def squash(text: str) -> str:
    """Lower case, one space between words, straight quotes: how quotes are compared."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in(quote: str | None, source: str, min_chars: int = 8) -> bool:
    """Whether `quote` appears in `source` word for word (case and spacing aside)."""
    if not quote:
        return False
    q = squash(quote).strip(" .\"'")
    return len(q) >= min_chars and q in squash(source)
