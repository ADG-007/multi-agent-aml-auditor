"""The Graph agent: transaction patterns and shared identifiers, then a pattern label.

    agent = GraphAgent(client)
    scores = agent.score(account_id, ctx, as_of, transaction, state)   # Python, before the judge
    agent.label(scores, ctx, state)                                    # the model, after it

`score` runs the required tools and gives the two weighted results: Transaction Pattern
(Stage 2: single-deposit count k, monthly count m, pass-through) and Graph (links to accounts
already flagged on their own evidence). No model takes part in a score.

`label` asks the model for one label from a fixed list, citing fact ids. The guard drops
unknown ids and falls back to the Python label when the facts do not support the model's
choice. The label feeds the rulebook search and the report outline. It never scores.
No signal, no model call.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from src.agents.base import Agent, AgentRun, AgentSpec, known_ids
from src.agents.graph import config, links
from src.llm import Role
from src.orchestration.run_state import RunState
from src.schemas import (
    ReasoningStep,
    Stage2TransactionPatternOutput,
    Stage5GraphOutput,
    Transaction,
)
from src.stages import stage2_transaction_pattern
from src.stages.stage0_triage import TriageContext
from src.stages.stage2_transaction_pattern.analyst import SubScores, sub_scores


NO_PATTERN, MIXED = "NO_PATTERN", "MIXED_PATTERN"


class PatternLabel(BaseModel):
    label: Literal[
        "NAIVE_CASH_STRUCTURING",
        "MONTHLY_CAP_STRUCTURING",
        "PASS_THROUGH",
        "LINKED_TO_FLAGGED_ACCOUNTS",
        "MIXED_PATTERN",
        "NO_PATTERN",
    ]
    fact_ids: list[str] = Field(description="The fact ids the label rests on, e.g. F1.")
    reason: str = Field(description="One short sentence.")
    more: list[str] = Field(
        default_factory=list, description="Optional tool names to run first, or empty."
    )


PROMPT = """Name the pattern in one bank account's facts. Pick exactly one label.

Labels:
- NAIVE_CASH_STRUCTURING: several single cash deposits each just under the reporting \
threshold.
- MONTHLY_CAP_STRUCTURING: many smaller cash deposits whose monthly total sits just under \
the monthly reporting threshold.
- PASS_THROUGH: money from unrelated payers sent on within hours.
- LINKED_TO_FLAGGED_ACCOUNTS: shares a device, IP address or address with accounts already \
flagged.
- MIXED_PATTERN: two or more of the patterns above.
- NO_PATTERN: none of them.

Use only the facts below and cite the fact ids your label rests on. The scores are already \
computed; do not change or recompute them. You may ask for optional tools in `more` \
("path_to_flagged", "counterparty_overlap") if they would help; otherwise leave it empty.

FACTS:
{facts}
"""


@dataclass
class GraphScores:
    """The Python results the judge reads, plus what the label step needs."""

    account_id: str
    transaction_pattern: Stage2TransactionPatternOutput
    sub: SubScores
    graph: Stage5GraphOutput
    link: links.LinkScore
    run: AgentRun
    transaction: Transaction | None = None
    label: str = NO_PATTERN
    label_by: str = "python"

    @property
    def has_signal(self) -> bool:
        return self.transaction_pattern.confidence > 0 or self.graph.confidence > 0


class GraphAgent(Agent):
    spec = AgentSpec(
        name="graph",
        role=Role.GRAPH,
        prompt_version=config.PROMPT_VERSION,
        required_tools=("structuring", "pass_through", "shared_identifiers", "flagged_neighbours"),
        optional_tools=("path_to_flagged", "counterparty_overlap"),
    )

    # ----------------------------------------------------------------------------------
    # Scores (Python)
    # ----------------------------------------------------------------------------------

    def score(
        self,
        account_id: str,
        ctx: TriageContext,
        as_of: datetime | None,
        transaction: Transaction | None = None,
        state: RunState | None = None,
    ) -> GraphScores:
        state = state or RunState()
        run = self.start(account_id)

        pattern = stage2_transaction_pattern.run(account_id, ctx, as_of, transaction)
        sub = sub_scores(account_id, ctx, as_of, transaction)
        self.tool(run, "structuring")
        self.tool(run, "pass_through")
        run.facts.add(
            "structuring",
            f"Single cash deposits in the band: k = {sub.k} in the busiest 30 days "
            f"(sub-score {sub.structuring:g}). "
            + ("; ".join(pattern.structuring_indicators[:2]) or "No in-band cash."),
            value=sub.k,
        )
        run.facts.add(
            "monthly",
            f"Monthly cash totals in the band: m = {sub.m} (sub-score {sub.monthly:g}).",
            value=sub.m,
        )
        run.facts.add(
            "pass_through",
            f"Pass-through sub-score {sub.pass_through:g}. "
            + ("; ".join(pattern.pass_through_indicators[:2]) or "No fast forwarding."),
            value=sub.pass_through,
        )

        g = links.build_graph(ctx.bank, transaction)
        link = links.score_links(g, account_id, state)
        self.tool(run, "shared_identifiers")
        self.tool(run, "flagged_neighbours")
        run.facts.add(
            "links",
            f"Shares a device, IP or address with {len(link.linked)} account(s); flagged on "
            f"their own evidence: by device/IP {', '.join(link.hard) or 'none'}, by address "
            f"only {', '.join(link.soft) or 'none'} (link score {link.confidence:g}).",
            value=link.confidence,
        )

        def tag(other: str) -> str:
            if other in link.flagged:
                return f" ({state.standing(other).value})"
            if other in link.echo:
                return f" ({state.standing(other).value} through links only: not counted)"
            return " (not flagged)"

        shared = [
            f"{kind} {value} shared with {other}" + tag(other)
            for other in sorted(link.linked)
            for kind in (*links.HARD, *links.SOFT)
            for value in sorted(link.linked[other].get(kind, ()))
        ]
        trail = [
            ReasoningStep(
                step="Shared identifiers",
                finding=f"Linked to {len(link.linked)} account(s): "
                f"{', '.join(sorted(link.linked)) or 'none'}.",
                impact="Device and IP are hard links, an address alone is soft (NET-002).",
            ),
            ReasoningStep(
                step="Flagged in this run",
                finding=f"Hard: {', '.join(link.hard) or 'none'}. "
                f"Soft: {', '.join(link.soft) or 'none'}. "
                f"Flagged through links only (not counted): {', '.join(link.echo) or 'none'}. "
                f"Not flagged: {', '.join(link.unflagged) or 'none'}.",
                impact="Only links to accounts already MONITOR or ESCALATE on their own "
                "evidence score; a chance link to an ordinary account adds nothing.",
            ),
            ReasoningStep(
                step="Confidence",
                finding=link.why,
                impact=f"Graph confidence {link.confidence:g}; links alone are capped at "
                f"{config.RING_ONLY_CAP:g} (no veto without a sanctions link).",
            ),
        ]
        graph = Stage5GraphOutput(
            shared_identifiers=shared,
            connected_flagged_accounts=link.flagged,
            confidence=link.confidence,
            reasoning_trail=trail,
        )
        run.trail += pattern.reasoning_trail[:4] + trail
        scores = GraphScores(account_id, pattern, sub, graph, link, run, transaction)
        scores.label = python_label(sub, link.confidence)
        return scores

    # ----------------------------------------------------------------------------------
    # Label (the model; never scored)
    # ----------------------------------------------------------------------------------

    def label(self, scores: GraphScores, ctx: TriageContext, state: RunState) -> str:
        run = scores.run
        fallback = python_label(scores.sub, scores.link.confidence)
        scores.label, scores.label_by = fallback, "python"
        if not scores.has_signal:
            run.step("Pattern label", "No signal from the tools.", "NO_PATTERN; no model call.")
            return scores.label

        what = "Pattern label"
        answer = self.ask(run, PROMPT.format(facts=run.facts.render()), PatternLabel, what)
        if answer is not None and answer.more:
            tools = {
                "path_to_flagged": lambda: self._path(scores, ctx, state),
                "counterparty_overlap": lambda: self._overlap(scores, ctx, state),
            }
            if self.optional(run, answer.more, tools):
                answer = self.ask(run, PROMPT.format(facts=run.facts.render()), PatternLabel, what)
        if answer is None:
            run.step(what, f"No model answer: Python label {fallback}.", "Never scored.")
            return scores.label

        cited, unknown = known_ids(answer.fact_ids, run.facts)
        allowed = supported_labels(scores.sub, scores.link.confidence)
        if not cited:
            verdict = f"cited no real fact id: Python label {fallback} kept."
        elif answer.label not in allowed:
            verdict = f"is not supported by the facts: Python label {fallback} kept."
        else:
            scores.label, scores.label_by = answer.label, "model"
            verdict = f"accepted, citing {', '.join(cited)}."
        run.step(
            what,
            f"Model said {answer.label} ({answer.reason.strip()[:160]}); it {verdict}"
            + (f" Unknown fact ids dropped: {', '.join(unknown)}." if unknown else ""),
            "The label feeds the rulebook search and the outline. Never scored.",
        )
        return scores.label

    def _path(self, scores: GraphScores, ctx: TriageContext, state: RunState) -> None:
        g = links.build_graph(ctx.bank, scores.transaction)
        path = links.path_to_flagged(g, scores.account_id, state)
        scores.run.facts.add(
            "path",
            "Path to a flagged account: " + (" -> ".join(path) if path else "none within reach."),
        )

    def _overlap(self, scores: GraphScores, ctx: TriageContext, state: RunState) -> None:
        found = links.counterparty_overlap(ctx.bank, scores.account_id, state)
        scores.run.facts.add(
            "overlap",
            "Rare payees shared with flagged accounts: "
            + (
                "; ".join(f"{cp} (also paid by {', '.join(a)})" for cp, a in found[:4])
                if found
                else "none."
            )
            + " Reported only, never scored.",
        )


def supported_labels(sub: SubScores, link_confidence: float) -> set[str]:
    """The labels the Python facts allow."""
    present = set()
    if sub.structuring > 0:
        present.add("NAIVE_CASH_STRUCTURING")
    if sub.monthly > 0:
        present.add("MONTHLY_CAP_STRUCTURING")
    if sub.pass_through > 0:
        present.add("PASS_THROUGH")
    if link_confidence > 0:
        present.add("LINKED_TO_FLAGGED_ACCOUNTS")
    if len(present) >= 2:
        present.add(MIXED)
    return present or {NO_PATTERN}


def python_label(sub: SubScores, link_confidence: float) -> str:
    """The label without a model: the strongest signal, or MIXED when two are real."""
    strengths = {
        "NAIVE_CASH_STRUCTURING": sub.structuring,
        "MONTHLY_CAP_STRUCTURING": sub.monthly,
        "PASS_THROUGH": sub.pass_through,
        "LINKED_TO_FLAGGED_ACCOUNTS": link_confidence,
    }
    real = {label: s for label, s in strengths.items() if s > 0}
    if not real:
        return NO_PATTERN
    # A naive structurer also fills a monthly band; that is one pattern, not two.
    cash = {"NAIVE_CASH_STRUCTURING", "MONTHLY_CAP_STRUCTURING"}
    kinds = {"cash" if label in cash else label for label in real}
    if len(kinds) >= 2:
        return MIXED
    return max(real, key=lambda label: (real[label], label))
