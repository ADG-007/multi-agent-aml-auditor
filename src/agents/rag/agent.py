"""The RAG agent: find the rulebook clauses that fit what the other agents found.

    RagAgent(client).run(account_id, evidence) -> RagResult

Python writes one search query per finding (a weighted stage at 0.40 or more, an ID field
flagged by both reads, the Graph agent's label) and searches the index. The model then reads
the clause text and says, for each retrieved clause, whether it applies, with a quote. It may
add up to two queries of its own. Guard: a clause must be in the retrieved set and its quote
must be in the clause's text, or the answer for that clause is dropped.

The citable set is what the Auditor may cite. If the model fails, the retrieved set stands.
Evidence with no clause in range sets the policy-gap flag. No finding, no model call.
The agent never scores.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from src.agents.base import Agent, AgentRun, AgentSpec, quote_in
from src.agents.rag import config, index
from src.llm import LLMStageFailure, Role
from src.schemas import (
    ReasoningStep,
    RulebookClause,
    Stage1VisionOutput,
    Stage2TransactionPatternOutput,
    Stage3KYCOutput,
    Stage4AdverseMediaOutput,
    Stage5GraphOutput,
    Stage6RulebookOutput,
    StageName,
)


Searcher = Callable[[str], list[tuple[str, float]]]

LABEL_QUERIES = {
    "NAIVE_CASH_STRUCTURING": "Several cash deposits each kept just below the cash reporting "
    "threshold to avoid the report.",
    "MONTHLY_CAP_STRUCTURING": "Connected cash deposits whose total in a calendar month is "
    "kept just below the monthly cash reporting threshold.",
    "PASS_THROUGH": "An account receiving money from unrelated payers and sending it on "
    "within hours, as a money mule does.",
    "LINKED_TO_FLAGGED_ACCOUNTS": "Accounts linked by a shared device, IP address or address "
    "to accounts already under suspicion.",
}


@dataclass(frozen=True)
class Query:
    source: str
    text: str


@dataclass
class Evidence:
    """What the other agents found, as the RAG agent sees it."""

    vision: Stage1VisionOutput | None = None
    transaction_pattern: Stage2TransactionPatternOutput | None = None
    kyc: Stage3KYCOutput | None = None
    adverse_media: Stage4AdverseMediaOutput | None = None
    graph: Stage5GraphOutput | None = None
    graph_label: str | None = None
    #: The confidences the judge was given (a set-aside finding is not searched for).
    judge_inputs: dict[str, float] = field(default_factory=dict)
    id_flagged: bool = False

    def strength(self, stage: StageName, own: float) -> float:
        return self.judge_inputs.get(stage.value, own)


class ClauseAnswer(BaseModel):
    clause_id: str
    applies: bool
    quote: str = Field(description="Exact words from the clause that fit the finding, or empty.")


class ClauseReview(BaseModel):
    clauses: list[ClauseAnswer]
    extra_queries: list[str] = Field(
        default_factory=list,
        description="At most 2 short search queries if a finding has no clause, else empty.",
    )


PROMPT = """A bank account review produced the findings below. A search returned rulebook \
clauses. For each clause, say whether it applies to a finding.

- applies: true only when the clause governs one of the findings.
- quote: the exact words from that clause's text that fit the finding, copied character for \
character (one sentence at most). Empty when the clause does not apply.
- Answer for every clause listed, by its id. Do not invent clause ids.
- extra_queries: if a finding has no fitting clause, give up to 2 short search queries for \
it. Otherwise leave it empty.

FINDINGS:
{findings}

CLAUSES:
{clauses}
"""


@dataclass
class AppliedClause:
    clause_id: str
    title: str
    quote: str
    distance: float


@dataclass
class RagResult:
    output: Stage6RulebookOutput
    policy_gap: bool
    applied: list[AppliedClause]
    queries: list[Query]
    run: AgentRun

    @property
    def clause_ids(self) -> list[str]:
        return list(self.output.retrieved_clause_ids)


def build_queries(e: Evidence) -> list[Query]:
    """One query per finding, from that agent's own lines."""
    queries = []
    if e.vision and e.id_flagged:
        queries.append(
            Query(
                "Vision",
                "Identity document field disagrees with the customer record, possible "
                "tampering: " + "; ".join(e.vision.field_discrepancies),
            )
        )
    tp = e.transaction_pattern
    if tp and e.strength(StageName.TRANSACTION_PATTERN, tp.confidence) >= config.SIGNAL_MIN:
        lines = tp.structuring_indicators + tp.pass_through_indicators
        queries.append(Query("TransactionPattern", "; ".join(lines) or tp.velocity_summary))
    if e.kyc and e.kyc.confidence >= config.SIGNAL_MIN:
        queries.append(
            Query(
                "KYC",
                "Declared customer profile inconsistent with observed account activity: "
                + "; ".join(e.kyc.profile_behavior_mismatches),
            )
        )
    am = e.adverse_media
    if am and am.confidence >= config.SIGNAL_MIN and am.severity_tier is not None:
        tier = am.severity_tier.value.replace("_", " ").lower()
        queries.append(
            Query(
                "AdverseMedia", f"Name screening match on the {tier} list: {am.matched_entries[0]}"
            )
        )
    if e.graph and e.graph.confidence >= config.SIGNAL_MIN:
        queries.append(
            Query(
                "Graph",
                "Accounts linked by shared device, IP address or address to flagged accounts: "
                + "; ".join(e.graph.shared_identifiers[:6]),
            )
        )
    if queries and e.graph_label in LABEL_QUERIES:
        queries.append(Query("GraphLabel", LABEL_QUERIES[e.graph_label]))
    return queries


class RagAgent(Agent):
    spec = AgentSpec(
        name="rag",
        role=Role.RAG,
        prompt_version=config.PROMPT_VERSION,
        required_tools=("python_queries", "search", "read_clause_text"),
        optional_tools=("model_query",),
    )

    def __init__(self, client, clauses: Iterable[RulebookClause] | None = None) -> None:
        super().__init__(client)
        self._clauses = {c.clause_id: c for c in (clauses or index.load_clauses())}

    def run(self, account_id: str, evidence: Evidence, searcher: Searcher | None = None):
        """`searcher` is the index in the pipeline; tests pass a fake."""
        run = self.start(account_id)
        search = searcher or (lambda text: index.search(self.client, text))
        queries = build_queries(evidence)
        self.tool(run, "python_queries")
        if self.client.off and searcher is None:
            # The search needs the embedding model: with the models off there is none.
            step = ReasoningStep(
                step="Citable clauses",
                finding=f"Models are off: {len(queries)} finding(s), no rulebook search.",
                impact="Nothing to cite in a Python-only run.",
            )
            run.trail.append(step)
            output = Stage6RulebookOutput(
                policy_violations=[f"{q.source}: {q.text[:160]}" for q in queries],
                reasoning_trail=[step],
            )
            return RagResult(output=output, policy_gap=False, applied=[], queries=queries, run=run)

        kept: dict[str, float] = {}
        trail: list[ReasoningStep] = []

        def look(q: Query) -> None:
            hits = search(q.text)
            fit = [(cid, d) for cid, d in hits if d <= config.MAX_DISTANCE and cid in self._clauses]
            for cid, d in fit:
                kept[cid] = min(d, kept.get(cid, d))
            trail.append(
                ReasoningStep(
                    step=f"Search for {q.source}",
                    finding=", ".join(f"{cid} {d:.3f}" for cid, d in hits) or "no clause returned",
                    impact=f"{len(fit)} within distance {config.MAX_DISTANCE:g}: "
                    f"{', '.join(c for c, _ in fit) or 'none'}.",
                )
            )

        for q in queries:
            look(q)
        self.tool(run, "search")
        self.tool(run, "read_clause_text")

        applied: list[AppliedClause] = []
        how = "No finding to cite."
        if queries and kept:
            review = self._review(run, queries, kept)
            if review is not None and review.extra_queries:
                extra = [
                    Query("Model", " ".join(text.split())[:300])
                    for text in review.extra_queries[: config.MAX_MODEL_QUERIES]
                    if text.strip()
                ]
                before = set(kept)

                def more() -> None:
                    for q in extra:
                        queries.append(q)
                        look(q)

                if extra and self.optional(run, ["model_query"], {"model_query": more}):
                    if set(kept) != before:
                        review = self._review(run, queries, kept) or review
            applied, how = self._guard(review, kept)

        gap = bool(queries) and not kept
        ids = (
            [a.clause_id for a in applied] if applied else sorted(kept, key=lambda c: (kept[c], c))
        )
        if not applied and kept:
            applied = [
                AppliedClause(cid, self._clauses[cid].title, "", kept[cid])
                for cid in ids[: config.TOP_K]
            ]
            ids = [a.clause_id for a in applied]
        trail.append(
            ReasoningStep(
                step="Citable clauses",
                finding=(", ".join(ids) if ids else "none") + f". {how}",
                impact=(
                    "Policy gap: real evidence, but no clause fits."
                    if gap
                    else "The Auditor may cite only these clauses."
                    if ids
                    else "Nothing to cite."
                ),
            )
        )
        run.trail += trail
        output = Stage6RulebookOutput(
            retrieved_clause_ids=ids,
            policy_violations=[f"{q.source}: {q.text[:160]}" for q in queries],
            reasoning_trail=trail,
        )
        return RagResult(output=output, policy_gap=gap, applied=applied, queries=queries, run=run)

    def _review(
        self, run: AgentRun, queries: list[Query], kept: dict[str, float]
    ) -> ClauseReview | None:
        shown = sorted(kept, key=lambda c: (kept[c], c))[: config.MAX_CLAUSES_IN_PROMPT]
        prompt = PROMPT.format(
            findings="\n".join(f"- {q.source}: {q.text[:400]}" for q in queries),
            clauses="\n".join(
                f"- {cid} ({self._clauses[cid].title}): {self._clauses[cid].text}" for cid in shown
            ),
        )
        try:
            return self.ask(run, prompt, ClauseReview, "Applies and quote", queue_on_failure=False)
        except LLMStageFailure:  # pragma: no cover - ask() already degrades
            return None

    def _guard(
        self, review: ClauseReview | None, kept: dict[str, float]
    ) -> tuple[list[AppliedClause], str]:
        """The clauses the model applied with a quote that is in the clause text."""
        if review is None:
            return [], "No model answer: the retrieved clauses stand."
        applied: list[AppliedClause] = []
        dropped: list[str] = []
        for answer in review.clauses:
            cid = answer.clause_id.strip().upper()
            if not answer.applies:
                continue
            if cid not in kept:
                dropped.append(f"{cid} (not retrieved)")
            elif not quote_in(answer.quote, self._clauses[cid].text):
                dropped.append(f"{cid} (quote not in the clause)")
            elif cid not in [a.clause_id for a in applied]:
                applied.append(
                    AppliedClause(cid, self._clauses[cid].title, answer.quote.strip(), kept[cid])
                )
        applied.sort(key=lambda a: (a.distance, a.clause_id))
        how = f"Model applied {len(applied)} of {len(kept)} retrieved clause(s) with a real quote."
        if dropped:
            how += f" Dropped by the guard: {', '.join(dropped)}."
        if not applied:
            how += " None survived, so the nearest retrieved clauses stand."
        return applied, how


def validate_citations(text: str, allowed: Iterable[str]) -> tuple[list[str], list[str]]:
    """(cited and allowed, cited but not allowed)."""
    allowed = set(allowed)
    cited = list(dict.fromkeys(re.findall(config.CLAUSE_ID_PATTERN, text)))
    return [c for c in cited if c in allowed], [c for c in cited if c not in allowed]


def strip_citations(text: str, invalid: Iterable[str]) -> str:
    """Remove invalid clause ids with their list joiner (", " or " and "), then empty brackets."""
    for cid in invalid:
        cid = re.escape(cid)
        text = re.sub(rf"\s*\(\s*\b{cid}\b\s*\)", "", text)
        text = re.sub(rf"\b{cid}\b\s*(?:,|\band\b|\bor\b)\s*", "", text)
        text = re.sub(rf"\s*(?:,|\band\b|\bor\b)\s*\b{cid}\b", "", text)
        text = re.sub(rf"\s*\b{cid}\b", "", text)
    text = re.sub(r"\s*\(\s*\)", "", text)
    return re.sub(r"\s+([.,;:)])", r"\1", text)
