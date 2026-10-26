"""Stage 6's entry point: find the rulebook clauses that fit what the other stages found.

    run(findings) -> RulebookResult

One ChromaDB query per stage with a finding (>= 0.40, or an ID mismatch). Keeps the top 3
within distance 0.75. Evidence with no clause in range sets the policy-gap flag. No LLM.
validate_citations() is the check Stage 7's text must pass.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

import chromadb

from src.schemas import (
    ReasoningStep,
    Stage1VisionOutput,
    Stage2TransactionPatternOutput,
    Stage3KYCOutput,
    Stage4AdverseMediaOutput,
    Stage5GraphOutput,
    Stage6RulebookOutput,
)
from src.stages.stage6_rulebook import config


@dataclass(frozen=True)
class RulebookResult:
    output: Stage6RulebookOutput
    policy_gap: bool


@dataclass(frozen=True)
class Query:
    stage: str
    text: str


def build_queries(
    vision: Stage1VisionOutput | None = None,
    transaction_pattern: Stage2TransactionPatternOutput | None = None,
    kyc: Stage3KYCOutput | None = None,
    adverse_media: Stage4AdverseMediaOutput | None = None,
    graph: Stage5GraphOutput | None = None,
) -> list[Query]:
    """One query per stage with a finding, from that stage's own lines."""
    queries = []
    if vision and vision.field_discrepancies:
        queries.append(
            Query(
                "Vision",
                "Identity document field disagrees with the customer record, possible "
                "tampering: " + "; ".join(vision.field_discrepancies),
            )
        )
    tp = transaction_pattern
    if tp and tp.confidence >= config.SIGNAL_MIN:
        lines = tp.structuring_indicators + tp.pass_through_indicators
        queries.append(Query("TransactionPattern", "; ".join(lines) or tp.velocity_summary))
    if kyc and kyc.confidence >= config.SIGNAL_MIN:
        queries.append(
            Query(
                "KYC",
                "Declared customer profile inconsistent with observed account activity: "
                + "; ".join(kyc.profile_behavior_mismatches),
            )
        )
    am = adverse_media
    if am and am.confidence >= config.SIGNAL_MIN and am.severity_tier is not None:
        tier = am.severity_tier.value.replace("_", " ").lower()
        queries.append(
            Query(
                "AdverseMedia", f"Name screening match on the {tier} list: {am.matched_entries[0]}"
            )
        )
    if graph and graph.confidence >= config.SIGNAL_MIN:
        queries.append(
            Query(
                "Graph",
                "Accounts linked by shared device, IP address or address to flagged accounts: "
                + "; ".join(graph.shared_identifiers[:6]),
            )
        )
    return queries


@lru_cache(maxsize=4)
def _collection(path: str):
    client = chromadb.PersistentClient(path=path)
    return client.get_collection(config.COLLECTION_NAME)


def search(text: str, k: int = config.TOP_K) -> list[tuple[str, float]]:
    """(clause id, cosine distance) for the k nearest clauses, nearest first."""
    result = _collection(str(config.CHROMA_DIR)).query(query_texts=[text], n_results=k)
    return list(zip(result["ids"][0], result["distances"][0], strict=True))


def run(queries: Iterable[Query], searcher=search) -> RulebookResult:
    """`searcher` is ChromaDB in the pipeline; tests pass a fake."""
    queries = list(queries)
    kept: dict[str, float] = {}
    trail: list[ReasoningStep] = []
    for q in queries:
        hits = searcher(q.text)
        fit = [(cid, d) for cid, d in hits if d <= config.MAX_DISTANCE]
        for cid, d in fit:
            kept[cid] = min(d, kept.get(cid, d))
        trail.append(
            ReasoningStep(
                step=f"Search for {q.stage}",
                finding=", ".join(f"{cid} {d:.3f}" for cid, d in hits) or "no clause returned",
                impact=f"{len(fit)} within distance {config.MAX_DISTANCE:g}: "
                f"{', '.join(c for c, _ in fit) or 'none'}.",
            )
        )
    gap = bool(queries) and not kept
    ids = sorted(kept, key=lambda c: (kept[c], c))
    trail.append(
        ReasoningStep(
            step="Citable clauses",
            finding=", ".join(ids) if ids else "none",
            impact=(
                "Policy gap: real evidence, but no clause fits."
                if gap
                else "Stage 7 may cite only these clauses."
                if ids
                else "No finding to cite."
            ),
        )
    )
    return RulebookResult(
        output=Stage6RulebookOutput(
            retrieved_clause_ids=ids,
            policy_violations=[f"{q.stage}: {q.text[:160]}" for q in queries],
            reasoning_trail=trail,
        ),
        policy_gap=gap,
    )


def validate_citations(text: str, allowed: Iterable[str]) -> tuple[list[str], list[str]]:
    """(cited and allowed, cited but not allowed)."""
    allowed = set(allowed)
    cited = list(dict.fromkeys(re.findall(config.CLAUSE_ID_PATTERN, text)))
    return [c for c in cited if c in allowed], [c for c in cited if c not in allowed]


def strip_citations(text: str, invalid: Iterable[str]) -> str:
    """Remove invalid clause ids with their list joiner (", " or " and "), then empty brackets."""
    for cid in invalid:
        cid = re.escape(cid)
        text = re.sub(rf"\s*\(\s*\b{cid}\b\s*\)", "", text)  # "(BAD-001)"
        text = re.sub(rf"\b{cid}\b\s*(?:,|\band\b|\bor\b)\s*", "", text)  # "BAD-001, X"
        text = re.sub(rf"\s*(?:,|\band\b|\bor\b)\s*\b{cid}\b", "", text)  # "X and BAD-001"
        text = re.sub(rf"\s*\b{cid}\b", "", text)
    text = re.sub(r"\s*\(\s*\)", "", text)
    return re.sub(r"\s+([.,;:)])", r"\1", text)
