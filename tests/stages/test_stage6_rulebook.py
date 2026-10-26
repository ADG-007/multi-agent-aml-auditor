"""Stage 6: the distance cut-off, the policy gap and the citation check."""

from src.stages.stage6_rulebook import run, validate_citations
from src.stages.stage6_rulebook.agent import Query, strip_citations


def _searcher(hits):
    return lambda text: hits


def test_only_clauses_within_075_are_kept():
    out = run(
        [Query("TransactionPattern", "cash")],
        searcher=_searcher([("STRUCT-001", 0.40), ("STRUCT-002", 0.74), ("KYC-001", 0.80)]),
    )
    assert out.output.retrieved_clause_ids == ["STRUCT-001", "STRUCT-002"]
    assert not out.policy_gap


def test_real_evidence_with_no_fitting_clause_is_a_policy_gap():
    out = run([Query("Graph", "links")], searcher=_searcher([("NET-002", 0.90)]))
    assert out.policy_gap and out.output.retrieved_clause_ids == []


def test_no_finding_is_no_gap():
    out = run([], searcher=_searcher([]))
    assert not out.policy_gap and out.output.reasoning_trail


def test_citation_check_splits_allowed_from_invented():
    ok, bad = validate_citations(
        "Breaches STRUCT-001 and MADE-999 (also INJECTED-2).", ["STRUCT-001"]
    )
    assert ok == ["STRUCT-001"] and bad == ["MADE-999", "INJECTED-2"]
    assert "MADE-999" not in strip_citations("See MADE-999 and STRUCT-001.", bad)


def test_strip_citations_leaves_clean_text():
    cases = [
        ("Relevant clauses (BAD-001, SANC-001) apply.", "Relevant clauses (SANC-001) apply."),
        ("Under KYC-002 and BAD-001.", "Under KYC-002."),
        ("Under BAD-001 and SANC-001.", "Under SANC-001."),
        ("Cash rules apply (BAD-001).", "Cash rules apply."),
        ("A, BAD-001, B.", "A, B."),
    ]
    for text, clean in cases:
        assert strip_citations(text, ["BAD-001"]) == clean


def test_real_index_finds_structuring_clauses():
    from pathlib import Path

    import pytest

    if not Path("data/rulebook/chroma").exists():
        pytest.skip("ChromaDB index not built (python -m src.generators.rulebook)")
    out = run(
        [Query("TransactionPattern", "6 cash deposits of ₹9,00,000–₹10,00,000, each under the CTR")]
    )
    assert "STRUCT-001" in out.output.retrieved_clause_ids
