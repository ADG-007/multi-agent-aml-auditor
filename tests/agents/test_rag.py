"""The RAG agent: Python queries, the applies + quote guard, policy gap, citation helpers."""

import json

from src.agents.rag import config, index
from src.agents.rag.agent import (
    Evidence,
    RagAgent,
    build_queries,
    strip_citations,
    validate_citations,
)
from src.llm import LLMMode
from src.schemas import (
    SeverityTier,
    Stage1VisionOutput,
    Stage2TransactionPatternOutput,
    Stage3KYCOutput,
    Stage4AdverseMediaOutput,
    Stage5GraphOutput,
)
from tests.factories import fake_client


CLAUSES = {c.clause_id: c for c in index.load_clauses()}
STRUCT_QUOTE = "Deposits deliberately kept below this ceiling"


def evidence(**kwargs) -> Evidence:
    base = dict(
        transaction_pattern=Stage2TransactionPatternOutput(
            structuring_indicators=["6 cash deposits of 9-10 lakh in 30 days"],
            velocity_summary="steady",
            confidence=0.8,
        ),
        kyc=Stage3KYCOutput(confidence=0.0),
        graph_label="NAIVE_CASH_STRUCTURING",
    )
    return Evidence(**(base | kwargs))


def searcher(hits):
    calls = []

    def search(text):
        calls.append(text)
        return hits

    search.calls = calls
    return search


def review(*answers, extra=()) -> str:
    return json.dumps(
        {
            "clauses": [{"clause_id": c, "applies": a, "quote": q} for c, a, q in answers],
            "extra_queries": list(extra),
        }
    )


HITS = [("STRUCT-001", 0.31), ("STRUCT-002", 0.40), ("KYC-001", 0.9)]


def test_one_query_per_finding_plus_the_label():
    e = evidence(
        kyc=Stage3KYCOutput(profile_behavior_mismatches=["money in 5x income"], confidence=0.65),
        adverse_media=Stage4AdverseMediaOutput(
            matched_entries=["AM-1 X"], severity_tier=SeverityTier.ADVERSE_NEWS, confidence=0.7
        ),
        graph=Stage5GraphOutput(shared_identifiers=["device D shared"], confidence=0.6),
        vision=Stage1VisionOutput(
            document_type="AADHAAR", field_discrepancies=["dob"], forgery_confidence=0.95
        ),
        id_flagged=True,
    )
    assert [q.source for q in build_queries(e)] == [
        "Vision",
        "TransactionPattern",
        "KYC",
        "AdverseMedia",
        "Graph",
        "GraphLabel",
    ]


def test_a_set_aside_finding_is_not_searched_for():
    e = evidence(judge_inputs={"TransactionPattern": 0.0})
    assert build_queries(e) == []


def test_no_finding_means_no_search_and_no_model_call(tmp_path):
    client = fake_client(tmp_path)
    search = searcher(HITS)
    weak = Stage2TransactionPatternOutput(velocity_summary="steady", confidence=0.1)
    result = RagAgent(client).run("ACC-1", Evidence(transaction_pattern=weak), search)
    assert result.clause_ids == [] and not result.policy_gap
    assert search.calls == [] and client.fake.requests == []


def test_applied_clause_needs_a_real_quote_and_a_retrieved_id(tmp_path):
    answer = review(
        ("STRUCT-001", True, STRUCT_QUOTE),
        ("STRUCT-002", True, "words that are not in the clause at all"),
        ("MULE-001", True, "A customer account"),
        ("KYC-001", False, ""),
    )
    result = RagAgent(fake_client(tmp_path, answer)).run("ACC-1", evidence(), searcher(HITS))
    assert result.clause_ids == ["STRUCT-001"]
    assert result.applied[0].quote == STRUCT_QUOTE
    finding = result.output.reasoning_trail[-1].finding
    assert "STRUCT-002 (quote not in the clause)" in finding
    assert "MULE-001 (not retrieved)" in finding
    assert not result.policy_gap


def test_clauses_beyond_the_distance_cutoff_are_not_citable(tmp_path):
    client = fake_client(tmp_path, review(("KYC-001", True, "anything")))
    result = RagAgent(client).run("ACC-1", evidence(), searcher(HITS))
    assert "KYC-001" not in result.clause_ids
    assert "KYC-001" not in client.fake.prompts()[0].split("CLAUSES:")[1]


def test_model_failure_keeps_the_retrieved_set(tmp_path):
    result = RagAgent(fake_client(tmp_path, "junk", "junk")).run(
        "ACC-1", evidence(), searcher(HITS)
    )
    assert result.clause_ids == ["STRUCT-001", "STRUCT-002"]
    off = RagAgent(fake_client(tmp_path, mode=LLMMode.OFF)).run("ACC-1", evidence(), searcher(HITS))
    assert off.clause_ids == ["STRUCT-001", "STRUCT-002"]


def test_evidence_with_no_clause_in_range_is_a_policy_gap(tmp_path):
    client = fake_client(tmp_path)
    result = RagAgent(client).run("ACC-1", evidence(), searcher([("KYC-001", 0.95)]))
    assert result.policy_gap and result.clause_ids == []
    assert client.fake.requests == []


def test_model_may_add_two_queries_at_most(tmp_path):
    first = review(("STRUCT-001", True, STRUCT_QUOTE), extra=["q one", "q two", "q three"])
    client = fake_client(tmp_path, first, first)
    search = searcher(HITS)
    result = RagAgent(client).run("ACC-1", evidence(), search)
    assert search.calls[-2:] == ["q one", "q two"]
    assert [q.source for q in result.queries][-2:] == ["Model", "Model"]
    assert len(client.fake.chats()) == 1  # no new clause came back, so no second review


def test_search_hits_are_saved_for_replay(tmp_path, monkeypatch):
    class FakeCollection:
        def query(self, query_embeddings, n_results):
            return {"ids": [["STRUCT-001", "KYC-001"]], "distances": [[0.25, 0.7]]}

    monkeypatch.setattr(index, "_collection", lambda: FakeCollection())

    def transport(path, payload):
        assert payload["input"][0].startswith(config.QUERY_INSTRUCTION)
        return {"embeddings": [[0.1, 0.2]]}

    live = fake_client(tmp_path)
    live._transport = transport
    assert index.search(live, "cash below the threshold") == [
        ("STRUCT-001", 0.25),
        ("KYC-001", 0.7),
    ]
    replay = fake_client(tmp_path, mode=LLMMode.CACHED)
    assert index.search(replay, "cash below the threshold") == [
        ("STRUCT-001", 0.25),
        ("KYC-001", 0.7),
    ]


def test_citation_helpers():
    text = "Breaches STRUCT-001 and FAKE-999 (KYC-002)."
    assert validate_citations(text, ["STRUCT-001"]) == (["STRUCT-001"], ["FAKE-999", "KYC-002"])
    assert strip_citations(text, ["FAKE-999", "KYC-002"]) == "Breaches STRUCT-001."
    assert STRUCT_QUOTE in CLAUSES["STRUCT-001"].text
