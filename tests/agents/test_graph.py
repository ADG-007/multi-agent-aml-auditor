"""The Graph agent: Python link and pattern scores, and the guarded label."""

import json

from src.agents.graph import links
from src.agents.graph.agent import GraphAgent, python_label, supported_labels
from src.llm import LLMMode
from src.orchestration.run_state import CaseRecord, RunState
from src.rulebook import RulebookLookup
from src.schemas import Disposition, TransactionDirection, TransactionType
from src.stages.stage0_triage import TriageContext
from src.stages.stage2_transaction_pattern.analyst import SubScores
from tests.factories import account, bank, customer, fake_client, txn


IDS = ["ACC-T001", "ACC-T002", "ACC-T003", "ACC-T004", "ACC-T005"]


def _bank(device_sharers=(), address_sharers=(), extra=()):
    accounts, customers, rows = [], [], list(extra)
    for i, a in enumerate(IDS):
        cid = f"CUST-T{i:03d}"
        address = "1 Shared Lane" if a in address_sharers else f"{i} Own Road"
        accounts.append(account(account_id=a, customer_id=cid))
        customers.append(customer(customer_id=cid, name=f"Person {i}", address=address))
        device = "DEV-RING" if a in device_sharers else f"DEV-{i}"
        rows.append(txn(account_id=a, device_id=device, direction=TransactionDirection.DEBIT))
    return bank(accounts=accounts, customers=customers, transactions=rows)


def _flag(state, account_id, disposition=Disposition.MONITOR, sanctions=False):
    state._history[account_id] = [
        CaseRecord(f"CASE-{account_id}-1", disposition, frozenset(), None, sanctions)
    ]


def _score(b, state, client=None, tmp_path=None):
    ctx = TriageContext.build(b, RulebookLookup.load())
    agent = GraphAgent(client or fake_client(tmp_path, mode=LLMMode.OFF))
    return agent, ctx, agent.score("ACC-T001", ctx, ctx.review_date, state=state)


def _say(label, fact_ids=("F4",), more=()) -> str:
    return json.dumps(
        {"label": label, "fact_ids": list(fact_ids), "reason": "because", "more": list(more)}
    )


def _structuring_rows():
    """Six cash deposits of 9.5 lakh inside one month: a naive structurer (k = 6)."""
    return [
        txn(
            amount=950_000,
            type=TransactionType.CASH_DEPOSIT,
            timestamp=f"2026-08-{day:02d}T10:00:00",
        )
        for day in (3, 6, 10, 14, 18, 22)
    ]


# --------------------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------------------


def test_a_link_to_an_unflagged_account_scores_nothing(tmp_path):
    _, _, scores = _score(_bank(device_sharers=IDS[:2]), RunState(), tmp_path=tmp_path)
    assert scores.graph.confidence == 0.0
    assert any("not flagged" in s for s in scores.graph.shared_identifiers)
    assert not scores.has_signal


def test_hard_links_climb_to_the_ring_only_cap(tmp_path):
    b = _bank(device_sharers=IDS)
    state = RunState()
    for n, value in {1: 0.60, 2: 0.75, 3: 0.85, 4: 0.85}.items():
        _flag(state, IDS[n])
        _, _, scores = _score(b, state, tmp_path=tmp_path)
        assert scores.graph.confidence == value
        assert len(scores.graph.connected_flagged_accounts) == n


def test_an_address_alone_is_a_soft_link(tmp_path):
    b = _bank(address_sharers=IDS[:3])
    state = RunState()
    _flag(state, "ACC-T002")
    assert _score(b, state, tmp_path=tmp_path)[2].graph.confidence == 0.30
    _flag(state, "ACC-T003")
    assert _score(b, state, tmp_path=tmp_path)[2].graph.confidence == 0.45


def test_an_account_flagged_through_links_only_is_not_counted(tmp_path):
    b = _bank(device_sharers=IDS[:2])
    state = RunState()
    state._history["ACC-T002"] = [
        CaseRecord("CASE-ACC-T002-1", Disposition.MONITOR, frozenset(), None, False, False)
    ]
    _, _, scores = _score(b, state, tmp_path=tmp_path)
    assert scores.graph.confidence == 0.0
    assert any("not counted" in s for s in scores.graph.shared_identifiers)


def test_sanctions_link_plus_a_second_hard_link_is_the_only_veto(tmp_path):
    b = _bank(device_sharers=IDS[:3])
    state = RunState()
    _flag(state, "ACC-T002", Disposition.SUSPICIOUS_ESCALATE, sanctions=True)
    assert _score(b, state, tmp_path=tmp_path)[2].graph.confidence == 0.60
    _flag(state, "ACC-T003")
    assert _score(b, state, tmp_path=tmp_path)[2].graph.confidence == 0.95


def test_structuring_tool_reports_k_and_m_from_stage_2(tmp_path):
    _, _, scores = _score(_bank(extra=_structuring_rows()), RunState(), tmp_path=tmp_path)
    assert scores.sub.k == 6
    assert scores.transaction_pattern.confidence == max(
        scores.sub.structuring, scores.sub.monthly, scores.sub.pass_through
    )
    assert "k = 6" in scores.run.facts.get("F1").text
    assert "m = " in scores.run.facts.get("F2").text
    assert scores.label == "NAIVE_CASH_STRUCTURING"


# --------------------------------------------------------------------------------------
# Label
# --------------------------------------------------------------------------------------


def test_label_rules():
    none = SubScores(0, 0, 0, 0, 0)
    cash = SubScores(0.8, 6, 0.3, 1, 0)
    assert python_label(none, 0.0) == "NO_PATTERN"
    assert python_label(cash, 0.0) == "NAIVE_CASH_STRUCTURING"
    assert python_label(SubScores(0, 0, 0.6, 2, 0), 0.0) == "MONTHLY_CAP_STRUCTURING"
    assert python_label(cash, 0.6) == "MIXED_PATTERN"
    assert supported_labels(none, 0.0) == {"NO_PATTERN"}
    assert "PASS_THROUGH" not in supported_labels(cash, 0.6)


def test_no_signal_means_no_model_call(tmp_path):
    client = fake_client(tmp_path)
    agent, ctx, scores = _score(_bank(), RunState(), client)
    assert agent.label(scores, ctx, RunState()) == "NO_PATTERN"
    assert client.fake.requests == []


def test_supported_label_is_accepted_and_never_changes_a_score(tmp_path):
    b = _bank(device_sharers=IDS[:2])
    state = RunState()
    _flag(state, "ACC-T002")
    agent, ctx, scores = _score(b, state, fake_client(tmp_path, _say("LINKED_TO_FLAGGED_ACCOUNTS")))
    before = (scores.graph.confidence, scores.transaction_pattern.confidence)
    assert agent.label(scores, ctx, state) == "LINKED_TO_FLAGGED_ACCOUNTS"
    assert scores.label_by == "model"
    assert (scores.graph.confidence, scores.transaction_pattern.confidence) == before


def test_unsupported_label_or_unknown_ids_fall_back_to_python(tmp_path):
    b = _bank(device_sharers=IDS[:2])
    state = RunState()
    _flag(state, "ACC-T002")
    agent, ctx, scores = _score(b, state, fake_client(tmp_path / "a", _say("PASS_THROUGH")))
    assert agent.label(scores, ctx, state) == "LINKED_TO_FLAGGED_ACCOUNTS"
    assert scores.label_by == "python"
    client = fake_client(tmp_path / "b", _say("LINKED_TO_FLAGGED_ACCOUNTS", ["F77"]))
    agent, ctx, scores = _score(b, state, client)
    agent.label(scores, ctx, state)
    assert scores.label_by == "python" and "no real fact id" in scores.run.trail[-1].finding


def test_optional_tools_add_facts_and_a_second_call(tmp_path):
    b = _bank(device_sharers=IDS[:2])
    state = RunState()
    _flag(state, "ACC-T002")
    client = fake_client(
        tmp_path,
        _say("NO_PATTERN", more=["path_to_flagged", "counterparty_overlap", "anything"]),
        _say("LINKED_TO_FLAGGED_ACCOUNTS", ["F5"]),
    )
    agent, ctx, scores = _score(b, state, client)
    assert agent.label(scores, ctx, state) == "LINKED_TO_FLAGGED_ACCOUNTS"
    assert "ACC-T001 -> device DEV-RING -> ACC-T002" in scores.run.facts.get("F5").text
    assert "never scored" in scores.run.facts.get("F6").text
    assert len(client.fake.chats()) == 2


def test_counterparty_overlap_ignores_hubs(monkeypatch):
    monkeypatch.setattr(links.config, "OVERLAP_MAX_ACCOUNTS", 3)  # CP-0001 is paid by all 5
    rows = [
        txn(account_id=a, direction=TransactionDirection.DEBIT, counterparty_id="CP-RARE")
        for a in IDS[:2]
    ]
    b = _bank(extra=rows)
    state = RunState()
    _flag(state, "ACC-T002")
    assert links.counterparty_overlap(b, "ACC-T001", state) == [("CP-RARE", ["ACC-T002"])]
    assert links.counterparty_overlap(b, "ACC-T003", state) == []


def test_explained_monthly_cash_is_not_labelled_as_structuring(tmp_path):
    client = fake_client(tmp_path, _say("MONTHLY_CAP_STRUCTURING", ["F2"]))
    agent, ctx, scores = _score(_bank(), RunState(), client)
    scores.sub = SubScores(0, 0, 0.6, 2, 0)
    assert agent.label(scores, ctx, RunState(), cash_explained=True) == "NO_PATTERN"
    assert client.fake.requests == []
    assert "explained" in scores.run.trail[-1].finding
