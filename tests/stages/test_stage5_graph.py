"""Stage 5: only links to accounts flagged in this run score; the ring-only cap; the veto."""

from src.orchestration.run_state import CaseRecord, RunState
from src.rulebook import RulebookLookup
from src.schemas import Disposition, TransactionDirection
from src.stages import stage5_graph as s5
from src.stages.stage0_triage import TriageContext
from tests.factories import account, bank, customer, txn


IDS = ["ACC-T001", "ACC-T002", "ACC-T003", "ACC-T004", "ACC-T005"]


def _bank(device_sharers=(), address_sharers=()):
    accounts, customers, rows = [], [], []
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


def _run(b, state):
    ctx = TriageContext.build(b, RulebookLookup.load())
    return s5.run("ACC-T001", ctx, ctx.review_date, state=state)


def test_a_link_to_an_unflagged_account_scores_nothing():
    out = _run(_bank(device_sharers=IDS[:2]), RunState())
    assert out.confidence == 0.0
    assert any("not flagged" in s for s in out.shared_identifiers)


def test_hard_links_climb_to_the_ring_only_cap():
    b = _bank(device_sharers=IDS)
    state = RunState()
    expected = {1: 0.60, 2: 0.75, 3: 0.85, 4: 0.85}
    for n, value in expected.items():
        _flag(state, IDS[n])
        out = _run(b, state)
        assert out.confidence == value
        assert len(out.connected_flagged_accounts) == n


def test_an_address_alone_is_a_soft_link():
    b = _bank(address_sharers=IDS[:3])
    state = RunState()
    _flag(state, "ACC-T002")
    assert _run(b, state).confidence == 0.30
    _flag(state, "ACC-T003")
    assert _run(b, state).confidence == 0.45


def test_veto_needs_a_sanctions_link_and_a_second_hard_link():
    b = _bank(device_sharers=IDS[:3])
    state = RunState()
    _flag(state, "ACC-T002", Disposition.SUSPICIOUS_ESCALATE, sanctions=True)
    assert _run(b, state).confidence == 0.60  # one hard link only: no veto
    _flag(state, "ACC-T003")
    assert _run(b, state).confidence == 0.95


def test_an_account_flagged_only_through_links_is_not_counted():
    # No echo: ACC-T002 is MONITOR only because of its links, so it adds nothing here.
    b = _bank(device_sharers=IDS[:2])
    state = RunState()
    state._history["ACC-T002"] = [
        CaseRecord("CASE-ACC-T002-1", Disposition.MONITOR, frozenset(), None, flagged_on_own=False)
    ]
    out = _run(b, state)
    assert out.confidence == 0.0 and out.connected_flagged_accounts == []
    assert any("through links only: not counted" in s for s in out.shared_identifiers)
