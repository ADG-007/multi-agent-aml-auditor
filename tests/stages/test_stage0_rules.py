"""Stage 0 rules, one at a time, at their boundaries."""

import pytest

from src.matching import NameIndex
from src.orchestration.run_state import RunState
from src.rulebook import RulebookLookup
from src.schemas import (
    Disposition,
    InvestigationReport,
    PipelineStatus,
    RulebookCategory,
    RulebookClause,
    TransactionDirection,
    TransactionType,
    TriageEventKind,
    TriageRule,
)
from src.stages.stage0_triage import rules
from tests.factories import bank, customer, media_entry, txn


CASH = TransactionType.CASH_DEPOSIT
CREDIT, DEBIT = TransactionDirection.CREDIT, TransactionDirection.DEBIT


@pytest.fixture(scope="module")
def rulebook():
    return RulebookLookup.load()


# --- cash threshold -------------------------------------------------------------------


@pytest.mark.parametrize(
    "amount, type, fires",
    [
        (899_999.99, CASH, False),
        (900_000, CASH, True),  # band floor is inclusive
        (950_000, CASH, True),
        (1_000_000, CASH, True),  # exactly at the threshold: top of the band, not a CTR
        (1_000_000.01, CASH, True),  # more than the threshold: a CTR
        (1_500_000, TransactionType.CASH_WITHDRAWAL, True),
        (950_000, TransactionType.NEFT, False),  # cash only
        (1_500_000, TransactionType.SWIFT, False),
    ],
)
def test_cash_threshold(rulebook, amount, type, fires):
    hit = rules.cash_threshold(txn(amount=amount, type=type), rulebook)
    assert (hit is not None) == fires
    if hit:
        assert hit.fired.clause_id == "STRUCT-001"
        assert hit.fired.transaction_id is not None


def test_cash_threshold_reports_only_more_than_the_threshold(rulebook):
    # PMLA Rule 3(1)(A) says "more than" ten lakh, so exactly ten lakh is band, not CTR.
    at = rules.cash_threshold(txn(amount=1_000_000, type=CASH), rulebook)
    above = rules.cash_threshold(txn(amount=1_000_001, type=CASH), rulebook)
    assert "structuring band" in at.reason
    assert "above the" in above.reason and "structuring band" not in above.reason


def test_cash_threshold_follows_the_rulebook():
    clause = RulebookClause(
        clause_id="STRUCT-001",
        title="t",
        text="t",
        category=RulebookCategory.STRUCTURING,
        sources=[],
        metadata={"threshold_inr": 500_000, "near_threshold_pct": 0.8},
    )
    custom = RulebookLookup([clause])
    assert rules.cash_threshold(txn(amount=400_000, type=CASH), custom) is not None
    assert rules.cash_threshold(txn(amount=399_999, type=CASH), custom) is None


# --- cross-border ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "amount, type, direction, fires",
    [
        (500_001, TransactionType.SWIFT, CREDIT, True),
        (500_001, TransactionType.SWIFT, DEBIT, True),
        (500_000, TransactionType.SWIFT, CREDIT, False),  # exactly at: law says "more than"
        (499_999, TransactionType.SWIFT, CREDIT, False),
        (900_000, TransactionType.RTGS, CREDIT, False),  # domestic rail
    ],
)
def test_cross_border(rulebook, amount, type, direction, fires):
    hit = rules.cross_border(txn(amount=amount, type=type, direction=direction), rulebook)
    assert (hit is not None) == fires


# --- KYC event ------------------------------------------------------------------------


def test_kyc_event_fires_for_submissions_and_reviews_only():
    assert rules.kyc_event(TriageEventKind.KYC_SUBMITTED).fired.rule == TriageRule.KYC_EVENT
    assert rules.kyc_event(TriageEventKind.PERIODIC_REVIEW) is not None
    assert rules.kyc_event(TriageEventKind.TRANSACTION) is None


# --- known-bad identifiers ------------------------------------------------------------


def _escalated_state(device="DEV-BAD", ip="10.9.9.9", source="ACC-SRC") -> RunState:
    state = RunState()
    src_bank = bank(
        accounts=None,
        transactions=[txn(account_id="ACC-T001", device_id=device, ip_address=ip)],
    )
    state.record_case(
        InvestigationReport(
            case_id="CASE-ACC-T001-1",
            target_account="ACC-T001",
            pipeline_status=PipelineStatus.COMPLETED,
            confidence=0.95,
            disposition=Disposition.SUSPICIOUS_ESCALATE,
        ),
        src_bank,
    )
    return state


def test_known_bad_device_and_ip_from_another_account():
    state = _escalated_state()
    t = txn(account_id="ACC-OTHER", device_id="DEV-BAD", ip_address="10.9.9.9")
    hits = rules.known_bad_on_transaction(t, "ACC-OTHER", state)
    assert len(hits) == 2
    assert all(h.fired.rule == TriageRule.KNOWN_BAD_IDENTIFIER for h in hits)


def test_known_bad_ignores_the_accounts_own_identifiers():
    state = _escalated_state()
    t = txn(account_id="ACC-T001", device_id="DEV-BAD")
    assert rules.known_bad_on_transaction(t, "ACC-T001", state) == []


def test_known_bad_address():
    state = _escalated_state()
    shared = customer(customer_id="CUST-X")  # same default address as the escalated customer
    assert rules.known_bad_address(shared, "ACC-OTHER", state) is not None
    elsewhere = customer(customer_id="CUST-Y", address="1 Other Street, Delhi 110001")
    assert rules.known_bad_address(elsewhere, "ACC-OTHER", state) is None


# --- name match -----------------------------------------------------------------------


def test_name_match_on_holder_counterparty_and_alias():
    names = NameIndex(
        [
            media_entry(entry_id="AM-1", entity_name="Asha Verma"),
            media_entry(entry_id="AM-2", entity_name="Rakesh Gill", aliases=["Rocky Gill"]),
        ]
    )
    history = [
        txn(counterparty_name="Rocky Gill"),
        txn(counterparty_name="Rocky Gill"),  # same name screened once
        txn(counterparty_name="Ordinary Payer"),
    ]
    hits = rules.name_match(customer(name="Asha Verma"), history, names)
    details = [h.fired.detail for h in hits]
    assert len(hits) == 2
    assert any("account holder" in d for d in details)
    assert any("Rocky Gill" in d for d in details)


def test_name_match_none():
    names = NameIndex([media_entry(entity_name="Vikram Sethi")])
    assert rules.name_match(customer(name="Asha Verma"), [txn()], names) == []


# --- pass-through ---------------------------------------------------------------------


def _burst(amounts=(5_000, 5_000, 5_000), forwarded=12_000, gap_hours=2):
    rows = []
    for i, amount in enumerate(amounts):
        rows.append(
            txn(
                amount=amount,
                timestamp=f"2026-08-01T{10 + i * gap_hours:02d}:00:00",
                counterparty_id=f"CP-NEW-{i}",
            )
        )
    rows.append(
        txn(
            amount=forwarded,
            direction=DEBIT,
            timestamp="2026-08-01T20:00:00",
            counterparty_id="CP-COLLECTOR",
        )
    )
    return rows


def test_pass_through_fires_on_a_forwarded_burst(rulebook):
    hits = rules.pass_through(_burst(), rulebook)
    assert len(hits) == 1
    assert hits[0].fired.clause_id == "MULE-002"


def test_pass_through_needs_three_first_time_credits(rulebook):
    assert rules.pass_through(_burst(amounts=(5_000, 5_000), forwarded=9_000), rulebook) == []


def test_pass_through_needs_seventy_percent_forwarded(rulebook):
    assert rules.pass_through(_burst(forwarded=10_499), rulebook) == []  # 69.99%
    assert rules.pass_through(_burst(forwarded=10_500), rulebook) != []  # exactly 70%


def test_pass_through_ignores_credits_below_the_mule_floor(rulebook):
    assert rules.pass_through(_burst(amounts=(999, 5_000, 5_000), forwarded=10_000), rulebook) == []


def test_pass_through_ignores_repeat_payers(rulebook):
    history = [txn(counterparty_id="CP-NEW-0", timestamp="2026-07-01T10:00:00")] + _burst()
    assert rules.pass_through(history, rulebook) == []


def test_pass_through_live_window(rulebook):
    rows = _burst()
    inside = rows[-1]  # the forwarding debit, inside the burst window
    hit = rules.pass_through(rows, rulebook, within=inside)
    assert hit and hit[0].fired.transaction_id == inside.transaction_id

    later = txn(amount=500, direction=DEBIT, timestamp="2026-08-03T10:00:00")
    assert rules.pass_through(rows + [later], rulebook, within=later) == []


# --- re-investigation -----------------------------------------------------------------


def test_reinvestigation_needs_a_flag_and_another_rule():
    state = _escalated_state()
    some_hit = rules.kyc_event(TriageEventKind.KYC_SUBMITTED)
    assert rules.reinvestigation("ACC-T001", state, [some_hit]) is not None
    assert rules.reinvestigation("ACC-T001", state, []) is None
    assert rules.reinvestigation("ACC-UNFLAGGED", state, [some_hit]) is None


def test_inr_formatting():
    assert rules.inr(950_000) == "₹9,50,000"
    assert rules.inr(10_000_000) == "₹1,00,00,000"
    assert rules.inr(999) == "₹999"
