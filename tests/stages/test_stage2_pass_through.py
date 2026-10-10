"""Stage 2 pass-through sub-score on small hand-built accounts (phase_3_execution.md step 4)."""

from datetime import datetime, timedelta

import pytest

from src.rulebook import RulebookLookup
from src.schemas import TransactionDirection, TransactionType
from src.stages.stage2_transaction_pattern import pass_through
from tests.factories import txn


AS_OF = datetime(2026, 9, 21, 23, 0)
CREDIT = TransactionDirection.CREDIT
DEBIT = TransactionDirection.DEBIT


@pytest.fixture(scope="module")
def rulebook() -> RulebookLookup:
    return RulebookLookup.load()


def _at(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def _credit(when, amount, payer, type=TransactionType.UPI):
    return txn(amount=amount, timestamp=_at(when), counterparty_id=payer, type=type)


def _debit(when, amount, payee, type=TransactionType.IMPS):
    return txn(
        amount=amount, direction=DEBIT, timestamp=_at(when), counterparty_id=payee, type=type
    )


def _salary_history(start: datetime) -> list:
    """A quiet ordinary history: monthly salary in, rent and groceries out to known payees."""
    rows = []
    for month in range(2):
        day = start + timedelta(days=30 * month)
        rows += [
            _credit(day, 60_000, "CP-EMPLOYER", TransactionType.NEFT),
            _debit(day + timedelta(days=1), 20_000, "CP-LANDLORD"),
            _debit(day + timedelta(days=3), 4_000, "CP-GROCER", TransactionType.UPI),
        ]
    return rows


def _mule_burst(start: datetime, n: int = 6, payee="CP-COLLECTOR", types=None) -> list:
    """n strangers pay 20,000 each, a day apart; each is swept on 2 hours later."""
    rows = []
    for i in range(n):
        when = start + timedelta(days=i)
        kind = types[i % len(types)] if types else TransactionType.UPI
        rows += [
            _credit(when, 20_000, f"CP-VICTIM-{i}", kind),
            _debit(when + timedelta(hours=2), 19_000, payee),
        ]
    return rows


def _score(rows, rulebook):
    rows = sorted(rows, key=lambda t: (t.timestamp, t.transaction_id))
    return pass_through.score(rows, AS_OF, rulebook)


def test_mule_sweeping_strangers_to_a_new_payee_scores_high(rulebook):
    rows = _salary_history(datetime(2026, 7, 1, 9)) + _mule_burst(datetime(2026, 9, 1, 10))
    result = _score(rows, rulebook)
    assert result.score >= 0.75
    best = result.best
    assert best.n == 6 and best.matched == 6 and best.novelty == 1.0
    assert best.median_fast_hold == timedelta(hours=2)
    assert best.top_payee_share == 1.0


def test_sweeping_to_a_payee_paid_before_keeps_only_the_base_factor(rulebook):
    rows = _salary_history(datetime(2026, 7, 1, 9)) + _mule_burst(
        datetime(2026, 9, 1, 10), payee="CP-LANDLORD"
    )
    result = _score(rows, rulebook)
    assert result.best.novelty == 0.0
    assert result.score == pytest.approx(0.7)


def test_merchant_holding_for_days_and_paying_repeat_suppliers_scores_zero(rulebook):
    start = datetime(2026, 6, 25, 9)
    rows = [_debit(start, 30_000, "CP-SUPPLIER")]
    for week in range(12):
        monday = start + timedelta(days=7 * week + 1)
        for i in range(5):
            rows.append(_credit(monday + timedelta(hours=8 * i), 6_000, f"CP-BUYER-{week}-{i}"))
        rows.append(_debit(monday + timedelta(days=4), 28_000, "CP-SUPPLIER"))
    result = _score(rows, rulebook)
    assert result.full.strangers == 60 and result.full.matched > 0
    assert result.full.n == 0
    assert result.score == 0.0


def test_a_single_life_event_is_gated_to_zero(rulebook):
    when = datetime(2026, 9, 10, 9)
    rows = _salary_history(datetime(2026, 7, 1, 9)) + [
        _credit(when, 50_000, "CP-UNCLE"),
        _credit(when + timedelta(hours=1), 50_000, "CP-AUNT"),
        _debit(when + timedelta(hours=3), 100_000, "CP-HOSPITAL"),
    ]
    result = _score(rows, rulebook)
    assert result.best.n == 2 and result.best.gate == 0.0
    assert result.score == 0.0


def test_credits_below_the_floor_are_ignored(rulebook):
    floor = rulebook.mule_floor()
    start = datetime(2026, 9, 1, 10)
    rows = []
    for i in range(6):
        when = start + timedelta(days=i)
        rows += [
            _credit(when, floor - 1, f"CP-VICTIM-{i}"),
            _debit(when + timedelta(hours=2), floor, "CP-COLLECTOR"),
        ]
    result = _score(rows, rulebook)
    assert result.full.strangers == 0 and result.full.credits_at_floor == 0
    assert result.score == 0.0


def test_a_burst_only_in_the_last_30_days_is_found_by_the_recent_window(rulebook):
    # 60 days of slow stranger money (held 3 days), then a fast burst in the last 30 days.
    rows = []
    start = datetime(2026, 6, 26, 9)
    for i in range(20):
        when = start + timedelta(days=3 * i)
        rows += [
            _credit(when, 10_000, f"CP-BUYER-{i}"),
            _debit(when + timedelta(days=2), 9_000, "CP-SUPPLIER"),
        ]
    rows += _mule_burst(datetime(2026, 9, 5, 10), n=5)
    result = _score(rows, rulebook)
    assert result.full.n == 5 and result.full.score == 0.0  # 5 of 25 is a share of 0.20
    assert result.recent.n == 5 and result.recent.score >= 0.75
    assert result.best is result.recent and result.best.label == "last 30 days"


def test_any_rail_counts(rulebook):
    rails = [TransactionType.NEFT, TransactionType.IMPS, TransactionType.RTGS]
    upi = _score(_mule_burst(datetime(2026, 9, 1, 10)), rulebook)
    mixed = _score(_mule_burst(datetime(2026, 9, 1, 10), types=rails), rulebook)
    assert mixed.score == upi.score == 1.0


def test_a_repeat_payer_is_not_a_stranger(rulebook):
    rows = _mule_burst(datetime(2026, 9, 1, 10))
    rows.append(_credit(datetime(2026, 9, 15, 10), 500, "CP-VICTIM-0"))  # pays again, small
    result = _score(rows, rulebook)
    assert result.full.strangers == 5


def test_run_closes_only_at_70_percent(rulebook):
    when = datetime(2026, 9, 1, 10)
    rows = [
        _credit(when, 10_000, "CP-A"),
        _credit(when + timedelta(hours=1), 10_000, "CP-B"),
        _debit(when + timedelta(hours=2), 13_000, "CP-OUT"),  # 65%: still open
        _debit(when + timedelta(hours=30), 1_000, "CP-OUT"),  # 70%: closes here
    ]
    rows = sorted(rows, key=lambda t: t.timestamp)
    strangers = pass_through.stranger_credits(rows, rulebook.mule_floor())
    (run,) = pass_through.close_runs(rows, strangers, rulebook.mule_floor())
    assert run.closed_at == when + timedelta(hours=30)
    assert run.holds() == [timedelta(hours=30), timedelta(hours=29)]


def test_gate_table():
    assert [pass_through.gate_for(n) for n in range(8)] == [0, 0, 0, 0.6, 0.8, 1.0, 1.0, 1.0]
