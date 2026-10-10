"""Stage 2 structuring sub-score: the band edges, cash only, per-direction k and the frame."""

from datetime import datetime, timedelta

import pytest

from src.rulebook import RulebookLookup
from src.schemas import RulebookCategory, RulebookClause, TransactionDirection, TransactionType
from src.stages.stage2_transaction_pattern import structuring
from tests.factories import txn


AS_OF = datetime(2026, 9, 21, 23, 0)
DEPOSIT = TransactionType.CASH_DEPOSIT
WITHDRAWAL = TransactionType.CASH_WITHDRAWAL
DEBIT = TransactionDirection.DEBIT


@pytest.fixture(scope="module")
def rulebook() -> RulebookLookup:
    return RulebookLookup.load()


def _deposit(amount=950_000, day="2026-08-01", hour=10):
    return txn(amount=amount, type=DEPOSIT, timestamp=f"{day}T{hour:02d}:00:00")


def _withdrawal(amount=950_000, day="2026-08-01", hour=10):
    return txn(amount=amount, type=WITHDRAWAL, direction=DEBIT, timestamp=f"{day}T{hour:02d}:00:00")


def _days(start, n, make=_deposit, **kw):
    first = datetime.fromisoformat(start)
    return [make(day=f"{first + timedelta(days=i):%Y-%m-%d}", **kw) for i in range(n)]


def test_top_edge_is_in_band_and_one_rupee_more_is_not(rulebook):
    assert structuring.score([_deposit(1_000_000)], AS_OF, rulebook).k == 1
    assert structuring.score([_deposit(1_000_001)], AS_OF, rulebook).k == 0


def test_bottom_edge_is_in_band_and_one_rupee_less_is_not(rulebook):
    assert structuring.score([_deposit(900_000)], AS_OF, rulebook).k == 1
    assert structuring.score([_deposit(899_999)], AS_OF, rulebook).k == 0


def test_in_band_non_cash_payment_is_not_counted(rulebook):
    rows = [txn(amount=950_000, type=t) for t in (TransactionType.NEFT, TransactionType.RTGS)]
    result = structuring.score(rows, AS_OF, rulebook)
    assert result.k == 0 and result.score == 0.0
    assert result.in_band_non_cash == 2


def test_three_deposits_score_040(rulebook):
    result = structuring.score(_days("2026-08-01", 3), AS_OF, rulebook)
    assert result.k == 3 and result.score == pytest.approx(0.40)


def test_deposits_and_withdrawals_are_counted_apart(rulebook):
    rows = _days("2026-08-01", 3) + _days("2026-08-10", 3, make=_withdrawal)
    rows.sort(key=lambda t: t.timestamp)
    result = structuring.score(rows, AS_OF, rulebook)
    assert result.k == 3
    assert result.busiest[DEPOSIT].count == 3 and result.busiest[WITHDRAWAL].count == 3


def test_burst_across_a_month_boundary_counts_whole(rulebook):
    rows = _days("2026-07-29", 4)  # 29, 30, 31 Jul and 1 Aug
    result = structuring.score(rows, AS_OF, rulebook)
    assert result.k == 4 and result.score == pytest.approx(0.50)
    assert [m.month.month for m in result.month_totals] == [7, 8]


def test_busiest_window_is_thirty_days(rulebook):
    rows = [_deposit(day="2026-07-01"), _deposit(day="2026-07-30"), _deposit(day="2026-07-31")]
    assert structuring.score(rows, AS_OF, rulebook).k == 2  # 1 Jul to 31 Jul is over 30 days


def test_rows_outside_the_frame_are_ignored(rulebook):
    before = _days("2026-05-01", 5)  # more than 90 days before AS_OF
    after = _days("2026-09-22", 5)  # after AS_OF
    inside = _days("2026-08-01", 2)
    result = structuring.score(before + inside + after, AS_OF, rulebook)
    assert result.k == 2


def test_map_points_and_cap():
    expected = {0: 0.0, 1: 0.15, 2: 0.30, 3: 0.40, 4: 0.50, 5: 0.65, 6: 0.80, 7: 0.90, 8: 1.0}
    for k, value in expected.items():
        assert structuring.structuring_map(k) == pytest.approx(value)
    assert structuring.structuring_map(15) == 1.0


def test_month_totals_cover_all_cash_both_directions(rulebook):
    rows = [
        _deposit(5_000, day="2026-08-02"),
        _deposit(950_000, day="2026-08-03"),
        _withdrawal(2_000, day="2026-08-04"),
        txn(amount=40_000, type=TransactionType.UPI, timestamp="2026-08-05T10:00:00"),
    ]
    (august,) = structuring.score(rows, AS_OF, rulebook).month_totals
    assert (august.deposits, august.withdrawals) == (955_000, 2_000)


def test_a_custom_rulebook_moves_the_band():
    custom = RulebookLookup(
        [
            RulebookClause(
                clause_id="STRUCT-001",
                category=RulebookCategory.STRUCTURING,
                title="test",
                text="test",
                sources=["test"],
                metadata={"threshold_inr": 500_000, "near_threshold_pct": 0.8},
            )
        ]
    )
    rows = [_deposit(400_000, day="2026-08-01"), _deposit(950_000, day="2026-08-02")]
    result = structuring.score(rows, AS_OF, custom)
    assert result.k == 1 and result.busiest[DEPOSIT].rows[0].amount == 400_000
