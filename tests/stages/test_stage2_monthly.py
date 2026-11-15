"""Stage 2 monthly cash sub-score: band months, sliding windows, the map and the band edges."""

from datetime import datetime, timedelta

import pytest

from src.rulebook import RulebookLookup
from src.schemas import RulebookCategory, RulebookClause, TransactionDirection, TransactionType
from src.stages.stage2_transaction_pattern import monthly
from tests.factories import txn


AS_OF = datetime(2026, 9, 21, 23, 0)
DEPOSIT = TransactionType.CASH_DEPOSIT
WITHDRAWAL = TransactionType.CASH_WITHDRAWAL


@pytest.fixture(scope="module")
def rulebook() -> RulebookLookup:
    return RulebookLookup.load()


def _month(total, start, n=10, make_type=DEPOSIT):
    """`n` equal cash rows of one direction, a day apart from `start`, adding up to `total`."""
    first = datetime.fromisoformat(start)
    direction = TransactionDirection.CREDIT if make_type == DEPOSIT else TransactionDirection.DEBIT
    return [
        txn(
            amount=total / n,
            type=make_type,
            direction=direction,
            timestamp=f"{first + timedelta(days=i):%Y-%m-%dT11:00:00}",
        )
        for i in range(n)
    ]


@pytest.mark.parametrize("m, expected", [(0, 0.0), (1, 0.30), (2, 0.60), (3, 0.85), (4, 0.85)])
def test_month_map(m, expected):
    assert monthly.monthly_map(m) == pytest.approx(expected)


def test_band_edges_are_inclusive(rulebook):
    assert monthly.score(_month(900_000, "2026-07-01"), AS_OF, rulebook).m == 1
    assert monthly.score(_month(1_000_000, "2026-07-01"), AS_OF, rulebook).m == 1
    assert monthly.score(_month(899_990, "2026-07-01"), AS_OF, rulebook).m == 0


def test_a_month_over_the_threshold_is_not_a_band_month(rulebook):
    """A month above ₹10 lakh was reported; neither test counts it."""
    result = monthly.score(_month(1_000_010, "2026-07-01"), AS_OF, rulebook)
    assert result.m == 0 and result.score == 0.0


def test_two_band_months_score_060(rulebook):
    rows = _month(950_000, "2026-07-01") + _month(940_000, "2026-08-03")
    result = monthly.score(rows, AS_OF, rulebook)
    deposits = result.by_direction[DEPOSIT]
    assert [b.month.month for b in deposits.band_months] == [7, 8]
    assert result.m == 2 and result.score == pytest.approx(0.60)


def test_window_across_a_month_end_counts(rulebook):
    """The Q5 example: ₹7 lakh late July + ₹7 lakh early August, each month under the band."""
    rows = _month(700_000, "2026-07-20", n=7) + _month(700_000, "2026-08-03", n=7)
    result = monthly.score(rows, AS_OF, rulebook)
    deposits = result.by_direction[DEPOSIT]
    assert deposits.band_months == () and len(deposits.windows) == 1
    assert result.m == 1 and result.score == pytest.approx(0.30)


def test_window_touching_a_reported_month_does_not_count(rulebook):
    rows = _month(1_200_000, "2026-07-01", n=12) + _month(500_000, "2026-08-01", n=5)
    assert monthly.score(rows, AS_OF, rulebook).m == 0


def test_directions_are_counted_apart(rulebook):
    rows = _month(500_000, "2026-07-01") + _month(500_000, "2026-07-02", make_type=WITHDRAWAL)
    assert monthly.score(rows, AS_OF, rulebook).m == 0
    rows = _month(950_000, "2026-07-01", make_type=WITHDRAWAL)
    result = monthly.score(rows, AS_OF, rulebook)
    assert result.m == 1 and result.leading[0] == WITHDRAWAL


def test_non_cash_is_not_counted(rulebook):
    rows = [
        txn(amount=95_000, type=TransactionType.NEFT, timestamp=f"2026-07-{d:02d}T11:00:00")
        for d in range(1, 11)
    ]
    assert monthly.score(rows, AS_OF, rulebook).m == 0


def test_band_comes_from_the_rulebook():
    clause = RulebookClause(
        clause_id="STRUCT-001",
        title="t",
        text="t",
        category=RulebookCategory.STRUCTURING,
        sources=[],
        metadata={
            "threshold_inr": 1_000_000,
            "near_threshold_pct": 0.9,
            "monthly_threshold_inr": 500_000,
            "monthly_near_threshold_pct": 0.8,
        },
    )
    custom = RulebookLookup([clause])
    assert monthly.score(_month(450_000, "2026-07-01"), AS_OF, custom).m == 1
    assert monthly.score(_month(950_000, "2026-07-01"), AS_OF, custom).m == 0
