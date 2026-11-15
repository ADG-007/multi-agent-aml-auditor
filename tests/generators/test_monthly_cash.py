"""Generator side of Amendment 4: the monthly tests, the band guard and the structurer planting."""

import random
from datetime import date, datetime, timedelta

from src.generators import config as cfg
from src.generators import monthly_cash
from src.generators.transactions import (
    _plant_monthly_structuring,
    branch_weekdays,
    history_months,
    keep_cash_out_of_band,
    style_max,
)
from src.schemas import TransactionType


DEPOSIT = TransactionType.CASH_DEPOSIT
WITHDRAWAL = TransactionType.CASH_WITHDRAWAL


def _row(amount, when, kind=DEPOSIT):
    return {"amount": amount, "type": kind, "timestamp": when}


def _spread(total, start, n=10, kind=DEPOSIT):
    first = datetime.fromisoformat(start)
    return [_row(total / n, first + timedelta(days=i), kind) for i in range(n)]


def _pairs(rows, kind=DEPOSIT):
    cash = sorted((r for r in rows if r["type"] == kind), key=lambda r: r["timestamp"])
    return [(r["timestamp"], r["amount"]) for r in cash]


def test_band_month_and_sliding_hit():
    assert monthly_cash.band_months(_pairs(_spread(950_000, "2026-07-01"))) == [date(2026, 7, 1)]
    across = _spread(700_000, "2026-07-20", n=7) + _spread(700_000, "2026-08-03", n=7)
    assert monthly_cash.band_months(_pairs(across)) == []
    assert len(monthly_cash.sliding_hits(_pairs(across))) == 1


def test_guard_moves_cash_until_no_test_fires():
    rows = _spread(950_000, "2026-07-01") + _spread(940_000, "2026-08-03", kind=WITHDRAWAL)
    keep_cash_out_of_band(rows)
    for kind in (DEPOSIT, WITHDRAWAL):
        assert monthly_cash.first_violation(_pairs(rows, kind)) is None
    moved = [r for r in rows if r["type"] == TransactionType.NEFT]
    assert moved and sum(r["amount"] for r in rows) == 950_000 + 940_000


def test_guard_can_keep_planted_deposits():
    rows = _spread(950_000, "2026-07-01")
    keep_cash_out_of_band(rows, keep_deposits=True)
    assert all(r["type"] == DEPOSIT for r in rows)


def test_monthly_cap_planting_keeps_each_month_in_the_band():
    for style in cfg.STRUCTURING_STYLES:
        for seed in range(5):
            rng = random.Random(seed)
            ordinary = _spread(250_000, "2026-07-06", n=2) + _spread(150_000, "2026-08-10", n=3)

            def row(amount, kind, day, counterparty, direction):
                return _row(
                    amount, datetime.combine(day, datetime.min.time()) + timedelta(hours=11), kind
                )

            planted = _plant_monthly_structuring(rng, row, ordinary, style, [None])
            deposits = _pairs(ordinary + planted)
            totals = monthly_cash.month_totals(deposits)
            months = monthly_cash.band_months(deposits)
            assert cfg.STRUCTURING_MONTHS[0] <= len(months) <= cfg.STRUCTURING_MONTHS[1]
            assert all(total <= cfg.CTR_THRESHOLD_INR for total in totals.values())
            assert all(
                cfg.STRUCTURING_DEPOSIT_MIN <= r["amount"] <= style_max(style) for r in planted
            )
            assert all(r["timestamp"].weekday() < 5 for r in planted)


def test_history_months_and_weekdays():
    assert [m.month for m in history_months()] == [6, 7, 8, 9]
    assert all(d.weekday() < 5 and d.month == 8 for d in branch_weekdays(date(2026, 8, 1)))
    assert date(2026, 8, 1) not in branch_weekdays(date(2026, 8, 1))  # a Saturday
