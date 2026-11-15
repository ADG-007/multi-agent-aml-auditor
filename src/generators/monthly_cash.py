"""Monthly cash totals against the CTR, as the generator and verify.py see them (Amendment 4).

The law counts connected cash over a calendar month (Rule 3(1)(B), rulebook STRUCT-002). Two
tests, per cash direction:

- a **band month**: one calendar month's cash of one direction totals ₹9–10 lakh (Q4);
- a **sliding hit**: a 30-day window holds ₹9 lakh or more of one direction, and no calendar
  month the window's rows fall in is above ₹10 lakh, so no cash report was due (Q10).

Stage 2 has its own implementation of the same tests; this one exists so the generator can
keep unplanted accounts clear of them and verify.py can check that it did.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta

from src.generators import config as cfg

WINDOW = timedelta(days=cfg.STRUCTURING_WINDOW_DAYS)
CASH_TYPES = ("CASH_DEPOSIT", "CASH_WITHDRAWAL")


def month_of(moment: datetime) -> date:
    return moment.date().replace(day=1)


def month_totals(rows: Iterable[tuple[datetime, float]]) -> dict[date, float]:
    totals: dict[date, float] = defaultdict(float)
    for when, amount in rows:
        totals[month_of(when)] += amount
    return dict(totals)


def in_band(total: float) -> bool:
    return cfg.MONTHLY_BAND_LOW <= total <= cfg.MONTHLY_BAND_HIGH


def band_months(rows: Sequence[tuple[datetime, float]]) -> list[date]:
    return sorted(m for m, total in month_totals(rows).items() if in_band(total))


def sliding_hits(rows: Sequence[tuple[datetime, float]]) -> list[tuple[int, int]]:
    """Non-overlapping 30-day windows meeting the Q10 test, as (start, end) row indices.

    `rows` must be in time order. Windows start at a row; the earliest qualifying one is
    taken and the next must start after it ends, which counts the most windows possible.
    """
    months = month_totals(rows)
    hits: list[tuple[int, int]] = []
    start = 0
    while start < len(rows):
        end = start
        total = 0.0
        while end < len(rows) and rows[end][0] < rows[start][0] + WINDOW:
            total += rows[end][1]
            end += 1
        touched = {month_of(when) for when, _ in rows[start:end]}
        if total >= cfg.MONTHLY_BAND_LOW and all(
            months[m] <= cfg.MONTHLY_BAND_HIGH for m in touched
        ):
            hits.append((start, end))
            boundary = rows[start][0] + WINDOW
            while start < len(rows) and rows[start][0] < boundary:
                start += 1
            continue
        start += 1
    return hits


def first_violation(rows: Sequence[tuple[datetime, float]]) -> list[int] | None:
    """Row indices of the first band month or sliding hit, or None if there is neither."""
    months = month_totals(rows)
    for month in sorted(months):
        if in_band(months[month]):
            return [i for i, (when, _) in enumerate(rows) if month_of(when) == month]
    hits = sliding_hits(rows)
    if hits:
        start, end = hits[0]
        return list(range(start, end))
    return None
