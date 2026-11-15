"""Monthly cash aggregates against the CTR, shared by Stage 0 and Stage 2 (Amendment 4).

The law treats connected cash in one calendar month as one transaction when the total is more
than the threshold (Rule 3(1)(B), STRUCT-002). Two findings, each for one cash direction
(deposits and withdrawals apart, as D3):

- a **band month** (Q4): a calendar month's total lies in [floor, threshold], both edges
  inclusive, so no report was due although the total came close;
- a **sliding window** (Q10): 30 days holding the floor or more, where no calendar month the
  window's rows fall in totals more than the threshold. Separate (non-overlapping) windows are
  counted, earliest first.

The band is passed in, read from the rulebook by the caller. Rows must be one direction, in
time order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from src.rulebook import StructuringThreshold
from src.schemas import Transaction


@dataclass(frozen=True)
class BandMonth:
    month: date
    total: float
    #: The row at which the month's running total first reached the floor.
    reached_by: Transaction


@dataclass(frozen=True)
class SlidingWindow:
    rows: tuple[Transaction, ...]
    #: The row at which the window's running total first reached the floor.
    reached_by: Transaction

    @property
    def total(self) -> float:
        return sum(t.amount for t in self.rows)

    @property
    def first(self) -> datetime:
        return self.rows[0].timestamp

    @property
    def last(self) -> datetime:
        return self.rows[-1].timestamp


def month_of(moment: datetime) -> date:
    return moment.date().replace(day=1)


def month_totals(rows: Sequence[Transaction]) -> dict[date, float]:
    totals: dict[date, float] = {}
    for t in rows:
        month = month_of(t.timestamp)
        totals[month] = totals.get(month, 0.0) + t.amount
    return totals


def band_months(rows: Sequence[Transaction], band: StructuringThreshold) -> list[BandMonth]:
    found = []
    totals = month_totals(rows)
    for month in sorted(totals):
        total = totals[month]
        if not band.band_floor_inr <= total <= band.threshold_inr:
            continue
        running = 0.0
        for t in rows:
            if month_of(t.timestamp) == month:
                running += t.amount
                if running >= band.band_floor_inr:
                    found.append(BandMonth(month, total, t))
                    break
    return found


def sliding_windows(
    rows: Sequence[Transaction], band: StructuringThreshold, span: timedelta
) -> list[SlidingWindow]:
    totals = month_totals(rows)
    found: list[SlidingWindow] = []
    start = 0
    while start < len(rows):
        end = start
        while end < len(rows) and rows[end].timestamp < rows[start].timestamp + span:
            end += 1
        window = rows[start:end]
        total = sum(t.amount for t in window)
        touched = {month_of(t.timestamp) for t in window}
        if total >= band.band_floor_inr and all(totals[m] <= band.threshold_inr for m in touched):
            running = 0.0
            reached = window[-1]
            for t in window:
                running += t.amount
                if running >= band.band_floor_inr:
                    reached = t
                    break
            found.append(SlidingWindow(tuple(window), reached))
            boundary = rows[start].timestamp + span
            while start < len(rows) and rows[start].timestamp < boundary:
                start += 1
            continue
        start += 1
    return found
