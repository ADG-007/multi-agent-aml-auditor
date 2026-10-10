"""Structuring sub-score: in-band cash in the busiest 30 days (phase_3_execution.md Q2-Q4, D3, D4).

The law's cash report (the CTR, STRUCT-001) covers cash only, so only cash deposits and cash
withdrawals are counted (Amendment 3, A1). A row is in band when its amount lies in
[threshold x near_threshold_pct, threshold], both edges inclusive: the CTR reports cash of
*more than* the threshold, so an amount exactly at it escapes the report and tops the band
(A3). The band is read from the rulebook at run time, never written here.

`k` is the largest number of in-band rows of one direction inside any sliding 30-day window
of the frame. Deposits and withdrawals are counted apart, each in its own busiest window, as
the RBI aggregates credits and debits apart (D3); the larger count is `k`. The legal figure -
each calendar month's cash totals - goes to the trail only (A2), as does the count of in-band
non-cash payments, which carry no amount report and so are not structuring.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from src.rulebook import RulebookLookup, StructuringThreshold
from src.schemas import Transaction, TransactionType
from src.stages.stage2_transaction_pattern import config


CASH_DIRECTIONS: tuple[TransactionType, ...] = (
    TransactionType.CASH_DEPOSIT,
    TransactionType.CASH_WITHDRAWAL,
)


@dataclass(frozen=True)
class BusiestWindow:
    """The rows of the 30-day window holding the most of some rows (empty if there are none)."""

    rows: tuple[Transaction, ...]

    @property
    def count(self) -> int:
        return len(self.rows)

    @property
    def first(self) -> datetime | None:
        return self.rows[0].timestamp if self.rows else None

    @property
    def last(self) -> datetime | None:
        return self.rows[-1].timestamp if self.rows else None

    @property
    def total(self) -> float:
        return sum(t.amount for t in self.rows)


@dataclass(frozen=True)
class MonthCash:
    """One calendar month's cash, all amounts, deposits and withdrawals apart (A2)."""

    month: date
    deposits: float
    withdrawals: float


@dataclass(frozen=True)
class StructuringResult:
    band: StructuringThreshold
    #: The busiest window of in-band rows, one per cash direction (D3).
    busiest: dict[TransactionType, BusiestWindow]
    k: int
    score: float
    month_totals: tuple[MonthCash, ...]
    in_band_non_cash: int

    @property
    def leading(self) -> BusiestWindow:
        """The direction that set k (deposits on a tie)."""
        return max(self.busiest.values(), key=lambda w: w.count)


def score(
    rows: Sequence[Transaction], as_of: datetime, rulebook: RulebookLookup
) -> StructuringResult:
    """Score the 90-day frame of `rows` ending at `as_of`."""
    band = rulebook.structuring()
    frame = in_window(rows, as_of, config.FRAME)

    def in_band(t: Transaction) -> bool:
        return band.band_floor_inr <= t.amount <= band.threshold_inr

    busiest = {
        kind: busiest_window([t for t in frame if t.type == kind and in_band(t)], config.WINDOW)
        for kind in CASH_DIRECTIONS
    }
    k = max(w.count for w in busiest.values())
    return StructuringResult(
        band=band,
        busiest=busiest,
        k=k,
        score=structuring_map(k),
        month_totals=_month_totals(frame),
        in_band_non_cash=sum(1 for t in frame if t.type not in CASH_DIRECTIONS and in_band(t)),
    )


def structuring_map(k: int) -> float:
    """The decided map (D4), linear between its points, flat beyond the last."""
    points = config.STRUCTURING_MAP
    if k <= points[0][0]:
        return points[0][1]
    for (k0, s0), (k1, s1) in zip(points, points[1:], strict=False):
        if k <= k1:
            return s0 + (s1 - s0) * (k - k0) / (k1 - k0)
    return points[-1][1]


# --------------------------------------------------------------------------------------
# Windows (shared with the pass-through sub-score and the velocity summary)
# --------------------------------------------------------------------------------------


def in_window(rows: Sequence[Transaction], end: datetime, span: timedelta) -> list[Transaction]:
    """The rows in (end - span, end], in the order given (the bank keeps them in time order)."""
    start = end - span
    return [t for t in rows if start < t.timestamp <= end]


def busiest_window(rows: Sequence[Transaction], span: timedelta) -> BusiestWindow:
    """The window [t, t + span), started at one of the rows, holding the most rows.

    The earliest such window wins a tie. `rows` must be in time order.
    """
    kept = list(rows)
    best: tuple[int, int] = (0, 0)  # (start index, end index exclusive)
    end = 0
    for start, first in enumerate(kept):
        end = max(end, start)
        while end < len(kept) and kept[end].timestamp < first.timestamp + span:
            end += 1
        if end - start > best[1] - best[0]:
            best = (start, end)
    return BusiestWindow(tuple(kept[best[0] : best[1]]))


def _month_totals(frame: Sequence[Transaction]) -> tuple[MonthCash, ...]:
    totals: dict[date, list[float]] = {}
    for t in frame:
        if t.type not in CASH_DIRECTIONS:
            continue
        month = t.timestamp.date().replace(day=1)
        pair = totals.setdefault(month, [0.0, 0.0])
        pair[0 if t.type == TransactionType.CASH_DEPOSIT else 1] += t.amount
    return tuple(MonthCash(m, d, w) for m, (d, w) in sorted(totals.items()))
