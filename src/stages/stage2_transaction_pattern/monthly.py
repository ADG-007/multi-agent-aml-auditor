"""Monthly cash sub-score: months of cash kept just under the CTR (Amendment 4, Q4, Q5, Q9, Q10).

A structurer who knows the monthly aggregate (STRUCT-002) keeps each calendar month's cash
under the threshold with many smaller deposits, so the per-deposit count `k` sees nothing.
This sub-score counts, per cash direction, the calendar months in the band and the separate
30-day windows over the floor with no month above the threshold (cash_aggregate.py). `m` is
the higher of the two counts, over both directions; the month map turns it into a score. The
band is read from STRUCT-001's monthly metadata at run time, never written here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from src.rulebook import RulebookLookup, StructuringThreshold
from src.schemas import Transaction, TransactionType
from src.stages.cash_aggregate import BandMonth, SlidingWindow, band_months, sliding_windows
from src.stages.stage2_transaction_pattern import config
from src.stages.stage2_transaction_pattern.structuring import CASH_DIRECTIONS, in_window, piecewise


@dataclass(frozen=True)
class DirectionMonthly:
    band_months: tuple[BandMonth, ...]
    windows: tuple[SlidingWindow, ...]

    @property
    def m(self) -> int:
        return max(len(self.band_months), len(self.windows))


@dataclass(frozen=True)
class MonthlyResult:
    band: StructuringThreshold
    by_direction: dict[TransactionType, DirectionMonthly]
    m: int
    score: float

    @property
    def leading(self) -> tuple[TransactionType, DirectionMonthly]:
        """The direction that set m (deposits on a tie)."""
        return max(self.by_direction.items(), key=lambda kv: kv[1].m)


def score(rows: Sequence[Transaction], as_of: datetime, rulebook: RulebookLookup) -> MonthlyResult:
    """Score the 90-day frame of `rows` ending at `as_of`."""
    band = rulebook.monthly_structuring()
    frame = in_window(rows, as_of, config.FRAME)
    by_direction = {}
    for kind in CASH_DIRECTIONS:
        cash = [t for t in frame if t.type == kind]
        by_direction[kind] = DirectionMonthly(
            band_months=tuple(band_months(cash, band)),
            windows=tuple(sliding_windows(cash, band, config.WINDOW)),
        )
    m = max(d.m for d in by_direction.values())
    return MonthlyResult(band=band, by_direction=by_direction, m=m, score=monthly_map(m))


def monthly_map(m: int) -> float:
    """The decided map (Q9), linear between its points, flat beyond the last."""
    return piecewise(config.MONTHLY_MAP, m)
