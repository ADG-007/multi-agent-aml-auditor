"""Pass-through sub-score: strangers' money sent on within hours (a mule's pattern).

A shop also has many one-off payers, but its money stays for days. So the score is led by how
many stranger credits leave fast. Every rail counts.

- Stranger credit: at or above the mule floor, from a payer seen once in the window.
- Run: stranger credits plus the debits after them. It closes when debits reach
  RUN_CLOSE_SHARE of its credits.
- Score: gate(n) x share(n / matched) x (BASE + BOOST x novelty), n = credits held <= 12 h.

The 90-day frame and its last 30 days are both scored; the higher wins.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.rulebook import RulebookLookup
from src.schemas import Transaction, TransactionDirection
from src.stages.stage2_transaction_pattern import config
from src.stages.stage2_transaction_pattern.structuring import in_window


CREDIT = TransactionDirection.CREDIT
DEBIT = TransactionDirection.DEBIT


@dataclass(frozen=True)
class Run:
    """A closed run: stranger credits, the debits that closed it, and when it closed."""

    credits: tuple[Transaction, ...]
    debits: tuple[Transaction, ...]
    closed_at: datetime

    def holds(self) -> list[timedelta]:
        return [self.closed_at - c.timestamp for c in self.credits]

    @property
    def fast(self) -> bool:
        return any(h <= config.FAST_HOLD for h in self.holds())


@dataclass(frozen=True)
class WindowScore:
    """One window's measurements and its score."""

    label: str
    end: datetime
    #: Credits at or above the floor, and how many of them came from strangers.
    credits_at_floor: int
    strangers: int
    #: Stranger credits in closed runs, and those held at most FAST_HOLD_HOURS (n).
    matched: int
    n: int
    fast_holds: tuple[timedelta, ...]
    novelty: float
    #: Of the money in the fast runs' debits, the share that went to the largest payee.
    top_payee_share: float
    gate: float
    share: float
    share_mapped: float
    score: float

    @property
    def stranger_share(self) -> float:
        return self.strangers / self.credits_at_floor if self.credits_at_floor else 0.0

    @property
    def median_fast_hold(self) -> timedelta | None:
        return statistics.median(self.fast_holds) if self.fast_holds else None


@dataclass(frozen=True)
class PassThroughResult:
    floor: float
    full: WindowScore
    recent: WindowScore

    @property
    def best(self) -> WindowScore:
        """The higher-scoring window; the full frame on a tie."""
        return self.recent if self.recent.score > self.full.score else self.full

    @property
    def score(self) -> float:
        return self.best.score


def score(
    rows: Sequence[Transaction], as_of: datetime, rulebook: RulebookLookup
) -> PassThroughResult:
    """Score the 90-day frame of `rows` ending at `as_of`, and its last 30 days."""
    floor = rulebook.mule_floor()
    frame = in_window(rows, as_of, config.FRAME)
    recent = in_window(rows, as_of, config.RECENT)
    return PassThroughResult(
        floor=floor,
        full=score_window(frame, frame, floor, f"{config.FRAME_DAYS} days", as_of),
        recent=score_window(recent, frame, floor, f"last {config.RECENT_DAYS} days", as_of),
    )


def score_window(
    window: Sequence[Transaction],
    frame: Sequence[Transaction],
    floor: float,
    label: str,
    end: datetime,
) -> WindowScore:
    """Score one window. `frame` is the full 90 days, the history novelty is judged against."""
    strangers = stranger_credits(window, floor)
    runs = close_runs(window, strangers, floor)
    holds = [h for run in runs for h in run.holds()]
    fast_holds = tuple(h for h in holds if h <= config.FAST_HOLD)
    fast_runs = [run for run in runs if run.fast]

    n, matched = len(fast_holds), len(holds)
    share = n / matched if matched else 0.0
    share_mapped = _share_map(share)
    novelty, top_payee = _destinations(fast_runs, frame)
    gate = gate_for(n)
    return WindowScore(
        label=label,
        end=end,
        credits_at_floor=sum(1 for t in window if t.direction == CREDIT and t.amount >= floor),
        strangers=len(strangers),
        matched=matched,
        n=n,
        fast_holds=fast_holds,
        novelty=novelty,
        top_payee_share=top_payee,
        gate=gate,
        share=share,
        share_mapped=share_mapped,
        score=gate * share_mapped * (config.NOVELTY_BASE + config.NOVELTY_BOOST * novelty),
    )


def stranger_credits(window: Sequence[Transaction], floor: float) -> set[str]:
    """Ids of credits at or above the floor whose payer appears once among the window's credits."""
    payers = Counter(t.counterparty_id for t in window if t.direction == CREDIT)
    return {
        t.transaction_id
        for t in window
        if t.direction == CREDIT and t.amount >= floor and payers[t.counterparty_id] == 1
    }


def close_runs(window: Sequence[Transaction], strangers: set[str], floor: float) -> list[Run]:
    """Run-close matching. A run still open at the window's end matches nothing."""
    runs: list[Run] = []
    credits: list[Transaction] = []
    debits: list[Transaction] = []
    for t in window:
        if t.transaction_id in strangers:
            credits.append(t)
        elif t.direction == DEBIT and t.amount >= floor and credits:
            debits.append(t)
            if sum(d.amount for d in debits) >= config.RUN_CLOSE_SHARE * sum(
                c.amount for c in credits
            ):
                runs.append(Run(tuple(credits), tuple(debits), t.timestamp))
                credits, debits = [], []
    return runs


def gate_for(n: int) -> float:
    """0 below the gate table's first count; else the factor of the highest count reached."""
    factor = 0.0
    for count, value in config.GATE:
        if n >= count:
            factor = value
    return factor


def _share_map(share: float) -> float:
    span = config.SHARE_HIGH - config.SHARE_LOW
    return min(1.0, max(0.0, (share - config.SHARE_LOW) / span))


def _destinations(fast_runs: Sequence[Run], frame: Sequence[Transaction]) -> tuple[float, float]:
    """(novelty, top payee's share) of the fast runs' debits. Novelty = share sent to payees
    not paid before the first fast run. Both 0 with no fast run."""
    if not fast_runs:
        return 0.0, 0.0
    started = min(c.timestamp for c in fast_runs[0].credits)
    known = {t.counterparty_id for t in frame if t.direction == DEBIT and t.timestamp < started}
    by_payee: Counter[str] = Counter()
    for run in fast_runs:
        for d in run.debits:
            by_payee[d.counterparty_id] += d.amount
    total = sum(by_payee.values())
    novel = sum(amount for payee, amount in by_payee.items() if payee not in known)
    return novel / total, max(by_payee.values()) / total
