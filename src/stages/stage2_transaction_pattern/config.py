"""Stage 2's own tunable numbers (phase_3_execution.md, Settled decisions and D4-D6).

Only numbers that no rulebook clause defines live here. The structuring band (STRUCT-001's
threshold_inr and near_threshold_pct) and the mule transaction floor (MULE-002's
txn_floor_inr) are read from the rulebook at run time and must never be copied into this file.
"""

from __future__ import annotations

from datetime import timedelta


#: The frame Stage 2 reads: the 90 days ending at the as-of time (D2).
FRAME_DAYS: int = 90
#: The sliding window the structuring count `k` and the velocity summary use (Q4, A2). The
#: law sets no window; 30 days matches the RBI's monthly aggregation in length.
WINDOW_DAYS: int = 30

#: Structuring map (D4): in-band cash count k -> score, linear between points. Every k at or
#: above the last point scores the last score.
STRUCTURING_MAP: tuple[tuple[int, float], ...] = (
    (0, 0.0),
    (1, 0.15),
    (2, 0.30),
    (3, 0.40),
    (4, 0.50),
    (5, 0.65),
    (6, 0.80),
    (7, 0.90),
    (8, 1.0),
)

#: A matched stranger credit counts towards n when sent on within this many hours (D6).
FAST_HOLD_HOURS: float = 12.0
#: A run of stranger credits closes when its debits reach this share of its credits (D5).
RUN_CLOSE_SHARE: float = 0.70

#: The gate on n, the fast stranger credits (D6): n below the first entry scores 0, and n at
#: or above an entry's count takes that entry's factor (highest matching entry wins).
GATE: tuple[tuple[int, float], ...] = (
    (3, 0.6),
    (4, 0.8),
    (5, 1.0),
)

#: The share map (D6): n / matched stranger credits, linear from 0 at the low edge to 1 at the
#: high edge, clamped.
SHARE_LOW: float = 0.25
SHARE_HIGH: float = 0.60

#: The novelty booster (D6): window score = gate x share x (BASE + BOOST x novelty).
NOVELTY_BASE: float = 0.7
NOVELTY_BOOST: float = 0.3

#: The recent window scored beside the full frame (D6): the last 30 days of it.
RECENT_DAYS: int = 30

#: Round-amount share (trail only, Q3): an amount that is a whole multiple of this is round.
ROUND_AMOUNT_INR: int = 1_000

#: The reasoning trail holds at most this many steps (D8).
MAX_TRAIL_STEPS: int = 8

#: Derived spans, kept beside the day counts they come from.
FRAME: timedelta = timedelta(days=FRAME_DAYS)
WINDOW: timedelta = timedelta(days=WINDOW_DAYS)
RECENT: timedelta = timedelta(days=RECENT_DAYS)
FAST_HOLD: timedelta = timedelta(hours=FAST_HOLD_HOURS)
