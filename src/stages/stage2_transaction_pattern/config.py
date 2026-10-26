"""Stage 2's own numbers. The cash band and mule floor come from the rulebook, never here."""

from __future__ import annotations

from datetime import timedelta


#: The 90 days ending at the as-of time.
FRAME_DAYS: int = 90
#: Sliding window for `k` and velocity (same length as the RBI's monthly total).
WINDOW_DAYS: int = 30

#: In-band cash count k -> score, linear between points, flat after the last.
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

#: A stranger credit counts towards n when sent on within this many hours.
FAST_HOLD_HOURS: float = 12.0
#: A run closes when its debits reach this share of its credits.
RUN_CLOSE_SHARE: float = 0.70

#: Gate on n: below the first count scores 0, else the highest count reached sets the factor.
GATE: tuple[tuple[int, float], ...] = (
    (3, 0.6),
    (4, 0.8),
    (5, 1.0),
)

#: n / matched credits -> 0..1, linear between the edges, clamped.
SHARE_LOW: float = 0.25
SHARE_HIGH: float = 0.60

#: Window score = gate x share x (BASE + BOOST x novelty).
NOVELTY_BASE: float = 0.7
NOVELTY_BOOST: float = 0.3

#: Recent window scored beside the full frame.
RECENT_DAYS: int = 30

#: Trail only: a whole multiple of this counts as a round amount.
ROUND_AMOUNT_INR: int = 1_000

#: Max trail steps.
MAX_TRAIL_STEPS: int = 8

#: Derived spans, kept beside the day counts they come from.
FRAME: timedelta = timedelta(days=FRAME_DAYS)
WINDOW: timedelta = timedelta(days=WINDOW_DAYS)
RECENT: timedelta = timedelta(days=RECENT_DAYS)
FAST_HOLD: timedelta = timedelta(hours=FAST_HOLD_HOURS)
