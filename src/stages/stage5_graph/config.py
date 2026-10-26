"""Stage 5's numbers. Device / IP = hard link, address = soft link. Only flagged accounts score."""

#: Confidence by the number of distinct flagged accounts with a hard link.
HARD_LINK_SCORES: tuple[float, ...] = (0.60, 0.75, 0.85)

#: Confidence by the number of distinct flagged accounts linked by address only.
SOFT_LINK_SCORES: tuple[float, ...] = (0.30, 0.45)

#: Links alone never reach the 0.90 veto.
RING_ONLY_CAP: float = 0.85

#: The one veto path: a hard link to a sanctions escalation plus a second hard link.
VETO_CONFIDENCE: float = 0.95
VETO_MIN_HARD_LINKS: int = 2
