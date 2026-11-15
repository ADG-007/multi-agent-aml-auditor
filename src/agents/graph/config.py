"""The Graph agent's numbers. Device / IP = hard link, address = soft link. Only accounts
flagged on their own evidence count as links. The label never scores."""

PROMPT_VERSION: str = "graph-v1"

#: Confidence by the number of distinct flagged accounts with a hard link.
HARD_LINK_SCORES: tuple[float, ...] = (0.60, 0.75, 0.85)

#: Confidence by the number of distinct flagged accounts linked by address only.
SOFT_LINK_SCORES: tuple[float, ...] = (0.30, 0.45)

#: Links alone never reach the 0.90 veto.
RING_ONLY_CAP: float = 0.85

#: The one veto path: a hard link to a sanctions escalation plus a second hard link.
VETO_CONFIDENCE: float = 0.95
VETO_MIN_HARD_LINKS: int = 2

#: The fixed label list. Each label needs the Python signal named beside it.
LABELS: tuple[str, ...] = (
    "NAIVE_CASH_STRUCTURING",  # single deposits in the band (k)
    "MONTHLY_CAP_STRUCTURING",  # monthly cash totals in the band (m)
    "PASS_THROUGH",  # credits from strangers sent on within hours
    "LINKED_TO_FLAGGED_ACCOUNTS",  # shared device, IP or address with a flagged account
    "MIXED_PATTERN",  # two or more of the above
    "NO_PATTERN",
)

#: A counterparty paid by more accounts than this is a shop or utility, not a link.
OVERLAP_MAX_ACCOUNTS: int = 15

#: How far the path tool looks: account - identifier - account - identifier - account.
PATH_MAX_HOPS: int = 4
