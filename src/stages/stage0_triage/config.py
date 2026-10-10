"""Stage 0's own tunable numbers (phase_2_execution.md D8).

Only numbers that no rulebook clause defines live here. The cash-reporting threshold and band
(STRUCT-001), the cross-border wire threshold (XB-004) and the mule transaction floor
(MULE-002) are read from the rulebook at run time and must never be copied into this file.
"""

from datetime import timedelta

from src.schemas import TransactionType


#: One fuzzy-match cutoff for every adverse-media entry. Deliberately below Stage 4's 92 for
#: NAME_SIMILARITY_ONLY entries, so the planted look-alike names still reach Stage 4 to be
#: judged there.
NAME_MATCH_CUTOFF: float = 85.0

#: Pass-through: at least this many first-time credits (each at or above MULE-002's floor)...
PASS_THROUGH_MIN_CREDITS: int = 3
#: ...within this window of the first of them...
PASS_THROUGH_WINDOW: timedelta = timedelta(hours=24)
#: ...with debits worth at least this share of those credits between the burst's first credit
#: and PASS_THROUGH_WINDOW after its last.
PASS_THROUGH_FORWARD_SHARE: float = 0.70

#: The CTR is a cash report, so only these types fire the cash-threshold rule.
CASH_TYPES: frozenset[TransactionType] = frozenset(
    {TransactionType.CASH_DEPOSIT, TransactionType.CASH_WITHDRAWAL}
)
