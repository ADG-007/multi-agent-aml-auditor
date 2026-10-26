"""Stage 0's own numbers. Thresholds a clause defines (cash, wire, mule floor) are read
from the rulebook, never copied here."""

from datetime import timedelta

from src.schemas import TransactionType


#: One name-match cutoff for every entry. Below Stage 4's 92, so look-alikes reach Stage 4.
NAME_MATCH_CUTOFF: float = 85.0

#: Pass-through: at least this many first-time credits (each at or above MULE-002's floor)...
PASS_THROUGH_MIN_CREDITS: int = 3
#: ...within this window of the first of them...
PASS_THROUGH_WINDOW: timedelta = timedelta(hours=24)
#: ...with debits worth at least this share of them, up to PASS_THROUGH_WINDOW after the last.
PASS_THROUGH_FORWARD_SHARE: float = 0.70

#: The CTR is a cash report, so only these types fire the cash-threshold rule.
CASH_TYPES: frozenset[TransactionType] = frozenset(
    {TransactionType.CASH_DEPOSIT, TransactionType.CASH_WITHDRAWAL}
)
