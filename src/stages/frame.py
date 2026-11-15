"""The 90-day window the evidence stages read, plus a live event's own transaction."""

from __future__ import annotations

from datetime import datetime

from src.schemas import Transaction
from src.stages.stage0_triage import TriageContext
from src.stages.stage2_transaction_pattern import config as s2


def frame_rows(
    ctx: TriageContext,
    account_id: str,
    as_of: datetime | None,
    transaction: Transaction | None = None,
) -> list[Transaction]:
    """The account's rows in (as_of - 90 days, as_of], oldest first. Empty with no as-of."""
    if as_of is None:
        return []
    rows = list(ctx.bank.transactions(account_id))
    if transaction is not None and all(
        t.transaction_id != transaction.transaction_id for t in rows
    ):
        rows.append(transaction)
        rows.sort(key=lambda t: (t.timestamp, t.transaction_id))
    start = as_of - s2.FRAME
    return [t for t in rows if start < t.timestamp <= as_of]
