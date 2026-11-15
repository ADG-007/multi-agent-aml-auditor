"""The KYC facts tool: what the account did in its 90 days. Python computes every number.

The facts are what the contradiction rules compare the profile's claims with. Velocity and
pass-through come from Stage 2's own code, so every account gets the same wording.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from src.agents.base import FactSheet
from src.schemas import Transaction, TransactionDirection, TransactionType
from src.stages import stage2_transaction_pattern
from src.stages.cash_aggregate import month_totals
from src.stages.frame import frame_rows
from src.stages.stage0_triage import TriageContext
from src.stages.stage0_triage.rules import inr
from src.stages.stage2_transaction_pattern import config as s2


MONTHS = s2.FRAME_DAYS / 30


@dataclass(frozen=True)
class Activity:
    """The numbers the rules read. Monthly figures are the 90 days divided by 3."""

    income: float
    money_in_month: float
    money_out_month: float
    cash_deposits: int
    cash_deposit_total: float
    busiest_cash_month: float
    cross_border_wires: int
    payers: int
    transactions: int

    @property
    def income_month(self) -> float:
        return self.income / 12

    @property
    def turnover_month(self) -> float:
        return self.money_in_month + self.money_out_month


def build(
    sheet: FactSheet,
    account_id: str,
    ctx: TriageContext,
    as_of: datetime | None,
    transaction: Transaction | None = None,
) -> Activity:
    """Add the account's facts to `sheet` and return the numbers behind them."""
    bank = ctx.bank
    customer = bank.customer_for(account_id)
    account = bank.account(account_id)
    rows = frame_rows(ctx, account_id, as_of, transaction)
    income = customer.stated_income
    credits = [t for t in rows if t.direction == TransactionDirection.CREDIT]
    debits = [t for t in rows if t.direction == TransactionDirection.DEBIT]
    money_in, money_out = sum(t.amount for t in credits), sum(t.amount for t in debits)
    deposits = [t for t in rows if t.type == TransactionType.CASH_DEPOSIT]
    busiest = max(month_totals(deposits).values(), default=0.0)
    wires = [t for t in rows if t.is_cross_border]
    payers = len({t.counterparty_id for t in credits})
    activity = Activity(
        income=income,
        money_in_month=money_in / MONTHS,
        money_out_month=money_out / MONTHS,
        cash_deposits=len(deposits),
        cash_deposit_total=sum(t.amount for t in deposits),
        busiest_cash_month=busiest,
        cross_border_wires=len(wires),
        payers=payers,
        transactions=len(rows),
    )

    def x_income(month: float) -> str:
        return f"{month / activity.income_month:.1f}x the declared monthly income" if income else ""

    sheet.add(
        "profile",
        f"Record: occupation '{customer.occupation}', declared income {inr(income)} a year "
        f"({inr(activity.income_month)} a month); {customer.residency_status.value} customer "
        f"in {customer.country}; {account.account_type.value} account opened "
        f"{account.opened_date}.",
        value=income,
    )
    sheet.add(
        "income",
        f"Money in: {inr(money_in)} in {s2.FRAME_DAYS} days from {len(credits)} credits, "
        f"{inr(activity.money_in_month)} a month = {x_income(activity.money_in_month)}.",
        value=activity.money_in_month,
    )
    sheet.add(
        "income",
        f"Money out: {inr(money_out)} in {s2.FRAME_DAYS} days in {len(debits)} debits, "
        f"{inr(activity.money_out_month)} a month.",
        value=activity.money_out_month,
    )
    sheet.add(
        "activity",
        f"Turnover (in + out) {inr(activity.turnover_month)} a month across about "
        f"{len(rows) / MONTHS:.0f} transactions a month.",
        value=activity.turnover_month,
    )
    sheet.add(
        "cash",
        _cash(deposits, "Cash deposits")
        + (f" Busiest calendar month {inr(busiest)}." if deposits else ""),
        value=activity.cash_deposit_total,
    )
    sheet.add(
        "cash",
        _cash([t for t in rows if t.type == TransactionType.CASH_WITHDRAWAL], "Cash withdrawals"),
    )
    sheet.add("foreign", _cross_border(wires, ctx), value=len(wires))
    sheet.add(
        "payers",
        f"Credits came from {payers} different payers; "
        f"{sum(1 for t in credits if t.type == TransactionType.UPI)} were UPI receipts.",
        value=payers,
    )
    if as_of is not None:
        s2_out = stage2_transaction_pattern.run(account_id, ctx, as_of, transaction)
        sheet.add("activity", f"Velocity: {s2_out.velocity_summary}")
        sheet.add(
            "pass_through",
            "Pass-through: " + "; ".join(s2_out.pass_through_indicators)
            if s2_out.pass_through_indicators
            else "Pass-through: no credits forwarded on within hours.",
        )
    return activity


def _cash(cash: Sequence[Transaction], label: str) -> str:
    if not cash:
        return f"{label}: none."
    return (
        f"{label}: {len(cash)}, total {inr(sum(t.amount for t in cash))}, largest "
        f"{inr(max(t.amount for t in cash))}."
    )


def _cross_border(wires: Sequence[Transaction], ctx: TriageContext) -> str:
    if not wires:
        return "Cross-border wires: none."
    parts = []
    for direction, word in (
        (TransactionDirection.CREDIT, "inward"),
        (TransactionDirection.DEBIT, "outward"),
    ):
        ws = [t for t in wires if t.direction == direction]
        if ws:
            parts.append(f"{len(ws)} {word} totalling {inr(sum(t.amount for t in ws))}")
    countries = sorted(
        {cp.country for t in wires if (cp := ctx.bank.counterparty(t.counterparty_id))}
    )
    codes = sorted({t.purpose_code for t in wires if t.purpose_code})
    return (
        f"Cross-border wires: {', '.join(parts)}; countries {', '.join(countries) or 'unknown'}; "
        f"purpose codes {', '.join(codes) or 'none'}."
    )
