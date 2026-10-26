"""The facts Stage 3's model reads (F1, F2 ...). Code computes every number.

Velocity and pass-through come from Stage 2's code, so every account gets the same facts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from src.schemas import Transaction, TransactionDirection, TransactionType
from src.stages import stage2_transaction_pattern
from src.stages.frame import frame_rows
from src.stages.stage0_triage import TriageContext
from src.stages.stage0_triage.rules import inr
from src.stages.stage2_transaction_pattern import config as s2


YEAR_SCALE = 365 / s2.FRAME_DAYS


@dataclass(frozen=True)
class Fact:
    fact_id: str
    topic: str
    text: str
    x_income: float | None = None  # money in per year / declared income (money-in fact only)


def build_facts(
    account_id: str,
    ctx: TriageContext,
    as_of: datetime | None,
    transaction: Transaction | None = None,
) -> list[Fact]:
    bank = ctx.bank
    customer = bank.customer_for(account_id)
    account = bank.account(account_id)
    rows = frame_rows(ctx, account_id, as_of, transaction)
    income = customer.stated_income
    credits = [t for t in rows if t.direction == TransactionDirection.CREDIT]
    debits = [t for t in rows if t.direction == TransactionDirection.DEBIT]
    money_in, money_out = sum(t.amount for t in credits), sum(t.amount for t in debits)

    def x_income(amount: float) -> str:
        return f"{amount * YEAR_SCALE / income:.1f}x the declared income" if income else "n/a"

    facts = [
        (
            "profile",
            f"Declared occupation '{customer.occupation}', declared income {inr(income)} a "
            f"year; {customer.residency_status.value} customer in {customer.country}.",
        ),
        (
            "income",
            f"Money in over {s2.FRAME_DAYS} days: {inr(money_in)} from {len(credits)} "
            f"credits; scaled to a year {inr(money_in * YEAR_SCALE)} = {x_income(money_in)}.",
            round(money_in * YEAR_SCALE / income, 1) if income else None,
        ),
        (
            "income",
            f"Money out over {s2.FRAME_DAYS} days: {inr(money_out)} in {len(debits)} "
            f"debits; scaled to a year {inr(money_out * YEAR_SCALE)} = {x_income(money_out)}.",
        ),
        ("income", f"Turnover (in + out) scaled to a year = {x_income(money_in + money_out)}."),
        (
            "activity",
            f"{len(rows)} transactions in {s2.FRAME_DAYS} days, about "
            f"{len(rows) / (s2.FRAME_DAYS / 30):.0f} a month.",
        ),
        ("cash", _cash(rows, TransactionType.CASH_DEPOSIT, "Cash deposits")),
        ("cash", _cash(rows, TransactionType.CASH_WITHDRAWAL, "Cash withdrawals")),
        ("foreign", _cross_border(rows, ctx)),
        ("upi", _upi(credits)),
        ("account", f"{account.account_type.value} account opened {account.opened_date}."),
    ]
    if as_of is not None:
        s2_out = stage2_transaction_pattern.run(account_id, ctx, as_of, transaction)
        facts.append(("activity", f"Velocity: {s2_out.velocity_summary}"))
        if s2_out.pass_through_indicators:
            facts.append(
                ("pass_through", "Pass-through: " + "; ".join(s2_out.pass_through_indicators))
            )
        else:
            facts.append(("pass_through", "Pass-through: no credits forwarded on within hours."))
    return [Fact(f"F{i}", *fact) for i, fact in enumerate(facts, start=1)]


def _cash(rows: Sequence[Transaction], kind: TransactionType, label: str) -> str:
    cash = [t for t in rows if t.type == kind]
    if not cash:
        return f"{label}: none."
    return (
        f"{label}: {len(cash)}, total {inr(sum(t.amount for t in cash))}, largest "
        f"{inr(max(t.amount for t in cash))}."
    )


def _cross_border(rows: Sequence[Transaction], ctx: TriageContext) -> str:
    wires = [t for t in rows if t.is_cross_border]
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


def _upi(credits: Sequence[Transaction]) -> str:
    upi = [t for t in credits if t.type == TransactionType.UPI]
    if not upi:
        return "UPI credits: none."
    payers = len({t.counterparty_id for t in upi})
    return (
        f"UPI credits: {len(upi)} totalling {inr(sum(t.amount for t in upi))} from {payers} "
        f"different payers."
    )
