"""Stage 0: decide which stages investigate an event, and record why.

- TRANSACTION: rules on that transaction, pass-through up to it, whole-account rules.
- KYC_SUBMITTED: the KYC rule plus whole-account rules.
- PERIODIC_REVIEW: rules on every transaction, every pass-through burst, whole-account and KYC.

Route = union of the fired rules' stages. Rulebook joins any weighted stage. Auditor always runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.bank import BankData
from src.llm.cache import LLMMode
from src.matching import NameIndex
from src.orchestration.run_state import RunState
from src.rulebook import RulebookLookup
from src.schemas import (
    STAGE_WEIGHTS,
    Stage0TriageOutput,
    StageName,
    Transaction,
    TriageEvent,
    TriageEventKind,
)
from src.stages.stage0_triage import rules
from src.stages.stage0_triage.rules import RULE_ROUTES, Hit


@dataclass(frozen=True)
class TriageContext:
    """Read-only inputs for one run. review_date is the "now" of every periodic review."""

    bank: BankData
    rulebook: RulebookLookup
    names: NameIndex
    review_date: datetime | None = None
    #: Model mode for Stage 3 (src/llm/cache.py). Off by default: no accidental model calls.
    llm_mode: LLMMode = LLMMode.OFF
    #: The same for Stage 7, the writer. The verdict never depends on it.
    writer_mode: LLMMode = LLMMode.OFF
    #: Where Stage 1 finds ID images and Stages 3 and 7 keep saved answers (None = data/).
    documents_dir: Path | None = None
    cache_root: Path | None = None

    @classmethod
    def build(
        cls,
        bank: BankData,
        rulebook: RulebookLookup,
        llm_mode: LLMMode = LLMMode.OFF,
        writer_mode: LLMMode = LLMMode.OFF,
        documents_dir: Path | None = None,
        cache_root: Path | None = None,
    ) -> TriageContext:
        return cls(
            bank=bank,
            rulebook=rulebook,
            names=NameIndex(bank.adverse_media),
            review_date=bank.review_date(),
            llm_mode=llm_mode,
            writer_mode=writer_mode,
            documents_dir=documents_dir,
            cache_root=cache_root,
        )


def triage(event: TriageEvent, ctx: TriageContext, state: RunState) -> Stage0TriageOutput:
    account_id = event.account_id
    customer = ctx.bank.customer_for(account_id)

    if event.kind == TriageEventKind.TRANSACTION:
        history = _history_until(ctx.bank, event.transaction)
        hits = _transaction_hits(event.transaction, account_id, ctx, state)
        hits += rules.pass_through(history, ctx.rulebook, within=event.transaction)
    elif event.kind == TriageEventKind.PERIODIC_REVIEW:
        history = list(ctx.bank.transactions(account_id))
        hits = []
        for t in history:
            hits += _transaction_hits(t, account_id, ctx, state)
        hits += rules.pass_through(history, ctx.rulebook)
    else:
        history = list(ctx.bank.transactions(account_id))
        hits = []

    kyc = rules.kyc_event(event.kind)
    if kyc:
        hits.append(kyc)
    hits += rules.name_match(customer, history, ctx.names)
    address = rules.known_bad_address(customer, account_id, state)
    if address:
        hits.append(address)

    reinvestigation = rules.reinvestigation(account_id, state, hits)
    if reinvestigation:
        hits.append(reinvestigation)

    return Stage0TriageOutput(
        routed_stages=_route(hits),
        trigger_reasons=[h.reason for h in hits],
        fired_rules=[h.fired for h in hits],
    )


def first_alert(output: Stage0TriageOutput, bank: BankData, account_id: str) -> Transaction | None:
    """The earliest transaction that fired a rule, or None if only whole-account rules fired."""
    ids = {f.transaction_id for f in output.fired_rules if f.transaction_id}
    if not ids:
        return None
    return next(t for t in bank.transactions(account_id) if t.transaction_id in ids)


def _transaction_hits(
    t: Transaction, account_id: str, ctx: TriageContext, state: RunState
) -> list[Hit]:
    candidates = (rules.cash_threshold(t, ctx.rulebook), rules.cross_border(t, ctx.rulebook))
    hits = [h for h in candidates if h]
    hits += rules.known_bad_on_transaction(t, account_id, state)
    return hits


def _history_until(bank: BankData, transaction: Transaction) -> list[Transaction]:
    """The account's history up to and including `transaction`, whether or not it is posted."""
    cutoff = (transaction.timestamp, transaction.transaction_id)
    history = [
        t
        for t in bank.transactions(transaction.account_id)
        if (t.timestamp, t.transaction_id) <= cutoff
        and t.transaction_id != transaction.transaction_id
    ]
    history.append(transaction)
    return history


def _route(hits: list[Hit]) -> list[StageName]:
    routed: set[StageName] = set()
    for h in hits:
        routed |= RULE_ROUTES[h.fired.rule]
    if routed & set(STAGE_WEIGHTS):
        routed.add(StageName.RULEBOOK)
    routed.add(StageName.AUDITOR)
    return [stage for stage in StageName if stage in routed]
