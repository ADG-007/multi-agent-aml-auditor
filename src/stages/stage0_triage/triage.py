"""Stage 0: decide which stages investigate an event, and record why.

One entry point for every event kind (phase_2_execution.md D2):

- TRANSACTION (live): the per-transaction rules on that transaction, pass-through on the
  history ending at it, and the whole-account rules.
- KYC_SUBMITTED: the KYC row plus the whole-account rules.
- PERIODIC_REVIEW (Phase 11): the per-transaction rules on every transaction in the history,
  every pass-through burst, the whole-account rules, and the KYC row for the profile on file.

The route is the union of every fired rule's stages. A flagged account that trips any rule is
re-investigated by every stage. Rulebook joins whenever a weighted stage is routed; Auditor is
always routed, so even a case with no fired rule gets its thin CLEAR report.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bank import BankData
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
    """The read-only inputs Stage 0 needs, built once per run."""

    bank: BankData
    rulebook: RulebookLookup
    names: NameIndex

    @classmethod
    def build(cls, bank: BankData, rulebook: RulebookLookup) -> TriageContext:
        return cls(bank=bank, rulebook=rulebook, names=NameIndex(bank.adverse_media))


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
