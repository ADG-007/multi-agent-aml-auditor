"""Stage 0's routing table, one small function per row (MASTER_PLAN.md Phase 2).

Each rule returns Hits: a machine-readable FiredRule plus the readable sentence that goes into
trigger_reasons. Per-transaction rules carry the transaction id; whole-account rules carry
None. No rule ever reads ground truth, and no rulebook-defined number is written here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.matching import NameIndex
from src.orchestration.run_state import RunState
from src.rulebook import (
    CROSS_BORDER_CLAUSE_ID,
    MULE_FLOOR_CLAUSE_ID,
    STRUCTURING_CLAUSE_ID,
    RulebookLookup,
)
from src.schemas import (
    Customer,
    FiredRule,
    StageName,
    Transaction,
    TransactionDirection,
    TriageEventKind,
    TriageRule,
)
from src.stages.stage0_triage import config


S = StageName

#: The routing table: which stages each rule sends the case to. Rulebook is added whenever a
#: weighted stage is routed and Auditor always (triage.py), so they are not repeated here
#: except where the table lists them explicitly.
RULE_ROUTES: dict[TriageRule, frozenset[StageName]] = {
    TriageRule.CASH_THRESHOLD: frozenset({S.TRANSACTION_PATTERN, S.ADVERSE_MEDIA, S.GRAPH}),
    TriageRule.CROSS_BORDER: frozenset({S.TRANSACTION_PATTERN, S.KYC, S.GRAPH}),
    TriageRule.KYC_EVENT: frozenset({S.KYC, S.VISION}),
    TriageRule.KNOWN_BAD_IDENTIFIER: frozenset({S.GRAPH, S.ADVERSE_MEDIA}),
    TriageRule.NAME_MATCH: frozenset({S.ADVERSE_MEDIA}),
    TriageRule.PASS_THROUGH: frozenset({S.TRANSACTION_PATTERN, S.KYC, S.GRAPH}),
    TriageRule.REINVESTIGATION: frozenset(
        {S.VISION, S.TRANSACTION_PATTERN, S.KYC, S.ADVERSE_MEDIA, S.GRAPH, S.RULEBOOK}
    ),
}


@dataclass(frozen=True)
class Hit:
    fired: FiredRule
    reason: str


def _hit(
    rule: TriageRule,
    detail: str,
    reason: str,
    transaction_id: str | None = None,
    clause_id: str | None = None,
) -> Hit:
    return Hit(
        fired=FiredRule(
            rule=rule, transaction_id=transaction_id, clause_id=clause_id, detail=detail
        ),
        reason=reason,
    )


# --------------------------------------------------------------------------------------
# Per-transaction rules
# --------------------------------------------------------------------------------------


def cash_threshold(t: Transaction, rulebook: RulebookLookup) -> Hit | None:
    """A cash transaction at or above the CTR threshold, or in the band just under it."""
    if t.type not in config.CASH_TYPES:
        return None
    s = rulebook.structuring()
    if t.amount < s.band_floor_inr:
        return None
    where = (
        f"at or above the {inr(s.threshold_inr)} cash-reporting threshold"
        if t.amount >= s.threshold_inr
        else f"in the structuring band {inr(s.band_floor_inr)}-{inr(s.threshold_inr)}, "
        "just under the cash-reporting threshold"
    )
    return _hit(
        TriageRule.CASH_THRESHOLD,
        detail=f"{t.type.value} {inr(t.amount)}",
        reason=f"{_label(t.type.value)} of {inr(t.amount)} on {_day(t)} is {where} "
        f"({STRUCTURING_CLAUSE_ID}).",
        transaction_id=t.transaction_id,
        clause_id=STRUCTURING_CLAUSE_ID,
    )


def cross_border(t: Transaction, rulebook: RulebookLookup) -> Hit | None:
    """A cross-border wire at or above the CBWTR threshold, inward or outward."""
    threshold = rulebook.cross_border_threshold()
    if not t.is_cross_border or t.amount < threshold:
        return None
    way = "inward" if t.direction == TransactionDirection.CREDIT else "outward"
    return _hit(
        TriageRule.CROSS_BORDER,
        detail=f"{way} {t.type.value} {inr(t.amount)}",
        reason=f"{way.capitalize()} cross-border wire of {inr(t.amount)} on {_day(t)} is at or "
        f"above the {inr(threshold)} wire-reporting threshold ({CROSS_BORDER_CLAUSE_ID}).",
        transaction_id=t.transaction_id,
        clause_id=CROSS_BORDER_CLAUSE_ID,
    )


def known_bad_on_transaction(t: Transaction, account_id: str, state: RunState) -> list[Hit]:
    """The transaction's device or IP was listed from another, escalated account.

    An account's own listed identifiers never count against it: the list exists to reach
    other, connected accounts.
    """
    hits = []
    for label, value, listed in (
        ("device", t.device_id, state.known_bad_devices),
        ("IP", t.ip_address, state.known_bad_ips),
    ):
        source = listed.get(value) if value else None
        if source and source != account_id:
            hits.append(
                _hit(
                    TriageRule.KNOWN_BAD_IDENTIFIER,
                    detail=f"{label} {value} listed from {source}",
                    reason=f"The {label} {value} on the transaction of {_day(t)} was also used "
                    f"by escalated account {source}.",
                    transaction_id=t.transaction_id,
                )
            )
    return hits


# --------------------------------------------------------------------------------------
# Whole-account rules
# --------------------------------------------------------------------------------------


def kyc_event(kind: TriageEventKind) -> Hit | None:
    if kind == TriageEventKind.KYC_SUBMITTED:
        return _hit(
            TriageRule.KYC_EVENT,
            detail="KYC profile submitted",
            reason="A new or updated KYC profile was submitted.",
        )
    if kind == TriageEventKind.PERIODIC_REVIEW:
        return _hit(
            TriageRule.KYC_EVENT,
            detail="periodic KYC review",
            reason="Periodic KYC review of the profile on file.",
        )
    return None


def known_bad_address(customer: Customer, account_id: str, state: RunState) -> Hit | None:
    source = state.known_bad_addresses.get(customer.address)
    if not source or source == account_id:
        return None
    return _hit(
        TriageRule.KNOWN_BAD_IDENTIFIER,
        detail=f"address listed from {source}",
        reason=f"The customer's address is shared with escalated account {source}.",
    )


def name_match(
    customer: Customer, history: Sequence[Transaction], names: NameIndex
) -> list[Hit]:
    """The holder or any counterparty in the history matches an entry's name or alias."""
    hits = []
    candidates = [("account holder", customer.name)]
    seen = {customer.name}
    for t in history:
        if t.counterparty_name not in seen:
            seen.add(t.counterparty_name)
            candidates.append(("counterparty", t.counterparty_name))

    for role, name in candidates:
        match = names.best_match(name, cutoff=config.NAME_MATCH_CUTOFF)
        if match is None:
            continue
        via = f" (alias of {match.entity_name})" if match.via_alias else ""
        hits.append(
            _hit(
                TriageRule.NAME_MATCH,
                detail=f"{role} '{name}' ~ '{match.matched_text}' {match.score:g} "
                f"[{match.entry_id}, {match.tier.value}]",
                reason=f"The {role} name '{name}' matches '{match.matched_text}'{via} on the "
                f"adverse-media list at {match.score:g}/100 ({match.tier.value}).",
            )
        )
    return hits


def pass_through(
    history: Sequence[Transaction],
    rulebook: RulebookLookup,
    within: Transaction | None = None,
) -> list[Hit]:
    """Bursts of first-time credits forwarded onward within hours.

    Review mode (`within` None): every non-overlapping qualifying burst in the history.
    Live mode: fires once if `within` lies inside a qualifying burst's window, from its first
    credit to PASS_THROUGH_WINDOW after its last, with the history ending at `within`.
    """
    floor = rulebook.mule_floor()
    window = config.PASS_THROUGH_WINDOW

    seen: set[str] = set()
    first_time: list[Transaction] = []
    for t in history:
        if t.direction != TransactionDirection.CREDIT:
            continue
        if t.counterparty_id not in seen and t.amount >= floor:
            first_time.append(t)
        seen.add(t.counterparty_id)
    debits = [t for t in history if t.direction == TransactionDirection.DEBIT]

    hits: list[Hit] = []
    i = 0
    while i < len(first_time):
        start = first_time[i].timestamp
        burst = [c for c in first_time[i:] if c.timestamp <= start + window]
        if len(burst) >= config.PASS_THROUGH_MIN_CREDITS:
            end = burst[-1].timestamp + window
            credited = sum(c.amount for c in burst)
            forwarded = sum(d.amount for d in debits if start <= d.timestamp <= end)
            if forwarded >= config.PASS_THROUGH_FORWARD_SHARE * credited:
                share = forwarded / credited
                detail = (
                    f"{len(burst)} first-time credits {inr(credited)} from {start:%Y-%m-%d %H:%M}, "
                    f"{inr(forwarded)} ({share:.0%}) sent on"
                )
                reason = (
                    f"{len(burst)} first-time credits of at least {inr(floor)} totalling "
                    f"{inr(credited)} arrived within {_hours(window)} hours from "
                    f"{start:%Y-%m-%d %H:%M}, and {inr(forwarded)} ({share:.0%}) left the "
                    f"account within {_hours(window)} hours of the last ({MULE_FLOOR_CLAUSE_ID})."
                )
                if within is None:
                    hits.append(
                        _hit(
                            TriageRule.PASS_THROUGH,
                            detail=detail,
                            reason=reason,
                            transaction_id=burst[0].transaction_id,
                            clause_id=MULE_FLOOR_CLAUSE_ID,
                        )
                    )
                    i += len(burst)
                    continue
                if start <= within.timestamp <= end:
                    return [
                        _hit(
                            TriageRule.PASS_THROUGH,
                            detail=detail,
                            reason=reason,
                            transaction_id=within.transaction_id,
                            clause_id=MULE_FLOOR_CLAUSE_ID,
                        )
                    ]
        i += 1
    return hits


def reinvestigation(account_id: str, state: RunState, other_hits: Sequence[Hit]) -> Hit | None:
    """An already-flagged account tripped another rule: every stage re-runs."""
    if not other_hits or not state.is_flagged(account_id):
        return None
    latest = state.history(account_id)[-1]
    return _hit(
        TriageRule.REINVESTIGATION,
        detail=f"standing {latest.disposition.value} from {latest.case_id}",
        reason=f"The account is already under {latest.disposition.value} from {latest.case_id} "
        "and a new rule fired, so every stage re-runs.",
    )


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------


def inr(amount: float) -> str:
    """Indian digit grouping: 950000 -> ₹9,50,000."""
    whole = round(amount)
    sign = "-" if whole < 0 else ""
    digits = str(abs(whole))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups) + "," + tail
    return f"{sign}₹{digits}"


def _day(t: Transaction) -> str:
    return f"{t.timestamp:%Y-%m-%d}"


def _label(type_value: str) -> str:
    return type_value.replace("_", " ").capitalize()


def _hours(window) -> int:
    return int(window.total_seconds() // 3600)
