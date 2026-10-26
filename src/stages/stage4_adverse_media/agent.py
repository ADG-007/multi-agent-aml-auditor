"""Stage 4's entry point: screen the holder and every counterparty against the list.

    run(account_id, ctx, as_of, transaction) -> Stage4AdverseMediaOutput

Holder and 90-day counterparty names vs every entry and alias (>= 85). Tier and DOB set each
match's score. Confidence = the strongest match. No model calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from src.matching import NameMatch
from src.schemas import (
    AUTO_VETO_THRESHOLD,
    CORROBORATION_STAGE_MIN,
    Customer,
    ReasoningStep,
    SeverityTier,
    Stage4AdverseMediaOutput,
    Transaction,
)
from src.stages.frame import frame_rows
from src.stages.stage0_triage import TriageContext
from src.stages.stage4_adverse_media import config


T = SeverityTier


@dataclass(frozen=True)
class Hit:
    role: str  # "holder" or "counterparty"
    name: str
    match: NameMatch
    score: float
    note: str


def run(
    account_id: str,
    ctx: TriageContext,
    as_of: datetime | None,
    transaction: Transaction | None = None,
) -> Stage4AdverseMediaOutput:
    customer = ctx.bank.customer_for(account_id)
    rows = frame_rows(ctx, account_id, as_of, transaction)
    counterparties = sorted({t.counterparty_name for t in rows} - {customer.name})

    hits = [
        _holder_hit(customer, m, ctx)
        for m in ctx.names.all_matches(customer.name, config.CANDIDATE_CUTOFF)
    ]
    for name in counterparties:
        hits += [
            _counterparty_hit(name, m) for m in ctx.names.all_matches(name, config.CANDIDATE_CUTOFF)
        ]
    hits.sort(key=lambda h: (-h.score, -h.match.score, h.match.entry_id))

    top = hits[0] if hits and hits[0].score > 0 else None
    confidence = top.score if top else 0.0
    trail = [
        ReasoningStep(
            step="Holder screening",
            finding=_found(customer.name, [h for h in hits if h.role == "holder"]),
            impact="Name and aliases matched at >= "
            f"{config.CANDIDATE_CUTOFF:g}; look-alike entries need >= {config.LOOKALIKE_CUTOFF:g}"
            " (SANC-002).",
        ),
        ReasoningStep(
            step="Counterparty screening",
            finding=f"{len(counterparties)} counterparty names in the frame; "
            f"{sum(1 for h in hits if h.role == 'counterparty')} matched.",
            impact="A sanctioned payee is itself reportable (SANC-003).",
        ),
    ]
    for h in hits[: config.MAX_TRAIL_STEPS - 3]:
        trail.append(
            ReasoningStep(
                step=f"{h.match.entry_id} ({h.match.tier.value})",
                finding=f"{h.role} '{h.name}' ~ '{h.match.matched_text}' {h.match.score:g}/100; "
                f"{h.note}",
                impact=f"Scores {h.score:g}.",
            )
        )
    trail.append(
        ReasoningStep(
            step="Confidence",
            finding=f"{len(hits)} match(es); strongest {confidence:g}"
            + (f" from {top.match.entry_id}" if top else ""),
            impact=_impact(confidence),
        )
    )
    return Stage4AdverseMediaOutput(
        # Only matches that score; a rejected look-alike stays in the trail.
        matched_entries=[_line(h) for h in hits if h.score > 0],
        severity_tier=top.match.tier if top else None,
        confidence=confidence,
        reasoning_trail=trail,
    )


def _holder_hit(customer: Customer, m: NameMatch, ctx: TriageContext) -> Hit:
    if m.tier == T.CONFIRMED_SANCTIONS:
        listed = ctx.names.entry(m.entry_id).date_of_birth
        agrees = dob_agrees(listed, customer.dob)
        if agrees is None:
            return Hit(
                "holder",
                customer.name,
                m,
                config.HOLDER_SANCTIONS_NO_DOB,
                "no date of birth on the entry: cannot be set aside on the name alone",
            )
        if agrees:
            return Hit(
                "holder",
                customer.name,
                m,
                config.HOLDER_SANCTIONS_DOB_AGREES,
                f"date of birth {listed} agrees",
            )
        return Hit(
            "holder",
            customer.name,
            m,
            config.HOLDER_SANCTIONS_DOB_DISAGREES,
            f"date of birth {listed} disagrees with {customer.dob}: likely name clash",
        )
    if m.tier == T.ADVERSE_NEWS:
        return Hit("holder", customer.name, m, config.HOLDER_NEWS, "adverse news, capped")
    return _lookalike("holder", customer.name, m, config.HOLDER_LOOKALIKE)


def _counterparty_hit(name: str, m: NameMatch) -> Hit:
    if m.tier == T.CONFIRMED_SANCTIONS:
        return Hit("counterparty", name, m, config.COUNTERPARTY_SANCTIONS, "sanctioned payee")
    if m.tier == T.ADVERSE_NEWS:
        return Hit("counterparty", name, m, config.COUNTERPARTY_NEWS, "news about a payee")
    return _lookalike("counterparty", name, m, config.COUNTERPARTY_LOOKALIKE)


def _lookalike(role: str, name: str, m: NameMatch, score: float) -> Hit:
    if m.score < config.LOOKALIKE_CUTOFF:
        return Hit(role, name, m, 0.0, f"name-only entry under {config.LOOKALIKE_CUTOFF:g}: no hit")
    return Hit(role, name, m, score, "name-only entry, no story behind it")


def dob_agrees(listed: str | None, dob: date) -> bool | None:
    """None when the entry has no date; a year-only entry compares the year."""
    if not listed:
        return None
    if len(listed) == 4:
        return int(listed) == dob.year
    return listed == dob.isoformat()


def _line(h: Hit) -> str:
    return (
        f"{h.match.entry_id} {h.match.entity_name} ({h.match.tier.value}): {h.role} "
        f"'{h.name}', name {h.match.score:g}, {h.note} -> {h.score:g}"
    )


def _found(name: str, hits: list[Hit]) -> str:
    if not hits:
        return f"'{name}' matches no entry."
    return f"'{name}' matches {len(hits)} entr{'y' if len(hits) == 1 else 'ies'}."


def _impact(confidence: float) -> str:
    if confidence >= AUTO_VETO_THRESHOLD:
        return "Veto range: the case escalates on this alone (SANC-001)."
    if confidence >= CORROBORATION_STAGE_MIN:
        return "A real signal; below the veto, it needs another stage to escalate."
    if confidence > 0:
        return "Weak name-only signal; never enough alone."
    return "No adverse-media signal."
