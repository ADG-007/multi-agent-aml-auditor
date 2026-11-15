"""The Web Research agent: screen the holder and every counterparty against the list.

    WebResearchAgent(client).run(account_id, ctx, as_of, transaction) -> WebResult

Python does the screening and sets every score: holder and 90-day counterparty names against
every entry and alias (>= 85); the tier and the date of birth set each hit's score. The model
answers one question per holder hit: is the listed person the same person as the customer
(same / different / unsure), citing fact ids. The guard is lower-only: "different" can lower
a sanctions hit whose date of birth does not agree; nothing can raise a score; a hit whose
date of birth agrees is never lowered; an adverse-news hit is never lowered (a news report's
date of birth is not an authoritative identifier), so the answer is context for the analyst.
No hit, no model call.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from src.agents.base import Agent, AgentRun, AgentSpec, known_ids
from src.agents.web_research import config
from src.llm import Role
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


T = SeverityTier
BACKUP_KIND = "Same-person call"


@dataclass(frozen=True)
class Hit:
    role: str  # "holder" or "counterparty"
    name: str
    match: NameMatch
    score: float
    note: str
    #: True / False when the entry's date of birth was compared with the holder's.
    dob_agrees: bool | None = None


class SamePerson(BaseModel):
    """The model's one answer per holder hit."""

    verdict: Literal["SAME", "DIFFERENT", "UNSURE"]
    fact_ids: list[str] = Field(description="The fact ids the answer rests on, e.g. F2.")
    reason: str = Field(description="One short sentence.")
    more: list[str] = Field(
        default_factory=list, description="Optional tool names to run first, or empty."
    )


PROMPT = """A name-screening tool matched a bank customer against one entry on a sanctions \
and adverse-news list. Decide whether the customer and the listed person are the same person.

- SAME: the facts point to one person (for example the dates of birth agree).
- DIFFERENT: a fact rules it out (for example the dates of birth disagree).
- UNSURE: the facts do not settle it.

Judge only from the facts below and cite the fact ids you used. A similar name alone is not \
proof. You may ask for the optional tool "alias_lookup" in `more` if the aliases would help; \
otherwise leave `more` empty.

FACTS:
{facts}
"""


@dataclass
class WebResult:
    output: Stage4AdverseMediaOutput
    hits: list[Hit]
    run: AgentRun


class WebResearchAgent(Agent):
    spec = AgentSpec(
        name="web_research",
        role=Role.WEB_RESEARCH,
        prompt_version=config.PROMPT_VERSION,
        required_tools=("holder_screen", "counterparty_screen", "entry_details"),
        optional_tools=("alias_lookup",),
    )

    def run(
        self,
        account_id: str,
        ctx: TriageContext,
        as_of: datetime | None,
        transaction: Transaction | None = None,
        role: Role | None = None,
    ) -> WebResult:
        """`role` is set to the backup model when the queue phase repeats the call."""
        run = self.start(account_id)
        customer = ctx.bank.customer_for(account_id)
        rows = frame_rows(ctx, account_id, as_of, transaction)
        counterparties = sorted({t.counterparty_name for t in rows} - {customer.name})

        hits = [
            holder_hit(customer, m, ctx)
            for m in ctx.names.all_matches(customer.name, config.CANDIDATE_CUTOFF)
        ]
        self.tool(run, "holder_screen")
        for name in counterparties:
            hits += [
                counterparty_hit(name, m)
                for m in ctx.names.all_matches(name, config.CANDIDATE_CUTOFF)
            ]
        self.tool(run, "counterparty_screen")
        self.tool(run, "entry_details")

        notes: dict[str, str] = {}
        judged = []
        for hit in hits:
            if hit.role == "holder" and hit.score > 0:
                hit, notes[hit.match.entry_id] = self._same_person(run, hit, customer, ctx, role)
            judged.append(hit)
        hits = sorted(judged, key=lambda h: (-h.score, -h.match.score, h.match.entry_id))

        top = hits[0] if hits and hits[0].score > 0 else None
        confidence = top.score if top else 0.0
        trail = [
            ReasoningStep(
                step="Holder screening",
                finding=_found(customer.name, [h for h in hits if h.role == "holder"]),
                impact=f"Name and aliases matched at >= {config.CANDIDATE_CUTOFF:g}; look-alike "
                f"entries need >= {config.LOOKALIKE_CUTOFF:g} (SANC-002).",
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
                    finding=f"{h.role} '{h.name}' ~ '{h.match.matched_text}' "
                    f"{h.match.score:g}/100; {h.note}"
                    + (f" {notes[h.match.entry_id]}" if h.match.entry_id in notes else ""),
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
        run.trail += trail
        output = Stage4AdverseMediaOutput(
            matched_entries=[_line(h) for h in hits if h.score > 0],
            severity_tier=top.match.tier if top else None,
            confidence=confidence,
            reasoning_trail=trail,
        )
        return WebResult(output=output, hits=hits, run=run)

    # ----------------------------------------------------------------------------------
    # The same-person call (lower-only)
    # ----------------------------------------------------------------------------------

    def _same_person(
        self, run: AgentRun, hit: Hit, customer: Customer, ctx: TriageContext, role: Role | None
    ) -> tuple[Hit, str]:
        entry = ctx.names.entry(hit.match.entry_id)
        facts = run.facts
        first = len(facts)
        facts.add(
            "customer",
            f"Customer: {customer.name}, born {customer.dob.isoformat()}, "
            f"{customer.occupation}, {customer.residency_status.value} in {customer.country}.",
        )
        facts.add(
            "entry",
            f"List entry {entry.entry_id} ({entry.severity_tier.value}): {entry.entity_name}, "
            f"born {entry.date_of_birth or 'not given'}, listed {entry.listing_date or 'n/a'}. "
            f"{entry.description}",
        )
        facts.add(
            "match",
            f"The customer's name matched '{hit.match.matched_text}' at {hit.match.score:g} "
            "out of 100"
            + (" (an alias of the entry)." if hit.match.via_alias else " (the entry's name)."),
        )
        facts.add(
            "dob",
            "Dates of birth: "
            + {True: "they agree.", False: "they disagree.", None: "the entry gives none."}[
                hit.dob_agrees
            ],
        )

        def render() -> str:
            return "\n".join(f"{f.fact_id}: {f.text}" for f in list(facts)[first:])

        what = f"Same-person call {entry.entry_id}"
        answer = self.ask(run, PROMPT.format(facts=render()), SamePerson, what, role=role)
        if answer is not None and answer.more:

            def aliases() -> None:
                facts.add(
                    "alias",
                    f"Aliases of {entry.entry_id}: {', '.join(entry.aliases) or 'none'}.",
                )

            if self.optional(run, answer.more, {"alias_lookup": aliases}):
                answer = self.ask(run, PROMPT.format(facts=render()), SamePerson, what, role=role)
        if answer is None:
            return hit, "Same-person call: no answer; the Python score stands."

        cited, unknown = known_ids(answer.fact_ids, facts)
        said = f"Same-person call: {answer.verdict} ({answer.reason.strip()[:160]})"
        if unknown:
            said += f" Unknown fact ids dropped: {', '.join(unknown)}."
        # Only a sanctions entry can be lowered: its identifiers are authoritative. A news
        # report's date of birth is not, so a news hit is never dismissed on it.
        can_lower = hit.match.tier == T.CONFIRMED_SANCTIONS and not hit.dob_agrees
        if answer.verdict == "UNSURE":
            if role is None and can_lower:
                run.queue(BACKUP_KIND, f"unsure about {entry.entry_id}")
            return hit, said + " The Python score stands."
        if answer.verdict == "SAME":
            return hit, said + " A 'same' answer never raises the score."
        if not cited:
            return hit, said + " No real fact id cited: ignored, the Python score stands."
        if hit.dob_agrees:
            return hit, said + " The dates of birth agree, so the hit cannot be dismissed."
        if not can_lower:
            return hit, said + (
                " Only a sanctions entry can be lowered on its identifiers: recorded for the "
                "analyst, the Python score stands."
            )
        lowered = min(hit.score, config.DISMISSED_SCORE)
        return (
            replace(hit, score=lowered, note=hit.note + "; judged a different person"),
            said + f" Lowered from {hit.score:g} to {lowered:g}.",
        )


# --------------------------------------------------------------------------------------
# Python screening (the scores)
# --------------------------------------------------------------------------------------


def holder_hit(customer: Customer, m: NameMatch, ctx: TriageContext) -> Hit:
    listed = ctx.names.entry(m.entry_id).date_of_birth
    agrees = dob_agrees(listed, customer.dob)
    if m.tier == T.CONFIRMED_SANCTIONS:
        if agrees is None:
            score, note = (
                config.HOLDER_SANCTIONS_NO_DOB,
                "no date of birth on the entry: cannot be set aside on the name alone",
            )
        elif agrees:
            score, note = config.HOLDER_SANCTIONS_DOB_AGREES, f"date of birth {listed} agrees"
        else:
            score, note = (
                config.HOLDER_SANCTIONS_DOB_DISAGREES,
                f"date of birth {listed} disagrees with {customer.dob}: likely name clash",
            )
        return Hit("holder", customer.name, m, score, note, agrees)
    if m.tier == T.ADVERSE_NEWS:
        return Hit("holder", customer.name, m, config.HOLDER_NEWS, "adverse news, capped", agrees)
    return replace(
        _lookalike("holder", customer.name, m, config.HOLDER_LOOKALIKE), dob_agrees=agrees
    )


def counterparty_hit(name: str, m: NameMatch) -> Hit:
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
