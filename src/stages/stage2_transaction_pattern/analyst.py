"""Stage 2's entry point: one account's 90 days in, one honest 0-1 confidence out.

    run(account_id, ctx, as_of) -> Stage2TransactionPatternOutput

Two sub-scores on the 90 days ending at `as_of`: cash structuring and pass-through. Confidence
is the stronger one. Velocity, round amounts and monthly cash go to the trail only.
Trail text is a fixed template. No model calls. Errors are left to the orchestrator.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from src.rulebook import MULE_FLOOR_CLAUSE_ID, STRUCTURING_CLAUSE_ID
from src.schemas import (
    SCORE_DECIMALS,
    ReasoningStep,
    Stage2TransactionPatternOutput,
    Transaction,
    TransactionType,
)
from src.stages.stage0_triage import TriageContext
from src.stages.stage0_triage.rules import inr
from src.stages.stage2_transaction_pattern import config, pass_through, structuring
from src.stages.stage2_transaction_pattern.pass_through import PassThroughResult
from src.stages.stage2_transaction_pattern.structuring import StructuringResult


#: Other clauses the trail cites: monthly cash total, turnover surge, non-cash rails.
MONTHLY_CASH_CLAUSE_ID = "STRUCT-002"
TURNOVER_CLAUSE_ID = "STRUCT-003"
NON_CASH_CLAUSE_ID = "STR-002"

_DIRECTION_WORDS = {
    TransactionType.CASH_DEPOSIT: ("cash deposit", "cash deposits"),
    TransactionType.CASH_WITHDRAWAL: ("cash withdrawal", "cash withdrawals"),
}


def run(
    account_id: str,
    ctx: TriageContext,
    as_of: datetime | None,
    transaction: Transaction | None = None,
) -> Stage2TransactionPatternOutput:
    """Analyse the 90 days ending at `as_of`, plus a live `transaction` even if not posted.

    `as_of` is None only for an empty bank: nothing to analyse.
    """
    rows = _rows(ctx, account_id, transaction) if as_of is not None else ()
    end = as_of or datetime.max  # no rows either way; any end with room for the frame
    frame = structuring.in_window(rows, end, config.FRAME)

    s = structuring.score(rows, end, ctx.rulebook)
    p = pass_through.score(rows, end, ctx.rulebook)
    confidence = round(max(s.score, p.score), SCORE_DECIMALS)
    velocity = _velocity_summary(frame, as_of)

    trail = [
        _structuring_step(s),
        _pass_through_step(p),
        ReasoningStep(
            step="Combined confidence",
            finding=f"Structuring {s.score:.4f}, pass-through {p.score:.4f}.",
            impact=f"Confidence is the stronger of the two: {confidence:.4f}.",
        ),
        ReasoningStep(
            step="Velocity",
            finding=velocity,
            impact=f"Reported, not scored; Stage 3 compares it with the profile "
            f"({TURNOVER_CLAUSE_ID}).",
        ),
        _round_share_step(frame),
        _month_totals_step(s),
        _non_cash_step(s),
        _stranger_share_step(p),
    ]
    return Stage2TransactionPatternOutput(
        structuring_indicators=_structuring_indicators(s),
        pass_through_indicators=_pass_through_indicators(p),
        velocity_summary=velocity,
        confidence=confidence,
        reasoning_trail=trail[: config.MAX_TRAIL_STEPS],
    )


def _rows(
    ctx: TriageContext, account_id: str, transaction: Transaction | None
) -> tuple[Transaction, ...]:
    """The account's posted rows, plus a live transaction not yet posted, in time order."""
    rows = ctx.bank.transactions(account_id)
    if transaction is None or any(t.transaction_id == transaction.transaction_id for t in rows):
        return rows
    return tuple(sorted((*rows, transaction), key=lambda t: (t.timestamp, t.transaction_id)))


# --------------------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------------------


def _structuring_indicators(s: StructuringResult) -> list[str]:
    lines = []
    band = f"{inr(s.band.band_floor_inr)}–{inr(s.band.threshold_inr)}"
    for kind, window in s.busiest.items():
        if not window.count:
            continue
        one, many = _DIRECTION_WORDS[kind]
        if window.count == 1:
            lines.append(
                f"1 {one} of {band} on {_day(window.first)}, under the CTR "
                f"({STRUCTURING_CLAUSE_ID})"
            )
        else:
            lines.append(
                f"{window.count} {many} of {band} between {_span(window.first, window.last)}, "
                f"each under the CTR ({STRUCTURING_CLAUSE_ID})"
            )
    return lines


def _pass_through_indicators(p: PassThroughResult) -> list[str]:
    w = p.best
    if w.score == 0 and w.n == 0:
        return []
    lines = [
        f"{w.n} of {w.matched} matched stranger credits of at least {inr(p.floor)} were sent on "
        f"within {config.FAST_HOLD_HOURS:g} hours ({w.share:.0%}) in the {w.label} to "
        f"{_day(w.end)}; median hold {_hours(w.median_fast_hold)} ({MULE_FLOOR_CLAUSE_ID})"
    ]
    if w.n:
        lines.append(
            f"{w.top_payee_share:.0%} of the money that closed those runs went to one payee, "
            f"{w.novelty:.0%} to payees not paid before ({MULE_FLOOR_CLAUSE_ID})"
        )
    return lines


# --------------------------------------------------------------------------------------
# Trail steps
# --------------------------------------------------------------------------------------


def _structuring_step(s: StructuringResult) -> ReasoningStep:
    deposits = s.busiest[TransactionType.CASH_DEPOSIT].count
    withdrawals = s.busiest[TransactionType.CASH_WITHDRAWAL].count
    finding = (
        f"k = {s.k}: at most {deposits} in-band cash deposits and {withdrawals} in-band cash "
        f"withdrawals in any {config.WINDOW_DAYS} days, band "
        f"{inr(s.band.band_floor_inr)}–{inr(s.band.threshold_inr)}."
    )
    if s.k:
        lead = s.leading
        finding += f" The busiest {lead.count} total {inr(lead.total)}."
    return ReasoningStep(
        step="Structuring",
        finding=finding,
        impact=f"Structuring sub-score {s.score:.4f} from the k map ({STRUCTURING_CLAUSE_ID}).",
    )


def _pass_through_step(p: PassThroughResult) -> ReasoningStep:
    w = p.best
    booster = config.NOVELTY_BASE + config.NOVELTY_BOOST * w.novelty
    return ReasoningStep(
        step="Pass-through",
        finding=f"{w.label}: {w.n} of {w.matched} matched stranger credits sent on within "
        f"{config.FAST_HOLD_HOURS:g} hours; novelty {w.novelty:.0%}.",
        impact=f"Pass-through sub-score {w.score:.4f} = gate {w.gate:g} x share map "
        f"{w.share_mapped:.2f} x novelty factor {booster:.2f} ({MULE_FLOOR_CLAUSE_ID}).",
    )


def _round_share_step(frame: Sequence[Transaction]) -> ReasoningStep:
    unit = config.ROUND_AMOUNT_INR
    rounded = sum(1 for t in frame if t.amount % unit == 0)
    share = rounded / len(frame) if frame else 0.0
    return ReasoningStep(
        step="Round-amount share",
        finding=f"{rounded} of {len(frame)} transactions ({share:.0%}) are whole multiples "
        f"of {inr(unit)}.",
        impact="Reported, not scored: round amounts are common in honest activity.",
    )


def _month_totals_step(s: StructuringResult) -> ReasoningStep:
    if s.month_totals:
        finding = (
            "; ".join(
                f"{m.month:%b %Y}: deposits {inr(m.deposits)}, withdrawals {inr(m.withdrawals)}"
                for m in s.month_totals
            )
            + "."
        )
    else:
        finding = "No cash in the frame."
    return ReasoningStep(
        step="Calendar-month cash totals",
        finding=finding,
        impact=f"The legal monthly cash aggregate ({MONTHLY_CASH_CLAUSE_ID}); reported, "
        "not scored.",
    )


def _non_cash_step(s: StructuringResult) -> ReasoningStep:
    band = f"{inr(s.band.band_floor_inr)}–{inr(s.band.threshold_inr)}"
    return ReasoningStep(
        step="In-band non-cash payments",
        finding=f"{s.in_band_non_cash} non-cash payments in the band {band}.",
        impact=f"Context only: no amount report applies to these rails, so they are not "
        f"structuring ({NON_CASH_CLAUSE_ID}).",
    )


def _stranger_share_step(p: PassThroughResult) -> ReasoningStep:
    w = p.best
    return ReasoningStep(
        step="Stranger share",
        finding=f"{w.label}: {w.strangers} of {w.credits_at_floor} credits of at least "
        f"{inr(p.floor)} came from one-time payers ({w.stranger_share:.0%}).",
        impact="Reported, not scored: one-time payers are common for merchants too.",
    )


def _velocity_summary(frame: Sequence[Transaction], as_of: datetime | None) -> str:
    if not frame:
        to = f" to {_day(as_of)}" if as_of else ""
        return f"No transactions in the {config.FRAME_DAYS} days{to}."
    busiest = structuring.busiest_window(frame, config.WINDOW)
    average = len(frame) / (config.FRAME_DAYS / config.WINDOW_DAYS)
    return (
        f"Busiest {config.WINDOW_DAYS} days from {_day(busiest.first)}: {busiest.count} "
        f"transactions against an average of {average:.1f} per {config.WINDOW_DAYS} days over "
        f"the {config.FRAME_DAYS}-day frame (ratio {busiest.count / average:.2f})."
    )


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------


def _day(moment: datetime, year: bool = True) -> str:
    return f"{moment.day} {moment:%b}" + (f" {moment.year}" if year else "")


def _span(first: datetime, last: datetime) -> str:
    return f"{_day(first, year=first.year != last.year)} and {_day(last)}"


def _hours(hold: timedelta | None) -> str:
    return f"{hold.total_seconds() / 3600:.1f} h" if hold is not None else "n/a"
