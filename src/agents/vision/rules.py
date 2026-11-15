"""The KYC contradiction rules. Python sets the level; the model only supplied the claims.

The rules are public red flags (RBI KYC Master Direction, FATF), not facts about this data:

- volume: money in against the declared base, which is the declared income, or the declared
  expected activity when the profile states a larger one. 1.5x to 3x is a moderate concern,
  3x or more a high one. Cash that the transaction-pattern check already scores is left out,
  so the same money is not counted as two findings. A declared one-off event that covers the
  excess cancels the concern.
- "no cash" against real cash deposits: a direct contradiction, a high concern.
- "no foreign dealings" against cross-border wires: a moderate concern.
- "no business" or "one source" against very many payers: a moderate concern.
- declared cash takings explain monthly cash up to 1.25x the declared amount.

Level: two high concerns SEVERE, one HIGH, two moderate HIGH, one moderate MODERATE.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.agents.vision import config
from src.agents.vision.claims import Claims
from src.agents.vision.kyc_facts import MONTHS, Activity
from src.stages.stage0_triage.rules import inr


HIGH, MODERATE = "HIGH", "MODERATE"


@dataclass(frozen=True)
class Concern:
    key: str
    level: str
    text: str


@dataclass
class KycVerdict:
    level: str = "NONE"
    score: float = 0.0
    concerns: list[Concern] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    cash_explained: bool = False
    cash_note: str = ""
    volume_multiple: float = 0.0


def judge(claims: Claims, activity: Activity, cash_scored: bool) -> KycVerdict:
    """`cash_scored`: the transaction-pattern check already scores this account's cash."""
    verdict = KycVerdict()
    concerns: list[Concern] = []

    # Volume
    declared_activity = (
        claims.amount("expected_activity") if claims.is_("expected_activity", "STATED") else None
    )
    base = max(activity.income_month, declared_activity or 0.0)
    cash_month = activity.cash_deposit_total / MONTHS if cash_scored else 0.0
    money_in = max(activity.money_in_month - cash_month, 0.0)
    multiple = round(money_in / base, 2) if base else 0.0
    verdict.volume_multiple = multiple
    if multiple >= config.INCOME_MODERATE_FROM:
        level = HIGH if multiple >= config.INCOME_HIGH_FROM else MODERATE
        against = "declared expected activity" if base > activity.income_month else "income"
        left_out = " (cash scored by the transaction-pattern check left out)" if cash_month else ""
        text = (
            f"Money in {inr(money_in)} a month is {multiple:g}x the {against} of "
            f"{inr(base)} a month{left_out}."
        )
        covered = _event_covers(claims, (money_in - base) * MONTHS, level)
        if covered:
            verdict.cancelled.append(f"{text} {covered}")
        else:
            concerns.append(Concern("volume", level, text))

    # "No cash" against cash deposits
    cash_share = (
        (activity.cash_deposit_total / MONTHS) / activity.income_month
        if activity.income_month
        else 0.0
    )
    if claims.is_("cash", "NO_CASH") and cash_share >= config.NO_CASH_MIN_INCOME_SHARE:
        concerns.append(
            Concern(
                "cash",
                HIGH,
                f"Profile says '{claims.quotes['cash']}' but the account has "
                f"{activity.cash_deposits} cash deposits totalling "
                f"{inr(activity.cash_deposit_total)}.",
            )
        )

    # Declared cash takings
    if claims.is_("cash", "CASH_DECLARED"):
        declared = claims.amount("cash")
        if declared and activity.busiest_cash_month <= declared * config.CASH_EXPLAINED_WITHIN:
            verdict.cash_explained = True
            verdict.cash_note = (
                f"Profile declares cash takings of {inr(declared)} a month "
                f"('{claims.quotes['cash']}'); the busiest month's cash is "
                f"{inr(activity.busiest_cash_month)}, within "
                f"{config.CASH_EXPLAINED_WITHIN:g}x: explained."
            )
        elif declared:
            verdict.cash_note = (
                f"Profile declares cash takings of {inr(declared)} a month, but the busiest "
                f"month's cash is {inr(activity.busiest_cash_month)}: not explained."
            )
        else:
            verdict.cash_note = "Profile declares cash use without an amount: not explained."

    # "No foreign dealings" against wires
    if claims.is_("foreign", "NO_FOREIGN") and activity.cross_border_wires > 0:
        concerns.append(
            Concern(
                "foreign",
                MODERATE,
                f"Profile says '{claims.quotes['foreign']}' but the account has "
                f"{activity.cross_border_wires} cross-border wires.",
            )
        )

    # "No business" / "one source" against very many payers
    narrow = next(
        (
            n
            for n, v in (("business", "NO_BUSINESS"), ("sources", "SINGLE_SOURCE"))
            if claims.is_(n, v)
        ),
        None,
    )
    if narrow and activity.payers >= config.MANY_PAYERS_FROM:
        concerns.append(
            Concern(
                "payers",
                MODERATE,
                f"Profile says '{claims.quotes[narrow]}' but credits came from "
                f"{activity.payers} different payers.",
            )
        )

    verdict.concerns = concerns
    verdict.level = level_of(concerns)
    verdict.score = config.LEVEL_SCORES[verdict.level]
    return verdict


def level_of(concerns: list[Concern]) -> str:
    high = sum(1 for c in concerns if c.level == HIGH)
    moderate = sum(1 for c in concerns if c.level == MODERATE)
    if high >= 2:
        return "SEVERE"
    if high == 1 or moderate >= 2:
        return "HIGH"
    return "MODERATE" if moderate == 1 else "NONE"


def drop_concern(verdict: KycVerdict, key: str, why: str) -> None:
    """Cancel one concern (the narrow check found a stated reason) and set the level again."""
    for concern in [c for c in verdict.concerns if c.key == key]:
        verdict.concerns.remove(concern)
        verdict.cancelled.append(f"{concern.text} {why}")
    verdict.level = level_of(verdict.concerns)
    verdict.score = config.LEVEL_SCORES[verdict.level]


def _event_covers(claims: Claims, excess: float, level: str) -> str | None:
    """Why a declared one-off event cancels the volume concern, or None when it does not."""
    if not claims.is_("one_off_event", "EVENT_DECLARED"):
        return None
    quote = claims.quotes["one_off_event"]
    amount = claims.amount("one_off_event")
    if amount is not None:
        if excess <= amount * config.EVENT_COVERS_WITHIN:
            return f"Cancelled: the declared one-off event ('{quote}') covers the excess."
        return None
    if level == MODERATE:
        return f"Cancelled: a declared one-off event ('{quote}') explains a moderate excess."
    return None
