"""KYC claim extraction: the model copies what the profile says; it never judges.

Two reads with two framings (a form, then a question list in the opposite order). A claim
counts only when both reads give the same value and its quote is in the profile word for
word. A disagreement is UNSURE: the claim is treated as not stated until the backup model
has read it. The model returns quotes only; which claim a quote proves is fixed by the field
it sits in, and Python checks that a "no ..." field really holds a denial. Amounts are parsed
by Python from the quoted words, never taken from the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from src.agents.base import quote_in


NOT_STATED = "NOT_STATED"


class ClaimsForm(BaseModel):
    """Read 1: the form. Every field is a quote from the profile, or an empty string.
    Every field is required, so the model must answer each one."""

    no_cash: str = Field(description="Words saying cash will NOT be deposited.")
    cash_takings: str = Field(description="Words saying cash IS deposited regularly.")
    no_foreign: str = Field(description="Words saying there are NO overseas dealings.")
    foreign_dealings: str = Field(description="Words describing money abroad.")
    no_business: str = Field(description="Words saying NO business use.")
    business_receipts: str = Field(description="Words saying customers pay in.")
    single_source: str = Field(description="Words saying one income source only.")
    one_off_event: str = Field(description="Words naming a one-off inflow.")
    expected_activity: str = Field(description="Expected rupee amount per month.")


class ClaimsAnswers(BaseModel):
    """Read 2: the same claims, asked as questions in the opposite order."""

    expected_activity: str
    one_off_event: str
    single_source: str
    business_receipts: str
    no_business: str
    foreign_dealings: str
    no_foreign: str
    cash_takings: str
    no_cash: str


class ReasonClaim(BaseModel):
    """The narrow question: does the profile give a reason for money above the income?"""

    value: Literal["REASON_GIVEN", "NO_REASON"]
    quote: str = ""


FIELDS = ("cash", "foreign", "business", "sources", "one_off_event", "expected_activity")

#: claim -> ((quote field, the value it proves, whether the words must deny something), ...)
QUOTE_FIELDS: dict[str, tuple[tuple[str, str, bool], ...]] = {
    "cash": (("no_cash", "NO_CASH", True), ("cash_takings", "CASH_DECLARED", False)),
    "foreign": (
        ("no_foreign", "NO_FOREIGN", True),
        ("foreign_dealings", "FOREIGN_DECLARED", False),
    ),
    "business": (
        ("no_business", "NO_BUSINESS", True),
        ("business_receipts", "BUSINESS_DECLARED", False),
    ),
    "sources": (("single_source", "SINGLE_SOURCE", False),),
    "one_off_event": (("one_off_event", "EVENT_DECLARED", False),),
    "expected_activity": (("expected_activity", "STATED", False),),
}

_RULES = """Rules:
- Every answer is words copied from the profile, character for character, one sentence at \
most. Never write your own words.
- If the profile does not say it, answer with an empty string "". Most profiles leave most \
of these unsaid.
- Do not judge the customer. Do not use outside knowledge. The occupation alone proves \
nothing."""

FORM_PROMPT = """Fill in a form from one bank customer's KYC profile.

{rules}

Fields:
- no_cash: the words where the profile says cash will NOT be deposited into the account.
- cash_takings: the words where the profile says cash IS deposited as normal use of the \
account (for example a shop's cash takings), with the amount if one is given. "Mostly \
digital" is not this.
- no_foreign: the words where the profile says the customer has NO overseas or foreign \
dealings.
- foreign_dealings: the words where the profile describes money sent to or received from \
another country.
- no_business: the words where the profile says the account has NO business use, or is for \
personal use only.
- business_receipts: the words where the profile says the account receives payments from \
the customers or clients of a business.
- single_source: the words where the profile says there is one income source only, or no \
other income.
- one_off_event: the words where the profile names a one-off inflow (a property sale, an \
inheritance, a loan, a gift, a maturity payout).
- expected_activity: the words where the profile gives the expected account activity as a \
rupee amount per month.

PROFILE:
<<<{profile}>>>
"""

QUESTIONS_PROMPT = """Sort the sentences of one bank customer's KYC profile into topics.

Go through the profile one sentence at a time. If a sentence is about one of the nine \
topics below, copy that sentence (or the part of it that matters) under the topic. A topic \
that no sentence is about stays "".

{rules}

Topics:
1. expected_activity: a sentence that gives the expected account activity as a rupee \
amount per month.
2. one_off_event: a sentence about a single, one-time inflow such as a property sale, an \
inheritance, a loan, a gift or a maturity payout.
3. single_source: a sentence saying the customer has one source of income only, or no \
other income.
4. business_receipts: a sentence saying the account takes payments from the customers or \
clients of a business.
5. no_business: a sentence saying the account will NOT be used for business, or is for \
personal use only.
6. foreign_dealings: a sentence describing money moving to or from another country.
7. no_foreign: a sentence saying the customer has NO dealings overseas.
8. cash_takings: a sentence saying cash IS deposited regularly as normal use, such as cash \
takings, with its amount if the sentence gives one.
9. no_cash: a sentence saying NO cash will be deposited.

PROFILE:
<<<{profile}>>>
"""

NARROW_PROMPT = """One question about the KYC profile below.

Does the profile give a reason why the account could receive more money than the declared \
income? Examples: a declared business turnover, a one-off sale or inheritance, family \
remittances, a second stated income. REASON_GIVEN with the exact words from the profile, or \
NO_REASON with an empty quote. The occupation alone is not a reason.

PROFILE:
<<<{profile}>>>
"""


def form_prompt(profile: str) -> str:
    return FORM_PROMPT.format(rules=_RULES, profile=profile.strip())


def questions_prompt(profile: str) -> str:
    return QUESTIONS_PROMPT.format(rules=_RULES, profile=profile.strip())


def narrow_prompt(profile: str) -> str:
    return NARROW_PROMPT.format(profile=profile.strip())


# --------------------------------------------------------------------------------------
# Guard and reconciliation
# --------------------------------------------------------------------------------------


@dataclass
class Claims:
    """The claims that survived: both reads agree and the quote is in the profile."""

    values: dict[str, str] = field(default_factory=lambda: dict.fromkeys(FIELDS, NOT_STATED))
    quotes: dict[str, str] = field(default_factory=dict)
    #: Fields the two reads disagreed on (treated as not stated until resolved).
    unsure: list[str] = field(default_factory=list)
    #: Quotes a read gave that are not in the profile, or that say the opposite of their field.
    dropped: list[str] = field(default_factory=list)
    #: The disagreement could change the KYC result, so the backup model is asked.
    matters: bool = False
    #: Both reads failed (not: the models are off), so nothing could be checked.
    unread: bool = False

    def is_(self, name: str, value: str) -> bool:
        return self.values[name] == value

    def amount(self, name: str) -> float | None:
        """The first rupee amount in the claim's quote, parsed by Python."""
        found = parse_inr(self.quotes.get(name, ""))
        return found[0] if found else None

    def stated(self) -> dict[str, str]:
        return {k: v for k, v in self.values.items() if v != NOT_STATED}


_DENIAL = re.compile(r"\b(no|not|never|without|nil|none|neither|nor|solely|only)\b", re.IGNORECASE)


def guarded(read: BaseModel | None, profile: str) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """One read as claim -> (value, quote). A quote counts only when it is in the profile word
    for word and its words fit its field: a "no ..." field must hold a denial."""
    out: dict[str, tuple[str, str]] = {}
    dropped: list[str] = []
    if read is None:
        return out, dropped
    for name, options in QUOTE_FIELDS.items():
        kept: list[tuple[str, str]] = []
        for quote_field, value, must_deny in options:
            quote = (getattr(read, quote_field) or "").strip()
            if not quote:
                continue
            if not quote_in(quote, profile):
                dropped.append(f"{value} (quote not in the profile)")
            elif len(options) > 1 and must_deny != bool(_DENIAL.search(quote)):
                dropped.append(f"{value} (the quoted words say the opposite)")
            else:
                kept.append((value, quote))
        out[name] = kept[0] if len(kept) == 1 else (NOT_STATED, "")
        if len(kept) > 1:
            dropped.append(f"{name} (both sides quoted)")
    return out, dropped


def reconcile(read_1: BaseModel | None, read_2: BaseModel | None, profile: str) -> Claims:
    """Keep a claim when both reads agree. One read alone can only say "not stated"."""
    a, dropped_a = guarded(read_1, profile)
    b, dropped_b = guarded(read_2, profile)
    claims = Claims(dropped=[f"read 1 {d}" for d in dropped_a] + [f"read 2 {d}" for d in dropped_b])
    if not a and not b:
        return claims
    for name in FIELDS:
        va, qa = a.get(name, (None, ""))
        vb, qb = b.get(name, (None, ""))
        if va == vb and va is not None:
            claims.values[name] = va
            if va != NOT_STATED:
                # The longer quote is kept: it is the one more likely to hold the amount.
                claims.quotes[name] = max(qa, qb, key=len)
        elif {va, vb} <= {NOT_STATED, None}:
            continue  # one read missing, the other says nothing: nothing to keep
        else:
            claims.unsure.append(name)
    return claims


def single(read: BaseModel | None, profile: str) -> Claims:
    """The claims one read alone would give (used to see whether a disagreement matters)."""
    claims = Claims()
    for name, (value, quote) in guarded(read, profile)[0].items():
        claims.values[name] = value
        if value != NOT_STATED:
            claims.quotes[name] = quote
    return claims


def resolve(claims: Claims, read_3: BaseModel | None, profile: str) -> list[str]:
    """Settle the unsure fields with the backup model's read. Returns the fields settled."""
    third, dropped = guarded(read_3, profile)
    claims.dropped += [f"backup {d}" for d in dropped]
    settled = []
    for name in list(claims.unsure):
        if name in third:
            value, quote = third[name]
            claims.values[name] = value
            if value != NOT_STATED:
                claims.quotes[name] = quote
            claims.unsure.remove(name)
            settled.append(name)
    return settled


_AMOUNT = re.compile(
    r"(?:rs\.?|₹|inr)\s*([\d,]+(?:\.\d+)?)\s*(crore|lakh|lac|thousand)?",
    re.IGNORECASE,
)
_UNITS = {"crore": 10_000_000, "lakh": 100_000, "lac": 100_000, "thousand": 1_000}


def parse_inr(text: str) -> list[float]:
    """Every rupee amount in `text`: "Rs 9.08 lakh" -> 908000, "Rs 89,583" -> 89583."""
    amounts = []
    for number, unit in _AMOUNT.findall(text):
        digits = number.replace(",", "")
        if digits and digits != ".":
            amounts.append(float(digits) * _UNITS.get((unit or "").lower(), 1))
    return amounts
