"""The Stage 3 prompt and answer format. Rules of thumb come from public RBI / FATF red flags,
never from this project's data, so the answer key cannot leak in."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.stages.stage3_kyc.summary import Fact


Level = Literal["NONE", "LOW", "MODERATE", "HIGH", "SEVERE"]


class Mismatch(BaseModel):
    fact_id: str = Field(description="The fact id that breaks the claim, e.g. F4.")
    profile_claim: str = Field(
        description="The profile words it breaks, copied from the profile. Never a claim the "
        "profile does not make."
    )
    why: str = Field(description="One short sentence.")


class KYCJudgement(BaseModel):
    level: Level = Field(description="How badly the profile fails to explain the activity.")
    mismatches: list[Mismatch] = Field(default_factory=list)
    notes: list[str] = Field(
        default_factory=list, description="At most 3 short sentences of reasoning."
    )


TEMPLATE = """You review one bank customer. Decide whether the customer's KYC profile explains \
what the account actually did. Judge meaning only: every number below is already computed, so \
do not recompute anything.

Most customers are ordinary. Start from NONE and move up only for a rule below.

What is normal (never a mismatch):
- Money in up to about 1.5x the declared income. Turnover counts money in and money out, so \
up to about 3x the declared income is normal.
- A few cash deposits or withdrawals that are small next to the declared income. "Mostly \
digital" does not rule out cash.
- A few credits sent on within hours, when most of the money stays.

Rules of thumb (general red flags):
- Money in several times (3x or more) the declared income, with no reason in the profile, is \
a high concern. Money in about 1.5x to 3x with no reason is a moderate concern.
- A profile that rules something out in its own words (no cash, no business use, one income \
source) while the account shows it is a direct contradiction: a high concern.
- Large or repeated cross-border money with no foreign link or purpose in the profile is a \
concern.
- Money from many unrelated payers, most of it sent on within hours, with no business reason \
in the profile, is a concern.
- A profile that declares the activity (a shop declaring many small UPI receipts, a stated \
one-off event such as a property sale) explains it: low or none.

Levels:
- NONE: the profile explains the activity.
- LOW: a small oddity, no rule above is met.
- MODERATE: one moderate concern.
- HIGH: one high concern, or two moderate concerns.
- SEVERE: two or more high concerns.

For every mismatch give the fact id it rests on (F1, F2 ...) and the profile words it breaks, \
copied from the profile. A normal fact is not a mismatch. At most 4 mismatches and at most 3 \
short notes. If the profile explains the activity, give NONE and no mismatches.

PROFILE:
{profile}

FACTS:
{facts}
"""


def render(profile: str, facts: list[Fact]) -> str:
    return TEMPLATE.format(
        profile=profile.strip() or "(no profile on file)",
        facts="\n".join(f"{f.fact_id}: {f.text}" for f in facts),
    )
