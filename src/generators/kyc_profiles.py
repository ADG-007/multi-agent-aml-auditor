"""Generator 4 of 9 - free-text KYC narratives.

The primary input to Stage 3, and a routing input to Stage 0.

These are deliberately unstructured prose, written in the register a branch officer actually
uses in an account-opening record. Stage 3's job is judging whether a human-written account
of someone's life is plausible against what their account really does, and that judgement is
only meaningful if the text reads like something a person wrote rather than a serialised
dict.

Narratives are built after transactions exist, so a clean profile can state a figure that
genuinely matches the account's behaviour, and a planted mismatch can contradict it by a
specific, checkable margin rather than vaguely.
"""

from __future__ import annotations

import random
from collections import defaultdict

from src.generators import config as cfg
from src.schemas import (
    Account,
    AccountType,
    Customer,
    KYCProfile,
    ResidencyStatus,
    Transaction,
    TransactionType,
)

#: Why an ordinary customer says they opened the account.
DOMESTIC_PURPOSES = (
    "routine salary credits and household expenses",
    "day-to-day family expenses and utility payments",
    "business receipts and supplier payments",
    "collection of shop takings and vendor settlements",
    "monthly pension credit and medical expenses",
    "savings towards a home loan down payment",
    "school fees, insurance premiums and regular savings",
    "receiving professional fees from clients",
)

NRI_PURPOSES = (
    "remitting monthly earnings home to support family in India",
    "maintaining savings in India while working abroad",
    "funding a property purchase in India from overseas earnings",
    "supporting parents and meeting household costs in India",
    "receiving overseas salary and managing Indian investments",
)

#: Length-of-relationship phrasing, so every narrative does not open identically.
OPENERS = (
    "Customer has been known to the branch since account opening.",
    "Profile captured at account opening and reviewed at the last periodic updation.",
    "Details recorded during in-person verification at the branch.",
    "Profile updated at the customer's most recent KYC refresh.",
    "Information confirmed against the officially valid documents on file.",
)


def _lakh(amount: float) -> str:
    """Format an INR amount the way an Indian banker writes it."""
    if amount >= 10_000_000:
        return f"Rs {amount / 10_000_000:.2f} crore"
    if amount >= 100_000:
        return f"Rs {amount / 100_000:.2f} lakh"
    return f"Rs {amount:,.0f}"


def _observed(transactions: list[Transaction]) -> dict:
    """Summarise what the account actually did over the 90-day window."""
    total = sum(t.amount for t in transactions)
    cash_in = sum(t.amount for t in transactions if t.type == TransactionType.CASH_DEPOSIT)
    largest = max((t.amount for t in transactions), default=0.0)
    foreign_like = sum(
        1 for t in transactions if t.type in (TransactionType.NEFT, TransactionType.RTGS)
    )
    return {
        "count": len(transactions),
        "total": total,
        "cash_in": cash_in,
        "largest": largest,
        "wire_count": foreign_like,
        # Extrapolated from 90 days to a year, which is the basis on which a stated annual
        # income can be compared to observed activity at all.
        "annualised": total * (365 / 90),
    }


def _purpose_for(rng: random.Random, customer: Customer, account: Account) -> str:
    """A stated purpose that fits the customer's occupation and account type.

    A junior engineer whose savings account is described as collecting shop takings reads as
    a mismatch in its own right. Since the mismatch channel is planted deliberately, a clean
    profile has to be genuinely coherent, or Stage 3's false-positive rate is decided by this
    function rather than by the model.
    """
    occupation = customer.occupation.lower()
    if customer.residency_status == ResidencyStatus.NRI:
        return rng.choice(NRI_PURPOSES)
    if "retired" in occupation:
        return "monthly pension credit and medical expenses"
    if "student" in occupation:
        return "school fees, insurance premiums and regular savings"
    if account.account_type == AccountType.CURRENT:
        return rng.choice(
            (
                "business receipts and supplier payments",
                "collection of shop takings and vendor settlements",
                "receiving professional fees from clients",
            )
        )
    return rng.choice(
        (
            "routine salary credits and household expenses",
            "day-to-day family expenses and utility payments",
            "savings towards a home loan down payment",
            "school fees, insurance premiums and regular savings",
        )
    )


def _clean_narrative(
    rng: random.Random,
    customer: Customer,
    account: Account,
    observed: dict,
) -> str:
    """A profile that holds up against the account's actual behaviour."""
    is_nri = customer.residency_status == ResidencyStatus.NRI
    purpose = _purpose_for(rng, customer, account)
    opener = rng.choice(OPENERS)

    residency_line = (
        f"Customer is a Non-Resident Indian currently residing in {customer.country}, "
        f"and visits India periodically."
        if is_nri
        else f"Customer resides at the address on record in India."
    )

    return (
        f"{opener} {customer.name}, aged {_age(customer)}, works as a "
        f"{customer.occupation.lower()} and declares an annual income of "
        f"{_lakh(customer.stated_income)}. {residency_line} The {account.account_type.value.lower()} "
        f"account is maintained for {purpose}. Expected activity is in the range of "
        f"{_lakh(customer.stated_income / 12 * 1.5)} per month across roughly "
        f"{max(1, observed['count'] // 3)} transactions, mostly through digital channels. "
        f"Source of funds is stated as {_source_of_funds(customer)}. No adverse information "
        f"was noted at the time of verification."
    )


def _age(customer: Customer) -> int:
    today = cfg.REFERENCE_DATE
    return (
        today.year
        - customer.dob.year
        - ((today.month, today.day) < (customer.dob.month, customer.dob.day))
    )


def _source_of_funds(customer: Customer) -> str:
    occupation = customer.occupation.lower()
    if "retired" in occupation:
        return "pension and accumulated retirement savings"
    if "student" in occupation:
        return "family support and a scholarship stipend"
    if "homemaker" in occupation:
        return "spouse's salary income"
    if any(word in occupation for word in ("proprietor", "trader", "contractor", "agent")):
        return "business receipts"
    if "overseas" in occupation or "gulf" in occupation:
        return "overseas salary income"
    return "salary income"


def _mismatch_narrative(
    rng: random.Random,
    customer: Customer,
    account: Account,
    observed: dict,
) -> str:
    """A profile that contradicts the account's behaviour in a specific, checkable way.

    Three shapes, all drawn from real KYC red flags rather than invented: an income that
    cannot support the throughput, a stated purpose that excludes the activity actually
    present, and an occupation that does not explain large cash handling. Each states a
    concrete figure, so Stage 3 has something to disagree with rather than a vague vibe.
    """
    is_nri = customer.residency_status == ResidencyStatus.NRI
    opener = rng.choice(OPENERS)
    shape = rng.choice(("income", "purpose", "occupation"))

    # A declared income well below what the account moves. Understating it is the point.
    understated = max(300_000, int(customer.stated_income * rng.uniform(0.25, 0.45)))

    if shape == "income":
        return (
            f"{opener} {customer.name}, aged {_age(customer)}, is employed as a "
            f"{customer.occupation.lower()} with a declared annual income of "
            f"{_lakh(understated)}. Customer stated that the account would see modest "
            f"activity of around {_lakh(understated / 24)} per month, consistent with a "
            f"single salary credit and routine household withdrawals. Source of funds is "
            f"stated as {_source_of_funds(customer)} only. Customer confirmed no other "
            f"income streams, business interests or third-party deposits."
        )

    if shape == "purpose":
        purpose = (
            "receiving monthly remittances from the customer's own overseas salary, with no "
            "third-party credits expected"
            if is_nri
            else "receiving the monthly pension credit and paying household utility bills, "
            "with no business use whatsoever"
        )
        return (
            f"{opener} {customer.name}, aged {_age(customer)}, described the "
            f"{account.account_type.value.lower()} account as being opened solely for "
            f"{purpose}. Customer declares an annual income of {_lakh(customer.stated_income)} "
            f"from {_source_of_funds(customer)} and stated that no cash deposits would be "
            f"made into the account at any point. Customer declined to nominate any "
            f"authorised signatory and confirmed the account is operated personally."
        )

    return (
        f"{opener} {customer.name}, aged {_age(customer)}, is recorded as a "
        f"{customer.occupation.lower()} earning {_lakh(understated)} annually. Customer "
        f"explained that the account is used for personal purposes only and that all "
        f"receipts are salary credits from a single employer. Customer stated they have no "
        f"involvement in trading, commission work or handling of cash on behalf of others, "
        f"and that they do not deal with overseas counterparties. Address and employment "
        f"details were verified against the officially valid document on file."
    )


def generate(
    rng: random.Random,
    customers: list[Customer],
    accounts: list[Account],
    transactions: list[Transaction],
    plan,
) -> list[KYCProfile]:
    """One narrative per customer, matched or deliberately mismatched per the planting plan."""
    by_account: dict[str, list[Transaction]] = defaultdict(list)
    for txn in transactions:
        by_account[txn.account_id].append(txn)

    account_by_customer = {a.customer_id: a for a in accounts}

    profiles: list[KYCProfile] = []
    for customer in customers:
        account = account_by_customer[customer.customer_id]
        observed = _observed(by_account.get(account.account_id, []))

        if account.account_id in plan.kyc_mismatch:
            narrative = _mismatch_narrative(rng, customer, account, observed)
        else:
            narrative = _clean_narrative(rng, customer, account, observed)

        profiles.append(KYCProfile(customer_id=customer.customer_id, narrative=narrative))

    return profiles
