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
    Counterparty,
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
    foreign_wires = sum(1 for t in transactions if t.type == TransactionType.SWIFT)
    return {
        "count": len(transactions),
        "total": total,
        "cash_in": cash_in,
        "largest": largest,
        "swift_count": foreign_wires,
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


def _foreign_line(
    cohort: str | None,
    is_nri: bool,
    is_round_trip_solo: bool,
    swift: list[tuple[Transaction, Counterparty]],
) -> str:
    """The sentence a branch officer records about expected foreign activity.

    A legitimate cross-border customer declares the activity up front - that declaration is
    exactly what separates them from the typology they resemble, and what Stage 3 weighs.
    """
    if is_nri:
        return (
            " Overseas earnings are credited to the NRE account by inward remittance from "
            "the customer's country of residence."
        )
    if not swift:
        return ""
    _first_txn, first_cp = swift[0]
    if cohort == "family_inward":
        return (
            f" Customer receives monthly family-maintenance remittances from a son or "
            f"daughter working in {_country_name(first_cp.country)}."
        )
    if cohort == "education_outward":
        return (
            f" Customer funds a child's university education at {first_cp.name}, "
            f"{first_cp.country}; fee remittances are made each term under the Liberalised "
            f"Remittance Scheme."
        )
    if cohort == "lrs_outward" or is_round_trip_solo:
        return (
            " Customer makes occasional remittances abroad under the Liberalised Remittance "
            "Scheme, for gifts to family and overseas investments."
        )
    return ""


def _clean_narrative(
    rng: random.Random,
    customer: Customer,
    account: Account,
    observed: dict,
    foreign_line: str = "",
) -> str:
    """A profile that holds up against the account's actual behaviour."""
    is_nri = customer.residency_status == ResidencyStatus.NRI
    purpose = _purpose_for(rng, customer, account)
    opener = rng.choice(OPENERS)

    residency_line = (
        f"Customer is a Non-Resident Indian currently residing in {_country_name(customer.country)}, "
        f"and visits India periodically."
        if is_nri
        else f"Customer resides at the address on record in India."
    )

    return (
        f"{opener} {customer.name}, aged {_age(customer)}, {_occupation_phrase(customer)} "
        f"and declares an annual income of "
        f"{_lakh(customer.stated_income)}. {residency_line} The {_account_label(account)} "
        f"account is maintained for {purpose}.{foreign_line} Expected activity is in the range of "
        f"{_lakh(customer.stated_income / 12 * 1.5)} per month across roughly "
        f"{max(1, observed['count'] // 3)} transactions, mostly through digital channels. "
        f"Source of funds is stated as {_source_of_funds(customer)}. No adverse information "
        f"was noted at the time of verification."
    )


def _occupation_phrase(customer: Customer) -> str:
    """The occupation as a clause that reads correctly for every occupation, including the
    ones that are not jobs - "works as a retired, state government pension" does not."""
    occupation = customer.occupation.lower()
    if occupation.startswith("retired"):
        return "is retired and draws a state government pension,"
    article = "an" if occupation[0] in "aeiou" else "a"
    if occupation in ("homemaker", "postgraduate student", "agricultural landholder"):
        return f"is {article} {occupation}"
    return f"works as {article} {occupation}"


#: Country names that take "the" in running prose.
_COUNTRIES_WITH_ARTICLE = frozenset({"United Arab Emirates", "United States", "United Kingdom"})


def _country_name(country: str) -> str:
    return f"the {country}" if country in _COUNTRIES_WITH_ARTICLE else country


def _account_label(account: Account) -> str:
    """How an officer names the account type in prose."""
    return "NRE" if account.account_type == AccountType.NRE else account.account_type.value.lower()


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
    cross_border: bool = False,
) -> str:
    """A profile that contradicts the account's behaviour in a specific, checkable way.

    Three shapes, all drawn from real KYC red flags rather than invented: an income that
    cannot support the throughput, a stated purpose that excludes the activity actually
    present, and an occupation that does not explain large cash handling. Each states a
    concrete figure, so Stage 3 has something to disagree with rather than a vague vibe.
    """
    is_nri = customer.residency_status == ResidencyStatus.NRI
    opener = rng.choice(OPENERS)
    shape = "cross_border" if cross_border else rng.choice(("income", "purpose", "occupation"))

    if shape == "cross_border":
        # The typology is foreign money the profile cannot explain, so the profile states,
        # specifically, that there will be none.
        return (
            f"{opener} {customer.name}, aged {_age(customer)}, {_occupation_phrase(customer)} "
            f"and declares an annual income of "
            f"{_lakh(customer.stated_income)} from {_source_of_funds(customer)}. The "
            f"{_account_label(account)} account is for domestic household use only. Customer "
            f"confirmed that no remittances to or from abroad are expected, that they hold no "
            f"overseas business interests or investments, and that no relative abroad sends "
            f"them money."
        )

    # The declared income is the one on the customer record. The contradiction lies in what
    # the account actually moves - many times that income - never in the profile disagreeing
    # with the record, which would let a plain number comparison flag every planted case
    # without any judgement at all.
    income = _lakh(customer.stated_income)

    if shape == "income":
        return (
            f"{opener} {customer.name}, aged {_age(customer)}, {_occupation_phrase(customer)} "
            f"with a declared annual income of {income}. Customer stated that the account "
            f"would see modest activity of around {_lakh(customer.stated_income / 24)} per "
            f"month, consistent with a single regular income credit and routine household "
            f"withdrawals. Source of funds is stated as {_source_of_funds(customer)} only. "
            f"Customer confirmed no other income streams, business interests or third-party "
            f"deposits."
        )

    if shape == "purpose":
        if is_nri:
            purpose = (
                "receiving monthly remittances from the customer's own overseas salary, with "
                "no third-party credits expected"
            )
        elif "retired" in customer.occupation.lower():
            purpose = (
                "receiving the monthly pension credit and paying household utility bills, "
                "with no business use whatsoever"
            )
        else:
            purpose = (
                "personal savings and paying household utility bills, with no business use "
                "whatsoever"
            )
        return (
            f"{opener} {customer.name}, aged {_age(customer)}, described the "
            f"{_account_label(account)} account as being opened solely for "
            f"{purpose}. Customer declares an annual income of {income} "
            f"from {_source_of_funds(customer)} and stated that no cash deposits would be "
            f"made into the account at any point. Customer declined to nominate any "
            f"authorised signatory and confirmed the account is operated personally."
        )

    return (
        f"{opener} {customer.name}, aged {_age(customer)}, {_occupation_phrase(customer)} "
        f"and declares an annual income of {income}. Customer explained that the account is "
        f"used for personal purposes only and that all receipts come from that single "
        f"source. Customer stated they have no involvement in trading, commission work or "
        f"handling of cash on behalf of others, and that they do not deal with overseas "
        f"counterparties. Address and employment details were verified against the "
        f"officially valid document on file."
    )


def generate(
    rng: random.Random,
    customers: list[Customer],
    accounts: list[Account],
    transactions: list[Transaction],
    counterparties: list[Counterparty],
    plan,
) -> list[KYCProfile]:
    """One narrative per customer, matched or deliberately mismatched per the planting plan."""
    by_account: dict[str, list[Transaction]] = defaultdict(list)
    for txn in transactions:
        by_account[txn.account_id].append(txn)
    counterparty_by_id = {c.counterparty_id: c for c in counterparties}

    account_by_customer = {a.customer_id: a for a in accounts}

    profiles: list[KYCProfile] = []
    for customer in customers:
        account = account_by_customer[customer.customer_id]
        observed = _observed(by_account.get(account.account_id, []))

        if account.account_id in plan.kyc_mismatch:
            narrative = _mismatch_narrative(
                rng,
                customer,
                account,
                observed,
                cross_border=account.account_id in plan.cross_border_mismatch,
            )
        else:
            swift = [
                (t, counterparty_by_id[t.counterparty_id])
                for t in by_account.get(account.account_id, [])
                if t.type == TransactionType.SWIFT
            ]
            foreign_line = _foreign_line(
                plan.cohort(account.account_id),
                customer.residency_status == ResidencyStatus.NRI,
                account.account_id in plan.round_trip_solo,
                swift,
            )
            narrative = _clean_narrative(rng, customer, account, observed, foreign_line)

        profiles.append(KYCProfile(customer_id=customer.customer_id, narrative=narrative))

    return profiles
