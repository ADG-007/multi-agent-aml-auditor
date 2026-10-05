"""Generator 1 of 9 - customers and their accounts.

Produces the reference data Stages 0, 1, 3, 5 and 7 all read. Runs first because every
other generator keys off customer_id / account_id.

Population blocks, in customer_id order (config.POPULATION_BLOCKS):
  - base: 225 domestic customers living in India, then 75 Non-Resident Indians
  - cross_border: 44 domestic customers - planted cross-border typologies and their
    legitimate look-alikes
  - mule: 26 domestic customers - planted money-mule typologies and their legitimate
    look-alikes (Amendment 2)

Every customer draws from a random stream keyed by its own id (streams.py), and each block
shuffles its cohorts from a stream of its own, so appending a block never changes anyone
already generated. A new block brings its own occupation list for the same reason: adding an
occupation to OCCUPATIONS would change what every base customer draws from it.

NRIs are modelled as Indian nationals resident abroad rather than as foreign nationals - that
is what the term actually means, and it is why an NRI here carries an Indian name, a foreign
residential address and a passport as the primary identity document, while a domestic
customer carries an Aadhaar. Every NRI holds an NRE account: FEMA does not allow a
non-resident to keep an ordinary resident savings account.

Which cross-border cohort each account belongs to is ground truth, so it is returned to the
caller rather than written anywhere a pipeline stage could read it.
"""

from __future__ import annotations

import random
from datetime import timedelta

from src.generators import config as cfg
from src.generators.addresses import foreign_address, indian_address
from src.generators.names import indian_name
from src.generators.streams import Streams
from src.generators.identifiers import (
    aadhaar_number,
    account_number,
    passport_number,
)
from src.schemas import Account, AccountType, Customer, Gender, IdDocumentType, ResidencyStatus

#: Occupation paired with the annual-income band that occupation plausibly earns, in INR,
#: and the account type it would normally open. Stage 3 judges whether a customer's stated
#: profile is behaviourally plausible, so the income attached to an occupation has to be
#: defensible on its own - otherwise a genuine mismatch is indistinguishable from generator
#: noise. CURRENT accounts go to business occupations, SAVINGS to salaried and retired ones,
#: which is how the two account types are actually used.
OCCUPATIONS: tuple[tuple[str, int, int, AccountType], ...] = (
    ("Government school teacher", 300_000, 750_000, AccountType.SAVINGS),
    ("Staff nurse", 300_000, 650_000, AccountType.SAVINGS),
    ("Clerk, public sector bank", 350_000, 800_000, AccountType.SAVINGS),
    ("Junior engineer, state electricity board", 400_000, 900_000, AccountType.SAVINGS),
    ("Software engineer", 800_000, 3_000_000, AccountType.SAVINGS),
    ("Senior software architect", 2_000_000, 5_000_000, AccountType.SAVINGS),
    ("Chartered accountant", 1_200_000, 4_500_000, AccountType.CURRENT),
    ("Practising advocate", 800_000, 3_500_000, AccountType.CURRENT),
    ("Consulting physician", 1_500_000, 5_000_000, AccountType.CURRENT),
    ("Kirana store proprietor", 350_000, 1_100_000, AccountType.CURRENT),
    ("Textile trader", 600_000, 2_400_000, AccountType.CURRENT),
    ("Jewellery shop proprietor", 900_000, 4_000_000, AccountType.CURRENT),
    ("Transport contractor", 700_000, 2_800_000, AccountType.CURRENT),
    ("Construction subcontractor", 500_000, 2_000_000, AccountType.CURRENT),
    ("Real estate agent", 600_000, 3_000_000, AccountType.CURRENT),
    ("Agricultural landholder", 300_000, 1_200_000, AccountType.SAVINGS),
    ("Retired, state government pension", 300_000, 600_000, AccountType.SAVINGS),
    ("Homemaker", 300_000, 500_000, AccountType.SAVINGS),
    ("Postgraduate student", 300_000, 450_000, AccountType.SAVINGS),
    ("Freelance graphic designer", 400_000, 1_400_000, AccountType.SAVINGS),
)

#: Occupations an NRI plausibly holds abroad, with income bands reflecting overseas earnings
#: converted to INR. Gulf-corridor blue-collar work and Western professional work are both
#: represented because the real diaspora is both, and a corpus of only high earners would
#: make every large remittance look normal.
NRI_OCCUPATIONS: tuple[tuple[str, int, int, AccountType], ...] = (
    ("Site supervisor, construction (Gulf)", 600_000, 1_600_000, AccountType.SAVINGS),
    ("Heavy vehicle driver (Gulf)", 400_000, 900_000, AccountType.SAVINGS),
    ("Registered nurse (overseas)", 900_000, 2_800_000, AccountType.SAVINGS),
    ("Software engineer (overseas)", 2_500_000, 5_000_000, AccountType.SAVINGS),
    ("Hotel operations manager (overseas)", 800_000, 2_200_000, AccountType.SAVINGS),
    ("Restaurant proprietor (overseas)", 1_000_000, 4_000_000, AccountType.CURRENT),
    ("Financial analyst (overseas)", 2_000_000, 5_000_000, AccountType.SAVINGS),
    ("Import-export trader (overseas)", 1_200_000, 5_000_000, AccountType.CURRENT),
    ("Postgraduate student (overseas)", 300_000, 500_000, AccountType.SAVINGS),
    ("Retail supervisor (overseas)", 500_000, 1_200_000, AccountType.SAVINGS),
)


#: Occupations each cross-border cohort draws from. The legitimate look-alikes need an
#: occupation that makes their foreign activity make sense - a parent paying overseas
#: university fees earns enough to, a household receiving family money is a pensioner or a
#: homemaker. The unexplained-inward cohort is deliberately modest earners, because that
#: typology is foreign money arriving that the stated profile cannot account for.
COHORT_OCCUPATIONS: dict[str, tuple[str, ...] | None] = {
    "round_trip": None,  # any occupation - round-tripping is not tied to a profession
    "unexplained_inward": (
        "Postgraduate student", "Homemaker", "Government school teacher", "Staff nurse",
        "Clerk, public sector bank", "Freelance graphic designer",
    ),
    "family_inward": (
        "Retired, state government pension", "Homemaker", "Agricultural landholder",
        "Government school teacher",
    ),
    "education_outward": (
        "Software engineer", "Senior software architect", "Chartered accountant",
        "Consulting physician", "Practising advocate",
    ),
    "lrs_outward": (
        "Senior software architect", "Chartered accountant", "Consulting physician",
        "Jewellery shop proprietor", "Software engineer",
    ),
}

#: Occupations for the mule block, kept apart from OCCUPATIONS so the base draws are
#: undisturbed. Mules are recruited from people with modest, irregular incomes - students,
#: gig workers, homemakers - which is who published typologies describe renting out an
#: account. The merchants run the kind of small business that takes customer payments on a
#: personal UPI ID linked to a savings account, which is exactly why they resemble a mule.
MULE_BLOCK_OCCUPATIONS: dict[str, tuple[tuple[str, int, int, AccountType], ...]] = {
    "mule": (
        ("Postgraduate student", 300_000, 450_000, AccountType.SAVINGS),
        ("Delivery partner, app-based platform", 300_000, 420_000, AccountType.SAVINGS),
        ("Homemaker", 300_000, 500_000, AccountType.SAVINGS),
        ("Retail sales assistant", 300_000, 480_000, AccountType.SAVINGS),
        ("Freelance graphic designer", 400_000, 900_000, AccountType.SAVINGS),
    ),
    "merchant": (
        ("Proprietor, retail fuel outlet", 600_000, 1_200_000, AccountType.SAVINGS),
        ("Proprietor, roadside mobile and electronics shop", 500_000, 1_000_000, AccountType.SAVINGS),
    ),
    "life_event": tuple(
        o for o in OCCUPATIONS
        if o[0] in (
            "Government school teacher", "Staff nurse", "Clerk, public sector bank",
            "Junior engineer, state electricity board", "Software engineer",
        )
    ),
}

#: Which occupation list each mule-block cohort draws from.
MULE_COHORT_OCCUPATIONS = {
    "network_mule": "mule",
    "solo_mule": "mule",
    "complicit_merchant": "merchant",
    "merchant_lookalike": "merchant",
    "life_event": "life_event",
}


def _weighted_country(rng: random.Random) -> str:
    """Pick an NRI's country of residence, weighted by real diaspora size."""
    roll = rng.random()
    cumulative = 0.0
    for country, share in cfg.NRI_COUNTRIES:
        cumulative += share
        if roll <= cumulative:
            return country
    return cfg.NRI_COUNTRIES[-1][0]


#: Each block's cohorts, in the order they are laid out before shuffling.
BLOCK_COHORTS: dict[str, tuple[tuple[str, int], ...]] = {
    "cross_border": cfg.CROSS_BORDER_COHORTS,
    "mule": cfg.MULE_BLOCK_COHORTS,
}


def _occupation_options(is_nri: bool, cohort: str | None):
    if is_nri:
        return NRI_OCCUPATIONS
    if cohort in MULE_COHORT_OCCUPATIONS:
        return MULE_BLOCK_OCCUPATIONS[MULE_COHORT_OCCUPATIONS[cohort]]
    if cohort and COHORT_OCCUPATIONS[cohort]:
        return tuple(o for o in OCCUPATIONS if o[0] in COHORT_OCCUPATIONS[cohort])
    return OCCUPATIONS


def _opened_date(rng: random.Random, cohort: str | None):
    """When the account was opened.

    Accounts open before the 90-day history window, so every account has a settled baseline
    for Stage 2 to compare a burst against. Mules are the exception that proves the rule:
    published red flags single out accounts opened within the last year, or long-dormant
    ones suddenly reactivated - a rented account is usually one of the two.
    """
    today = cfg.REFERENCE_DATE
    if cohort in ("network_mule", "solo_mule"):
        if rng.random() < 0.3:
            return today - timedelta(days=rng.randint(1500, 3000))  # dormant, reactivated
        return today - timedelta(days=rng.randint(cfg.HISTORY_DAYS + 5, 330))
    return today - timedelta(days=rng.randint(cfg.HISTORY_DAYS + 60, 3650))


def generate(
    streams: Streams,
    blocks: tuple[str, ...] | None = None,
) -> tuple[list[Customer], list[Account], dict[str, str]]:
    """Build the customer and account population.

    Returns the customers, their accounts, and the cohort of each account in a cohort block
    (ground truth - see the module docstring). `blocks` restricts generation to the named
    population blocks; it exists so the isolation guarantee can be checked by generating
    with and without a block and comparing.
    """
    included = set(blocks) if blocks else {name for name, _ in cfg.POPULATION_BLOCKS}

    customers: list[Customer] = []
    accounts: list[Account] = []
    cohorts: dict[str, str] = {}

    today = cfg.REFERENCE_DATE

    index = 0
    for block, block_size in cfg.POPULATION_BLOCKS:
        block_start = index
        index += block_size
        if block not in included:
            continue

        # Cohorts are shuffled across their block rather than laid out in order, so a
        # customer's position in the file says nothing about which typology it carries.
        cohort_labels: list[str] = []
        if block in BLOCK_COHORTS:
            cohort_labels = [label for label, count in BLOCK_COHORTS[block] for _ in range(count)]
            streams.rng("customers", "cohorts", block).shuffle(cohort_labels)

        for position in range(block_size):
            customer, account = _customer(
                streams,
                block_start + position,
                is_nri=block == "base" and position >= cfg.NUM_BASE_DOMESTIC,
                cohort=cohort_labels[position] if cohort_labels else None,
            )
            customers.append(customer)
            accounts.append(account)
            if cohort_labels:
                cohorts[account.account_id] = cohort_labels[position]

    return customers, accounts, cohorts


def _customer(
    streams: Streams,
    index: int,
    is_nri: bool,
    cohort: str | None,
) -> tuple[Customer, Account]:
    """One customer and their account, drawn from that customer's own stream."""
    customer_id = f"CUST-{index + 1:04d}"
    account_id = f"ACC-{index + 1:04d}"
    rng = streams.rng("customer", customer_id)
    today = cfg.REFERENCE_DATE

    # Indian name for both populations: an NRI is an Indian national living abroad. Gender
    # is drawn first so the name can agree with it - a card reading "Aryan Maharaj / FEMALE"
    # is exactly the kind of detail a reader notices.
    gender = Gender.MALE if rng.random() < 0.5 else Gender.FEMALE
    name = indian_name(streams.faker("en_IN", "customer", customer_id), gender.value)

    occupation, income_low, income_high, account_type = rng.choice(
        _occupation_options(is_nri, cohort)
    )
    if is_nri:
        account_type = AccountType.NRE
    stated_income = rng.randrange(
        max(income_low, cfg.INCOME_MIN),
        min(income_high, cfg.INCOME_MAX) + 1,
        10_000,
    )

    # Students skew young, retirees old; everyone else spans normal working age.
    if "student" in occupation.lower():
        age = rng.randint(21, 28)
    elif "retired" in occupation.lower():
        age = rng.randint(60, 78)
    else:
        age = rng.randint(24, 58)
    dob = today - timedelta(days=age * 365 + rng.randint(0, 364))

    if is_nri:
        country = _weighted_country(rng)
        address = foreign_address(rng, country)
        id_document_type = IdDocumentType.PASSPORT
        id_number = passport_number(rng)
        # Indian passports are issued with ten years' validity. A small slice is left
        # already expired so Stage 1's expiry check has something real to catch.
        if rng.random() < 0.08:
            id_expiration_date = today - timedelta(days=rng.randint(30, 900))
        else:
            id_expiration_date = today + timedelta(days=rng.randint(200, 3650))
    else:
        country = "India"
        address = indian_address(rng)
        id_document_type = IdDocumentType.AADHAAR
        id_number = aadhaar_number(rng)
        # Aadhaar does not expire, so the field stays empty rather than being faked.
        id_expiration_date = None

    customer = Customer(
        customer_id=customer_id,
        name=name,
        dob=dob,
        gender=gender,
        occupation=occupation,
        stated_income=stated_income,
        address=address,
        residency_status=ResidencyStatus.NRI if is_nri else ResidencyStatus.DOMESTIC,
        country=country,
        id_document_type=id_document_type,
        id_number=id_number,
        id_expiration_date=id_expiration_date,
    )
    account = Account(
        account_id=account_id,
        customer_id=customer_id,
        opened_date=_opened_date(rng, cohort),
        account_type=account_type,
    )
    return customer, account


def account_numbers_for(accounts: list[Account], rng: random.Random) -> dict[str, str]:
    """Printable bank account numbers, used on rendered documents only.

    Kept out of the Account model deliberately: account_id is the pipeline's key, and giving
    the model a second identifier would invite stages to join on the wrong one.
    """
    return {account.account_id: account_number(rng) for account in accounts}
