"""Generator 1 of 9 - customers and their accounts.

Produces the reference data Stages 0, 1, 3, 5 and 7 all read. Runs first because every
other generator keys off customer_id / account_id.

Two populations, 75/25: domestic customers living in India, and Non-Resident Indians. NRIs
are modelled as Indian nationals resident abroad rather than as foreign nationals - that is
what the term actually means, and it is why an NRI here carries an Indian name, a foreign
residential address and a passport as the primary identity document, while a domestic
customer carries an Aadhaar.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from faker import Faker

from src.generators import config as cfg
from src.generators.identifiers import (
    aadhaar_number,
    account_number,
    passport_number,
)
from src.schemas import Account, AccountType, Customer, IdDocumentType, ResidencyStatus

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


def gender_for(customer_id: str) -> str:
    """Derive a customer's gender deterministically from their id.

    Computed rather than stored, so that customers.py can pick a gendered name and
    id_documents.py can print the matching gender on the card without either one holding a
    field the approved schema does not have. A card reading "Aryan Maharaj / FEMALE" is
    exactly the kind of detail a reader notices immediately, so the two have to agree.

    Uses the numeric part of the id rather than a hash, because Python's string hashing is
    randomised per process and would break reproducibility across runs.
    """
    digits = int("".join(ch for ch in customer_id if ch.isdigit()) or 0)
    return "MALE" if digits % 2 == 0 else "FEMALE"


def _weighted_country(rng: random.Random) -> str:
    """Pick an NRI's country of residence, weighted by real diaspora size."""
    roll = rng.random()
    cumulative = 0.0
    for country, share in cfg.NRI_COUNTRIES:
        cumulative += share
        if roll <= cumulative:
            return country
    return cfg.NRI_COUNTRIES[-1][0]


def _foreign_address(faker_intl: Faker, rng: random.Random, country: str) -> str:
    """A residential address abroad, ending in the country of residence."""
    street = faker_intl.street_address().replace("\n", ", ")
    city = faker_intl.city()
    return f"{street}, {city}, {country}"


def _domestic_address(faker_in: Faker) -> str:
    """An Indian residential address, flattened to a single line with a PIN code."""
    return faker_in.address().replace("\n", ", ")


def generate(rng: random.Random) -> tuple[list[Customer], list[Account]]:
    """Build the customer and account population.

    The domestic block is emitted first and the NRI block second, so customer_id ordering is
    stable and a reader can tell the two populations apart at a glance in the JSON.
    """
    faker_in = Faker("en_IN")
    faker_intl = Faker("en_US")
    faker_in.seed_instance(cfg.SEED)
    faker_intl.seed_instance(cfg.SEED + 1)

    customers: list[Customer] = []
    accounts: list[Account] = []

    today = cfg.REFERENCE_DATE

    for index in range(cfg.NUM_CUSTOMERS):
        is_nri = index >= cfg.NUM_DOMESTIC
        customer_id = f"CUST-{index + 1:04d}"
        account_id = f"ACC-{index + 1:04d}"

        # Indian name for both populations: an NRI is an Indian national living abroad.
        # Gendered so the name agrees with the gender printed on the rendered Aadhaar.
        gender = gender_for(customer_id)
        name = faker_in.name_male() if gender == "MALE" else faker_in.name_female()
        surname = name.split()[-1]

        occupation, income_low, income_high, account_type = rng.choice(
            NRI_OCCUPATIONS if is_nri else OCCUPATIONS
        )
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
            address = _foreign_address(faker_intl, rng, country)
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
            address = _domestic_address(faker_in)
            id_document_type = IdDocumentType.AADHAAR
            id_number = aadhaar_number(rng)
            # Aadhaar does not expire, so the field stays empty rather than being faked.
            id_expiration_date = None

        customers.append(
            Customer(
                customer_id=customer_id,
                name=name,
                dob=dob,
                occupation=occupation,
                stated_income=stated_income,
                address=address,
                residency_status=ResidencyStatus.NRI if is_nri else ResidencyStatus.DOMESTIC,
                country=country,
                id_document_type=id_document_type,
                id_number=id_number,
                id_expiration_date=id_expiration_date,
            )
        )

        # Accounts open before the 90-day history window, so every account has a settled
        # baseline for Stage 2 to compare a burst against.
        opened_date = today - timedelta(days=rng.randint(cfg.HISTORY_DAYS + 60, 3650))
        accounts.append(
            Account(
                account_id=account_id,
                customer_id=customer_id,
                opened_date=opened_date,
                account_type=account_type,
            )
        )

    return customers, accounts


def account_numbers_for(accounts: list[Account], rng: random.Random) -> dict[str, str]:
    """Printable bank account numbers, used on rendered documents only.

    Kept out of the Account model deliberately: account_id is the pipeline's key, and giving
    the model a second identifier would invite stages to join on the wrong one.
    """
    return {account.account_id: account_number(rng) for account in accounts}
