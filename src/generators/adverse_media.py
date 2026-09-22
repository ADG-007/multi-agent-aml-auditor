"""Generator 6 of 9 - the adverse-media and sanctions corpus.

The primary input to Stage 4, and a cheap pre-filter input to Stage 0.

Modelled on the UN Security Council Consolidated List as India gives it effect through UAPA
Section 51A notifications: each entry carries the entity name, a set of aliases, a date of
birth where the subject is an individual, a listing date and a short narrative summary. The
alias list is functional rather than decorative - alias matching is precisely what real
sanctions screening uses to stop a subject slipping through under a variant spelling, and
Stage 4's fuzzy matcher is meant to exercise it.

Three tiers, and the distinction between them carries the entire false-positive test:
  CONFIRMED_SANCTIONS  a designated party; a legal certainty
  ADVERSE_NEWS         credible reporting of wrongdoing; real but not adjudicated
  NAME_SIMILARITY_ONLY a name collision and nothing more; the decoy

Phase 6 screens at a similarity of 85 for the first two tiers and a stricter 92 for the
third, so the decoys here are built to sit just inside that stricter band - close enough to
register as a hit, far enough to be discriminable.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from difflib import SequenceMatcher

from faker import Faker

from src.generators import config as cfg
from src.schemas import AdverseMediaEntry, Customer, SeverityTier

#: Narrative shapes borrowed from how designations and enforcement reporting are actually
#: written - a designation states the basis, a news item states the allegation and source.
SANCTIONS_DESCRIPTIONS = (
    "Designated under Section 51A of the Unlawful Activities (Prevention) Act following "
    "inclusion in the UN Security Council Consolidated List. Funds and financial assets "
    "held by or on behalf of this party are subject to freezing.",
    "Listed pursuant to a Ministry of Home Affairs notification giving effect to a UNSC "
    "1267 Committee designation. Reporting entities are required to freeze assets and "
    "report any attempted transaction without delay.",
    "Subject of a targeted financial sanction for involvement in the financing of "
    "proscribed activity. All dealings are prohibited and any match must be reported to "
    "the Financial Intelligence Unit.",
    "Named in a designation order for acting on behalf of, or at the direction of, an "
    "already-listed entity. Asset freezing obligations apply to accounts held directly or "
    "through associates.",
)

ADVERSE_NEWS_DESCRIPTIONS = (
    "Reported in national financial press as being under investigation by the Directorate "
    "of Enforcement in connection with alleged irregular fund movement through shell "
    "entities. No charges have been framed at the time of reporting.",
    "Named in an income tax survey report covering undisclosed cash receipts routed through "
    "multiple current accounts. Matter reported as under assessment.",
    "Subject of press reporting alleging diversion of trade advances to unrelated parties. "
    "The party has publicly denied the allegation.",
    "Referenced in a regulatory enforcement order for accepting deposits without "
    "authorisation. Penalty proceedings reported as pending.",
    "Reported as a person of interest in an ongoing inquiry into over-invoiced import "
    "consignments and associated outward remittances.",
    "Named in investigative reporting on a network of entities sharing common registered "
    "addresses and directors, used to layer cash receipts.",
    "Reported by a state police economic offences wing as under inquiry for a deposit "
    "collection scheme. Proceedings reported as at an early stage.",
)

NAME_SIMILARITY_DESCRIPTIONS = (
    "Entry retained for screening completeness only. The record originates from an "
    "unverified aggregated watchlist feed and carries no corroborating detail beyond a "
    "name. Not to be treated as adverse information on its own.",
    "Low-confidence record sourced from a bulk name feed with no date of birth, "
    "identification number or jurisdiction attached. Retained to avoid silent suppression, "
    "not as evidence.",
    "Name-only record with no supporting narrative, no issuing authority and no reference "
    "to any proceeding. Requires independent corroboration before any action.",
)

#: Alias spelling variants that mirror how Indian names genuinely vary in transliteration.
TRANSLITERATION_SWAPS = (
    ("aa", "a"),
    ("ee", "i"),
    ("oo", "u"),
    ("ksh", "x"),
    ("th", "t"),
    ("sh", "s"),
    ("v", "w"),
    ("y", "i"),
)


def _similar(left: str, right: str) -> float:
    """Cheap similarity proxy used at generation time only.

    Stage 4 uses rapidfuzz token-sort-ratio; that package is a Phase 6 dependency and is not
    needed here. difflib is close enough to keep a randomly generated background name from
    accidentally colliding with a real customer, which would silently create an unplanted
    adverse-media hit and corrupt the ground-truth labels.
    """
    return SequenceMatcher(None, left.lower(), right.lower()).ratio() * 100


def _aliases(rng: random.Random, name: str, count: int) -> list[str]:
    """Plausible transliteration variants of a name."""
    aliases: list[str] = []
    for _ in range(count):
        variant = name
        original, replacement = rng.choice(TRANSLITERATION_SWAPS)
        if original in variant.lower():
            index = variant.lower().index(original)
            variant = variant[:index] + replacement + variant[index + len(original) :]
        else:
            parts = variant.split()
            if len(parts) > 1:
                variant = " ".join(reversed(parts))
        if variant != name and variant not in aliases:
            aliases.append(variant)
    return aliases


def _near_collision(rng: random.Random, name: str) -> str:
    """A name close enough to trip a strict fuzzy matcher but not the same name.

    A doubled letter or a single-character substitution inside a name of this length lands
    in the low-to-mid nineties on a token-sort ratio, which is exactly the band Stage 4's
    NAME_SIMILARITY_ONLY cutoff is meant to sit inside.
    """
    parts = name.split()
    target_index = rng.randrange(len(parts))
    word = parts[target_index]
    if len(word) < 3:
        return name + "a"

    position = rng.randrange(1, len(word) - 1)
    if rng.random() < 0.5:
        word = word[:position] + word[position] + word[position:]  # double a letter
    else:
        word = word[:position] + rng.choice("aeiou") + word[position + 1 :]  # swap a vowel
    parts[target_index] = word
    return " ".join(parts)


def _dob_string(rng: random.Random, reference: date | None = None) -> str:
    """Listings quote a date of birth, and often only a year."""
    if reference is not None and rng.random() < 0.6:
        return reference.isoformat()
    if rng.random() < 0.25:
        return str(rng.randint(1955, 1995))
    born = date(rng.randint(1955, 1995), rng.randint(1, 12), rng.randint(1, 28))
    return born.isoformat()


def generate(
    rng: random.Random,
    customers: list[Customer],
    accounts,
    plan,
) -> list[AdverseMediaEntry]:
    """Build the 160-entry corpus.

    Linked entries come first so that every planted adverse-media account is guaranteed a
    matching record; decoys and background noise fill the rest.
    """
    faker_in = Faker("en_IN")
    faker_intl = Faker("en_US")
    faker_in.seed_instance(cfg.SEED + 4)
    faker_intl.seed_instance(cfg.SEED + 5)

    customer_by_account = {a.account_id: a.customer_id for a in accounts}
    customers_by_id = {c.customer_id: c for c in customers}
    all_customer_names = [c.name for c in customers]

    entries: list[AdverseMediaEntry] = []
    serial = 0

    def next_id() -> str:
        nonlocal serial
        serial += 1
        return f"AM-{serial:04d}"

    today = cfg.REFERENCE_DATE

    # ---------------------------------------------------------------------------------
    # Linked entries: one per planted actor, plus extra records for a subset of them.
    # Real subjects rarely appear exactly once in a corpus, and a second entry under an
    # alias is what makes Stage 4's alias handling worth having.
    # ---------------------------------------------------------------------------------
    real_hits = {
        account_id: tier
        for account_id, tier in plan.adverse_media.items()
        if tier != SeverityTier.NAME_SIMILARITY_ONLY
    }

    linked_accounts = sorted(real_hits)
    for account_id in linked_accounts:
        tier = real_hits[account_id]
        customer = customers_by_id[customer_by_account[account_id]]
        descriptions = (
            SANCTIONS_DESCRIPTIONS
            if tier == SeverityTier.CONFIRMED_SANCTIONS
            else ADVERSE_NEWS_DESCRIPTIONS
        )
        entries.append(
            AdverseMediaEntry(
                entry_id=next_id(),
                entity_name=customer.name,
                severity_tier=tier,
                description=rng.choice(descriptions),
                linked_actor_id=customer.customer_id,
                aliases=_aliases(rng, customer.name, rng.randint(1, 3)),
                date_of_birth=_dob_string(rng, customer.dob),
                listing_date=today - timedelta(days=rng.randint(60, 2200)),
            )
        )

    # Extra records for a subset, to reach the 32 linked entries the plan calls for.
    extra_needed = cfg.NUM_LINKED_ENTRIES - len(entries)
    for account_id in rng.sample(linked_accounts, k=min(extra_needed, len(linked_accounts))):
        tier = real_hits[account_id]
        customer = customers_by_id[customer_by_account[account_id]]
        alias_name = _aliases(rng, customer.name, 1)
        entries.append(
            AdverseMediaEntry(
                entry_id=next_id(),
                entity_name=alias_name[0] if alias_name else customer.name,
                severity_tier=SeverityTier.ADVERSE_NEWS,
                description=rng.choice(ADVERSE_NEWS_DESCRIPTIONS),
                linked_actor_id=customer.customer_id,
                aliases=[customer.name],
                date_of_birth=_dob_string(rng, customer.dob),
                listing_date=today - timedelta(days=rng.randint(30, 1500)),
            )
        )

    # ---------------------------------------------------------------------------------
    # Decoys: a near-collision with a clean customer's name, and nothing else. No linked
    # actor, no supporting detail. These are what the false-positive rate is measured on.
    # ---------------------------------------------------------------------------------
    for account_id in sorted(plan.name_similarity_decoys):
        customer = customers_by_id[customer_by_account[account_id]]
        entries.append(
            AdverseMediaEntry(
                entry_id=next_id(),
                entity_name=_near_collision(rng, customer.name),
                severity_tier=SeverityTier.NAME_SIMILARITY_ONLY,
                description=rng.choice(NAME_SIMILARITY_DESCRIPTIONS),
                linked_actor_id=None,
                aliases=[],
                date_of_birth=None,
                listing_date=today - timedelta(days=rng.randint(30, 1800)),
            )
        )

    # ---------------------------------------------------------------------------------
    # Background corpus. Names are checked against every real customer before being
    # accepted: an accidental collision here would create an adverse-media hit nobody
    # planted, and the ground-truth labels would silently stop matching the data.
    # ---------------------------------------------------------------------------------
    indian_so_far = len(entries)  # every entry above uses an Indian customer name
    indian_background = cfg.NUM_INDIAN_ENTITY_ENTRIES - indian_so_far
    foreign_background = cfg.NUM_FOREIGN_ENTITY_ENTRIES

    background_tiers = (
        [SeverityTier.CONFIRMED_SANCTIONS] * 18
        + [SeverityTier.ADVERSE_NEWS] * 60
        + [SeverityTier.NAME_SIMILARITY_ONLY] * (indian_background + foreign_background - 78)
    )
    rng.shuffle(background_tiers)

    def safe_name(generator) -> str:
        for _ in range(40):
            candidate = generator()
            if all(_similar(candidate, existing) < 80 for existing in all_customer_names):
                return candidate
        return candidate

    for index in range(indian_background + foreign_background):
        is_foreign = index >= indian_background
        name = safe_name(faker_intl.name if is_foreign else faker_in.name)
        tier = background_tiers[index]
        if tier == SeverityTier.CONFIRMED_SANCTIONS:
            description = rng.choice(SANCTIONS_DESCRIPTIONS)
        elif tier == SeverityTier.ADVERSE_NEWS:
            description = rng.choice(ADVERSE_NEWS_DESCRIPTIONS)
        else:
            description = rng.choice(NAME_SIMILARITY_DESCRIPTIONS)

        entries.append(
            AdverseMediaEntry(
                entry_id=next_id(),
                entity_name=name,
                severity_tier=tier,
                description=description,
                linked_actor_id=None,
                aliases=(
                    _aliases(rng, name, rng.randint(1, 2))
                    if tier != SeverityTier.NAME_SIMILARITY_ONLY
                    else []
                ),
                date_of_birth=(
                    _dob_string(rng) if tier != SeverityTier.NAME_SIMILARITY_ONLY else None
                ),
                listing_date=today - timedelta(days=rng.randint(30, 3000)),
            )
        )

    return entries
