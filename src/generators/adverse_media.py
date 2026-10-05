"""Generator 4 of 9 - the adverse-media and sanctions corpus.

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

The corpus is a fixed external list. Its background names are screened against the founding
population (config.CORPUS_SCREENED_BLOCKS) and every entry draws from a stream of its own, so
it never changes when a block is added; a later block instead keeps its own customer and
counterparty names clear of the corpus (clear_later_block_names).
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from difflib import SequenceMatcher

from src.generators import config as cfg
from src.generators.names import WESTERN_FIRST_NAMES, indian_name
from src.generators.streams import Streams
from src.schemas import AdverseMediaEntry, Counterparty, Customer, SeverityTier, Transaction

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


def _founding(entity_id: str) -> bool:
    """Whether a customer or counterparty belongs to the population the corpus screens."""
    if entity_id.startswith(("CUST-", "ACC-")):
        return cfg.block_of(entity_id) in cfg.CORPUS_SCREENED_BLOCKS
    return not entity_id.startswith("CPM-")  # the mule block's own counterparties


def generate(
    streams: Streams,
    customers: list[Customer],
    accounts,
    plan,
    counterparties: list[Counterparty],
) -> list[AdverseMediaEntry]:
    """Build the 160-entry corpus.

    Linked entries come first so that every planted adverse-media account is guaranteed a
    matching record; decoys and background noise fill the rest.
    """
    customer_by_account = {a.account_id: a.customer_id for a in accounts}
    customers_by_id = {c.customer_id: c for c in customers}
    # Stage 4 screens counterparties as well as account holders, so a background name must
    # stay clear of both - a counterparty colliding with a corpus entry would hand every
    # account that pays it a hit nobody planted.
    screened_names = [c.name for c in customers if _founding(c.customer_id)] + [
        c.name for c in counterparties if _founding(c.counterparty_id)
    ]

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
        rng = streams.rng("adverse_media", "linked", account_id)
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
    extras = streams.rng("adverse_media", "extras").sample(
        linked_accounts, k=min(extra_needed, len(linked_accounts))
    )
    for account_id in extras:
        rng = streams.rng("adverse_media", "extra", account_id)
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
        rng = streams.rng("adverse_media", "decoy", account_id)
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
    # Background corpus. Names are checked against every customer and counterparty before
    # being accepted: an accidental collision here would create an adverse-media hit nobody
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
    streams.rng("adverse_media", "background_tiers").shuffle(background_tiers)

    def safe_name(generator) -> str:
        for _ in range(40):
            candidate = generator()
            if all(_similar(candidate, existing) < 80 for existing in screened_names):
                return candidate
        return candidate

    for index in range(indian_background + foreign_background):
        rng = streams.rng("adverse_media", "background", index)
        is_foreign = index >= indian_background
        if is_foreign:
            name = safe_name(streams.faker("en_US", "adverse_media", "background", index).name)
        else:
            faker_in = streams.faker("en_IN", "adverse_media", "background", index)
            name = safe_name(lambda: indian_name(faker_in))
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


#: How close a later block's name may come to a corpus name. Stricter than Stage 4's 85
#: cutoff, and the same bar the background corpus is held to against the founding population.
CLEARANCE_SIMILARITY = 80


def clear_later_block_names(
    streams: Streams,
    entries: list[AdverseMediaEntry],
    customers: list[Customer],
    counterparties: list[Counterparty],
    transactions: list[Transaction],
) -> tuple[list[Customer], list[Counterparty], list[Transaction], int]:
    """Rename any later-block customer or counterparty whose name sits near a corpus entry.

    The corpus is fixed, so a block added after it has to stay out of its way: a mule-block
    payer who happened to share a sanctioned person's name would hand that account a hit
    nobody planted. A colliding name keeps its surname (so relatives still share one) and
    redraws the first word from the entity's own stream until it is clear. Transactions carry
    the counterparty's name too, so they are updated to match.

    Returns the updated lists and how many names changed.
    """
    corpus = [n.lower() for m in entries for n in [m.entity_name, *m.aliases]]

    def collides(name: str) -> bool:
        lowered = name.lower()
        for other in corpus:
            matcher = SequenceMatcher(None, lowered, other)
            if matcher.real_quick_ratio() * 100 >= CLEARANCE_SIMILARITY and (
                matcher.quick_ratio() * 100 >= CLEARANCE_SIMILARITY
                and matcher.ratio() * 100 >= CLEARANCE_SIMILARITY
            ):
                return True
        return False

    def cleared(entity_id: str, name: str, gender: str | None = None) -> str:
        attempt = 0
        while collides(name):
            attempt += 1
            faker = streams.faker("en_IN", "clearance", entity_id, attempt)
            if gender:
                name = indian_name(faker, gender)
                continue
            first = faker.first_name()
            if first in WESTERN_FIRST_NAMES:
                continue
            name = " ".join([first, *name.split()[1:]])
        return name

    renamed = 0
    new_customers = []
    for customer in customers:
        if not _founding(customer.customer_id):
            name = cleared(customer.customer_id, customer.name, customer.gender.value)
            if name != customer.name:
                customer = customer.model_copy(update={"name": name})
                renamed += 1
        new_customers.append(customer)

    new_names: dict[str, str] = {}
    new_counterparties = []
    for counterparty in counterparties:
        if not _founding(counterparty.counterparty_id):
            name = cleared(counterparty.counterparty_id, counterparty.name)
            if name != counterparty.name:
                counterparty = counterparty.model_copy(update={"name": name})
                new_names[counterparty.counterparty_id] = name
                renamed += 1
        new_counterparties.append(counterparty)

    new_transactions = [
        t.model_copy(update={"counterparty_name": new_names[t.counterparty_id]})
        if t.counterparty_id in new_names
        else t
        for t in transactions
    ]
    return new_customers, new_counterparties, new_transactions, renamed
