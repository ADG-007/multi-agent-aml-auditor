"""Generator 3 of 9 - 90 days of transaction history, and the counterparty pool.

The primary input to Stage 2, and a reference input to Stages 0, 5 and 7.

Amounts are constrained per payment rail rather than drawn from one global range, because
the rails really are constrained in India: RTGS has a two-lakh floor, UPI a one-lakh per
transaction ceiling, IMPS a five-lakh ceiling, ATM withdrawals a daily cap in the tens of
thousands. Honouring those bounds costs nothing and means a reader cannot find a ninety-rupee
RTGS or a fifteen-lakh UPI payment anywhere in the data.

Every transaction carries a direction - CREDIT for money in, DEBIT for money out. Money that
crosses the border moves only as a SWIFT wire, with the RBI purpose code the customer
declared: an S-code on the way out, a P-code on the way in. Domestic rails never touch a
foreign counterparty.

Patterns planted here, all driven by the shared planting plan rather than decided locally:
  - structuring: a burst of deposits parked just under the ten-lakh CTR threshold
  - cross-border funnelling: international ring members repeatedly wiring one offshore cluster
  - round-tripping: money wired to one foreign company comes back from a second company
    registered at the same address
  - unexplained inward money: foreign wires a modest earner's profile cannot account for
and the legitimate look-alikes each one needs:
  - benign round-amount clustering, against structuring
  - NRIs' own remittances into their NRE accounts, family money sent home, university fees
    and LRS gifts or investments abroad, against the cross-border typologies
The mule block's flows - money-mule pass-through and its look-alikes - live in mules.py.

Every account draws from a stream keyed by its own id, and transaction ids are built from
each transaction's own posting time and account number (TXN-20260722-143512-0042) rather
than a bank-wide serial, so appending a block of customers changes no existing transaction
(streams.py).
"""

from __future__ import annotations

import random
from collections import defaultdict
from datetime import date, datetime, time, timedelta

from faker import Faker

from src.generators import config as cfg
from src.generators.addresses import foreign_business_address
from src.generators.identifiers import device_id, ip_address
from src.generators.names import indian_name
from src.generators.rings import FOREIGN_RING_CLUSTER_IDS, ROUND_TRIP_ENTITY_PAIRS
from src.generators.streams import Streams
from src.schemas import (
    Account,
    Counterparty,
    Customer,
    FraudRing,
    RingType,
    Transaction,
    TransactionDirection,
    TransactionType,
)

CREDIT = TransactionDirection.CREDIT
DEBIT = TransactionDirection.DEBIT

#: Per-rail amount bounds in INR, reflecting the real limits each rail operates under. SWIFT
#: is absent on purpose: it is never chosen for ordinary domestic activity, and its amounts
#: are set by the cross-border flow that uses it.
RAIL_BOUNDS: dict[TransactionType, tuple[int, int]] = {
    TransactionType.UPI: (500, 100_000),
    TransactionType.IMPS: (1_000, 500_000),
    TransactionType.NEFT: (1_000, 1_500_000),
    TransactionType.RTGS: (200_000, 2_500_000),
    TransactionType.CASH_DEPOSIT: (500, 1_200_000),
    TransactionType.CASH_WITHDRAWAL: (500, 800_000),
    TransactionType.CHEQUE: (1_000, 1_500_000),
    TransactionType.ATM: (500, 25_000),
    TransactionType.POS: (500, 200_000),
}

#: How often each rail is used day to day. UPI dominates Indian retail payments by volume by
#: a very wide margin; RTGS is rare because it is a high-value rail.
RAIL_WEIGHTS: dict[TransactionType, float] = {
    TransactionType.UPI: 0.42,
    TransactionType.ATM: 0.11,
    TransactionType.POS: 0.11,
    TransactionType.IMPS: 0.10,
    TransactionType.NEFT: 0.09,
    TransactionType.CASH_WITHDRAWAL: 0.06,
    TransactionType.CASH_DEPOSIT: 0.06,
    TransactionType.CHEQUE: 0.03,
    TransactionType.RTGS: 0.02,
}

#: Rails whose direction is fixed by what they are: a cash deposit is always money in, a
#: withdrawal, an ATM draw or a card payment always money out. The transfer rails (UPI,
#: IMPS, NEFT, RTGS, cheque) carry both, and their direction is set by _assign_directions.
FIXED_DIRECTION: dict[TransactionType, TransactionDirection] = {
    TransactionType.CASH_DEPOSIT: CREDIT,
    TransactionType.CASH_WITHDRAWAL: DEBIT,
    TransactionType.ATM: DEBIT,
    TransactionType.POS: DEBIT,
}

#: Rails a customer can initiate from their own phone or computer - mobile or net banking.
#: A device and IP exist only on an outgoing transaction over one of these. Money coming in
#: was initiated by the sender, on the sender's device; a branch teller, an ATM, a shop's card
#: terminal or a paper cheque involves no customer device at all.
DIGITAL_RAILS = frozenset(
    {
        TransactionType.UPI,
        TransactionType.IMPS,
        TransactionType.NEFT,
        TransactionType.RTGS,
        TransactionType.SWIFT,
    }
)

#: Rails the planted structuring uses. Cash only (Amendment 3): structuring means splitting
#: cash to stay under the CTR, and no other domestic rail carries an amount-based report
#: (rulebook STR-002), so an in-band NEFT, RTGS or cheque payment avoids nothing. Kept as a
#: weighted table so the draw sequence below stays the same shape.
STRUCTURING_RAILS: tuple[tuple[TransactionType, float], ...] = (
    (TransactionType.CASH_DEPOSIT, 1.00),
)

#: Rails that only post during banking hours on weekdays. SWIFT is here because foreign
#: wires are processed by the bank's treasury desk in business hours, not in real time.
BRANCH_HOUR_RAILS = frozenset(
    {
        TransactionType.CASH_DEPOSIT,
        TransactionType.CASH_WITHDRAWAL,
        TransactionType.CHEQUE,
        TransactionType.SWIFT,
    }
)

#: The RBI purpose codes this dataset uses, with their official descriptions. S-codes are
#: outward remittances, P-codes inward. Checked against published RBI purpose-code lists.
OUTWARD_PURPOSE_CODES: dict[str, str] = {
    "S0001": "Indian investment abroad in equity capital (shares)",
    "S0305": "Travel for education (including fees, hostel expenses etc.)",
    "S1006": "Business and management consultancy and public relations services",
    "S1302": "Personal gifts and donations",
}
INWARD_PURPOSE_CODES: dict[str, str] = {
    "P0012": "Loans from non-residents to India",
    "P0014": "Non-resident deposit receipts (FCNRB/NRERA etc.)",
    "P0802": "Software implementation/consultancy services",
    "P1006": "Business and management consultancy and public relations services",
    "P1301": "Family maintenance and savings",
    "P1302": "Personal gifts and donations",
}

#: Business suffixes that read as real Indian counterparties.
INDIAN_BUSINESS_SUFFIXES = (
    "Traders",
    "Enterprises",
    "& Sons",
    "Industries",
    "Agencies",
    "Textiles",
    "Exports",
    "Motors",
    "Distributors",
    "Constructions",
)

#: The legal-form suffix a company carries in each jurisdiction.
COMPANY_SUFFIX_BY_COUNTRY: dict[str, tuple[str, ...]] = {
    "United Arab Emirates": ("FZE", "FZ-LLC", "General Trading LLC"),
    "Hong Kong": ("Limited", "Trading Limited"),
    "Mauritius": ("Ltd", "Holdings Ltd"),
    "Singapore": ("Pte Ltd",),
    "Cyprus": ("Ltd", "Holdings Ltd"),
    "Seychelles": ("Ltd",),
    "Malaysia": ("Sdn Bhd",),
    "United States": ("LLC", "Inc"),
    "United Kingdom": ("Ltd",),
}

#: Remittance providers the diaspora actually uses to send money home, one or two per NRI
#: country. Gulf remittances overwhelmingly move through exchange houses; the legal-form
#: suffix follows each country's convention. Names are invented.
REMITTANCE_PROVIDERS: tuple[tuple[str, str], ...] = (
    ("Al Noor Exchange LLC", "United Arab Emirates"),
    ("Gulf Crescent Exchange LLC", "United Arab Emirates"),
    ("Meridian Remit Inc", "United States"),
    ("Al Waha Exchange Co.", "Saudi Arabia"),
    ("Thameside Money Transfer Ltd", "United Kingdom"),
    ("Straits Remittance Pte Ltd", "Singapore"),
    ("Maple Leaf Money Services Inc", "Canada"),
    ("Southern Cross Remit Pty Ltd", "Australia"),
    ("Pearl Exchange W.L.L.", "Qatar"),
)

#: Universities abroad that fee remittances go to. Invented names.
UNIVERSITIES: tuple[tuple[str, str], ...] = (
    ("Northbridge University", "United States"),
    ("Great Plains State University", "United States"),
    ("University of Westmoor", "United Kingdom"),
    ("St Aldhelm's University", "United Kingdom"),
    ("Lakeshore College of Technology", "Canada"),
    ("Pacific Coast University", "Australia"),
)

#: Overseas brokerages a resident invests through under the LRS. Invented names.
BROKERAGES: tuple[tuple[str, str], ...] = (
    ("Crescent Wealth Brokerage FZE", "United Arab Emirates"),
    ("Harbourline Capital Pte Ltd", "Singapore"),
    ("Meridian Securities LLC", "United States"),
)

#: Registered addresses shared by unrelated legitimate companies - the look-alike for a
#: round-trip pair. A Dubai business centre or a Singapore office tower hosts many unrelated
#: registrations, so two linked-looking foreign companies are a lead, not proof.
SHARED_ADDRESS_DECOYS: tuple[tuple[str, ...], ...] = (
    ("Al Noor Exchange LLC", "Gulf Crescent Exchange LLC", "Crescent Wealth Brokerage FZE"),
    ("Straits Remittance Pte Ltd", "Harbourline Capital Pte Ltd"),
)

#: Where the offshore cluster and the round-trip pairs are registered: the jurisdictions a
#: real scheme of each kind favours.
CLUSTER_JURISDICTIONS = ("United Arab Emirates",) * 3 + ("Hong Kong",) * 2
ROUND_TRIP_JURISDICTIONS = ("United Arab Emirates", "Mauritius", "Singapore")

#: Jurisdictions the unexplained foreign payers sit in.
UNEXPLAINED_PAYER_JURISDICTIONS = (
    "United Arab Emirates", "Hong Kong", "Singapore", "Cyprus", "Seychelles", "Malaysia",
)

#: Countries universities and relatives abroad are drawn from for the look-alikes.
EDUCATION_COUNTRIES = ("United States", "United Kingdom", "Canada", "Australia")


#: How far an account's annualised throughput may sit from the customer's declared income
#: before the profile stops being defensible. Money flows both in and out, so turnover above
#: income is entirely normal; an order of magnitude above it is not.
CLEAN_THROUGHPUT_MULTIPLE = (0.6, 2.5)

#: What a planted KYC mismatch looks like: an account moving many times what its holder says
#: they earn. Stage 3's judgement only means something if the gap is this unambiguous - a
#: profile off by twenty percent is noise, not a finding.
MISMATCH_THROUGHPUT_MULTIPLE = (8.0, 25.0)


def _rail_for_amount(rng: random.Random, amount: float) -> TransactionType:
    """Pick a domestic payment rail that can actually carry this amount.

    Amount is drawn first and the rail second, rather than the other way round, because the
    account's total turnover has to track the customer's declared income - and that only
    works if the amount is chosen against a throughput target. Picking the rail first would
    let its bounds dictate the total.
    """
    candidates = {
        rail: RAIL_WEIGHTS[rail]
        for rail, (low, high) in RAIL_BOUNDS.items()
        if low <= amount <= high
    }
    if not candidates:
        # Above every retail rail's ceiling: RTGS is the only real option.
        return TransactionType.RTGS
    total = sum(candidates.values())
    roll = rng.random() * total
    cumulative = 0.0
    for rail, weight in candidates.items():
        cumulative += weight
        if roll <= cumulative:
            return rail
    return next(iter(candidates))


def _avoid_structuring_band(rng: random.Random, amount: float) -> float:
    """Keep ordinary activity out of the sub-threshold band.

    Without this, a routine large transfer lands between nine and ten lakh often enough that
    the band fills with innocent traffic and stops distinguishing anything. Real banks do see
    incidental amounts there, but this is a labelled benchmark: an unplanted transaction in
    the band would be an unlabelled positive, and Phase 11 would score the pipeline against
    ground truth that does not describe the data.

    The deliberate false positives come from the decoy types instead, where they are labelled
    and measurable.
    """
    if cfg.STRUCTURING_BAND_LOW <= amount <= cfg.STRUCTURING_BAND_HIGH:
        if rng.random() < 0.5:
            return float(rng.randint(400_000, cfg.STRUCTURING_BAND_LOW - 1))
        return float(rng.randint(cfg.STRUCTURING_BAND_HIGH + 1, 2_000_000))
    return amount


def _in_band(amount: float) -> bool:
    return cfg.STRUCTURING_BAND_LOW <= amount <= cfg.STRUCTURING_BAND_HIGH


def _timestamp_for(rng: random.Random, day: date, txn_type: TransactionType) -> datetime:
    """Place a transaction at a plausible hour of its day."""
    if txn_type in BRANCH_HOUR_RAILS:
        # Banking hours. If the drawn day is a weekend, shift back to the Friday - a cash
        # deposit cannot post at a closed branch, and a SWIFT wire is not processed on one.
        while day.weekday() >= 5:
            day = day - timedelta(days=1)
        hour = rng.randint(10, 15)
    else:
        # Digital rails peak in the evening and never fully stop.
        hour = rng.choices(
            population=list(range(24)),
            weights=[1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 7, 6, 6, 7, 8, 9, 9, 8, 6, 4, 2],
            k=1,
        )[0]
    return datetime.combine(day, time(hour, rng.randint(0, 59), rng.randint(0, 59)))


def _company_name(faker_intl: Faker, rng: random.Random, country: str) -> str:
    suffixes = COMPANY_SUFFIX_BY_COUNTRY.get(country, ("Ltd",))
    return f"{faker_intl.last_name()} {rng.choice(suffixes)}"


def generate_counterparties(streams: Streams) -> list[Counterparty]:
    """The base pool every transaction draws its other side from.

    Foreign counterparties are purpose-built rather than a generic pool, because every
    foreign wire in the data belongs to a specific flow: the offshore cluster, a round-trip
    pair, a remittance provider, a university, a brokerage, a relative abroad, or an
    unexplained foreign payer. The reserved CPF block comes first, in the order rings.py
    expects, so both generators agree on those IDs without either importing the other's
    output.

    This list is fixed: a later population block brings counterparties of its own rather
    than drawing new ones here, so the base pool never changes.
    """
    rng = streams.rng("counterparties", "base")
    faker_in = Faker("en_IN")
    faker_intl = Faker("en_US")
    faker_in.seed_instance(streams.seed_for("counterparties", "base", "en_IN"))
    faker_intl.seed_instance(streams.seed_for("counterparties", "base", "en_US"))

    counterparties: list[Counterparty] = []

    def add_foreign(name: str, country: str, address: str | None) -> None:
        counterparties.append(
            Counterparty(
                counterparty_id=f"CPF-{sum(1 for c in counterparties if c.is_foreign) + 1:04d}",
                name=name,
                country=country,
                is_foreign=True,
                registered_address=address,
            )
        )

    # The offshore cluster the international rings funnel into. Companies in one
    # jurisdiction share a registered address - a single formation agent's office, which is
    # how shell companies set up together usually look.
    # Sorted, not set order: Python randomises string hashing per process, so iterating the
    # set directly drew these addresses in a different order from run to run.
    cluster_addresses = {
        country: foreign_business_address(rng, country) for country in sorted(set(CLUSTER_JURISDICTIONS))
    }
    for country in CLUSTER_JURISDICTIONS:
        add_foreign(_company_name(faker_intl, rng, country), country, cluster_addresses[country])

    # Round-trip pairs: the company the money is sent to, and the one it comes back from,
    # registered at the same premises.
    for country in ROUND_TRIP_JURISDICTIONS:
        address = foreign_business_address(rng, country)
        add_foreign(_company_name(faker_intl, rng, country), country, address)
        add_foreign(_company_name(faker_intl, rng, country), country, address)

    assert [c.counterparty_id for c in counterparties] == list(FOREIGN_RING_CLUSTER_IDS) + [
        cp for pair in ROUND_TRIP_ENTITY_PAIRS for cp in pair
    ], "reserved CPF ids out of step with rings.py"

    # Legitimate foreign institutions. Members of a shared-address decoy group get one
    # common registered address; everything else gets its own.
    shared_address_for: dict[str, str] = {}
    for group in SHARED_ADDRESS_DECOYS:
        country = next(c for n, c in REMITTANCE_PROVIDERS + BROKERAGES if n == group[0])
        address = foreign_business_address(rng, country)
        for name in group:
            shared_address_for[name] = address
    for name, country in REMITTANCE_PROVIDERS + UNIVERSITIES + BROKERAGES:
        add_foreign(name, country, shared_address_for.get(name) or foreign_business_address(rng, country))

    # Unexplained foreign payers: ordinary-looking trading and consultancy companies.
    for country in UNEXPLAINED_PAYER_JURISDICTIONS:
        add_foreign(_company_name(faker_intl, rng, country), country, foreign_business_address(rng, country))

    # Relatives abroad, for LRS gifts. Private individuals, so no registered address.
    for _ in range(cfg.NUM_CROSS_BORDER_CUSTOMERS // 4):
        add_foreign(indian_name(faker_in), rng.choice(EDUCATION_COUNTRIES + ("United Arab Emirates",)), None)

    for index in range(cfg.NUM_DOMESTIC_COUNTERPARTIES):
        # A mix of businesses and individuals, as a real statement would show.
        if rng.random() < 0.6:
            name = f"{faker_in.last_name()} {rng.choice(INDIAN_BUSINESS_SUFFIXES)}"
        else:
            name = indian_name(faker_in)
        counterparties.append(
            Counterparty(
                counterparty_id=f"CPD-{index + 1:04d}",
                name=name,
                country="India",
                is_foreign=False,
            )
        )

    return counterparties


def _account_identifiers(
    streams: Streams,
    accounts: list[Account],
    fraud_rings: list[FraudRing],
    coincidence_clusters: list[dict],
) -> dict[str, dict]:
    """Assign each account its own devices and IPs, plus any shared ones.

    An ordinary customer has one or two devices and a couple of IPs. A ring member also has
    the ring's shared handset, and uses it often enough for the shared edge to be visible in
    the graph without the account's own device disappearing. A mule network's handler runs
    the rented accounts' banking themselves, so there the shared handset dominates.
    """
    identifiers: dict[str, dict] = {}
    for account in accounts:
        rng = streams.rng("identifiers", account.account_id)
        identifiers[account.account_id] = {
            "own_devices": [device_id(rng) for _ in range(rng.randint(1, 2))],
            "own_ips": [ip_address(rng) for _ in range(rng.randint(1, 3))],
            "shared_device": None,
            "shared_ip": None,
            "shared_use_rate": 0.0,
        }

    for ring in fraud_rings:
        for account_id in ring.member_account_ids:
            entry = identifiers[account_id]
            rng = streams.rng("identifiers", account_id, "shared")
            entry["shared_device"] = ring.shared_device_id
            entry["shared_ip"] = ring.shared_ip_address
            entry["shared_use_rate"] = (
                rng.uniform(0.75, 0.95) if ring.ring_type == RingType.MULE else rng.uniform(0.35, 0.60)
            )

    for cluster in coincidence_clusters:
        for account_id in cluster["member_account_ids"]:
            entry = identifiers[account_id]
            rng = streams.rng("identifiers", account_id, "shared")
            entry["shared_device"] = cluster["shared_device_id"]
            entry["shared_ip"] = cluster["shared_ip_address"]
            # A household shares a handset heavily - the decoy has to be at least as strong
            # a graph signal as a ring, or the false-positive test is trivially easy.
            entry["shared_use_rate"] = rng.uniform(0.40, 0.65)

    return identifiers


def _pick_device_ip(rng: random.Random, entry: dict) -> tuple[str, str]:
    if entry["shared_device"] and rng.random() < entry["shared_use_rate"]:
        device = entry["shared_device"]
        ip = entry["shared_ip"] or rng.choice(entry["own_ips"])
    else:
        device = rng.choice(entry["own_devices"])
        ip = rng.choice(entry["own_ips"])
    return device, ip


def _assign_directions(pending: list[dict]) -> None:
    """Set CREDIT or DEBIT on an account's ordinary activity, in place.

    Cash deposits, withdrawals, ATM and card payments have a fixed direction. The transfer
    rails carry both, and are assigned so that money in and money out stay roughly level
    over the window - what an ordinary account looks like, where income arrives and is spent.
    Largest transfers are placed first, each onto whichever side is currently lighter, which
    tends to make the big transfers the credits (salary, client fees) and leaves the many
    small ones as spending, as on a real statement. No randomness is used, so this adds
    nothing to the shared random stream.
    """
    totals = {CREDIT: 0.0, DEBIT: 0.0}
    flexible: list[dict] = []
    for row in pending:
        fixed = FIXED_DIRECTION.get(row["type"])
        if fixed is None:
            flexible.append(row)
        else:
            row["direction"] = fixed
            totals[fixed] += row["amount"]
    for row in sorted(flexible, key=lambda r: -r["amount"]):
        side = CREDIT if totals[CREDIT] <= totals[DEBIT] else DEBIT
        row["direction"] = side
        totals[side] += row["amount"]


def _swift_amount(rng: random.Random, target: float) -> float:
    """A foreign-wire amount near `target`, kept out of the structuring band."""
    return _avoid_structuring_band(rng, float(round(max(10_000.0, target), 2)))


def ordinary_rows(
    rng: random.Random,
    row,
    count: int,
    mean_amount: float,
    domestic_cps: list[Counterparty],
    is_round_decoy: bool = False,
) -> list[dict]:
    """Everyday domestic activity spread across the 90 days, directions not yet assigned.

    Shared with mules.py, so every account's everyday life is drawn the same way whatever
    else is planted on top of it.
    """
    window_start = cfg.REFERENCE_DATE - timedelta(days=cfg.HISTORY_DAYS)
    rows: list[dict] = []
    for _ in range(count):
        # Transaction sizes are heavily right-skewed in real accounts - many small payments
        # and a few large ones - so a lognormal draw around the mean fits far better than a
        # uniform one.
        amount = mean_amount * rng.lognormvariate(-0.3, 0.75)
        amount = min(max(amount, 500.0), 2_500_000.0)

        if is_round_decoy and rng.random() < 0.55:
            # The benign round-amount decoy: a shopkeeper banking takings in neat figures.
            # Round, repetitive and entirely legitimate - and well below the structuring
            # band, which is the only thing separating it from a real case.
            amount = float(min(cfg.ROUND_AMOUNTS, key=lambda value: abs(value - amount)))
        elif rng.random() < 0.18:
            # Everyone transacts at round figures sometimes.
            amount = float(min(cfg.ROUND_AMOUNTS, key=lambda value: abs(value - amount)))

        amount = _avoid_structuring_band(rng, float(round(amount, 2)))
        txn_type = _rail_for_amount(rng, amount)
        day = window_start + timedelta(days=rng.randint(0, cfg.HISTORY_DAYS - 1))
        rows.append(row(amount, txn_type, day, rng.choice(domestic_cps)))
    return rows


def _transaction_id(txn: Transaction, clash: int) -> str:
    """TXN-<posting date>-<posting time>-<account number>, e.g. TXN-20260722-143512-0042.

    Built only from the transaction's own posting time and account, the way a real payment
    reference encodes its date and time (a UPI reference number opens with the year, day of
    year and hour), so ids still sort bank-wide in time order - and adding customers never
    renumbers anyone else's. `clash` numbers a second posting by one account in one second.
    """
    stamp = txn.timestamp.strftime("%Y%m%d-%H%M%S")
    suffix = f"-{clash}" if clash > 1 else ""
    return f"TXN-{stamp}-{txn.account_id.split('-')[1]}{suffix}"


def generate(
    streams: Streams,
    customers: list[Customer],
    accounts: list[Account],
    fraud_rings: list[FraudRing],
    plan,
) -> tuple[list[Transaction], list[Counterparty]]:
    """Build the full transaction history for every account.

    `plan` is the PlantingPlan from evidence_ledger; it is typed loosely here only to avoid
    a circular import between the two generator modules.
    """
    # Imported here: mules.py builds on this module's helpers.
    from src.generators import mules as mules_module

    counterparties = generate_counterparties(streams)
    by_name = {c.name: c for c in counterparties}
    by_id = {c.counterparty_id: c for c in counterparties}
    domestic_cps = [c for c in counterparties if not c.is_foreign]
    cluster_cps = [by_id[cp] for cp in FOREIGN_RING_CLUSTER_IDS]
    providers_by_country: dict[str, list[Counterparty]] = defaultdict(list)
    for name, country in REMITTANCE_PROVIDERS:
        providers_by_country[country].append(by_name[name])
    universities = [by_name[name] for name, _ in UNIVERSITIES]
    brokerages = [by_name[name] for name, _ in BROKERAGES]
    reserved = set(FOREIGN_RING_CLUSTER_IDS) | {cp for pair in ROUND_TRIP_ENTITY_PAIRS for cp in pair}
    institution_names = {n for n, _ in REMITTANCE_PROVIDERS + UNIVERSITIES + BROKERAGES}
    foreign_companies = [
        c for c in counterparties
        if c.is_foreign and c.counterparty_id not in reserved and c.name not in institution_names
    ]
    unexplained_payers = [c for c in foreign_companies if c.registered_address]
    relatives_abroad = [c for c in foreign_companies if not c.registered_address]

    identifiers = _account_identifiers(streams, accounts, fraud_rings, plan.coincidence_clusters)

    # The mule block's collectors, and the counterparties each of its accounts brings.
    collectors = {c.counterparty_id: c for c in mules_module.collector_counterparties(streams)}
    if any(cfg.block_of(a.account_id) == "mule" for a in accounts):
        counterparties.extend(collectors.values())
    ring_of = {m: r.ring_id for r in fraud_rings for m in r.member_account_ids}

    customer_by_id = {c.customer_id: c for c in customers}
    international_ring_members = {
        account_id
        for ring in fraud_rings
        if ring.ring_type.value == "INTERNATIONAL"
        for account_id in ring.member_account_ids
    }
    round_trip_pair_of = {
        account_id: (by_id[ring.foreign_counterparty_ids[0]], by_id[ring.foreign_counterparty_ids[1]])
        for ring in fraud_rings
        if ring.ring_type.value == "ROUND_TRIP"
        for account_id in ring.member_account_ids
    }

    window_end = cfg.REFERENCE_DATE
    window_start = window_end - timedelta(days=cfg.HISTORY_DAYS)

    transactions: list[Transaction] = []

    def day_at(offset: int) -> date:
        return window_start + timedelta(days=offset)

    for account in accounts:
        account_id = account.account_id
        rng = streams.rng("transactions", account_id)
        entry = identifiers[account_id]
        customer = customer_by_id[account.customer_id]
        income = customer.stated_income
        is_nri = customer.residency_status.value == "NRI"
        cohort = plan.cohort(account_id)

        # Rows are collected as dicts first because ordinary activity needs its directions
        # assigned as a group, once all of it exists.
        ordinary: list[dict] = []
        planted: list[dict] = []

        def row(
            amount: float,
            txn_type: TransactionType,
            when: date | datetime,
            counterparty: Counterparty,
            direction: TransactionDirection | None = None,
            purpose_code: str | None = None,
        ) -> dict:
            """One pending row. A bare date gets a plausible hour for its rail; a datetime
            is used as given, for flows whose timing is the point (a mule's sweep)."""
            device, ip = _pick_device_ip(rng, entry)
            return {
                "amount": amount,
                "type": txn_type,
                "timestamp": when if isinstance(when, datetime) else _timestamp_for(rng, when, txn_type),
                "device_id": device,
                "ip_address": ip,
                "counterparty": counterparty,
                "direction": direction,
                "purpose_code": purpose_code,
            }

        if cfg.block_of(account_id) == "mule":
            ordinary, planted, own_counterparties = mules_module.account_activity(
                streams, rng, account_id, customer, cohort, plan, collectors, domestic_cps, row,
                mules_module.burst_start_day(streams, account_id, ring_of.get(account_id)),
            )
            counterparties.extend(own_counterparties)
            _assign_directions(ordinary)
            _finish_account(rng, account_id, entry, ordinary + planted, transactions)
            continue

        baseline_count = rng.randint(cfg.TXNS_PER_ACCOUNT_MIN, cfg.TXNS_PER_ACCOUNT_MAX)
        is_round_decoy = account_id in plan.round_amount_decoys
        is_structuring = account_id in plan.structuring
        # A cross-border mismatch is carried by the foreign money itself, so its ordinary
        # activity stays at a clean level rather than being inflated as well.
        inflate = account_id in plan.kyc_mismatch and account_id not in plan.cross_border_mismatch

        # An account's turnover is anchored to what its holder says they earn. Without this
        # anchor a clean account could plausibly move twenty times its declared income, and
        # the KYC-mismatch channel would be indistinguishable from ordinary variation.
        low_mult, high_mult = MISMATCH_THROUGHPUT_MULTIPLE if inflate else CLEAN_THROUGHPUT_MULTIPLE
        annual_target = income * rng.uniform(low_mult, high_mult)
        window_target = annual_target * (cfg.HISTORY_DAYS / 365)
        mean_amount = max(600.0, window_target / baseline_count)

        # ---------------------------------------------------------------------------
        # Ordinary domestic activity across the full 90 days.
        # ---------------------------------------------------------------------------
        ordinary = ordinary_rows(rng, row, baseline_count, mean_amount, domestic_cps, is_round_decoy)
        _assign_directions(ordinary)

        # ---------------------------------------------------------------------------
        # Planted structuring: a cluster of deposits just under the CTR threshold, inside
        # a single 30-day window. Rule 3 aggregates integrally connected cash transactions
        # over a month, so a month is the window a real analyst would look at. Always money
        # in, always from domestic sources.
        # ---------------------------------------------------------------------------
        if is_structuring:
            burst_start = rng.randint(0, cfg.HISTORY_DAYS - cfg.STRUCTURING_WINDOW_DAYS)
            for _ in range(rng.randint(cfg.STRUCTURING_TXN_MIN, cfg.STRUCTURING_TXN_MAX)):
                roll = rng.random()
                cumulative = 0.0
                txn_type = TransactionType.CASH_DEPOSIT
                for candidate, weight in STRUCTURING_RAILS:
                    cumulative += weight
                    if roll <= cumulative:
                        txn_type = candidate
                        break
                # Just under ten lakh, never over - the entire point is staying below the
                # reporting ceiling.
                amount = float(rng.randint(cfg.STRUCTURING_BAND_LOW, cfg.STRUCTURING_BAND_HIGH - 1))
                day = day_at(burst_start + rng.randint(0, cfg.STRUCTURING_WINDOW_DAYS - 1))
                planted.append(row(amount, txn_type, day, rng.choice(domestic_cps), CREDIT))

        # ---------------------------------------------------------------------------
        # NRIs: overseas earnings wired into the NRE account through a remittance provider
        # in their country of residence - the ordinary cross-border traffic every other
        # cross-border signal has to stand out from.
        # ---------------------------------------------------------------------------
        if is_nri:
            providers = providers_by_country.get(customer.country) or [by_name[REMITTANCE_PROVIDERS[0][0]]]
            for month in range(rng.randint(2, 3)):
                amount = _swift_amount(rng, income / 12 * rng.uniform(0.5, 0.9))
                day = day_at(month * 30 + rng.randint(0, 27))
                planted.append(row(amount, TransactionType.SWIFT, day, rng.choice(providers), CREDIT, "P0014"))

        # ---------------------------------------------------------------------------
        # Cross-border funnelling: international ring members wire the same small offshore
        # cluster again and again. Stage 5 sees a shared foreign counterparty cluster, and
        # Stage 6 cites the cross-border clause category afterwards.
        # ---------------------------------------------------------------------------
        if account_id in international_ring_members:
            for _ in range(rng.randint(5, 12)):
                # Capped below the structuring band, which would otherwise plant an
                # unlabelled structuring signal on every international ring member.
                amount = float(rng.randint(150_000, 880_000))
                day = day_at(rng.randint(0, cfg.HISTORY_DAYS - 1))
                purpose = rng.choice(("S1006", "S0001", "S1302"))
                planted.append(row(amount, TransactionType.SWIFT, day, rng.choice(cluster_cps), DEBIT, purpose))

        # ---------------------------------------------------------------------------
        # Round-tripping. A core member wires money to company A and, two to four weeks
        # later, receives nearly the same amount back from company B - registered at A's
        # address - declared as consultancy income, a loan or a gift. The ring's solo member
        # only sends money out: the one placing funds into the scheme.
        # ---------------------------------------------------------------------------
        if account_id in round_trip_pair_of:
            outbound, returning = round_trip_pair_of[account_id]
            if account_id in plan.round_trip_core:
                cycles = rng.randint(2, 3)
                per_cycle = income * rng.uniform(2.5, 5.0) / cycles
                for cycle in range(cycles):
                    # Resampled until neither leg lands in the structuring band - a loop
                    # that also looked like structuring would carry an unlabelled channel.
                    while True:
                        out_amount = float(round(per_cycle * rng.uniform(0.8, 1.2), 2))
                        back_amount = float(round(out_amount * rng.uniform(0.90, 0.97), 2))
                        if not _in_band(out_amount) and not _in_band(back_amount):
                            break
                    out_day = rng.randint(cycle * 25, cycle * 25 + 10)
                    back_day = min(out_day + rng.randint(12, 30), cfg.HISTORY_DAYS - 1)
                    planted.append(row(out_amount, TransactionType.SWIFT, day_at(out_day),
                                       outbound, DEBIT, rng.choice(("S0001", "S1006"))))
                    planted.append(row(back_amount, TransactionType.SWIFT, day_at(back_day),
                                       returning, CREDIT, rng.choice(("P1006", "P0012", "P1302"))))
            else:
                total = income * rng.uniform(0.5, 1.0)
                sends = rng.randint(1, 2)
                for _ in range(sends):
                    amount = _swift_amount(rng, total / sends)
                    day = day_at(rng.randint(0, cfg.HISTORY_DAYS - 1))
                    planted.append(row(amount, TransactionType.SWIFT, day, outbound, DEBIT, "S0001"))

        # ---------------------------------------------------------------------------
        # Unexplained inward money: foreign companies paying a modest earner "consultancy"
        # or "software" fees, or "gifts", far beyond anything the profile explains.
        # ---------------------------------------------------------------------------
        if cohort == "unexplained_inward":
            payers = rng.sample(unexplained_payers, k=rng.randint(1, 2))
            count = rng.randint(4, 8)
            total = income * rng.uniform(2.0, 4.0)
            for _ in range(count):
                amount = _swift_amount(rng, total / count * rng.uniform(0.7, 1.3))
                day = day_at(rng.randint(0, cfg.HISTORY_DAYS - 1))
                purpose = rng.choice(("P1006", "P0802", "P1302"))
                planted.append(row(amount, TransactionType.SWIFT, day, rng.choice(payers), CREDIT, purpose))

        # ---------------------------------------------------------------------------
        # The legitimate look-alikes.
        # ---------------------------------------------------------------------------
        if cohort == "family_inward":
            # A son or daughter working abroad sending household money home each month.
            country = rng.choice([c for c, _ in cfg.NRI_COUNTRIES])
            provider = rng.choice(providers_by_country[country])
            for month in range(3):
                amount = _swift_amount(rng, rng.uniform(15_000, 60_000))
                planted.append(row(amount, TransactionType.SWIFT, day_at(month * 30 + rng.randint(0, 27)),
                                   provider, CREDIT, "P1301"))

        if cohort == "education_outward":
            # A term's university fees, in one or two instalments.
            university = rng.choice(universities)
            instalments = rng.randint(1, 2)
            total = income * rng.uniform(0.25, 0.5)
            for _ in range(instalments):
                amount = _swift_amount(rng, total / instalments)
                planted.append(row(amount, TransactionType.SWIFT, day_at(rng.randint(0, cfg.HISTORY_DAYS - 1)),
                                   university, DEBIT, "S0305"))

        if cohort == "lrs_outward":
            # A gift to a relative abroad, or an investment through an overseas brokerage.
            total = income * rng.uniform(0.2, 0.5)
            sends = rng.randint(1, 2)
            for _ in range(sends):
                if rng.random() < 0.5:
                    counterparty, purpose = rng.choice(relatives_abroad), "S1302"
                else:
                    counterparty, purpose = rng.choice(brokerages), "S0001"
                amount = _swift_amount(rng, total / sends)
                planted.append(row(amount, TransactionType.SWIFT, day_at(rng.randint(0, cfg.HISTORY_DAYS - 1)),
                                   counterparty, DEBIT, purpose))

        _finish_account(rng, account_id, entry, ordinary + planted, transactions)

    # Ids come from each transaction's own posting time and account, so they still sort
    # bank-wide in time order without a serial that a new block would renumber.
    transactions.sort(key=lambda t: (t.timestamp, t.account_id, t.amount))
    clashes: dict[str, int] = defaultdict(int)
    numbered = []
    for txn in transactions:
        key = f"{txn.account_id}|{txn.timestamp.isoformat()}"
        clashes[key] += 1
        numbered.append(txn.model_copy(update={"transaction_id": _transaction_id(txn, clashes[key])}))

    # Time-ordered batches, not isolated rows - Stage 2's velocity comparison reads the
    # history in sequence, so the committed file is sorted the way it will be consumed.
    numbered.sort(key=lambda t: (t.account_id, t.timestamp))
    return numbered, counterparties


def _finish_account(
    rng: random.Random,
    account_id: str,
    entry: dict,
    rows: list[dict],
    transactions: list[Transaction],
) -> None:
    """Apply the device rule to one account's rows and append them as transactions."""
    # A device and IP are drawn for every row, so the account's stream is the same either
    # way, and then kept only where the customer's own device took part.
    for item in rows:
        if not (item["direction"] == DEBIT and item["type"] in DIGITAL_RAILS):
            item["device_id"] = item["ip_address"] = None

    # A ring member or household look-alike must actually show the shared phone. With
    # phones recorded only on payments the customer sent, an account that sends few of
    # them could otherwise miss it by chance, and the ring would silently lose a member
    # in the graph. The earliest sent payment is moved onto the shared phone if needed.
    if entry["shared_device"]:
        sent = sorted((r for r in rows if r["device_id"]), key=lambda r: r["timestamp"])
        if sent and not any(r["device_id"] == entry["shared_device"] for r in sent):
            sent[0]["device_id"] = entry["shared_device"]
            sent[0]["ip_address"] = entry["shared_ip"] or sent[0]["ip_address"]

    for item in rows:
        counterparty = item["counterparty"]
        transactions.append(
            Transaction(
                transaction_id="TXN-PENDING",
                account_id=account_id,
                amount=item["amount"],
                direction=item["direction"],
                type=item["type"],
                timestamp=item["timestamp"],
                device_id=item["device_id"],
                ip_address=item["ip_address"],
                counterparty_id=counterparty.counterparty_id,
                counterparty_name=counterparty.name,
                purpose_code=item["purpose_code"],
            )
        )
