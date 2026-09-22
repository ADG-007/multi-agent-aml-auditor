"""Generator 3 of 9 - 90 days of transaction history, and the counterparty pool.

The primary input to Stage 2, and a reference input to Stages 0, 5 and 7.

Amounts are constrained per payment rail rather than drawn from one global range, because
the rails really are constrained in India: RTGS has a two-lakh floor, UPI a one-lakh per
transaction ceiling, IMPS a five-lakh ceiling, ATM withdrawals a daily cap in the tens of
thousands. Honouring those bounds costs nothing and means a reader cannot find a ninety-rupee
RTGS or a fifteen-lakh UPI payment anywhere in the data.

Three patterns are planted here, all driven by the shared planting plan rather than decided
locally:
  - structuring: a burst of deposits parked just under the ten-lakh CTR threshold
  - cross-border funnelling: international ring members repeatedly paying one offshore cluster
  - benign round-amount clustering: the decoy that looks like structuring and is not
"""

from __future__ import annotations

import random
from datetime import date, datetime, time, timedelta

from faker import Faker

from src.generators import config as cfg
from src.generators.identifiers import device_id, ip_address
from src.generators.rings import FOREIGN_RING_CLUSTER_IDS
from src.schemas import (
    Account,
    Counterparty,
    Customer,
    FraudRing,
    Transaction,
    TransactionType,
)

#: Per-rail amount bounds in INR, reflecting the real limits each rail operates under.
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

#: Rails that can legitimately carry an amount in the structuring band. Cash deposits are
#: the classic vehicle, which is why they dominate the planted pattern.
STRUCTURING_RAILS: tuple[tuple[TransactionType, float], ...] = (
    (TransactionType.CASH_DEPOSIT, 0.70),
    (TransactionType.NEFT, 0.15),
    (TransactionType.RTGS, 0.10),
    (TransactionType.CHEQUE, 0.05),
)

#: Branch-hour rails only post during banking hours on weekdays; digital rails post at any
#: hour. Timestamps feed Stage 2's velocity view, so getting the daily shape right is worth
#: the few lines it takes.
BRANCH_HOUR_RAILS = frozenset(
    {TransactionType.CASH_DEPOSIT, TransactionType.CASH_WITHDRAWAL, TransactionType.CHEQUE}
)

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

#: Foreign counterparty suffixes by jurisdiction convention.
FOREIGN_BUSINESS_SUFFIXES = ("FZE", "FZ-LLC", "Pte Ltd", "Ltd", "LLC", "Holdings Ltd", "Trading DMCC")


#: How far an account's annualised throughput may sit from the customer's declared income
#: before the profile stops being defensible. Money flows both in and out, so turnover above
#: income is entirely normal; an order of magnitude above it is not.
CLEAN_THROUGHPUT_MULTIPLE = (0.6, 2.5)

#: What a planted KYC mismatch looks like: an account moving many times what its holder says
#: they earn. Stage 3's judgement only means something if the gap is this unambiguous - a
#: profile off by twenty percent is noise, not a finding.
MISMATCH_THROUGHPUT_MULTIPLE = (8.0, 25.0)


def _weighted_choice(rng: random.Random, weights: dict[TransactionType, float]) -> TransactionType:
    roll = rng.random()
    cumulative = 0.0
    for option, weight in weights.items():
        cumulative += weight
        if roll <= cumulative:
            return option
    return next(iter(weights))


def _rail_for_amount(rng: random.Random, amount: float) -> TransactionType:
    """Pick a payment rail that can actually carry this amount.

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

    The deliberate false positives come from the three decoy types instead, where they are
    labelled and measurable.
    """
    if cfg.STRUCTURING_BAND_LOW <= amount < cfg.STRUCTURING_BAND_HIGH:
        if rng.random() < 0.5:
            return float(rng.randint(400_000, cfg.STRUCTURING_BAND_LOW - 1))
        return float(rng.randint(cfg.STRUCTURING_BAND_HIGH + 1, 2_000_000))
    return amount


def _timestamp_for(rng: random.Random, day: date, txn_type: TransactionType) -> datetime:
    """Place a transaction at a plausible hour of its day."""
    if txn_type in BRANCH_HOUR_RAILS:
        # Branch counter hours. If the drawn day is a weekend, shift back to the Friday -
        # a cash deposit cannot post at a closed branch.
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


def generate_counterparties(rng: random.Random) -> list[Counterparty]:
    """The pool every transaction draws its other side from.

    The reserved CPF-#### block is created first and always foreign: rings.py hands those
    same IDs to the international rings, so both generators agree on the offshore cluster
    without either importing the other's output.
    """
    faker_in = Faker("en_IN")
    faker_intl = Faker("en_US")
    faker_in.seed_instance(cfg.SEED + 2)
    faker_intl.seed_instance(cfg.SEED + 3)

    counterparties: list[Counterparty] = []

    # The offshore cluster the two international rings funnel into.
    for index, cluster_id in enumerate(FOREIGN_RING_CLUSTER_IDS):
        country = cfg.FOREIGN_COUNTERPARTY_COUNTRIES[index % len(cfg.FOREIGN_COUNTERPARTY_COUNTRIES)]
        counterparties.append(
            Counterparty(
                counterparty_id=cluster_id,
                name=f"{faker_intl.last_name()} {rng.choice(FOREIGN_BUSINESS_SUFFIXES)}",
                country=country,
                is_foreign=True,
            )
        )

    for index in range(cfg.NUM_DOMESTIC_COUNTERPARTIES):
        # A mix of businesses and individuals, as a real statement would show.
        if rng.random() < 0.6:
            name = f"{faker_in.last_name()} {rng.choice(INDIAN_BUSINESS_SUFFIXES)}"
        else:
            name = faker_in.name()
        counterparties.append(
            Counterparty(
                counterparty_id=f"CPD-{index + 1:04d}",
                name=name,
                country="India",
                is_foreign=False,
            )
        )

    remaining_foreign = cfg.NUM_FOREIGN_COUNTERPARTIES - len(FOREIGN_RING_CLUSTER_IDS)
    for index in range(remaining_foreign):
        counterparties.append(
            Counterparty(
                counterparty_id=f"CPF-{index + 1 + len(FOREIGN_RING_CLUSTER_IDS):04d}",
                name=f"{faker_intl.last_name()} {rng.choice(FOREIGN_BUSINESS_SUFFIXES)}",
                country=rng.choice(cfg.FOREIGN_COUNTERPARTY_COUNTRIES),
                is_foreign=True,
            )
        )

    return counterparties


def _account_identifiers(
    rng: random.Random,
    accounts: list[Account],
    fraud_rings: list[FraudRing],
    coincidence_clusters: list[dict],
) -> dict[str, dict]:
    """Assign each account its own devices and IPs, plus any shared ones.

    An ordinary customer has one or two devices and a couple of IPs. A ring member also has
    the ring's shared handset, and uses it often enough for the shared edge to be visible in
    the graph without the account's own device disappearing.
    """
    identifiers: dict[str, dict] = {}
    for account in accounts:
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
            entry["shared_device"] = ring.shared_device_id
            entry["shared_ip"] = ring.shared_ip_address
            entry["shared_use_rate"] = rng.uniform(0.35, 0.60)

    for cluster in coincidence_clusters:
        for account_id in cluster["member_account_ids"]:
            entry = identifiers[account_id]
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


def generate(
    rng: random.Random,
    customers: list[Customer],
    accounts: list[Account],
    fraud_rings: list[FraudRing],
    plan,
) -> tuple[list[Transaction], list[Counterparty]]:
    """Build the full transaction history for every account.

    `plan` is the PlantingPlan from evidence_ledger; it is typed loosely here only to avoid
    a circular import between the two generator modules.
    """
    counterparties = generate_counterparties(rng)
    domestic_cps = [c for c in counterparties if not c.is_foreign]
    foreign_cps = [c for c in counterparties if c.is_foreign]
    cluster_cps = [c for c in counterparties if c.counterparty_id in FOREIGN_RING_CLUSTER_IDS]

    identifiers = _account_identifiers(rng, accounts, fraud_rings, plan.coincidence_clusters)

    residency_by_customer = {c.customer_id: c.residency_status for c in customers}
    income_by_customer = {c.customer_id: c.stated_income for c in customers}
    international_ring_members = {
        account_id
        for ring in fraud_rings
        if ring.foreign_counterparty_ids
        for account_id in ring.member_account_ids
    }

    window_end = cfg.REFERENCE_DATE
    window_start = window_end - timedelta(days=cfg.HISTORY_DAYS)

    transactions: list[Transaction] = []
    serial = 0

    for account in accounts:
        account_id = account.account_id
        entry = identifiers[account_id]
        is_nri = residency_by_customer[account.customer_id].value == "NRI"

        # An NRI account plausibly transacts abroad; a domestic one occasionally does.
        foreign_rate = 0.35 if is_nri else 0.05

        baseline_count = rng.randint(cfg.TXNS_PER_ACCOUNT_MIN, cfg.TXNS_PER_ACCOUNT_MAX)
        is_round_decoy = account_id in plan.round_amount_decoys
        is_structuring = account_id in plan.structuring
        is_kyc_mismatch = account_id in plan.kyc_mismatch

        # An account's turnover is anchored to what its holder says they earn. Without this
        # anchor a clean account could plausibly move twenty times its declared income, and
        # the KYC-mismatch channel would be indistinguishable from ordinary variation.
        low_mult, high_mult = (
            MISMATCH_THROUGHPUT_MULTIPLE if is_kyc_mismatch else CLEAN_THROUGHPUT_MULTIPLE
        )
        annual_target = income_by_customer[account.customer_id] * rng.uniform(low_mult, high_mult)
        window_target = annual_target * (cfg.HISTORY_DAYS / 365)
        mean_amount = max(600.0, window_target / baseline_count)

        # ---------------------------------------------------------------------------
        # Ordinary activity across the full 90 days.
        # ---------------------------------------------------------------------------
        for _ in range(baseline_count):
            # Transaction sizes are heavily right-skewed in real accounts - many small
            # payments and a few large ones - so a lognormal draw around the mean fits far
            # better than a uniform one.
            amount = mean_amount * rng.lognormvariate(-0.3, 0.75)
            amount = min(max(amount, 500.0), 2_500_000.0)

            if is_round_decoy and rng.random() < 0.55:
                # The benign round-amount decoy: a shopkeeper banking takings in neat
                # figures. Round, repetitive and entirely legitimate - and well below the
                # structuring band, which is the only thing separating it from a real case.
                amount = float(min(cfg.ROUND_AMOUNTS, key=lambda value: abs(value - amount)))
            elif rng.random() < 0.18:
                # Everyone transacts at round figures sometimes.
                amount = float(min(cfg.ROUND_AMOUNTS, key=lambda value: abs(value - amount)))

            amount = _avoid_structuring_band(rng, float(round(amount, 2)))
            txn_type = _rail_for_amount(rng, amount)

            if rng.random() < foreign_rate:
                counterparty = rng.choice(foreign_cps)
            else:
                counterparty = rng.choice(domestic_cps)

            day = window_start + timedelta(days=rng.randint(0, cfg.HISTORY_DAYS - 1))
            device, ip = _pick_device_ip(rng, entry)
            serial += 1
            transactions.append(
                Transaction(
                    transaction_id=f"TXN-{serial:07d}",
                    account_id=account_id,
                    amount=amount,
                    type=txn_type,
                    timestamp=_timestamp_for(rng, day, txn_type),
                    device_id=device,
                    ip_address=ip,
                    counterparty_id=counterparty.counterparty_id,
                    counterparty_name=counterparty.name,
                )
            )

        # ---------------------------------------------------------------------------
        # Planted structuring: a cluster of deposits just under the CTR threshold, inside
        # a single 30-day window. Rule 3 aggregates integrally connected cash transactions
        # over a month, so a month is the window a real analyst would look at.
        # ---------------------------------------------------------------------------
        if is_structuring:
            burst_start_offset = rng.randint(0, cfg.HISTORY_DAYS - cfg.STRUCTURING_WINDOW_DAYS)
            burst_start = window_start + timedelta(days=burst_start_offset)
            burst_count = rng.randint(cfg.STRUCTURING_TXN_MIN, cfg.STRUCTURING_TXN_MAX)

            for _ in range(burst_count):
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
                day = burst_start + timedelta(days=rng.randint(0, cfg.STRUCTURING_WINDOW_DAYS - 1))
                device, ip = _pick_device_ip(rng, entry)

                if account_id in international_ring_members and cluster_cps:
                    counterparty = rng.choice(cluster_cps)
                else:
                    counterparty = rng.choice(domestic_cps)

                serial += 1
                transactions.append(
                    Transaction(
                        transaction_id=f"TXN-{serial:07d}",
                        account_id=account_id,
                        amount=amount,
                        type=txn_type,
                        timestamp=_timestamp_for(rng, day, txn_type),
                        device_id=device,
                        ip_address=ip,
                        counterparty_id=counterparty.counterparty_id,
                        counterparty_name=counterparty.name,
                    )
                )

        # ---------------------------------------------------------------------------
        # Cross-border funnelling: international ring members pay the same small offshore
        # cluster again and again. No new pipeline stage detects this - Stage 5 sees it as
        # a shared counterparty cluster with is_foreign set, and Stage 6 cites the
        # cross-border clause category afterwards.
        # ---------------------------------------------------------------------------
        if account_id in international_ring_members and cluster_cps:
            for _ in range(rng.randint(5, 12)):
                txn_type = rng.choice((TransactionType.NEFT, TransactionType.RTGS))
                low, high = RAIL_BOUNDS[txn_type]
                # Capped below the structuring band: this is the cross-border channel, and
                # letting it spill into the band would plant an unlabelled structuring
                # signal on every international ring member.
                amount = float(rng.randint(max(low, 150_000), min(high, 880_000)))
                day = window_start + timedelta(days=rng.randint(0, cfg.HISTORY_DAYS - 1))
                device, ip = _pick_device_ip(rng, entry)
                counterparty = rng.choice(cluster_cps)
                serial += 1
                transactions.append(
                    Transaction(
                        transaction_id=f"TXN-{serial:07d}",
                        account_id=account_id,
                        amount=amount,
                        type=txn_type,
                        timestamp=_timestamp_for(rng, day, txn_type),
                        device_id=device,
                        ip_address=ip,
                        counterparty_id=counterparty.counterparty_id,
                        counterparty_name=counterparty.name,
                    )
                )

    # Time-ordered batches, not isolated rows - Stage 2's velocity comparison reads the
    # history in sequence, so the committed file is sorted the way it will be consumed.
    transactions.sort(key=lambda t: (t.account_id, t.timestamp))
    return transactions, counterparties
