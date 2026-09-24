"""Generator 2 of 9 - fraud ring definitions.

No pipeline stage ever reads this file. It shapes what transactions.py and graph.py emit,
and Stage 5 has to rediscover the cluster from graph topology alone. That separation is the
whole point: if the ring label reached the graph, Stage 5 would be reading the answer rather
than finding it, and Phase 11's per-stage recall number would mean nothing.

Eleven rings: six local, two international and three round-trip. A ring is a set of
accounts sharing an identifier with no legitimate reason to share one - the same handset, the
same broadband connection, sometimes the same registered address.

A round-trip ring sends money abroad to one foreign company and receives it back from a
second one registered at the same premises, so that the return looks like unrelated foreign
income. Its members come from the cross-border block in customers.py.
"""

from __future__ import annotations

import random

from src.generators import config as cfg
from src.generators.addresses import indian_address
from src.generators.identifiers import device_id, ip_address
from src.schemas import Account, Customer, FraudRing, ResidencyStatus, RingType

#: Reserved counterparty IDs for the offshore cluster the international rings funnel into.
#: transactions.py guarantees these exist as foreign counterparties, so both generators
#: agree on the cluster without either one importing the other's output.
FOREIGN_RING_CLUSTER_IDS: tuple[str, ...] = tuple(
    f"CPF-{i + 1:04d}" for i in range(cfg.FOREIGN_RING_CLUSTER_SIZE)
)

#: Reserved counterparty IDs for the round-trip rings: one (outbound, return) pair per ring,
#: each pair registered at the same foreign address. Numbered straight after the offshore
#: cluster so transactions.py can create them in a fixed position.
ROUND_TRIP_ENTITY_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (
        f"CPF-{cfg.FOREIGN_RING_CLUSTER_SIZE + 2 * i + 1:04d}",
        f"CPF-{cfg.FOREIGN_RING_CLUSTER_SIZE + 2 * i + 2:04d}",
    )
    for i in range(cfg.NUM_ROUND_TRIP_RINGS)
)

#: Addresses a ring re-registers several accounts against. Deliberately mundane commercial
#: premises - a shared address is only suspicious because of who shares it, and a
#: conspicuous address would hand Stage 5 the answer for the wrong reason.
RING_SHARED_ADDRESSES = (
    "Shop 14, Ganesh Complex, MG Road, Pune, Maharashtra 411001",
    "2nd Floor, Sai Arcade, Old Madras Road, Bengaluru, Karnataka 560016",
    "Unit 7, Laxmi Industrial Estate, Andheri West, Mumbai, Maharashtra 400053",
    "Plot 22, Sector 18, Udyog Vihar, Gurugram, Haryana 122015",
    "B-9, Mahalaxmi Chambers, Ashram Road, Ahmedabad, Gujarat 380009",
    "34, Netaji Subhash Road, Salt Lake, Kolkata, West Bengal 700064",
    "Flat 3C, Green Meadows, Gachibowli, Hyderabad, Telangana 500032",
    "12, Anna Salai, Teynampet, Chennai, Tamil Nadu 600018",
)


def generate(
    rng: random.Random,
    customers: list[Customer],
    accounts: list[Account],
    cohorts: dict[str, str],
) -> list[FraudRing]:
    """Assemble the rings and assign their members.

    International rings are seeded preferentially from NRI accounts, because a cross-border
    funnel run entirely through resident accounts would be an odd shape for the cross-border
    clause category to cite. Local rings draw from domestic accounts outside the cross-border
    block, and round-trip rings from that block's round_trip cohort.
    """
    residency_by_customer = {c.customer_id: c.residency_status for c in customers}
    domestic_accounts = [
        a.account_id
        for a in accounts
        if residency_by_customer[a.customer_id] == ResidencyStatus.DOMESTIC
        and a.account_id not in cohorts
    ]
    nri_accounts = [
        a.account_id
        for a in accounts
        if residency_by_customer[a.customer_id] == ResidencyStatus.NRI
    ]

    domestic_pool = list(domestic_accounts)
    nri_pool = list(nri_accounts)
    rng.shuffle(domestic_pool)
    rng.shuffle(nri_pool)

    rings: list[FraudRing] = []

    for index in range(cfg.NUM_LOCAL_RINGS + cfg.NUM_INTERNATIONAL_RINGS):
        is_international = index >= cfg.NUM_LOCAL_RINGS
        ring_id = f"RING-{index + 1:02d}"

        if is_international:
            # Two NRI accounts abroad plus two resident accounts feeding them - the shape a
            # real cross-border funnel takes, rather than an all-NRI or all-domestic cluster.
            members = [nri_pool.pop() for _ in range(2)]
            members += [domestic_pool.pop() for _ in range(cfg.RING_SIZE - 2)]
        else:
            members = [domestic_pool.pop() for _ in range(cfg.RING_SIZE)]

        members.sort()

        # Every ring shares a handset. A shared device is the hard identifier Stage 5 is
        # meant to weight most heavily, so it is the one signal present in every ring.
        shared_device = device_id(rng)

        # Roughly half also share a connection, and a quarter also share a registered
        # address. Varying this matters: a ring that shares all three every time would let
        # Stage 5 succeed on a single rule instead of on topology.
        shared_ip = ip_address(rng) if rng.random() < 0.55 else None
        shared_address = RING_SHARED_ADDRESSES[index] if rng.random() < 0.30 else None

        rings.append(
            FraudRing(
                ring_id=ring_id,
                ring_type=RingType.INTERNATIONAL if is_international else RingType.LOCAL,
                member_account_ids=members,
                shared_device_id=shared_device,
                shared_ip_address=shared_ip,
                shared_address=shared_address,
                foreign_counterparty_ids=(
                    list(FOREIGN_RING_CLUSTER_IDS) if is_international else []
                ),
            )
        )

    # Round-trip rings. They share a handset like every other ring - the money moves through
    # foreign companies, but the accounts sending it are still run together.
    round_trip_pool = sorted(a for a, cohort in cohorts.items() if cohort == "round_trip")
    rng.shuffle(round_trip_pool)
    for pair_index, entity_pair in enumerate(ROUND_TRIP_ENTITY_PAIRS):
        members = sorted(round_trip_pool[pair_index * cfg.RING_SIZE : (pair_index + 1) * cfg.RING_SIZE])
        rings.append(
            FraudRing(
                ring_id=f"RING-{len(rings) + 1:02d}",
                ring_type=RingType.ROUND_TRIP,
                member_account_ids=members,
                shared_device_id=device_id(rng),
                shared_ip_address=ip_address(rng) if rng.random() < 0.55 else None,
                foreign_counterparty_ids=list(entity_pair),
            )
        )

    return rings


def generate_coincidence_clusters(
    rng: random.Random,
    accounts: list[Account],
    ring_member_ids: set[str],
) -> list[dict]:
    """Legitimate shared-identifier clusters - the graph false-positive decoy.

    A married couple on one handset, or two small traders in one shop sharing a broadband
    line, produce exactly the topology a ring does. Phase 11 measures the false-positive
    rate against these, so they have to be genuinely indistinguishable at the graph level
    and separated only by the absence of any other planted channel.
    """
    available = [a.account_id for a in accounts if a.account_id not in ring_member_ids]
    rng.shuffle(available)

    clusters: list[dict] = []
    for index in range(cfg.NUM_GRAPH_COINCIDENCE_PAIRS):
        pair = sorted([available.pop(), available.pop()])
        share_address = index % 2 == 0
        clusters.append(
            {
                "cluster_id": f"COINCIDENCE-{index + 1:02d}",
                "member_account_ids": pair,
                "shared_device_id": device_id(rng),
                "shared_ip_address": ip_address(rng),
                # Half are a household (shared address too), half a shared business line. A
                # household's address is a home of its own, never one of the rings' premises -
                # reusing those would wire an innocent couple straight into a real ring.
                "shared_address": indian_address(rng) if share_address else None,
                "reason": "household" if share_address else "shared business premises",
            }
        )
    return clusters
