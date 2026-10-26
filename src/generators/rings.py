"""The 13 fraud rings (answer key): 6 local, 2 international, 3 round-trip, 2 mule networks.

A ring = accounts sharing a device, connection or address for no good reason. No stage reads
this file: Stage 5 must find the shape in the graph. Round-trip rings send money abroad and
get it back via a second company at the same address. Mule networks share one phone and one
collector. Each ring has its own stream.
"""

from __future__ import annotations

from src.generators import config as cfg
from src.generators.addresses import indian_address
from src.generators.identifiers import device_id, ip_address
from src.generators.streams import Streams, opaque_id
from src.schemas import Account, Customer, FraudRing, ResidencyStatus, RingType

#: Reserved ids for the offshore cluster (transactions.py creates them).
FOREIGN_RING_CLUSTER_IDS: tuple[str, ...] = tuple(
    f"CPF-{i + 1:04d}" for i in range(cfg.FOREIGN_RING_CLUSTER_SIZE)
)

#: Reserved (outbound, return) pairs for the round-trip rings, one shared address each.
ROUND_TRIP_ENTITY_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (
        f"CPF-{cfg.FOREIGN_RING_CLUSTER_SIZE + 2 * i + 1:04d}",
        f"CPF-{cfg.FOREIGN_RING_CLUSTER_SIZE + 2 * i + 2:04d}",
    )
    for i in range(cfg.NUM_ROUND_TRIP_RINGS)
)

#: Reserved opaque ids for the collectors (one per network, then per solo mule).
MULE_NETWORK_COLLECTOR_IDS: tuple[str, ...] = tuple(
    opaque_id("CPM", "collector", "network", i) for i in range(cfg.NUM_MULE_RINGS)
)
SOLO_MULE_COLLECTOR_IDS: tuple[str, ...] = tuple(
    opaque_id("CPM", "collector", "solo", i) for i in range(dict(cfg.MULE_BLOCK_COHORTS)["solo_mule"])
)

#: Mundane commercial addresses rings share (only who shares them is suspicious).
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
    streams: Streams,
    customers: list[Customer],
    accounts: list[Account],
    cohorts: dict[str, str],
) -> list[FraudRing]:
    """Build the rings: international from NRIs, local from base domestic accounts, round-trip
    and mule rings from their own blocks."""
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
    streams.rng("rings", "pool", "base_domestic").shuffle(domestic_pool)
    streams.rng("rings", "pool", "base_nri").shuffle(nri_pool)

    rings: list[FraudRing] = []

    for index in range(cfg.NUM_LOCAL_RINGS + cfg.NUM_INTERNATIONAL_RINGS):
        is_international = index >= cfg.NUM_LOCAL_RINGS
        ring_id = f"RING-{index + 1:02d}"
        rng = streams.rng("ring", ring_id)

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

        # About half also share an IP, a quarter an address, so no single rule finds them all.
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
    streams.rng("rings", "pool", "round_trip").shuffle(round_trip_pool)
    for pair_index, entity_pair in enumerate(ROUND_TRIP_ENTITY_PAIRS):
        ring_id = f"RING-{cfg.NUM_LOCAL_RINGS + cfg.NUM_INTERNATIONAL_RINGS + pair_index + 1:02d}"
        rng = streams.rng("ring", ring_id)
        members = sorted(round_trip_pool[pair_index * cfg.RING_SIZE : (pair_index + 1) * cfg.RING_SIZE])
        rings.append(
            FraudRing(
                ring_id=ring_id,
                ring_type=RingType.ROUND_TRIP,
                member_account_ids=members,
                shared_device_id=device_id(rng),
                shared_ip_address=ip_address(rng) if rng.random() < 0.55 else None,
                foreign_counterparty_ids=list(entity_pair),
            )
        )

    # Mule networks: one handler phone, one shared collector. Skipped without the mule block.
    mule_pool = sorted(a for a, cohort in cohorts.items() if cohort == "network_mule")
    streams.rng("rings", "pool", "network_mule").shuffle(mule_pool)
    first_mule_ring = cfg.NUM_LOCAL_RINGS + cfg.NUM_INTERNATIONAL_RINGS + cfg.NUM_ROUND_TRIP_RINGS
    for network_index, collector_id in enumerate(MULE_NETWORK_COLLECTOR_IDS):
        members = sorted(mule_pool[network_index * cfg.RING_SIZE : (network_index + 1) * cfg.RING_SIZE])
        if not members:
            continue
        ring_id = f"RING-{first_mule_ring + network_index + 1:02d}"
        rng = streams.rng("ring", ring_id)
        rings.append(
            FraudRing(
                ring_id=ring_id,
                ring_type=RingType.MULE,
                member_account_ids=members,
                shared_device_id=device_id(rng),
                shared_ip_address=ip_address(rng) if rng.random() < 0.55 else None,
                collector_counterparty_id=collector_id,
            )
        )

    return rings


def generate_coincidence_clusters(
    streams: Streams,
    accounts: list[Account],
    ring_member_ids: set[str],
) -> list[dict]:
    """Legit shared-identifier clusters (a couple on one phone, traders on one line): decoys
    that look exactly like a ring in the graph."""
    available = [a.account_id for a in accounts if a.account_id not in ring_member_ids]
    streams.rng("coincidence", "pool").shuffle(available)

    clusters: list[dict] = []
    for index in range(cfg.NUM_GRAPH_COINCIDENCE_PAIRS):
        rng = streams.rng("coincidence", index)
        pair = sorted([available.pop(), available.pop()])
        share_address = index % 2 == 0
        clusters.append(
            {
                "cluster_id": f"COINCIDENCE-{index + 1:02d}",
                "member_account_ids": pair,
                "shared_device_id": device_id(rng),
                "shared_ip_address": ip_address(rng),
                # Half households (own home address), half a shared business line.
                "shared_address": indian_address(rng) if share_address else None,
                "reason": "household" if share_address else "shared business premises",
            }
        )
    return clusters
