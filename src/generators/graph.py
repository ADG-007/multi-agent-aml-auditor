"""The entity graph (data/graph.graphml): accounts, devices, IPs, addresses, counterparties.

Edges record who used what, and how often. No ring label is stored: a ring is only accounts
sharing a node. Account-counterparty edges keep money in and out apart, and a foreign
company's registered address is a node, so round trips show as paths.
"""

from __future__ import annotations

from collections import defaultdict

import networkx as nx

from src.generators import config as cfg
from src.schemas import (
    Account,
    Counterparty,
    Customer,
    FraudRing,
    ResidencyStatus,
    Transaction,
    TransactionDirection,
)


def _address_key(address: str) -> str:
    """A stable node id for an address (normalised, so case and spaces do not split it)."""
    normalised = " ".join(address.split()).lower()
    return f"ADDR::{normalised}"


def build(
    customers: list[Customer],
    accounts: list[Account],
    transactions: list[Transaction],
    counterparties: list[Counterparty],
    plan,
    fraud_rings: list[FraudRing],
) -> nx.Graph:
    """Assemble the graph from the generated data."""
    graph = nx.Graph()

    customers_by_id = {c.customer_id: c for c in customers}
    counterparty_by_id = {c.counterparty_id: c for c in counterparties}

    # -----------------------------------------------------------------------------------
    # Account nodes.
    # -----------------------------------------------------------------------------------
    for account in accounts:
        customer = customers_by_id[account.customer_id]
        graph.add_node(
            account.account_id,
            node_type="account",
            customer_id=customer.customer_id,
            account_type=account.account_type.value,
            residency_status=customer.residency_status.value,
            country=customer.country,
            is_foreign=customer.residency_status == ResidencyStatus.NRI,
        )

        address_node = _address_key(customer.address)
        if address_node not in graph:
            graph.add_node(address_node, node_type="address", country=customer.country)
        graph.add_edge(account.account_id, address_node, relation="RESIDES_AT", weight=1)

    # -----------------------------------------------------------------------------------
    # Shared ring / cluster addresses are added here (customer records do not change).
    # -----------------------------------------------------------------------------------
    shared_address_sources: list[tuple[list[str], str | None]] = [
        (ring.member_account_ids, ring.shared_address) for ring in fraud_rings
    ]
    for cluster in plan.coincidence_clusters:
        shared_address_sources.append((cluster["member_account_ids"], cluster["shared_address"]))

    for member_ids, shared_address in shared_address_sources:
        if not shared_address:
            continue
        address_node = _address_key(shared_address)
        if address_node not in graph:
            graph.add_node(address_node, node_type="address", country="India")
        for account_id in member_ids:
            graph.add_edge(account_id, address_node, relation="RESIDES_AT", weight=1)

    # -----------------------------------------------------------------------------------
    # Device, IP and counterparty edges, weighted by use count (own phone vs touched once).
    # -----------------------------------------------------------------------------------
    device_use: dict[tuple[str, str], int] = defaultdict(int)
    ip_use: dict[tuple[str, str], int] = defaultdict(int)
    counterparty_use: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"count_in": 0, "count_out": 0, "amount_in": 0.0, "amount_out": 0.0}
    )

    for txn in transactions:
        # Only customer-initiated transactions carry a device and IP - see Transaction.
        if txn.device_id:
            device_use[(txn.account_id, txn.device_id)] += 1
        if txn.ip_address:
            ip_use[(txn.account_id, txn.ip_address)] += 1
        flow = counterparty_use[(txn.account_id, txn.counterparty_id)]
        if txn.direction == TransactionDirection.CREDIT:
            flow["count_in"] += 1
            flow["amount_in"] += txn.amount
        else:
            flow["count_out"] += 1
            flow["amount_out"] += txn.amount

    for (account_id, device), count in device_use.items():
        if device not in graph:
            graph.add_node(device, node_type="device")
        graph.add_edge(account_id, device, relation="USES_DEVICE", weight=count)

    for (account_id, ip), count in ip_use.items():
        if ip not in graph:
            graph.add_node(ip, node_type="ip")
        graph.add_edge(account_id, ip, relation="USES_IP", weight=count)

    for (account_id, counterparty_id), flow in counterparty_use.items():
        counterparty = counterparty_by_id[counterparty_id]
        if counterparty_id not in graph:
            graph.add_node(
                counterparty_id,
                node_type="counterparty",
                name=counterparty.name,
                country=counterparty.country,
                is_foreign=counterparty.is_foreign,
            )
            if counterparty.registered_address:
                address_node = _address_key(counterparty.registered_address)
                if address_node not in graph:
                    graph.add_node(address_node, node_type="address", country=counterparty.country)
                graph.add_edge(counterparty_id, address_node, relation="REGISTERED_AT", weight=1)
        graph.add_edge(
            account_id,
            counterparty_id,
            relation="TRANSACTED_WITH",
            weight=int(flow["count_in"] + flow["count_out"]),
            total_amount=float(flow["amount_in"] + flow["amount_out"]),
            count_in=int(flow["count_in"]),
            count_out=int(flow["count_out"]),
            amount_in=float(round(flow["amount_in"], 2)),
            amount_out=float(round(flow["amount_out"], 2)),
        )

    return graph


def write(graph: nx.Graph) -> None:
    """Save as GraphML: plain text, diffable, opens in Gephi or yEd."""
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, cfg.GRAPH_FILE)


def known_bad_identifiers(graph: nx.Graph, plan) -> dict[str, list[str]]:
    """Unused. It derives a known-bad list from ring membership (answer key), which would leak
    the answers to Stage 0. The run's list is built from escalations instead (RunState)."""
    devices: set[str] = set()
    ips: set[str] = set()
    addresses: set[str] = set()

    for account_id in plan.ring_membership:
        for neighbour in graph.neighbors(account_id):
            node_type = graph.nodes[neighbour].get("node_type")
            # Only identifiers shared by more than one account are worth listing: a ring
            # member's personal handset is not a known-bad identifier.
            if graph.degree(neighbour) < 2:
                continue
            if node_type == "device":
                devices.add(neighbour)
            elif node_type == "ip":
                ips.add(neighbour)
            elif node_type == "address":
                addresses.add(neighbour)

    return {
        "devices": sorted(devices),
        "ips": sorted(ips),
        "addresses": sorted(addresses),
    }
