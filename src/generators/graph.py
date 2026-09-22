"""Generator 7 of 9 - the entity graph.

Used exclusively by Stage 5, plus Stage 0's static known-bad identifier lookup, which is
derived from this graph rather than traversing it (Stage 0 never runs graph algorithms).

Five node types - accounts, devices, IPs, addresses and counterparties - with edges
recording which account used which identifier and how often. The ring label never appears
anywhere in this file. What a ring leaves behind is a set of accounts attached to the same
device or address node, and Stage 5 has to find that shape itself.

Counterparty nodes carry country and is_foreign. That is what makes the cross-border clause
category reachable without adding a pipeline stage: Stage 5 reads the attribute off the
node, and Stage 6 cites the relevant clause afterwards.
"""

from __future__ import annotations

from collections import defaultdict

import networkx as nx

from src.generators import config as cfg
from src.schemas import Account, Counterparty, Customer, ResidencyStatus, Transaction


def _address_key(address: str) -> str:
    """A stable node id for an address.

    Normalised so that two accounts registered at the same premises land on one node -
    whitespace and case differences must not split a genuinely shared address in two.
    """
    normalised = " ".join(address.split()).lower()
    return f"ADDR::{normalised}"


def build(
    customers: list[Customer],
    accounts: list[Account],
    transactions: list[Transaction],
    counterparties: list[Counterparty],
    plan,
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
    # Shared addresses that come from a planted ring or a legitimate cluster are registered
    # explicitly. A ring re-registers accounts against one address without the customer
    # records changing, so the edge has to be added here rather than inferred from
    # customer.address.
    # -----------------------------------------------------------------------------------
    shared_address_sources: list[tuple[list[str], str | None]] = []
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
    # Device, IP and counterparty edges, weighted by how often each was used. Weight is what
    # separates a customer's own handset from one they touched once, and Stage 5's
    # shared-edge count is only meaningful if that difference survives into the graph.
    # -----------------------------------------------------------------------------------
    device_use: dict[tuple[str, str], int] = defaultdict(int)
    ip_use: dict[tuple[str, str], int] = defaultdict(int)
    counterparty_use: dict[tuple[str, str], list[float]] = defaultdict(list)

    for txn in transactions:
        device_use[(txn.account_id, txn.device_id)] += 1
        ip_use[(txn.account_id, txn.ip_address)] += 1
        counterparty_use[(txn.account_id, txn.counterparty_id)].append(txn.amount)

    for (account_id, device), count in device_use.items():
        if device not in graph:
            graph.add_node(device, node_type="device")
        graph.add_edge(account_id, device, relation="USES_DEVICE", weight=count)

    for (account_id, ip), count in ip_use.items():
        if ip not in graph:
            graph.add_node(ip, node_type="ip")
        graph.add_edge(account_id, ip, relation="USES_IP", weight=count)

    for (account_id, counterparty_id), amounts in counterparty_use.items():
        counterparty = counterparty_by_id[counterparty_id]
        if counterparty_id not in graph:
            graph.add_node(
                counterparty_id,
                node_type="counterparty",
                name=counterparty.name,
                country=counterparty.country,
                is_foreign=counterparty.is_foreign,
            )
        graph.add_edge(
            account_id,
            counterparty_id,
            relation="TRANSACTED_WITH",
            weight=len(amounts),
            total_amount=float(sum(amounts)),
        )

    return graph


def write(graph: nx.Graph) -> None:
    """Persist the graph as GraphML.

    GraphML is chosen over a pickle because it is plain text, diffable in git and openable
    in Gephi or yEd - which matters for a project whose central claim is auditability.
    """
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, cfg.GRAPH_FILE)


def known_bad_identifiers(graph: nx.Graph, plan) -> dict[str, list[str]]:
    """Candidate contents for Stage 0's static known-bad lookup. NOT WRITTEN IN PHASE 1.

    Stage 0 is explicitly forbidden from running graph algorithms - it is a cheap routing
    check, not an analyst - so it needs a flat lookup rather than the graph itself.

    This implementation derives that lookup from ring membership, which is a ground-truth
    field, and that is exactly the problem: it would hand Stage 0 a perfect oracle for which
    accounts to route, and Phase 11's recall would measure the planting plan rather than the
    pipeline. Stage 0 only routes and never scores, so the leak is narrower than it first
    looks, but it is still a leak.

    Phase 2 owns Stage 0's routing table and therefore owns this decision. The honest
    alternatives are to seed the lookup from a small hand-picked subset that does not cover
    every ring, or to populate it from accounts already elevated in prior runs the way a
    real bank's list accumulates. Left here, unused and uncalled, so the question is visible
    when Phase 2 reaches it rather than being quietly answered now.
    """
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
