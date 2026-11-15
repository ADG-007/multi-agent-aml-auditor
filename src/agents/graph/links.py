"""Shared-identifier tools: the account graph, the link score and the two optional lookups.

Graph = account - device / IP / address, built live from the bank. Counterparties are hubs
(a shop is paid by a hundred accounts), so they are left out of the graph. Only links to
accounts flagged on their own evidence earlier in this run score, so a ring cannot flag
itself by echo and batch order is fixed.
"""

from __future__ import annotations

import weakref
from collections import defaultdict
from dataclasses import dataclass

import networkx as nx

from src.agents.graph import config
from src.bank import BankData
from src.orchestration.run_state import RunState
from src.schemas import Transaction, TransactionDirection


HARD = ("device", "ip")
SOFT = ("address",)


def build_graph(bank: BankData, extra: Transaction | None = None) -> nx.Graph:
    """Accounts joined to their devices, IPs and address.

    Built once per bank state and reused; `extra` goes on a copy, never on the saved graph.
    """
    size = bank.transaction_count
    saved = _GRAPHS.get(bank)
    if saved is None or saved[0] != size:
        g = nx.Graph()
        for account_id in bank.account_ids():
            g.add_node(account_id, kind="account")
            for t in bank.transactions(account_id):
                _link(g, t)
            g.add_edge(account_id, ("address", bank.customer_for(account_id).address))
        saved = _GRAPHS[bank] = (size, g)
    if extra is None:
        return saved[1]
    g = saved[1].copy()
    _link(g, extra)
    return g


_GRAPHS: weakref.WeakKeyDictionary[BankData, tuple[int, nx.Graph]] = weakref.WeakKeyDictionary()
_PAYERS: weakref.WeakKeyDictionary[BankData, tuple[int, dict[str, set[str]]]] = (
    weakref.WeakKeyDictionary()
)


def _link(g: nx.Graph, t: Transaction) -> None:
    if t.device_id:
        g.add_edge(t.account_id, ("device", t.device_id))
    if t.ip_address:
        g.add_edge(t.account_id, ("ip", t.ip_address))


def links(g: nx.Graph, account_id: str) -> dict[str, dict[str, set[str]]]:
    """Other account -> link kind -> the shared values."""
    out: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for node in g.neighbors(account_id):
        kind, value = node
        for other in g.neighbors(node):
            if other != account_id:
                out[other][kind].add(value)
    return out


@dataclass(frozen=True)
class LinkScore:
    confidence: float
    why: str
    linked: dict[str, dict[str, set[str]]]
    hard: list[str]
    soft: list[str]
    echo: list[str]
    unflagged: list[str]

    @property
    def flagged(self) -> list[str]:
        return sorted({*self.hard, *self.soft})


def score_links(g: nx.Graph, account_id: str, state: RunState) -> LinkScore:
    linked = links(g, account_id)
    flagged = {a: kinds for a, kinds in linked.items() if state.is_flagged_on_own(a)}
    echo = sorted(a for a in linked if a not in flagged and state.is_flagged(a))
    hard = sorted(a for a, kinds in flagged.items() if any(k in kinds for k in HARD))
    soft = sorted(a for a in flagged if a not in hard)
    unflagged = sorted(a for a in linked if a not in flagged and a not in echo)
    sanctioned = [a for a in hard if (rec := state.latest(a)) and rec.sanctions_escalation]

    if sanctioned and len(hard) >= config.VETO_MIN_HARD_LINKS:
        confidence = config.VETO_CONFIDENCE
        why = (
            f"hard link to sanctions escalation {sanctioned[0]} plus "
            f"{len(hard) - 1} more hard link(s): veto range"
        )
    elif hard:
        confidence = _ladder(config.HARD_LINK_SCORES, len(hard))
        why = f"{len(hard)} flagged account(s) by device/IP"
    elif soft:
        confidence = _ladder(config.SOFT_LINK_SCORES, len(soft))
        why = f"{len(soft)} flagged account(s) by address only (soft link)"
    else:
        confidence, why = 0.0, "no link to a flagged account"
    if confidence < config.VETO_CONFIDENCE:
        confidence = min(confidence, config.RING_ONLY_CAP)
    return LinkScore(round(confidence, 4), why, linked, hard, soft, echo, unflagged)


def _ladder(scores: tuple[float, ...], n: int) -> float:
    return scores[min(n, len(scores)) - 1]


# --------------------------------------------------------------------------------------
# Optional tools (facts only, never scored)
# --------------------------------------------------------------------------------------


def path_to_flagged(g: nx.Graph, account_id: str, state: RunState) -> list[str] | None:
    """The shortest chain of shared identifiers to an account flagged on its own evidence."""
    lengths = nx.single_source_shortest_path(g, account_id, cutoff=config.PATH_MAX_HOPS)
    found = [
        path
        for node, path in lengths.items()
        if isinstance(node, str) and node != account_id and state.is_flagged_on_own(node)
    ]
    if not found:
        return None
    best = min(found, key=lambda p: (len(p), p[-1]))
    return [n if isinstance(n, str) else f"{n[0]} {n[1]}" for n in best]


def counterparty_overlap(
    bank: BankData, account_id: str, state: RunState
) -> list[tuple[str, list[str]]]:
    """Payees this account pays that few accounts pay and a flagged account also pays."""
    payers = _payers(bank)
    out = []
    paid = {
        t.counterparty_id
        for t in bank.transactions(account_id)
        if t.direction == TransactionDirection.DEBIT
    }
    for counterparty_id in sorted(paid):
        accounts = payers.get(counterparty_id, set())
        if len(accounts) > config.OVERLAP_MAX_ACCOUNTS:
            continue
        others = sorted(a for a in accounts if a != account_id and state.is_flagged_on_own(a))
        if others:
            out.append((counterparty_id, others))
    return out


def _payers(bank: BankData) -> dict[str, set[str]]:
    """Counterparty id -> the accounts that paid it."""
    size = bank.transaction_count
    saved = _PAYERS.get(bank)
    if saved is None or saved[0] != size:
        index: dict[str, set[str]] = defaultdict(set)
        for account_id in bank.account_ids():
            for t in bank.transactions(account_id):
                if t.direction == TransactionDirection.DEBIT:
                    index[t.counterparty_id].add(account_id)
        saved = _PAYERS[bank] = (size, index)
    return saved[1]
