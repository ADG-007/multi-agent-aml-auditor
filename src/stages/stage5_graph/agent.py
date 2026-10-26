"""Stage 5's entry point: links to accounts already flagged in this run.

    run(account_id, ctx, as_of, transaction, state) -> Stage5GraphOutput

Graph = account - device / IP / address, built live from the bank. Only links to accounts
flagged on their own evidence earlier in this run score (no echo), so batch order is fixed.
No model calls.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import networkx as nx

from src.bank import BankData
from src.orchestration.run_state import RunState
from src.schemas import ReasoningStep, Stage5GraphOutput, Transaction
from src.stages.stage0_triage import TriageContext
from src.stages.stage5_graph import config


HARD = ("device", "ip")
SOFT = ("address",)


def build_graph(bank: BankData, extra: Transaction | None = None) -> nx.Graph:
    """Accounts joined to their devices, IPs and address. Counterparties are hubs, left out."""
    g = nx.Graph()
    for account_id in bank.account_ids():
        g.add_node(account_id, kind="account")
        rows = list(bank.transactions(account_id))
        if extra is not None and extra.account_id == account_id:
            rows.append(extra)
        for t in rows:
            if t.device_id:
                g.add_edge(account_id, ("device", t.device_id))
            if t.ip_address:
                g.add_edge(account_id, ("ip", t.ip_address))
        g.add_edge(account_id, ("address", bank.customer_for(account_id).address))
    return g


def links(g: nx.Graph, account_id: str) -> dict[str, dict[str, set[str]]]:
    """Other account -> link kind -> the shared values."""
    out: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for node in g.neighbors(account_id):
        kind, value = node
        for other in g.neighbors(node):
            if other != account_id:
                out[other][kind].add(value)
    return out


def run(
    account_id: str,
    ctx: TriageContext,
    as_of: datetime | None,
    transaction: Transaction | None = None,
    state: RunState | None = None,
) -> Stage5GraphOutput:
    state = state or RunState()
    g = build_graph(ctx.bank, transaction)
    linked = links(g, account_id)

    flagged = {a: kinds for a, kinds in linked.items() if state.is_flagged_on_own(a)}
    echo = sorted(a for a in linked if a not in flagged and state.is_flagged(a))
    hard = sorted(a for a, kinds in flagged.items() if any(k in kinds for k in HARD))
    soft = sorted(a for a in flagged if a not in hard)
    unflagged = sorted(a for a in linked if a not in flagged and a not in echo)
    sanctioned = [a for a in hard if (rec := state.latest(a)) and rec.sanctions_escalation]

    if sanctioned and len(hard) >= config.VETO_MIN_HARD_LINKS:
        confidence, why = (
            config.VETO_CONFIDENCE,
            (
                f"hard link to sanctions escalation {sanctioned[0]} plus "
                f"{len(hard) - 1} more hard link(s): veto range"
            ),
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

    def tag(other: str) -> str:
        if other in flagged:
            return f" ({state.standing(other).value})"
        if other in echo:
            return f" ({state.standing(other).value} through links only: not counted)"
        return " (not flagged)"

    shared = [
        f"{kind} {value} shared with {other}" + tag(other)
        for other in sorted(linked)
        for kind in (*HARD, *SOFT)
        for value in sorted(linked[other].get(kind, ()))
    ]
    trail = [
        ReasoningStep(
            step="Shared identifiers",
            finding=f"Linked to {len(linked)} account(s): {', '.join(sorted(linked)) or 'none'}.",
            impact="Device and IP are hard links, an address alone is soft (NET-002).",
        ),
        ReasoningStep(
            step="Flagged in this run",
            finding=f"Hard: {', '.join(hard) or 'none'}. Soft: {', '.join(soft) or 'none'}. "
            f"Flagged through links only (not counted): {', '.join(echo) or 'none'}. "
            f"Not flagged: {', '.join(unflagged) or 'none'}.",
            impact="Only links to accounts already MONITOR or ESCALATE on their own evidence "
            "score; a chance link to an ordinary account adds nothing.",
        ),
        ReasoningStep(
            step="Confidence",
            finding=why,
            impact=f"Graph confidence {confidence:g}; links alone are capped at "
            f"{config.RING_ONLY_CAP:g} (no veto without a sanctions link).",
        ),
    ]
    return Stage5GraphOutput(
        shared_identifiers=shared,
        connected_flagged_accounts=sorted(flagged),
        confidence=round(confidence, 4),
        reasoning_trail=trail,
    )


def _ladder(scores: tuple[float, ...], n: int) -> float:
    return scores[min(n, len(scores)) - 1]
