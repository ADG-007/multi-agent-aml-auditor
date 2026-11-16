"""The evidence the site draws for one account, read from the bank's own data.

Monthly money (cash apart), the monthly cash cap from the rulebook, the biggest
counterparties, the KYC narrative, the ID documents on file, and the accounts that share a
device, IP or address. Nothing here scores anything.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.agents.graph.links import build_graph, links
from src.bank import BankData
from src.rulebook import RulebookLookup
from src.schemas import TransactionDirection, TransactionType


MAX_GRAPH_NODES = 40
TOP_COUNTERPARTIES = 8


class DocumentIndex:
    """ID images by account. Only file names listed in the manifest can be served."""

    def __init__(self, data_dir: Path) -> None:
        self.folder = Path(data_dir) / "documents"
        manifest = self.folder / "manifest.json"
        rows = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
        self._by_account: dict[str, list[dict]] = defaultdict(list)
        self._files: set[str] = set()
        for row in rows:
            self._by_account[row["account_id"]].append(
                {"type": row["document_type"], "file": row["file"]}
            )
            self._files.add(row["file"])

    def for_account(self, account_id: str) -> list[dict]:
        return [
            d for d in self._by_account.get(account_id, []) if (self.folder / d["file"]).exists()
        ]

    def path(self, file: str) -> Path:
        if file not in self._files:
            raise KeyError(file)
        return self.folder / file


def build_evidence(
    bank: BankData,
    rulebook: RulebookLookup,
    documents: DocumentIndex,
    account_id: str,
    verdicts: dict[str, str] | None = None,
) -> dict:
    account = bank.account(account_id)
    customer = bank.customer_for(account_id)
    profile = bank.kyc_profile_for(account_id)
    cap = rulebook.monthly_structuring()
    rows = bank.transactions(account_id)
    return {
        "account": account_id,
        "customer": {
            "name": customer.name,
            "occupation": customer.occupation,
            "stated_income": customer.stated_income,
            "residency": customer.residency_status.value,
            "country": customer.country,
            "account_type": account.account_type.value,
            "opened": account.opened_date.isoformat(),
        },
        "transactions": len(rows),
        "months": _months(rows),
        "cap": {"threshold_inr": cap.threshold_inr, "band_floor_inr": cap.band_floor_inr},
        "counterparties": _counterparties(bank, rows),
        "kyc_narrative": profile.narrative if profile else "",
        "documents": documents.for_account(account_id),
        "graph": _graph(bank, account_id, verdicts or {}),
    }


def _months(rows) -> list[dict]:
    months: dict[str, dict] = {}
    for t in rows:
        m = months.setdefault(
            t.timestamp.strftime("%Y-%m"),
            {"cash_in": 0.0, "cash_out": 0.0, "credits": 0.0, "debits": 0.0, "count": 0},
        )
        m["count"] += 1
        credit = t.direction == TransactionDirection.CREDIT
        m["credits" if credit else "debits"] += t.amount
        if t.type == TransactionType.CASH_DEPOSIT:
            m["cash_in"] += t.amount
        elif t.type in (TransactionType.CASH_WITHDRAWAL, TransactionType.ATM):
            m["cash_out"] += t.amount
    return [
        {"month": month, **{k: round(v, 2) for k, v in m.items()}}
        for month, m in sorted(months.items())
    ]


def _counterparties(bank: BankData, rows) -> list[dict]:
    totals: dict[str, dict] = {}
    for t in rows:
        c = totals.setdefault(
            t.counterparty_id,
            {"id": t.counterparty_id, "name": t.counterparty_name, "total": 0.0, "count": 0},
        )
        c["total"] += t.amount
        c["count"] += 1
    top = sorted(totals.values(), key=lambda c: (-c["total"], c["id"]))[:TOP_COUNTERPARTIES]
    for c in top:
        known = bank.counterparty(c["id"])
        c["total"] = round(c["total"], 2)
        c["country"] = known.country if known else ""
        c["is_foreign"] = bool(known and known.is_foreign)
    return top


def _graph(bank: BankData, account_id: str, verdicts: dict[str, str]) -> dict:
    """The target, the accounts that share a device, IP or address with it, and what they share."""
    linked = links(build_graph(bank), account_id)
    ranked = sorted(linked, key=lambda a: (-sum(len(v) for v in linked[a].values()), a))
    nodes = [_account_node(bank, account_id, verdicts, True)]
    edges, shared = [], {}
    for other in ranked:
        new = {(k, v) for k, vals in linked[other].items() for v in vals} - set(shared)
        if len(nodes) + 1 + len(new) > MAX_GRAPH_NODES:
            break
        nodes.append(_account_node(bank, other, verdicts, False))
        for kind, vals in sorted(linked[other].items()):
            for value in sorted(vals):
                node_id = f"{kind}:{value}"
                if (kind, value) not in shared:
                    shared[(kind, value)] = node_id
                    nodes.append(
                        {
                            "id": node_id,
                            "kind": kind,
                            "label": value,
                            "verdict": None,
                            "is_target": False,
                        }
                    )
                    edges.append({"source": account_id, "target": node_id, "relation": kind})
                edges.append({"source": other, "target": node_id, "relation": kind})
    return {"nodes": nodes, "edges": edges, "linked_total": len(linked)}


def _account_node(bank: BankData, account_id: str, verdicts: dict[str, str], target: bool) -> dict:
    return {
        "id": account_id,
        "kind": "account",
        "label": bank.customer_for(account_id).name,
        "verdict": verdicts.get(account_id),
        "is_target": target,
    }
