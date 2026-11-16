"""Saved runs, as the site reads them.

A run is any folder under the outputs directory that holds a reports.jsonl. The store lists
runs, turns reports into table rows, and keeps the admin decisions and analyst sign-offs in
the run folder. A run id is the folder's relative path with "/" written as "~"; an id is only
ever looked up in the list of folders found, so no id can point outside the outputs directory.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from src.bank import BankData
from src.schemas import FORGERY_PAUSE_THRESHOLD, InvestigationReport


SORT_KEYS = ("account", "customer_name", "verdict", "confidence", "label")
VERDICT_ORDER = {"CLEAR": 0, "MONITOR": 1, "SUSPICIOUS-ESCALATE": 2, "BLOCKED_BY_ADMIN_REVIEW": 3}
DECISIONS = ("approve", "block")


class UnknownRun(KeyError):
    pass


class UnknownAccount(KeyError):
    pass


@dataclass
class _Loaded:
    stamp: float
    reports: dict[str, InvestigationReport]


class RunStore:
    def __init__(self, outputs: Path, bank: BankData) -> None:
        self.outputs = Path(outputs)
        self.bank = bank
        self._cache: dict[str, _Loaded] = {}

    # ----------------------------------------------------------------------------------
    # Runs
    # ----------------------------------------------------------------------------------

    def folders(self) -> dict[str, Path]:
        """Run id -> folder, newest first; a run with a scoreboard comes before one without."""
        if not self.outputs.exists():
            return {}
        found = [p.parent for p in self.outputs.rglob("reports.jsonl")]
        found.sort(
            key=lambda p: ((p / "scoreboard.json").exists(), p.stat().st_mtime), reverse=True
        )
        return {p.relative_to(self.outputs).as_posix().replace("/", "~"): p for p in found}

    def folder(self, run_id: str) -> Path:
        try:
            return self.folders()[run_id]
        except KeyError:
            raise UnknownRun(run_id) from None

    def list_runs(self) -> list[dict]:
        runs = []
        for run_id, folder in self.folders().items():
            summary = _read(folder / "summary.json", {})
            board = _read(folder / "scoreboard.json", None)
            runs.append(
                {
                    "id": run_id,
                    "label": run_id.replace("~", " / "),
                    "accounts": summary.get("accounts"),
                    "set": board.get("set") if board else None,
                    "mode": board.get("mode") if board else None,
                    "has_scoreboard": board is not None,
                    "modified": int((folder / "reports.jsonl").stat().st_mtime),
                }
            )
        return runs

    def reports(self, run_id: str) -> dict[str, InvestigationReport]:
        path = self.folder(run_id) / "reports.jsonl"
        stamp = path.stat().st_mtime
        loaded = self._cache.get(run_id)
        if loaded is None or loaded.stamp != stamp:
            reports = {}
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        report = InvestigationReport.model_validate_json(line)
                        reports[report.target_account] = report
            loaded = self._cache[run_id] = _Loaded(stamp, reports)
        return loaded.reports

    def report(self, run_id: str, account_id: str) -> InvestigationReport:
        try:
            return self.reports(run_id)[account_id]
        except KeyError:
            raise UnknownAccount(account_id) from None

    def summary(self, run_id: str) -> dict:
        folder = self.folder(run_id)
        return {
            "run": run_id,
            "summary": _read(folder / "summary.json", {}),
            "scoreboard": _read(folder / "scoreboard.json", None),
        }

    def verdicts(self, run_id: str) -> dict[str, str]:
        return {a: r.disposition.value for a, r in self.reports(run_id).items()}

    # ----------------------------------------------------------------------------------
    # Rows
    # ----------------------------------------------------------------------------------

    def all_rows(self, run_id: str) -> list[dict]:
        decisions, signed = self.decisions(run_id), self.signed(run_id)
        return [self._row(r, decisions, signed) for _, r in sorted(self.reports(run_id).items())]

    def _row(self, r: InvestigationReport, decisions: dict, signed: set[str]) -> dict:
        customer = self.bank.customer_for(r.target_account)
        vision = r.vision_findings
        return {
            "account": r.target_account,
            "case_id": r.case_id,
            "customer_name": customer.name,
            "occupation": customer.occupation,
            "verdict": r.disposition.value,
            "confidence": r.confidence,
            "label": r.graph_label or "",
            "id_pause": bool(vision and vision.forgery_confidence >= FORGERY_PAUSE_THRESHOLD),
            "inputs": dict(r.judge_inputs),
            "clauses": list(
                r.rulebook_findings.retrieved_clause_ids if r.rulebook_findings else []
            ),
            "signed": r.case_id in signed,
            "admin_decision": decisions.get(r.target_account),
        }

    def rows(
        self,
        run_id: str,
        *,
        verdicts: set[str] | None = None,
        label: str | None = None,
        id_pause: bool | None = None,
        q: str | None = None,
        sort: str = "account",
        order: str = "asc",
        page: int = 1,
        page_size: int = 25,
    ) -> dict:
        rows = self.all_rows(run_id)
        labels = sorted({r["label"] for r in rows if r["label"]})
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        rows = filter_rows(rows, verdicts=verdicts, label=label, id_pause=id_pause, q=q)
        rows = sort_rows(rows, sort, order)
        start = (max(page, 1) - 1) * page_size
        return {
            "rows": rows[start : start + page_size],
            "total": len(rows),
            "page": max(page, 1),
            "page_size": page_size,
            "labels": labels,
            "counts": counts,
        }

    # ----------------------------------------------------------------------------------
    # Admin decisions and sign-offs (saved in the run folder)
    # ----------------------------------------------------------------------------------

    def decisions(self, run_id: str) -> dict[str, str]:
        return _read(self.folder(run_id) / "admin_decisions.json", {})

    def set_decision(self, run_id: str, account_id: str, choice: str) -> None:
        if choice not in DECISIONS:
            raise ValueError(choice)
        self.report(run_id, account_id)
        decisions = self.decisions(run_id)
        decisions[account_id] = choice
        _write(self.folder(run_id) / "admin_decisions.json", decisions)

    def signed(self, run_id: str) -> set[str]:
        return set(_read(self.folder(run_id) / "signoffs.json", []))

    def sign(self, run_id: str, account_id: str) -> None:
        report = self.report(run_id, account_id)
        signed = self.signed(run_id) | {report.case_id}
        _write(self.folder(run_id) / "signoffs.json", sorted(signed))

    def admin_queue(self, run_id: str) -> list[dict]:
        decisions = self.decisions(run_id)
        queue = _read(self.folder(run_id) / "admin_queue.json", [])
        return [
            {
                **item,
                "batch_decision": item.get("decision"),
                "decision": decisions.get(item["account_id"]),
                "customer_name": self.bank.customer_for(item["account_id"]).name,
            }
            for item in queue
        ]


def filter_rows(
    rows: list[dict],
    *,
    verdicts: set[str] | None = None,
    label: str | None = None,
    id_pause: bool | None = None,
    q: str | None = None,
) -> list[dict]:
    needle = (q or "").strip().lower()
    out = []
    for r in rows:
        if verdicts and r["verdict"] not in verdicts:
            continue
        if label and r["label"] != label:
            continue
        if id_pause is not None and r["id_pause"] != id_pause:
            continue
        if needle and needle not in f"{r['account']} {r['customer_name']}".lower():
            continue
        out.append(r)
    return out


def sort_rows(rows: list[dict], sort: str, order: str) -> list[dict]:
    key = sort if sort in SORT_KEYS else "account"

    def value(r: dict):
        if key == "verdict":
            return (VERDICT_ORDER.get(r["verdict"], 9), r["account"])
        return (r[key], r["account"])

    return sorted(rows, key=value, reverse=order == "desc")


def _read(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _write(path: Path, value) -> None:
    """Write then rename, so a crash cannot leave half a file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
