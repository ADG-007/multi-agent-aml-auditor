"""Label audit and difficulty report for the dataset (Amendment 4, V4 and V6).

    python -m src.evaluation.label_audit

V4 - the answer key against the raw data:

- the labels are 49 escalate-worthy, 73 monitor-worthy and 248 clean;
- the labels the scoring design cannot reach: the forged-document channel feeds no weighted
  stage (invariant 2), so an account whose label needs it cannot be scored up to it;
- every planted channel is visible in the raw data, checked from the data files only, without
  any pipeline stage.

V6 - how hard the data is, reported only: the class mix, and how close each decoy comes to the
line its look-alike crosses.

Reads data/_ground_truth/, so pipeline code never imports this module. Exits non-zero if a V4
check fails. The forged-document check reads the rendered images with Tesseract; it is skipped,
with a note, if the images or Tesseract are missing.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from src.generators.rings import MULE_NETWORK_COLLECTOR_IDS, SOLO_MULE_COLLECTOR_IDS
from src.schemas import (
    Customer,
    Disposition,
    EvidenceLedgerEntry,
    FraudRing,
    SeverityTier,
    Transaction,
)


DATA = Path("data")
GROUND_TRUTH = DATA / "_ground_truth"

#: The labels the amendment must keep (49 / 73 / 248).
EXPECTED_LABELS = {
    Disposition.SUSPICIOUS_ESCALATE: 49,
    Disposition.MONITOR: 73,
    Disposition.CLEAR: 248,
}
#: The band and thresholds, from the committed rulebook (STRUCT-001, XB-004).
CASH = ("CASH_DEPOSIT", "CASH_WITHDRAWAL")


@dataclass
class Audit:
    lines: list[str]
    failures: list[str]


def _load(name: str, model) -> list:
    with (DATA / name).open(encoding="utf-8") as f:
        return [model(**r) for r in json.load(f)]


def _ground_truth(name: str, model) -> list:
    with (GROUND_TRUTH / name).open(encoding="utf-8") as f:
        return [model(**r) for r in json.load(f)]


def _clauses() -> dict[str, dict]:
    with (DATA / "rulebook" / "clauses.json").open(encoding="utf-8") as f:
        return {c["clause_id"]: c["metadata"] for c in json.load(f)}


def label_without_forgery(e: EvidenceLedgerEntry) -> Disposition:
    """The label the six channels give once the forged-document channel is taken away."""
    return e.model_copy(update={"forged_document_planted": False}).expected_disposition


def _month(moment: datetime):
    return moment.date().replace(day=1)


def _monthly_band_hit(rows: list[Transaction], floor: float, ceiling: float) -> bool:
    totals: dict = defaultdict(float)
    for t in rows:
        totals[_month(t.timestamp)] += t.amount
    if any(floor <= v <= ceiling for v in totals.values()):
        return True
    window = timedelta(days=30)
    for i, start in enumerate(rows):
        inside = [t for t in rows[i:] if t.timestamp < start.timestamp + window]
        if sum(t.amount for t in inside) >= floor and all(
            totals[_month(t.timestamp)] <= ceiling for t in inside
        ):
            return True
    return False


def _forged_visible(
    customers: list[Customer], ledger: list[EvidenceLedgerEntry]
) -> tuple[int, int] | None:
    """How many forged documents print a date of birth that contradicts the record."""
    try:
        import pytesseract
        from PIL import Image

        pytesseract.get_tesseract_version()
    except Exception:
        return None
    by_id = {c.customer_id: c for c in customers}
    forged = [e for e in ledger if e.forged_document_planted]
    seen = 0
    for e in forged:
        c = by_id[e.customer_id]
        path = DATA / "documents" / f"{c.customer_id}_{c.id_document_type.value}.png"
        if not path.exists():
            return None
        text = " ".join(pytesseract.image_to_string(Image.open(path)).split())
        seen += c.dob.strftime("%d/%m/%Y") not in text
    return seen, len(forged)


def audit() -> Audit:
    customers = _load("customers.json", Customer)
    transactions = _load("transactions.json", Transaction)
    ledger = _ground_truth("evidence_ledger.json", EvidenceLedgerEntry)
    rings = _ground_truth("fraud_rings.json", FraudRing)
    with (DATA / "adverse_media.json").open(encoding="utf-8") as f:
        media = json.load(f)
    meta = _clauses()
    threshold = meta["STRUCT-001"]["threshold_inr"]
    floor = threshold * meta["STRUCT-001"]["near_threshold_pct"]
    monthly_ceiling = meta["STRUCT-001"]["monthly_threshold_inr"]
    monthly_floor = monthly_ceiling * meta["STRUCT-001"]["monthly_near_threshold_pct"]
    wire_threshold = meta["XB-004"]["threshold_inr"]

    rows: dict[str, list[Transaction]] = defaultdict(list)
    for t in transactions:
        rows[t.account_id].append(t)
    for r in rows.values():
        r.sort(key=lambda t: t.timestamp)
    income = {c.customer_id: c.stated_income for c in customers}
    lines = ["Label audit (V4)", ""]
    failures: list[str] = []

    def report(ok: bool, text: str) -> None:
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {text}")
        if not ok:
            failures.append(text)

    # Labels
    labels = Counter(e.expected_disposition for e in ledger)
    report(
        all(labels[d] == n for d, n in EXPECTED_LABELS.items()),
        f"labels {labels[Disposition.SUSPICIOUS_ESCALATE]} / {labels[Disposition.MONITOR]} / "
        f"{labels[Disposition.CLEAR]} (escalate / monitor / clear), expected 49 / 73 / 248",
    )

    # Labels the design cannot reach
    unreachable = [e for e in ledger if label_without_forgery(e) != e.expected_disposition]
    moves = Counter(
        f"{e.expected_disposition.value} -> {label_without_forgery(e).value}" for e in unreachable
    )
    lines.append(
        f"  ----  {len(unreachable)} labels the scoring design cannot reach (forged document feeds "
        f"no weighted stage): " + ", ".join(f"{n} {k}" for k, n in sorted(moves.items()))
    )

    # Every planted channel visible in the raw data
    ring_devices = {m: r.shared_device_id for r in rings for m in r.member_account_ids}
    collectors = set(MULE_NETWORK_COLLECTOR_IDS) | set(SOLO_MULE_COLLECTOR_IDS)
    linked = {m["linked_actor_id"] for m in media if m["linked_actor_id"]}
    turnover = {a: sum(t.amount for t in r) * 365 / 90 for a, r in rows.items()}
    checks = {
        "structuring": (
            lambda e: e.structuring_planted,
            lambda e: (
                any(
                    t.type.value == "CASH_DEPOSIT" and floor <= t.amount <= threshold
                    for t in rows[e.account_id]
                )
                if e.structuring_style == "naive"
                else _monthly_band_hit(
                    [t for t in rows[e.account_id] if t.type.value == "CASH_DEPOSIT"],
                    monthly_floor,
                    monthly_ceiling,
                )
            ),
            "naive: a cash deposit in the band; others: a cash month in the band",
        ),
        "KYC mismatch": (
            lambda e: e.kyc_mismatch_planted,
            lambda e: turnover[e.account_id] / income[e.customer_id] > 6,
            "turnover above 6x declared income",
        ),
        "adverse media": (
            lambda e: e.adverse_media_hit not in (None, SeverityTier.NAME_SIMILARITY_ONLY),
            lambda e: e.customer_id in linked,
            "a corpus entry linked to the customer",
        ),
        "ring membership": (
            lambda e: e.graph_ring_membership is not None,
            lambda e: any(t.device_id == ring_devices[e.account_id] for t in rows[e.account_id]),
            "the ring's shared phone on the account's own payments",
        ),
        "pass-through": (
            lambda e: e.pass_through_planted,
            lambda e: any(
                t.direction.value == "DEBIT" and t.counterparty_id in collectors
                for t in rows[e.account_id]
            ),
            "money sent on to a mule collector",
        ),
    }
    for name, (planted, visible, how) in checks.items():
        group = [e for e in ledger if planted(e)]
        seen = sum(bool(visible(e)) for e in group)
        report(seen == len(group), f"{name}: {seen}/{len(group)} visible in the raw data ({how})")

    forged = _forged_visible(customers, ledger)
    if forged is None:
        lines.append("  skip  forged document: images or Tesseract not available")
    else:
        seen, total = forged
        report(
            seen == total,
            f"forged document: {seen}/{total} visible (printed date of birth "
            "contradicts the record, read by Tesseract)",
        )

    lines += ["", "Difficulty report (V6, reported only)", ""]
    lines += _difficulty(ledger, rows, labels, threshold, floor, monthly_floor, wire_threshold)
    return Audit(lines=lines, failures=failures)


def _difficulty(ledger, rows, labels, threshold, floor, monthly_floor, wire_threshold) -> list[str]:
    total = len(ledger)
    out = [
        "  Class mix: "
        + ", ".join(
            f"{d.value} {labels[d]} ({labels[d] / total:.1%})"
            for d in (Disposition.SUSPICIOUS_ESCALATE, Disposition.MONITOR, Disposition.CLEAR)
        ),
    ]
    decoys = Counter(d for e in ledger for d in e.decoy_types)
    out.append(
        f"  Decoys: {sum(1 for e in ledger if e.decoy_types)} accounts carry one, "
        + ", ".join(f"{k} {v}" for k, v in sorted(decoys.items()))
    )

    def by(tag):
        return [e for e in ledger if tag in e.decoy_types]

    def month_max(a):
        totals: dict = defaultdict(float)
        for t in rows[a]:
            if t.type.value == "CASH_DEPOSIT":
                totals[_month(t.timestamp)] += t.amount
        return max(totals.values(), default=0.0)

    cash_months = [month_max(e.account_id) for e in by("cash_business")]
    out.append(
        f"  Cash businesses: busiest month of cash deposits {_lakh(min(cash_months))}–"
        f"{_lakh(max(cash_months))} (band starts at {_lakh(monthly_floor)}; "
        f"{sum(m >= monthly_floor for m in cash_months)} of {len(cash_months)} reach it)"
    )
    round_max = [max(t.amount for t in rows[e.account_id]) for e in by("round_amount")]
    out.append(
        f"  Round-amount decoys: largest single payment {_lakh(min(round_max))}–"
        f"{_lakh(max(round_max))} (band starts at {_lakh(floor)})"
    )
    wires = [
        t.amount
        for e in by("cross_border_legitimate")
        for t in rows[e.account_id]
        if t.type.value == "SWIFT"
    ]
    out.append(
        f"  Legitimate cross-border: {sum(w > wire_threshold for w in wires)} of {len(wires)} "
        f"wires above the {_lakh(wire_threshold)} wire report"
    )
    others = [
        month_max(a)
        for a in rows
        if not any(
            e.account_id == a and (e.structuring_planted or "cash_business" in e.decoy_types)
            for e in ledger
        )
    ]
    out.append(
        f"  Everyone else: busiest month of cash deposits up to {_lakh(max(others))}; "
        f"{sum(m >= 0.8 * monthly_floor for m in others)} account(s) at 80% of the band floor "
        "or more"
    )
    structurer_months = []
    per_month = []
    for e in ledger:
        if e.structuring_style in (None, "naive"):
            continue
        totals: dict = defaultdict(list)
        for t in rows[e.account_id]:
            if t.type.value == "CASH_DEPOSIT":
                totals[_month(t.timestamp)].append(t.amount)
        for amounts in totals.values():
            if monthly_floor <= sum(amounts) <= threshold:
                structurer_months.append(sum(amounts))
                per_month.append(len(amounts))
    out.append(
        f"  Monthly-cap structurers: {len(structurer_months)} band months, totals "
        f"{_lakh(min(structurer_months))}–{_lakh(max(structurer_months))}, "
        f"{min(per_month)}–{max(per_month)} deposits a month "
        f"(median {statistics.median(per_month):g})"
    )
    return out


def _lakh(amount: float) -> str:
    return f"₹{amount / 100_000:.2f} lakh"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    result = audit()
    print("\n".join(result.lines))
    if result.failures:
        print("\nFAILED: " + "; ".join(result.failures))
        return 1
    print("\nAll V4 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
