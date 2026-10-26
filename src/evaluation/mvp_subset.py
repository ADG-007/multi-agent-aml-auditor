"""The 50 test accounts, checked against the answer key and saved once.

    python -m src.evaluation.mvp_subset

29 named accounts (20 bad, 9 traps) are checked against the ledger, then 21 clean ones are
drawn with a fixed seed. Where the ledger label needs the ID check to score (0008, 0121), the
group's own answer is used and the ledger label kept beside it. Writes
data/_ground_truth/mvp_accounts.json. Same output every run.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.schemas import Disposition, EvidenceLedgerEntry, SeverityTier


GROUND_TRUTH = Path("data") / "_ground_truth"
LEDGER_PATH = GROUND_TRUTH / "evidence_ledger.json"
SUBSET_PATH = GROUND_TRUTH / "mvp_accounts.json"

CLEAN_COUNT = 21
CLEAN_SEED = 42

E, M, C = Disposition.SUSPICIOUS_ESCALATE, Disposition.MONITOR, Disposition.CLEAR
Check = Callable[[EvidenceLedgerEntry], bool]


@dataclass(frozen=True)
class Group:
    name: str
    kind: str  # "bad" or "trap"
    accounts: tuple[str, ...]
    answer: Disposition
    check: Check
    show_case: str | None = None
    mvp_answer: Disposition | None = None  # set only where the ledger label cannot be reached
    why: str = ""


def _ring(ring_id: str) -> Check:
    return lambda e: e.graph_ring_membership == ring_id


def _decoy(decoy: str) -> Check:
    return lambda e: decoy in e.decoy_types


def _both(a: Check, b: Check) -> Check:
    return lambda e: a(e) and b(e)


def _only(e: EvidenceLedgerEntry) -> bool:
    return e.real_channel_count == 1


_news: Check = lambda e: e.adverse_media_hit == SeverityTier.ADVERSE_NEWS  # noqa: E731
_sanctions: Check = lambda e: e.adverse_media_hit == SeverityTier.CONFIRMED_SANCTIONS  # noqa: E731
_struct: Check = lambda e: e.structuring_planted  # noqa: E731
_kyc: Check = lambda e: e.kyc_mismatch_planted  # noqa: E731
_forged: Check = lambda e: e.forged_document_planted  # noqa: E731
_mule: Check = lambda e: e.pass_through_planted  # noqa: E731
_r01 = _ring("RING-01")

_NO_ID_SCORE = (
    "The ID check is never scored (invariant 2): after the admin approves, it adds nothing."
)

#: The named groups.
GROUPS: tuple[Group, ...] = (
    Group("RING-01 news + ring", "bad", ("ACC-0022",), E, _both(_r01, _news)),
    Group("RING-01 ring only", "bad", ("ACC-0045",), M, _both(_r01, _only)),
    Group(
        "RING-01 ring + fake ID",
        "bad",
        ("ACC-0121",),
        E,
        _both(_r01, _forged),
        mvp_answer=M,
        why=_NO_ID_SCORE + " Left: the ring link alone, as for ACC-0045.",
    ),
    Group("RING-01 ring + structuring", "bad", ("ACC-0154",), E, _both(_r01, _struct)),
    Group(
        "Mule ring RING-12",
        "bad",
        ("ACC-0347", "ACC-0366", "ACC-0367", "ACC-0368"),
        E,
        _both(_ring("RING-12"), _mule),
    ),
    Group("Structurer", "bad", ("ACC-0037", "ACC-0068", "ACC-0125"), M, _both(_struct, _only)),
    Group("Structurer + KYC lie", "bad", ("ACC-0044",), E, _both(_struct, _kyc)),
    Group("KYC lie", "bad", ("ACC-0036", "ACC-0055"), M, _both(_kyc, _only)),
    Group("Sanctions", "bad", ("ACC-0074", "ACC-0171"), E, _sanctions),
    Group("Bad news", "bad", ("ACC-0034",), M, _both(_news, _only)),
    Group("Fake ID", "bad", ("ACC-0004",), M, _both(_forged, _only)),
    Group(
        "Fake ID + KYC lie",
        "bad",
        ("ACC-0008",),
        E,
        _both(_forged, _kyc),
        mvp_answer=M,
        why=_NO_ID_SCORE + " Left: the KYC lie alone, and one stage never escalates alone.",
    ),
    Group("Solo mule", "bad", ("ACC-0352",), M, _both(_mule, _only)),
    Group(
        "Name like a listed name", "trap", ("ACC-0010", "ACC-0025"), C, _decoy("name_similarity")
    ),
    Group("Merchant lookalike", "trap", ("ACC-0346", "ACC-0350"), C, _decoy("merchant_lookalike")),
    Group("Life event", "trap", ("ACC-0345",), C, _decoy("life_event")),
    Group("Round amounts", "trap", ("ACC-0003",), C, _decoy("round_amount")),
    Group("Legit cross-border", "trap", ("ACC-0301",), C, _decoy("cross_border_legitimate")),
    Group(
        "Shared device by chance", "trap", ("ACC-0007", "ACC-0012"), C, _decoy("graph_coincidence")
    ),
)

#: The show cases.
SHOW_CASES: dict[str, str] = {
    "A": "ACC-0044",
    "B": "ACC-0045",
    "C": "ACC-0346",
    "D": "ACC-0121",
    "STAR": "ACC-0074",
}


def is_clean(e: EvidenceLedgerEntry) -> bool:
    return e.real_channel_count == 0 and not e.decoy_types and e.graph_ring_membership is None


def build(ledger: dict[str, EvidenceLedgerEntry]) -> tuple[dict, list[str]]:
    """The subset file's content, and every problem found while checking the plan's table."""
    problems: list[str] = []
    rows: list[dict] = []
    for g in GROUPS:
        for account_id in g.accounts:
            entry = ledger.get(account_id)
            if entry is None:
                problems.append(f"{account_id}: not in the ledger")
                continue
            if not g.check(entry):
                problems.append(f"{account_id}: ledger does not fit group '{g.name}'")
            if entry.expected_disposition != g.answer:
                problems.append(
                    f"{account_id}: plan says {g.answer.value}, ledger label is "
                    f"{entry.expected_disposition.value}"
                )
            row = {
                "account_id": account_id,
                "kind": g.kind,
                "group": g.name,
                "expected": (g.mvp_answer or entry.expected_disposition).value,
            }
            if g.mvp_answer is not None:
                row |= {"ledger": entry.expected_disposition.value, "why": g.why}
            rows.append(row)

    named = {r["account_id"] for r in rows}
    pool = sorted(a for a, e in ledger.items() if is_clean(e) and a not in named)
    clean = sorted(random.Random(CLEAN_SEED).sample(pool, CLEAN_COUNT))
    rows += [
        {"account_id": a, "kind": "clean", "group": "Clean", "expected": C.value} for a in clean
    ]
    rows.sort(key=lambda r: r["account_id"])
    return {"seed": CLEAN_SEED, "show_cases": SHOW_CASES, "accounts": rows}, problems


def load_ledger(path: Path = LEDGER_PATH) -> dict[str, EvidenceLedgerEntry]:
    with path.open(encoding="utf-8") as f:
        return {r["account_id"]: EvidenceLedgerEntry(**r) for r in json.load(f)}


def load_subset(path: Path = SUBSET_PATH) -> dict:
    """The saved subset. Evaluation and the CLI read it; pipeline stages never do."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    subset, problems = build(load_ledger())
    if problems:
        print("The plan's table does not match the answer key:")
        print("\n".join(f"  - {p}" for p in problems))
        return 1

    SUBSET_PATH.write_text(json.dumps(subset, indent=2) + "\n", encoding="utf-8")
    rows = subset["accounts"]
    kinds = Counter(r["kind"] for r in rows)
    answers = Counter(r["expected"] for r in rows)
    print(f"Saved {len(rows)} accounts to {SUBSET_PATH}")
    print(f"  bad {kinds['bad']} + traps {kinds['trap']} + clean {kinds['clean']}")
    print(
        f"  {answers[E.value]} ESCALATE + {answers[M.value]} MONITOR + "
        f"{answers[C.value]} CLEAR = {len(rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
