"""Stage 2's test subset, picked by rule from the answer key and saved once.

    python -m src.evaluation.stage2_subset

Phases 3-7 are build-tested on small per-stage subsets plus one shared set of linked accounts
(decisions_and_progress.md Part 11); the full 370-account run happens once, in Phase 11. This
script picks Stage 2's accounts (phase_3_execution.md D7) and writes them to
data/_ground_truth/stage_subsets.json, in the answer-key folder no pipeline module reads.

The rule, so the pick is reproducible and nobody chooses accounts by hand:

1. The shared set: every member of RING-04 (one structurer, one KYC-mismatch member) and the
   two lowest-id members of RING-12 (mules that share one collector).
2. Each slot, in order, takes a shared account that fits it, else the lowest account id that
   fits; an account fills at most one slot. The k = 4 structurer slot prefers an account whose
   only real channel is structuring.

`k` is computed with Stage 2's own structuring code at the bank's review date. Running the
script twice writes the same file. Other stages' keys in the file are left as they are.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from src.bank import BankData
from src.rulebook import RulebookLookup
from src.schemas import EvidenceLedgerEntry, FraudRing, RingType, TransactionDirection
from src.stages.stage2_transaction_pattern import structuring


GROUND_TRUTH = Path("data") / "_ground_truth"
LEDGER_PATH = GROUND_TRUTH / "evidence_ledger.json"
RINGS_PATH = GROUND_TRUTH / "fraud_rings.json"
SUBSET_PATH = GROUND_TRUTH / "stage_subsets.json"

SHARED_RING = "RING-04"
SHARED_MULE_RING = "RING-12"
SHARED_MULE_MEMBERS = 2


@dataclass(frozen=True)
class Truth:
    """What the answer key says about one account, plus Stage 2's own k for the slot rules."""

    entry: EvidenceLedgerEntry
    k: int
    #: The account paid a MULE ring's collector: a complicit merchant when it is in no ring.
    pays_mule_collector: bool

    @property
    def structuring_only(self) -> bool:
        return self.entry.structuring_planted and self.entry.real_channel_count == 1

    @property
    def role(self) -> str:
        """The account's Stage 2 role in the answer key (first match wins)."""
        e = self.entry
        if e.structuring_planted:
            return "structurer"
        if e.pass_through_planted:
            if e.graph_ring_membership is not None:
                return "network_mule"
            return "complicit_merchant" if self.pays_mule_collector else "solo_mule"
        for decoy in (
            "merchant_lookalike",
            "life_event",
            "round_amount",
            "cross_border_legitimate",
        ):
            if decoy in e.decoy_types:
                return decoy
        return "other"


def _is(role: str) -> Callable[[Truth], bool]:
    return lambda t: t.role == role


def _structurer(k: int) -> Callable[[Truth], bool]:
    return lambda t: t.role == "structurer" and t.k == k


#: The slots, in filling order (D7): (slot name, fits).
SLOTS: tuple[tuple[str, Callable[[Truth], bool]], ...] = (
    ("structurer_k4", _structurer(4)),
    ("structurer_k6", _structurer(6)),
    ("structurer_k8", _structurer(8)),
    ("network_mule", _is("network_mule")),
    ("solo_mule", _is("solo_mule")),
    ("solo_mule", _is("solo_mule")),
    ("complicit_merchant", _is("complicit_merchant")),
    ("merchant_lookalike", _is("merchant_lookalike")),
    ("merchant_lookalike", _is("merchant_lookalike")),
    ("life_event", _is("life_event")),
    ("round_amount", _is("round_amount")),
    ("cross_border_legitimate", _is("cross_border_legitimate")),
)

#: The slot that prefers a structuring-only account.
PREFERS_STRUCTURING_ONLY = "structurer_k4"


def load_truth(
    bank: BankData, rulebook: RulebookLookup, account_ids: Iterable[str] | None = None
) -> dict[str, Truth]:
    """The answer key's view of `account_ids` (default: every account)."""
    ledger = load_ledger()
    collectors = {
        r.collector_counterparty_id
        for r in load_rings()
        if r.ring_type == RingType.MULE and r.collector_counterparty_id
    }
    as_of = bank.review_date()
    truth = {}
    for account_id in sorted(account_ids or bank.account_ids()):
        rows = bank.transactions(account_id)
        truth[account_id] = Truth(
            entry=ledger[account_id],
            k=structuring.score(rows, as_of, rulebook).k,
            pays_mule_collector=any(
                t.direction == TransactionDirection.DEBIT and t.counterparty_id in collectors
                for t in rows
            ),
        )
    return truth


def pick(truth: dict[str, Truth]) -> dict[str, list]:
    rings = {r.ring_id: r for r in load_rings()}
    shared = (
        sorted(rings[SHARED_RING].member_account_ids)
        + sorted(rings[SHARED_MULE_RING].member_account_ids)[:SHARED_MULE_MEMBERS]
    )

    taken: set[str] = set()
    stage2 = []
    for slot, fits in SLOTS:
        candidates = [a for a in shared if a not in taken and fits(truth[a])]
        if not candidates:
            candidates = [a for a in sorted(truth) if a not in taken and fits(truth[a])]
        if slot == PREFERS_STRUCTURING_ONLY:
            preferred = [a for a in candidates if truth[a].structuring_only]
            candidates = preferred or candidates
        if not candidates:
            raise ValueError(f"no account fits slot {slot}")
        taken.add(candidates[0])
        stage2.append({"account_id": candidates[0], "slot": slot})

    structurers = [s["account_id"] for s in stage2 if s["slot"].startswith("structurer")]
    if not any(truth[a].structuring_only for a in structurers):
        raise ValueError("no structuring-only account among the structurer slots")
    return {"shared": shared, "stage2": stage2}


def save(subsets: dict[str, list], path: Path = SUBSET_PATH) -> None:
    """Write Stage 2's keys, keeping any key another stage has added."""
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing.update(subsets)
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8", newline="\n")


def load_subsets(path: Path = SUBSET_PATH) -> dict[str, list]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ledger(path: Path = LEDGER_PATH) -> dict[str, EvidenceLedgerEntry]:
    with path.open(encoding="utf-8") as f:
        return {r["account_id"]: EvidenceLedgerEntry(**r) for r in json.load(f)}


def load_rings(path: Path = RINGS_PATH) -> list[FraudRing]:
    with path.open(encoding="utf-8") as f:
        return [FraudRing(**r) for r in json.load(f)]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    truth = load_truth(BankData.load(), RulebookLookup.load())
    subsets = pick(truth)
    save(subsets)
    print(f"Shared set: {', '.join(subsets['shared'])}")
    print("Stage 2 subset:")
    for s in subsets["stage2"]:
        t = truth[s["account_id"]]
        print(
            f"  {s['account_id']}  {s['slot']:<24} k={t.k}  channels={t.entry.real_channel_count}"
        )
    print(f"\nWritten to {SUBSET_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
