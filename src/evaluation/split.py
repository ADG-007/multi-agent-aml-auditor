"""The 100 / 270 split: a tuning set with a fixed mix, picked by rule from the answer key.

    python -m src.evaluation.split

The 100 accounts are tuned on; the other 270 are graded once, at the end. No account moves
between the two. The rule, so nobody picks an account by hand:

1. Whole rings: RING-01, then the lowest ring id of every other ring type.
2. Named slots (escalate and monitor accounts outside rings, the two cash-business traps):
   the lowest account ids that fit the slot.
3. Bulk groups (the other trap kinds, clean accounts): a seeded sample.

The file is written to data/_ground_truth/v2_split.json, where no pipeline module reads.
Running the script twice writes the same file.
"""

from __future__ import annotations

import json
import random
import sys
from collections.abc import Callable
from pathlib import Path

from src.bank import BankData
from src.evaluation.stage2_subset import GROUND_TRUTH, Truth, load_rings, load_truth
from src.rulebook import RulebookLookup
from src.schemas import Disposition, SeverityTier


SPLIT_PATH = GROUND_TRUTH / "v2_split.json"
SEED = 42
TUNING_SIZE = 100
FIRST_RING = "RING-01"

Fits = Callable[[Truth], bool]


def _channels(t: Truth) -> frozenset[str]:
    e = t.entry
    found = set()
    if e.structuring_planted:
        found.add("structuring")
    if e.kyc_mismatch_planted:
        found.add("kyc")
    if e.adverse_media_hit == SeverityTier.ADVERSE_NEWS:
        found.add("news")
    if e.adverse_media_hit == SeverityTier.CONFIRMED_SANCTIONS:
        found.add("sanctions")
    if e.forged_document_planted:
        found.add("fake_id")
    if e.pass_through_planted:
        found.add("pass_through")
    return frozenset(found)


def _only(*channels: str) -> Fits:
    want = frozenset(channels)
    return lambda t: t.entry.graph_ring_membership is None and _channels(t) == want


def _structurer(style: str) -> Fits:
    return lambda t: _only("structuring")(t) and t.entry.structuring_style == style


def _solo_mule(t: Truth) -> bool:
    return _only("pass_through")(t) and not t.pays_mule_collector


def _decoy(kind: str, without: str | None = None) -> Fits:
    return lambda t: kind in t.entry.decoy_types and (without not in t.entry.decoy_types)


def _cash_business(merchant: bool) -> Fits:
    return lambda t: (
        "cash_business" in t.entry.decoy_types
        and ("merchant_lookalike" in t.entry.decoy_types) == merchant
    )


def _clean(t: Truth) -> bool:
    return t.entry.expected_disposition == Disposition.CLEAR and not t.entry.decoy_types


#: (group, slot, how many, fits), filled by lowest account id.
NAMED: tuple[tuple[str, str, int, Fits], ...] = (
    ("escalate", "structuring+kyc", 1, _only("structuring", "kyc")),
    ("escalate", "kyc+fake_id", 1, _only("kyc", "fake_id")),
    ("escalate", "structuring+news", 1, _only("structuring", "news")),
    ("escalate", "sanctions", 2, _only("sanctions")),
    ("monitor", "kyc", 5, _only("kyc")),
    ("monitor", "structuring_naive", 1, _structurer("naive")),
    ("monitor", "structuring_small", 1, _structurer("small")),
    ("monitor", "structuring_medium", 1, _structurer("medium")),
    ("monitor", "structuring_mixed", 1, _structurer("mixed")),
    ("monitor", "news", 2, _only("news")),
    ("monitor", "fake_id", 1, _only("fake_id")),
    ("monitor", "solo_mule", 1, _solo_mule),
    ("trap", "cash_business_merchant", 1, _cash_business(merchant=True)),
    ("trap", "cash_business_shop", 1, _cash_business(merchant=False)),
)

#: (group, slot, how many, fits), filled by a seeded sample.
SAMPLED: tuple[tuple[str, str, int, Fits], ...] = (
    ("trap", "cross_border_legitimate", 6, _decoy("cross_border_legitimate")),
    ("trap", "round_amount", 4, _decoy("round_amount")),
    ("trap", "name_similarity", 4, _decoy("name_similarity")),
    ("trap", "graph_coincidence", 3, _decoy("graph_coincidence")),
    ("trap", "merchant_lookalike", 2, _decoy("merchant_lookalike", without="cash_business")),
    ("trap", "life_event", 1, _decoy("life_event")),
    ("clean", "clean", 44, _clean),
)


def pick(truth: dict[str, Truth]) -> dict:
    """The split, as the saved file holds it."""
    rng = random.Random(SEED)
    tuning: list[dict[str, str]] = []
    taken: set[str] = set()

    def take(account_id: str, group: str, slot: str) -> None:
        taken.add(account_id)
        tuning.append({"account_id": account_id, "group": group, "slot": slot})

    rings = sorted(load_rings(), key=lambda r: r.ring_id)
    chosen = [next(r for r in rings if r.ring_id == FIRST_RING)]
    for ring in rings:
        if ring.ring_type not in {r.ring_type for r in chosen}:
            chosen.append(ring)
    for ring in chosen:
        for account_id in sorted(ring.member_account_ids):
            take(account_id, "ring", f"{ring.ring_id} {ring.ring_type.value}")

    for group, slot, count, fits in NAMED:
        fit = [a for a in sorted(truth) if a not in taken and fits(truth[a])]
        if len(fit) < count:
            raise ValueError(f"slot {slot} needs {count} accounts, {len(fit)} fit")
        for account_id in fit[:count]:
            take(account_id, group, slot)

    for group, slot, count, fits in SAMPLED:
        fit = [a for a in sorted(truth) if a not in taken and fits(truth[a])]
        if len(fit) < count:
            raise ValueError(f"slot {slot} needs {count} accounts, {len(fit)} fit")
        for account_id in sorted(rng.sample(fit, count)):
            take(account_id, group, slot)

    if len(tuning) != TUNING_SIZE:
        raise ValueError(f"the tuning set has {len(tuning)} accounts, not {TUNING_SIZE}")
    return {
        "seed": SEED,
        "tuning": sorted(tuning, key=lambda s: s["account_id"]),
        "held_out": sorted(a for a in truth if a not in taken),
    }


def save(split: dict, path: Path = SPLIT_PATH) -> None:
    path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8", newline="\n")


def load_split(path: Path = SPLIT_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tuning_ids(path: Path = SPLIT_PATH) -> list[str]:
    return [s["account_id"] for s in load_split(path)["tuning"]]


def held_out_ids(path: Path = SPLIT_PATH) -> list[str]:
    return list(load_split(path)["held_out"])


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    truth = load_truth(BankData.load(), RulebookLookup.load())
    split = pick(truth)
    save(split)
    groups: dict[str, int] = {}
    for s in split["tuning"]:
        groups[s["group"]] = groups.get(s["group"], 0) + 1
    labels: dict[str, int] = {}
    for s in split["tuning"]:
        label = truth[s["account_id"]].entry.expected_disposition.value
        labels[label] = labels.get(label, 0) + 1
    print(f"Tuning set: {len(split['tuning'])} accounts; held out: {len(split['held_out'])}")
    print("  by group: " + ", ".join(f"{g} {n}" for g, n in sorted(groups.items())))
    print("  by label: " + ", ".join(f"{g} {n}" for g, n in sorted(labels.items())))
    print(f"Written to {SPLIT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
