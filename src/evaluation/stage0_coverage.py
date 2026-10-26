"""Stage 0 coverage against the answer key - how much planted evidence reaches the right stage.

    python -m src.evaluation.stage0_coverage

Runs Stage 0 on every account with fresh run memory and joins the routes with the ledger.
Each evidence channel is checked against the stage that can see it. A few hard checks fail
the run (exit non-zero); the rest is reported. Evaluation only.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.bank import BankData
from src.orchestration.run_state import RunState
from src.rulebook import RulebookLookup
from src.schemas import (
    Disposition,
    EvidenceLedgerEntry,
    SeverityTier,
    StageName,
    TriageEvent,
    TriageEventKind,
    TriageRule,
)
from src.stages.stage0_triage import TriageContext, triage
from src.stages.stage0_triage.config import NAME_MATCH_CUTOFF


LEDGER_PATH = Path("data") / "_ground_truth" / "evidence_ledger.json"

S = StageName
REAL_MEDIA_TIERS = {SeverityTier.CONFIRMED_SANCTIONS, SeverityTier.ADVERSE_NEWS}

#: Evidence channel -> (does this ledger entry carry it, the stage that can see it).
CHANNELS = {
    "structuring": (lambda e: e.structuring_planted, S.TRANSACTION_PATTERN),
    "pass-through": (lambda e: e.pass_through_planted, S.TRANSACTION_PATTERN),
    "KYC mismatch": (lambda e: e.kyc_mismatch_planted, S.KYC),
    "forged document": (lambda e: e.forged_document_planted, S.VISION),
    "adverse media": (lambda e: e.adverse_media_hit in REAL_MEDIA_TIERS, S.ADVERSE_MEDIA),
    "ring membership": (lambda e: e.graph_ring_membership is not None, S.GRAPH),
}


@dataclass
class Coverage:
    lines: list[str]
    failures: list[str]


def channels_of(entry: EvidenceLedgerEntry) -> list[str]:
    return [name for name, (carries, _) in CHANNELS.items() if carries(entry)]


def measure(bank: BankData, ledger: list[EvidenceLedgerEntry]) -> Coverage:
    ctx = TriageContext.build(bank, RulebookLookup.load())
    state = RunState()
    routes, rules = {}, {}
    for account_id in bank.account_ids():
        out = triage(
            TriageEvent(kind=TriageEventKind.PERIODIC_REVIEW, account_id=account_id), ctx, state
        )
        routes[account_id] = set(out.routed_stages)
        rules[account_id] = {f.rule for f in out.fired_rules}

    lines = ["Stage 0 coverage (periodic review, empty known-bad list)", ""]
    failures: list[str] = []

    # Evidence channels
    lines.append(f"  {'Channel':<17} {'planted':>7} {'reached':>7}  stage")
    reached_by: dict[str, tuple[int, int]] = {}
    for name, (carries, stage) in CHANNELS.items():
        planted = [e for e in ledger if carries(e)]
        reached = [e for e in planted if stage in routes[e.account_id]]
        reached_by[name] = (len(reached), len(planted))
        lines.append(f"  {name:<17} {len(planted):>7} {len(reached):>7}  {stage.value}")

    mules = [e for e in ledger if e.pass_through_planted]
    mules_fired = sum(TriageRule.PASS_THROUGH in rules[e.account_id] for e in mules)
    lines.append(f"  pass-through rule fired on {mules_fired} of {len(mules)} planted mules")

    # Classes
    lines += [
        "",
        "  By label (reached = routed to a weighted stage beyond the KYC review)",
    ]
    weighted_beyond_kyc = {S.TRANSACTION_PATTERN, S.ADVERSE_MEDIA, S.GRAPH}
    for label, test in (
        ("escalate-worthy", lambda e: e.expected_disposition == Disposition.SUSPICIOUS_ESCALATE),
        ("monitor-worthy", lambda e: e.expected_disposition == Disposition.MONITOR),
        ("clean", lambda e: e.expected_disposition == Disposition.CLEAR),
    ):
        group = [e for e in ledger if test(e)]
        beyond = sum(bool(routes[e.account_id] & weighted_beyond_kyc) for e in group)
        lines.append(f"  {label:<16} {len(group):>4} accounts, {beyond:>3} reached")

    # Ring members only the batch's re-screening rounds can reach
    ring_gap = [
        e
        for e in ledger
        if channels_of(e) == ["ring membership"] and S.GRAPH not in routes[e.account_id]
    ]
    lines += [
        "",
        f"  Reported gap: {len(ring_gap)} ring-only accounts are not routed to Graph on a first "
        "pass; the batch's re-screening rounds reach them through the known-bad list.",
    ]

    # Decoys
    lines += ["", "  Decoys routed to the stage that must clear them"]
    decoy_checks = {
        "name_similarity": (TriageRule.NAME_MATCH, S.ADVERSE_MEDIA),
        "merchant_lookalike": (TriageRule.PASS_THROUGH, S.TRANSACTION_PATTERN),
        "cross_border_legitimate": (TriageRule.CROSS_BORDER, S.TRANSACTION_PATTERN),
    }
    decoy_counts = Counter(d for e in ledger for d in e.decoy_types)
    for decoy, total in sorted(decoy_counts.items()):
        group = [e for e in ledger if decoy in e.decoy_types]
        if decoy in decoy_checks:
            rule, stage = decoy_checks[decoy]
            n = sum(rule in rules[e.account_id] for e in group)
            lines.append(f"  {decoy:<24} {n:>3} of {total:<3} fire {rule.value} -> {stage.value}")
        else:
            n = sum(bool(routes[e.account_id] & weighted_beyond_kyc) for e in group)
            lines.append(f"  {decoy:<24} {n:>3} of {total:<3} routed beyond the KYC review")

    # Name-similarity decoys under the real matcher
    scores = []
    for e in ledger:
        if "name_similarity" in e.decoy_types:
            match = ctx.names.best_match(bank.customer_for(e.account_id).name, cutoff=0)
            scores.append(match.score)
    lines += [
        "",
        f"  Name-similarity decoys, rapidfuzz score of the holder's best match: "
        f"min {min(scores):g}, max {max(scores):g}; "
        f"{sum(s >= NAME_MATCH_CUTOFF for s in scores)} at or above Stage 0's "
        f"{NAME_MATCH_CUTOFF:g}, "
        f"{sum(s >= 92 for s in scores)} at or above Stage 4's 92, "
        f"{sum(s == 100 for s in scores)} identical to the customer's name",
    ]

    # Hard checks
    structuring_reached, structuring_planted = reached_by["structuring"]
    if structuring_reached != structuring_planted:
        failures.append(
            f"structuring: {structuring_reached} of {structuring_planted} reach TransactionPattern"
        )
    if mules_fired < len(mules) - 1:
        failures.append(
            f"pass-through rule caught {mules_fired} of {len(mules)} mules (need >= 11)"
        )
    missing_review = [a for a, r in routes.items() if not {S.KYC, S.VISION} <= r]
    if missing_review:
        failures.append(f"{len(missing_review)} accounts not routed to KYC and Vision")

    lines += ["", "  Hard checks"]
    lines.append(
        f"  {'PASS' if structuring_reached == structuring_planted else 'FAIL'}  "
        f"structuring {structuring_reached}/{structuring_planted} -> TransactionPattern"
    )
    lines.append(
        f"  {'PASS' if mules_fired >= len(mules) - 1 else 'FAIL'}  "
        f"pass-through rule on {mules_fired}/{len(mules)} mules (>= {len(mules) - 1})"
    )
    lines.append(
        f"  {'FAIL' if missing_review else 'PASS'}  "
        f"every account -> KYC + Vision ({len(routes) - len(missing_review)}/{len(routes)})"
    )
    return Coverage(lines=lines, failures=failures)


def load_ledger(path: Path = LEDGER_PATH) -> list[EvidenceLedgerEntry]:
    with path.open(encoding="utf-8") as f:
        return [EvidenceLedgerEntry(**r) for r in json.load(f)]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    coverage = measure(BankData.load(), load_ledger())
    print("\n".join(coverage.lines))
    if coverage.failures:
        print("\nFAILED: " + "; ".join(coverage.failures))
        return 1
    print("\nAll hard checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
