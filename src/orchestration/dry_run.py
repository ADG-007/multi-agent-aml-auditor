"""Phase 2 dry run: review every account end to end with stubbed stages.

    python -m src.orchestration.dry_run                  # all 370 accounts
    python -m src.orchestration.dry_run --account ACC-0001   # one case, printed in full

Runs Phase 11's periodic-review mode (phase_2_execution.md): accounts in order of their
earliest alerting transaction (ties by account id, accounts nothing alerted on last), each
through Stage 0 -> stubs -> scoring -> report. Every report is written as one JSON line to
outputs/phase2_dry_run.jsonl.

With neutral stubs (every stage RAN at 0.0) every verdict is CLEAR - that is correct. The dry
run proves routing and plumbing on the real bank; scoring correctness is proven by the tests.
No model is called.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.bank import BankData
from src.orchestration.case_runner import run_case
from src.orchestration.run_state import RunState
from src.rulebook import RulebookLookup
from src.schemas import InvestigationReport, TriageEvent, TriageEventKind
from src.stages.stage0_triage import TriageContext, first_alert, triage


DEFAULT_OUTPUT = Path("outputs") / "phase2_dry_run.jsonl"


def review_order(ctx: TriageContext) -> list[str]:
    """Accounts by earliest alerting transaction, then account id; no-alert accounts last.

    Computed on a fresh RunState, so the order depends only on the bank's own data, never on
    how earlier cases in this run turned out.
    """
    probe = RunState()
    keyed: list[tuple[int, datetime | None, str]] = []
    for account_id in ctx.bank.account_ids():
        out = triage(_review(account_id), ctx, probe)
        alert = first_alert(out, ctx.bank, account_id)
        keyed.append((0, alert.timestamp, account_id) if alert else (1, None, account_id))
    keyed.sort(key=lambda k: (k[0], k[1] or datetime.min, k[2]))
    return [account_id for _, _, account_id in keyed]


def run_all(ctx: TriageContext) -> list[InvestigationReport]:
    state = RunState()
    return [run_case(_review(account_id), ctx, state) for account_id in review_order(ctx)]


def summarise(reports: list[InvestigationReport], seconds: float) -> str:
    accounts_per_rule: Counter[str] = Counter()
    hits_per_rule: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    for r in reports:
        fired = r.triage_findings.fired_rules
        hits_per_rule.update(f.rule.value for f in fired)
        accounts_per_rule.update({f.rule.value for f in fired})
        routes[" + ".join(s.value for s in r.triage_findings.routed_stages)] += 1
    dispositions = Counter(r.disposition.value for r in reports)
    alerted = sum(1 for r in reports if r.triggering_transaction_id)

    lines = [
        f"Phase 2 dry run - {len(reports)} periodic-review cases in {seconds:.1f}s, 0 model calls",
        f"  accounts with an alerting transaction: {alerted}; "
        f"review-only: {len(reports) - alerted}",
        "",
        "  Rule                   accounts   hits",
    ]
    for rule, n in accounts_per_rule.most_common():
        lines.append(f"  {rule:<22} {n:>8}   {hits_per_rule[rule]:>4}")
    lines += ["", "  Route mix"]
    lines += [f"  {n:>5}  {route}" for route, n in routes.most_common()]
    lines += ["", "  Dispositions (neutral stubs, so CLEAR is expected)"]
    lines += [f"  {n:>5}  {d}" for d, n in dispositions.most_common()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--account", help="review one account and print its full report")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    # Rupee signs and Indian names in the output; Windows pipes default to cp1252.
    sys.stdout.reconfigure(encoding="utf-8")

    bank = BankData.load(args.data_dir)
    rulebook = RulebookLookup.load(Path(args.data_dir) / "rulebook" / "clauses.json")
    ctx = TriageContext.build(bank, rulebook)

    if args.account:
        report = run_case(_review(args.account), ctx, RunState())
        print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return 0

    started = time.perf_counter()
    reports = run_all(ctx)
    elapsed = time.perf_counter() - started

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in reports:
            f.write(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n")

    print(summarise(reports, elapsed))
    print(f"\n  {len(reports)} reports written to {args.out}")
    return 0


def _review(account_id: str) -> TriageEvent:
    return TriageEvent(kind=TriageEventKind.PERIODIC_REVIEW, account_id=account_id)


if __name__ == "__main__":
    raise SystemExit(main())
