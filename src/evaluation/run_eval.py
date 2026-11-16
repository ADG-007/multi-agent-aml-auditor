"""Run a batch on the tuning set or the held-out set and grade it against the answer key.

    python -m src.evaluation.run_eval --set tuning                    # the 100, live
    python -m src.evaluation.run_eval --set tuning --mode cached      # replay saved answers
    python -m src.evaluation.run_eval --set tuning --judge-only       # verdicts only, fast
    python -m src.evaluation.run_eval --set held_out --final          # the 270, once
    python -m src.evaluation.run_eval --set held_out --mode cached    # replay the 270

The tuning set is what every threshold and prompt was set on. The held-out set is run once,
at the end: a live run on it needs --final, and nothing is changed afterwards. A cached
replay calls no model and gives the same reports.

Writes outputs/<run>/: reports.jsonl, cases.jsonl, admin_queue.json, summary.json, grade.txt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.bank import BankData
from src.evaluation import grader, split
from src.evaluation.stage2_subset import load_ledger
from src.llm import LLMClient, LLMMode
from src.orchestration import batch
from src.orchestration.steps import Agents
from src.rulebook import RulebookLookup
from src.schemas import FORGERY_PAUSE_THRESHOLD, InvestigationReport, StageName
from src.stages.stage0_triage import TriageContext


SHOW_CASES = ("ACC-0044", "ACC-0045", "ACC-0346", "ACC-0121", "ACC-0074")
S = StageName


def outcome(report: InvestigationReport) -> grader.Outcome:
    def confidence(findings) -> float | None:
        return findings.confidence if findings is not None else None

    vision = report.vision_findings
    return grader.Outcome(
        account_id=report.target_account,
        disposition=report.disposition.value,
        paused=vision is not None and vision.forgery_confidence >= FORGERY_PAUSE_THRESHOLD,
        transaction_pattern=confidence(report.transaction_pattern_findings),
        kyc=confidence(report.kyc_findings),
        adverse_media=confidence(report.adverse_media_findings),
        graph=confidence(report.graph_findings),
    )


def show_cases(latest: dict[str, InvestigationReport], ledger) -> list[str]:
    lines = ["Show cases (account: key label / design label -> got, judge inputs, ID check)"]
    for account_id in SHOW_CASES:
        report = latest.get(account_id)
        if report is None:
            continue
        entry = ledger[account_id]
        inputs = ", ".join(f"{k} {v:g}" for k, v in report.judge_inputs.items()) or "none"
        vision = report.vision_findings
        lines.append(
            f"  {account_id}: {entry.expected_disposition.value} / "
            f"{grader.design_label(entry)} -> {report.disposition.value} "
            f"{report.confidence:g} [{inputs}] "
            f"ID {vision.forgery_confidence if vision else 'n/a'}"
            + (f" label {report.graph_label}" if report.graph_label else "")
        )
    return lines


def timing_lines(stats: dict[str, dict]) -> list[str]:
    lines = ["Model calls (seconds are as measured when each answer was made)"]
    for key, row in sorted(stats.items()):
        per_call = row["seconds_per_call"]
        lines.append(
            f"  {key:<44} calls {row['calls']:>4}  from cache {row['from_cache']:>4}  "
            f"failed {row['failed']:>3}  retried {row['retried']:>3}  "
            f"s/call {per_call if per_call is not None else 'n/a'}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", choices=("tuning", "held_out"), default="tuning")
    parser.add_argument("--mode", choices=("live", "cached", "off"), default="live")
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument("--final", action="store_true", help="allow the one live held-out run")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    if args.set == "held_out" and args.mode == "live" and not args.final:
        print("The held-out set is run live once, at the end. Add --final to do it.")
        return 2
    account_ids = split.tuning_ids() if args.set == "tuning" else split.held_out_ids()
    out_dir = args.out or Path("outputs") / f"v2_{args.set}" / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = TriageContext.build(BankData.load(), RulebookLookup.load())
    client = LLMClient(LLMMode(args.mode), audit_path=out_dir / "llm_calls.jsonl")
    agents = Agents.build(client)
    log = (out_dir / "progress.log").open("a", encoding="utf-8")

    def progress(line: str) -> None:
        log.write(f"{time.strftime('%H:%M:%S')} {line}\n")
        log.flush()
        if not args.quiet:
            print(line, flush=True)

    started = time.perf_counter()
    result = batch.run_batch(
        ctx, account_ids, agents, progress=progress, judge_only=args.judge_only
    )
    wall = time.perf_counter() - started
    batch.save(result, out_dir, client.calls)

    ledger = load_ledger()
    grade = grader.grade((outcome(r) for r in result.latest.values()), ledger)
    title = f"v2 on the {args.set} set ({len(account_ids)} accounts, mode {args.mode})"
    lines = [grader.render(grade, title), ""]
    lines += show_cases(result.latest, ledger)
    lines += [
        "",
        f"Rounds: {[len(r) for r in result.rounds]}; cases {len(result.cases)}; "
        f"admin queue {len(result.admin_queue)}; backup model {len(result.backup_accounts)} "
        f"account(s) ({result.backup_share:.1%}), changed {len(result.changed_by_backup)}",
        f"Phase seconds: {result.phase_seconds}; wall {wall:.0f} s",
    ]
    lines += result.warnings
    lines += [""] + timing_lines(batch.call_stats(client.calls))
    lines += ["", "Misses (key label differs from the result)"] + [f"  {m}" for m in grade.misses]
    text = "\n".join(lines)
    (out_dir / "grade.txt").write_text(text + "\n", encoding="utf-8")
    (out_dir / "grade.json").write_text(
        json.dumps(
            {
                "set": args.set,
                "mode": args.mode,
                "accounts": grade.accounts,
                "exact_key": grade.exact_key,
                "exact_design": grade.exact_design,
                "caught": grade.caught,
                "bad": grade.bad,
                "false_alarms": grade.false_alarms,
                "clean": grade.clean,
                "kyc_found": grade.kyc_found,
                "kyc_planted": grade.kyc_planted,
                "kyc_false_high": grade.kyc_false_high,
                "cash_business_alarms": grade.cash_business_alarms,
                "cash_business": grade.cash_business,
                "fake_ids_paused": grade.fake_ids_paused,
                "fake_ids": grade.fake_ids,
                "false_pauses": grade.false_pauses,
                "structurers": dict(grade.structurers),
                "structurers_caught": dict(grade.structurers_caught),
                "wall_seconds": round(wall, 1),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(text)
    print(f"\nWritten to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
