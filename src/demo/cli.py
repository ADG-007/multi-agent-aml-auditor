"""Command line: one live case, or a batch.

    python -m src.demo.cli case --account ACC-0044                 # live graph, node by node
    python -m src.demo.cli case --account ACC-0121 --admin block   # decide the ID pause
    python -m src.demo.cli case --account ACC-0044 --mode cached   # saved answers only
    python -m src.demo.cli batch --accounts ACC-0044 ACC-0045      # the 6-phase batch
    python -m src.demo.cli batch --all --mode off                  # Python only, no model

Modes: live (call the local models, reuse saved answers), cached (saved answers only),
off (no model: every agent gives its Python-only result). Every model is local, on Ollama.
On this 4 GB laptop a new answer takes seconds to tens of seconds (performance reduced by
hardware limits).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.bank import BankData
from src.llm import LLMClient, LLMMode
from src.orchestration import batch
from src.orchestration.pipeline import Paused, Pipeline
from src.orchestration.steps import Agents, CaseWork
from src.rulebook import RulebookLookup
from src.schemas import InvestigationReport
from src.stages.stage0_triage import TriageContext


def build(mode: str, audit: Path | None = None) -> tuple[TriageContext, Agents]:
    ctx = TriageContext.build(BankData.load(), RulebookLookup.load())
    client = LLMClient(LLMMode(mode), audit_path=audit)
    return ctx, Agents.build(client)


def render(report: InvestigationReport) -> str:
    lines = [
        f"{report.case_id}  {report.disposition.value}  confidence {report.confidence:g}"
        f"  ({report.pipeline_status.value})",
        "Stages: " + ", ".join(f"{k} {v.value}" for k, v in report.stage_status.items()),
        "Judge inputs: "
        + (", ".join(f"{k} {v:g}" for k, v in report.judge_inputs.items()) or "none"),
    ]
    lines += [f"  judge - {s.step}: {s.finding} -> {s.impact}" for s in report.judge_trail]
    if report.forgery_notice:
        lines.append(f"ID check: {report.forgery_notice}")
    if report.graph_label:
        lines.append(f"Pattern label (never scored): {report.graph_label}")
    if report.rulebook_findings and report.rulebook_findings.retrieved_clause_ids:
        lines.append("Clauses: " + ", ".join(report.rulebook_findings.retrieved_clause_ids))
    lines += [
        "",
        "Summary: " + report.executive_summary,
        "Narrative: " + report.investigation_narrative,
        "Conclusion: " + report.compliance_conclusion,
    ]
    if report.str_narrative:
        lines.append("STR draft: " + report.str_narrative)
    return "\n".join(lines)


def cmd_case(args: argparse.Namespace) -> int:
    ctx, agents = build(args.mode, Path("outputs") / "audit" / "llm_calls.jsonl")
    pipeline = Pipeline(ctx, agents)

    def on_node(name: str, work: CaseWork, seconds: float) -> None:
        extra = ""
        if name == "judge" and work.score is not None:
            extra = f"  -> {work.score.disposition.value} {work.score.confidence:g}"
        print(f"  [{seconds:6.1f} s] {name}{extra}", flush=True)

    started = time.perf_counter()
    result = pipeline.run(batch.review(args.account), on_node=on_node)
    if isinstance(result, Paused):
        print("\nPAUSED for the admin (ID check):")
        for line in result.evidence["field_discrepancies"]:
            print(f"  {line}")
        decision = args.admin
        if decision == "ask":
            decision = input("approve or block? ").strip().lower()
        print(f"Admin decision: {decision}")
        result = pipeline.resume(result, decision, on_node=on_node)
    print(f"\n{render(result)}")
    print(f"\n{time.perf_counter() - started:.1f} s in all; {len(agents.client.calls)} model calls")
    agents.client.unload()
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    out_dir = args.out or Path("outputs") / "batch"
    ctx, agents = build(args.mode, out_dir / "llm_calls.jsonl")
    account_ids = ctx.bank.account_ids() if args.all else args.accounts
    result = batch.run_batch(
        ctx, account_ids, agents, admin=args.admin, progress=None if args.quiet else print
    )
    batch.save(result, out_dir, agents.client.calls)
    for account_id in sorted(result.latest):
        report = result.latest[account_id]
        print(f"{report.case_id}  {report.disposition.value}  {report.confidence:g}")
    print(f"\n{len(result.latest)} accounts, {len(result.cases)} cases, rounds "
          f"{[len(r) for r in result.rounds]}; admin queue {len(result.admin_queue)}; "
          f"backup model {len(result.backup_accounts)}")  # fmt: skip
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    print(f"Written to {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    case = sub.add_parser("case", help="one case through the live graph")
    case.add_argument("--account", required=True)
    case.add_argument("--mode", choices=("live", "cached", "off"), default="live")
    case.add_argument("--admin", choices=("approve", "block", "ask"), default="approve")
    case.set_defaults(run=cmd_case)
    many = sub.add_parser("batch", help="the 6-phase batch")
    group = many.add_mutually_exclusive_group(required=True)
    group.add_argument("--accounts", nargs="+")
    group.add_argument("--all", action="store_true")
    many.add_argument("--mode", choices=("live", "cached", "off"), default="live")
    many.add_argument("--admin", choices=("approve", "block"), default="approve")
    many.add_argument("--out", type=Path)
    many.add_argument("--quiet", action="store_true")
    many.set_defaults(run=cmd_batch)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
