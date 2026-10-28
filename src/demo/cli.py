"""The command line.

    python -m src.demo.cli run50                 # all 50 accounts -> right/wrong table + metrics
    python -m src.demo.cli show                  # the 5 show cases, full reports
    python -m src.demo.cli case ACC-0044         # one account's full report
    python -m src.demo.cli kyc-precompute        # Stage 3 on the 50, once, saved (needs Ollama)
    python -m src.demo.cli generate --kind mix --seed 7   # new accounts -> data/packs/mix-seed7
    python -m src.demo.cli pack mix-seed7        # run a pack on the bank, graded (KYC live)

The 50-account commands run the whole batch underneath (Stage 5 needs the flagged accounts).
--kyc defaults to cached (a missing answer makes Stage 3 FAILED). --writer defaults to off for
run50 and live for show and case. The verdict never depends on the writer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.bank import BankData
from src.evaluation.mvp_report import format_summary, format_table, rows
from src.evaluation.mvp_subset import load_subset
from src.llm.cache import LLMMode
from src.newdata import KINDS, PACKS_DIR, Pack, list_packs, make_pack, run_pack
from src.orchestration.batch import run_batch
from src.rulebook import RulebookLookup
from src.schemas import InvestigationReport
from src.stages.stage0_triage import TriageContext
from src.stages.stage3_kyc.agent import stale_warning


OUT_DIR = Path("outputs")


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="AML Auditor")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, writer: str) -> None:
        p.add_argument("--kyc", choices=[m.value for m in LLMMode], default="cached")
        p.add_argument("--writer", choices=[m.value for m in LLMMode], default=writer)
        p.add_argument(
            "--admin",
            choices=["approve", "block", "ask"],
            default="approve",
            help="decision on a case the ID check paused",
        )

    common(sub.add_parser("run50", help="all 50 accounts, graded"), "off")
    common(sub.add_parser("show", help="the 5 show cases"), "live")
    p = sub.add_parser("case", help="one account's full report")
    p.add_argument("account_id")
    p.add_argument("--json", action="store_true", help="print the whole report as JSON")
    common(p, "live")
    p = sub.add_parser("kyc-precompute", help="Stage 3 on the 50 accounts, saved")
    p.add_argument("accounts", nargs="*", help="default: the 50")
    p = sub.add_parser("generate", help="make a pack of new accounts")
    p.add_argument("--kind", choices=[*KINDS, "mix"], default="mix")
    p.add_argument("--count", type=int, default=1, help="accounts (ring size for mule_ring)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--name", help="pack folder name (default <kind>-seed<seed>)")
    p = sub.add_parser("pack", help="run a pack on the bank, graded against its answer key")
    p.add_argument("name", help="folder under data/packs/")
    p.add_argument("--kyc", choices=[m.value for m in LLMMode], default="live")
    p.add_argument("--writer", choices=[m.value for m in LLMMode], default="off")
    args = parser.parse_args(argv)

    if args.command == "generate":
        return _generate(args)
    if args.command == "pack":
        return _pack(args)

    bank = BankData.load()
    rulebook = RulebookLookup.load()
    subset = load_subset()
    accounts = [r["account_id"] for r in subset["accounts"]]

    if args.command == "kyc-precompute":
        return _precompute(
            TriageContext.build(bank, rulebook, llm_mode=LLMMode.LIVE), args.accounts or accounts
        )

    ctx = TriageContext.build(bank, rulebook, llm_mode=LLMMode(args.kyc))
    if ctx.llm_mode == LLMMode.CACHED and (warning := stale_warning(len(accounts))):
        print(f"WARNING: {warning}\n")
    writer = LLMMode(args.writer)
    if args.command == "run50":
        focus: set[str] = set()
    elif args.command == "show":
        focus = set(subset["show_cases"].values())
    else:
        focus = {args.account_id}
        if args.account_id not in accounts:
            accounts.append(args.account_id)

    started = time.perf_counter()
    result = run_batch(
        ctx,
        accounts,
        admin=_ask if args.admin == "ask" else args.admin,
        writer_for=(lambda a: writer if (not focus or a in focus) else LLMMode.OFF),
    )
    seconds = time.perf_counter() - started
    OUT_DIR.mkdir(exist_ok=True)
    name = f"case_{args.account_id}" if args.command == "case" else args.command
    out = OUT_DIR / f"{name}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for a in result.order:
            f.write(json.dumps(result.latest[a].model_dump(mode="json"), ensure_ascii=False) + "\n")

    print(
        f"Ran {len(result.order)} accounts, {len(result.cases)} cases, "
        f"{len(result.rounds)} round(s) in {seconds:.0f}s. KYC {args.kyc}, writer "
        f"{args.writer}, admin {args.admin}. Reports: {out}\n"
    )

    if args.command == "run50":
        table = rows(subset, result.latest)
        print(format_table(table))
        print()
        print(format_summary(table))
        return 0

    expected = {r["account_id"]: r for r in subset["accounts"]}
    labels = {v: k for k, v in subset["show_cases"].items()}
    for account_id in [*subset["show_cases"].values()] if args.command == "show" else focus:
        report = result.latest[account_id]
        if getattr(args, "json", False):
            print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
            continue
        e = expected.get(account_id)
        tag = f"Show case {labels[account_id]}: " if account_id in labels else ""
        answer = f" | expected {e['expected']} ({e['group']})" if e else ""
        print("=" * 100)
        print(f"{tag}{account_id}{answer}")
        print(render(report))
    return 0


def render(r: InvestigationReport) -> str:
    """A readable report for the terminal."""
    lines = [
        f"{r.case_id}  {r.disposition.value}  confidence {r.confidence:.4f}  "
        f"({r.pipeline_status.value})",
        "",
        "Stages: " + ", ".join(f"{k} {v.value}" for k, v in r.stage_status.items()),
    ]
    flags = [
        n
        for n in ("evidence_incomplete", "single_signal_capped", "policy_gap_identified")
        if getattr(r, n)
    ]
    if flags:
        lines.append("Flags: " + ", ".join(flags))
    if r.forgery_notice:
        lines.append(r.forgery_notice)
    lines.append("")
    lines.append("Why it was opened:")
    lines += [f"  - {t}" for t in r.triage_findings.trigger_reasons[:4]]
    if len(r.triage_findings.trigger_reasons) > 4:
        lines.append(f"  ... {len(r.triage_findings.trigger_reasons) - 4} more")
    for title, f in (
        ("ID check (never scored)", r.vision_findings),
        ("Money moves", r.transaction_pattern_findings),
        ("Profile", r.kyc_findings),
        ("Blacklist", r.adverse_media_findings),
        ("Links", r.graph_findings),
        ("Rules", r.rulebook_findings),
    ):
        if f is None:
            continue
        score = getattr(f, "confidence", getattr(f, "forgery_confidence", None))
        lines.append("")
        lines.append(f"{title}" + (f": {score:g}" if score is not None else ""))
        lines += [f"  {s.step}: {s.finding}" for s in f.reasoning_trail[:5]]
    if r.auditor_trail:
        lines += ["", "Writer"] + [f"  {s.step}: {s.finding}" for s in r.auditor_trail]
    lines += [
        "",
        "Summary: " + r.executive_summary,
        "",
        "Narrative: " + r.investigation_narrative,
        "",
        "Conclusion: " + r.compliance_conclusion,
    ]
    if r.str_narrative:
        lines += ["", "STR draft: " + r.str_narrative]
    return "\n".join(lines)


def _generate(args) -> int:
    try:
        pack = make_pack(args.kind, args.count, args.seed, args.name)
    except FileExistsError as exc:
        print(exc)
        return 1
    print(f"Pack written: {pack.path}")
    for r in pack.answers["accounts"]:
        print(f"  {r['account_id']}  {r['group']:<28} expected {r['expected']}")
    print(f"Run it:  python -m src.demo.cli pack {pack.name}")
    return 0


def _pack(args) -> int:
    pack = Pack(PACKS_DIR / args.name)
    if pack.name not in {p.name for p in list_packs()}:
        print(f"No pack {pack.path}. Packs: {', '.join(p.name for p in list_packs()) or 'none'}")
        return 1
    print(
        f"{pack.name}: {len(pack.account_ids)} accounts. KYC {args.kyc}: a new account calls "
        "the local 8B (about 30 s each, performance reduced by hardware limits)."
    )
    started = time.perf_counter()
    _, result = run_pack(
        pack,
        RulebookLookup.load(),
        kyc=LLMMode(args.kyc),
        writer=LLMMode(args.writer),
        progress=lambda s: print("  " + s, flush=True),
    )
    print(
        f"\n{len(result.order)} accounts, {len(result.cases)} cases, {len(result.rounds)} "
        f"round(s) in {time.perf_counter() - started:.0f}s. Reports: "
        f"{pack.path / 'outputs' / 'run.jsonl'}\n"
    )
    table = rows(pack.answers, result.latest)
    print(format_table(table))
    print()
    print(format_summary(table))
    return 0


_asked: dict[str, str] = {}


def _ask(work) -> str:
    """Ask once per account: a re-screening round reuses the first answer."""
    account_id = work.event.account_id
    if account_id in _asked:
        return _asked[account_id]
    vision = work.findings["vision_findings"]
    print(f"\nID check paused {account_id}: {vision.field_discrepancies}")
    while True:
        answer = input("Admin decision [approve/block]: ").strip().lower()
        if answer in ("approve", "block"):
            _asked[account_id] = answer
            return answer


def _precompute(ctx: TriageContext, accounts: list[str]) -> int:
    from src.stages import stage3_kyc  # noqa: PLC0415

    print(
        f"Stage 3 on {len(accounts)} accounts (local 8B; performance reduced by hardware "
        "limits). Saved answers are reused."
    )
    failed = 0
    levels: dict[str, int] = {}
    for i, account_id in enumerate(accounts, 1):
        started = time.perf_counter()
        try:
            out = stage3_kyc.run(account_id, ctx, ctx.review_date)
            how = next(s.finding for s in out.reasoning_trail if s.step == "How it ran")
            final = next(s.finding for s in out.reasoning_trail if s.step == "Level to score")
            level = final.split(";")[0].removeprefix("Model level ")
            levels[level] = levels.get(level, 0) + 1
            print(
                f"[{i:>2}/{len(accounts)}] {account_id} {level:<8} {out.confidence:.2f} "
                f"({time.perf_counter() - started:.0f}s) {how[:60]}"
            )
        except Exception as exc:  # noqa: BLE001 - log and continue
            failed += 1
            print(f"[{i:>2}/{len(accounts)}] {account_id} FAILED: {exc}")
    print(f"Done. {len(accounts) - failed} saved or reused, {failed} failed.")
    print("Levels: " + ", ".join(f"{k} {v}" for k, v in sorted(levels.items())))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
