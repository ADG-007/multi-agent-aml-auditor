"""Step 3's check: the Vision agent on the tuning set, with the real model.

    python -m src.evaluation.vision_check            # the 100, live (saved answers reused)
    python -m src.evaluation.vision_check --cached   # replay only

Reports KYC lie recall and false HIGH, the claims behind the cash-business traps, and the ID
pause on the fake-ID accounts. Reads the answer key, so it lives here, not in the pipeline.
"""

from __future__ import annotations

import argparse
import sys
import time

from src.agents.vision.agent import VisionAgent
from src.bank import BankData
from src.evaluation import split
from src.evaluation.stage2_subset import load_ledger
from src.llm import LLMClient, LLMMode
from src.rulebook import RulebookLookup
from src.schemas import CORROBORATION_STAGE_MIN, FORGERY_PAUSE_THRESHOLD
from src.stages.stage0_triage import TriageContext


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    ctx = TriageContext.build(BankData.load(), RulebookLookup.load())
    ledger = load_ledger()
    agent = VisionAgent(LLMClient(LLMMode.CACHED if args.cached else LLMMode.LIVE))
    started = time.perf_counter()
    lies = found = honest = false_high = fakes = paused = false_pauses = queued = failed = 0
    traps = explained = 0
    for n, account_id in enumerate(split.tuning_ids(), start=1):
        entry = ledger[account_id]
        result = agent.run(account_id, ctx, ctx.review_date)
        high = result.kyc.confidence >= CORROBORATION_STAGE_MIN
        pause = result.vision.forgery_confidence >= FORGERY_PAUSE_THRESHOLD
        lies += entry.kyc_mismatch_planted
        found += entry.kyc_mismatch_planted and high
        honest += not entry.kyc_mismatch_planted
        false_high += (not entry.kyc_mismatch_planted) and high
        fakes += entry.forged_document_planted
        paused += entry.forged_document_planted and pause
        false_pauses += (not entry.forged_document_planted) and pause
        queued += bool(result.run.queued)
        failed += result.run.model_failed
        if "cash_business" in entry.decoy_types:
            traps += 1
            explained += result.cash_explained
        odd = (
            high != entry.kyc_mismatch_planted
            or pause != entry.forged_document_planted
            or result.run.queued
            or "cash_business" in entry.decoy_types
        )
        if args.verbose or odd:
            print(
                f"{n:>3} {account_id} kyc {result.verdict.level:<8} "
                f"x{result.verdict.volume_multiple:<5g} "
                f"id {result.vision.forgery_confidence:g} "
                f"{'LIE ' if entry.kyc_mismatch_planted else ''}"
                f"{'FAKE-ID ' if entry.forged_document_planted else ''}"
                f"{'explained ' if result.cash_explained else ''}"
                f"claims {result.claims.stated()} unsure {result.claims.unsure} "
                f"{'QUEUED' if result.run.queued else ''}"
            )
    calls = [c for c in agent.client.calls if not c["cached"] and c["ok"]]
    seconds = sum(c["seconds"] for c in calls)
    print(
        f"\nKYC lie recall {found}/{lies}; false HIGH {false_high}/{honest}; "
        f"cash-business traps explained {explained}/{traps}"
    )
    print(f"fake IDs paused {paused}/{fakes}; false pauses {false_pauses}")
    print(f"accounts queued for the backup model {queued}; with a failed call {failed}")
    print(
        f"{len(calls)} new model calls, {seconds:.0f} s"
        + (f" ({seconds / len(calls):.1f} s per call)" if calls else "")
        + f"; wall time {time.perf_counter() - started:.0f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
