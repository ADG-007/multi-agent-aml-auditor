"""Step 4's check: the Web Research agent on the tuning set, with the real model.

    python -m src.evaluation.web_check            # live (saved answers reused)
    python -m src.evaluation.web_check --cached   # replay only

Reports the score of every account with a list hit, against what the answer key planted:
real sanctions and news hits must keep their score, look-alike names must stay under 0.35.
"""

from __future__ import annotations

import argparse
import sys

from src.agents.web_research.agent import WebResearchAgent
from src.bank import BankData
from src.evaluation import split
from src.evaluation.stage2_subset import load_ledger
from src.llm import LLMClient, LLMMode
from src.rulebook import RulebookLookup
from src.schemas import CLEAR_BELOW, CORROBORATION_STAGE_MIN, SeverityTier
from src.stages.stage0_triage import TriageContext


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cached", action="store_true")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    ctx = TriageContext.build(BankData.load(), RulebookLookup.load())
    ledger = load_ledger()
    agent = WebResearchAgent(LLMClient(LLMMode.CACHED if args.cached else LLMMode.LIVE))
    real = found = lookalikes = quiet = others = other_hits = queued = 0
    for account_id in split.tuning_ids():
        entry = ledger[account_id]
        result = agent.run(account_id, ctx, ctx.review_date)
        score = result.output.confidence
        planted = entry.adverse_media_hit
        is_real = planted in (SeverityTier.CONFIRMED_SANCTIONS, SeverityTier.ADVERSE_NEWS)
        is_lookalike = planted == SeverityTier.NAME_SIMILARITY_ONLY
        real += is_real
        found += is_real and score >= CORROBORATION_STAGE_MIN
        lookalikes += is_lookalike
        quiet += is_lookalike and score < CLEAR_BELOW
        others += planted is None
        other_hits += planted is None and score >= CORROBORATION_STAGE_MIN
        queued += bool(result.run.queued)
        if result.hits:
            said = next((s.finding for s in result.run.trail if "Same-person" in s.finding), "")
            print(
                f"{account_id} planted {planted.value if planted else '-':<22} score {score:<5g} "
                f"{len(result.hits)} hit(s) {said[said.find('Same-person') :][:150]}"
            )
    calls = [c for c in agent.client.calls if not c["cached"] and c["ok"]]
    print(
        f"\nreal list hits scored >= 0.40: {found}/{real}; look-alike names under 0.35: "
        f"{quiet}/{lookalikes}; accounts with no planted hit scored >= 0.40: {other_hits}/{others}"
    )
    print(f"queued for the backup model {queued}; {len(calls)} new model calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
