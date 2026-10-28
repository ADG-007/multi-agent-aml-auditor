"""Run a pack's accounts through the real batch, on the bank plus the pack.

New accounts have no saved KYC answer, so the first live run calls the local 8B: about 30 s
an account on this laptop (performance reduced by hardware limits). Answers are saved in the
pack's own llm_cache/, never in data/llm_cache/.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from src.llm.cache import LLMMode
from src.newdata.pack import Pack
from src.orchestration.batch import BatchResult, run_batch
from src.rulebook import RulebookLookup
from src.stages.stage0_triage import TriageContext


def run_pack(
    pack: Pack,
    rulebook: RulebookLookup,
    kyc: LLMMode = LLMMode.LIVE,
    writer: LLMMode = LLMMode.OFF,
    progress: Callable[[str], None] | None = None,
) -> tuple[TriageContext, BatchResult]:
    """Reviews only the pack's accounts (re-check rounds included); writes outputs/run.jsonl."""
    ctx = pack.context(rulebook, kyc=kyc, writer=writer)
    result = run_batch(ctx, pack.account_ids, progress=progress)
    out = pack.path / "outputs" / "run.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for a in result.order:
            f.write(json.dumps(result.latest[a].model_dump(mode="json"), ensure_ascii=False) + "\n")
    return ctx, result
