"""The baseline v2 starts from: what `main` could decide on the tuning set.

    python -m src.evaluation.baseline

`main` has Stage 0 and Stage 2 only. Every other routed weighted stage was a stand-in at 0.0,
and `main`'s judge averaged those zeros in. This script repeats that arithmetic on its own
(Stage 2's real confidence, zeros for the other routed stages), so the figure stays the same
after v2 changes the judge. No model is called.
"""

from __future__ import annotations

import sys

from src.bank import BankData
from src.evaluation import grader, split
from src.orchestration.run_state import RunState
from src.rulebook import RulebookLookup
from src.schemas import (
    CLEAR_BELOW,
    ESCALATE_FROM,
    SCORE_DECIMALS,
    SOLO_SIGNAL_MIN,
    STAGE_WEIGHTS,
    StageName,
    TriageEvent,
    TriageEventKind,
)
from src.stages import stage2_transaction_pattern
from src.stages.stage0_triage import TriageContext, triage


def outcome(account_id: str, ctx: TriageContext) -> grader.Outcome:
    event = TriageEvent(kind=TriageEventKind.PERIODIC_REVIEW, account_id=account_id)
    routed = [s for s in triage(event, ctx, RunState()).routed_stages if s in STAGE_WEIGHTS]
    pattern = None
    score = 0.0
    if StageName.TRANSACTION_PATTERN in routed:
        pattern = stage2_transaction_pattern.run(account_id, ctx, ctx.review_date).confidence
        weight = STAGE_WEIGHTS[StageName.TRANSACTION_PATTERN]
        score = round(weight * pattern / sum(STAGE_WEIGHTS[s] for s in routed), SCORE_DECIMALS)
    if score < CLEAR_BELOW:
        disposition = grader.CLEAR
    elif score >= ESCALATE_FROM and pattern is not None and pattern >= SOLO_SIGNAL_MIN:
        disposition = grader.ESCALATE
    else:
        disposition = grader.MONITOR
    return grader.Outcome(account_id, disposition, transaction_pattern=pattern)


def run(account_ids: list[str]) -> grader.Grade:
    ctx = TriageContext.build(BankData.load(), RulebookLookup.load())
    return grader.grade(outcome(a, ctx) for a in account_ids)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    grade = run(split.tuning_ids())
    print(grader.render(grade, "Baseline on the 100 (main: Stage 0 + Stage 2, diluted judge)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
