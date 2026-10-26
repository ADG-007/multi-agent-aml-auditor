"""Stage 3's entry point: does the KYC profile explain the money?

    run(account_id, ctx, as_of, transaction) -> Stage3KYCOutput

Code builds the facts, the 8B picks a level and cites fact ids, code maps the level to a score
and drops citations of facts that do not exist. Money in ≥ 3x income lifts the level to HIGH.
ctx.llm_mode picks off / cached / live.
A new answer takes tens of seconds on this 4 GB laptop (performance reduced by hardware limits).
"""

from __future__ import annotations

import platform
import time
from datetime import datetime

from src.llm import LLMFailureReason, LLMStageFailure, get_stage_config, get_structured_response
from src.llm.cache import LLMMode, ResponseCache, fingerprint
from src.schemas import ReasoningStep, Stage3KYCOutput, Transaction
from src.stages.stage0_triage import TriageContext
from src.stages.stage3_kyc import config
from src.stages.stage3_kyc.prompt import KYCJudgement, render
from src.stages.stage3_kyc.summary import Fact, build_facts


STAGE = "KYC"


def run(
    account_id: str,
    ctx: TriageContext,
    as_of: datetime | None,
    transaction: Transaction | None = None,
    cache: ResponseCache | None = None,
    model_call=None,
) -> Stage3KYCOutput:
    """`model_call(prompt, schema, stage)` defaults to the shared client; tests pass a fake."""
    facts = build_facts(account_id, ctx, as_of, transaction)
    profile = ctx.bank.kyc_profile_for(account_id)
    prompt = render(profile.narrative if profile else "", facts)

    if ctx.llm_mode == LLMMode.OFF and model_call is None:
        return _output(facts, None, "Model off for this run: no judgement made (stand-in 0.0).")

    model_id = get_stage_config(STAGE).model_id or "unset"
    key = fingerprint(model_id, config.PROMPT_VERSION, prompt)
    cache = cache or ResponseCache("kyc")
    name = f"{account_id}-{key[:12]}"  # one file per input: a new prompt keeps the old answer
    saved = cache.load(name, key)
    if saved is not None:
        judgement = KYCJudgement.model_validate(saved["judgement"])
        how = (
            f"Saved answer from {model_id}, {config.PROMPT_VERSION} ({saved['seconds']:.0f} s "
            f"when made on {saved['machine']})."
        )
        return _output(facts, judgement, how)

    if ctx.llm_mode == LLMMode.CACHED and model_call is None:
        raise LLMStageFailure(
            LLMFailureReason.NOT_CONFIGURED,
            STAGE,
            f"no saved KYC answer for {account_id} with these inputs (cached mode); "
            "run the KYC pre-compute first",
        )

    started = time.perf_counter()
    judgement = (model_call or get_structured_response)(prompt, KYCJudgement, STAGE)
    seconds = time.perf_counter() - started
    cache.save(
        name,
        {
            "key": key,
            "account_id": account_id,
            "model": model_id,
            "prompt_version": config.PROMPT_VERSION,
            "seconds": round(seconds, 1),
            "machine": platform.node(),
            "made_at": datetime.now().isoformat(timespec="seconds"),
            "judgement": judgement.model_dump(),
        },
    )
    how = (
        f"New answer from {model_id}, {config.PROMPT_VERSION}, {seconds:.0f} s on this "
        "laptop (performance reduced by hardware limits)."
    )
    return _output(facts, judgement, how)


def stale_warning(accounts: int) -> str | None:
    """A warning when fewer saved answers match the current prompt than there are accounts."""
    saved = ResponseCache("kyc").count(config.PROMPT_VERSION)
    if saved >= accounts:
        return None
    return (
        f"Only {saved} saved KYC answers match prompt {config.PROMPT_VERSION} "
        f"({accounts} accounts). In cached mode a missing answer makes Stage 3 FAILED. "
        "Run `python -m src.demo.cli kyc-precompute` first."
    )


def score(judgement: KYCJudgement, facts: list[Fact]) -> tuple[float, list[str], str | None]:
    """(confidence, kept mismatch lines, note). The guard needs a real fact id, then the income
    floor applies."""
    ids = {f.fact_id for f in facts}
    kept = [m for m in judgement.mismatches if m.fact_id.strip().upper() in ids]
    kept = kept[: config.MAX_MISMATCHES]
    level = judgement.level
    note = None
    if config.LEVEL_SCORES[level] >= config.GUARDED_FROM and not kept:
        note = (
            f"Model said {level} but cited no real fact id: lowered to "
            f"{config.GUARD_FALLBACK_LEVEL}."
        )
        level = config.GUARD_FALLBACK_LEVEL
    dropped = len(judgement.mismatches) - len(kept)
    if dropped > 0 and note is None:
        note = f"{dropped} mismatch(es) dropped: fact id missing or unknown."
    lines = [f"{m.fact_id.upper()} vs '{m.profile_claim}': {m.why}" for m in kept]

    money_in = next((f for f in facts if f.x_income is not None), None)
    floor = config.INCOME_FLOOR_LEVEL
    if (
        money_in is not None
        and money_in.x_income >= config.INCOME_FLOOR_MULTIPLE
        and config.LEVEL_SCORES[level] < config.LEVEL_SCORES[floor]
    ):
        lines.insert(
            0,
            f"{money_in.fact_id} vs declared income: money in = {money_in.x_income:g}x "
            f"(code floor at {config.INCOME_FLOOR_MULTIPLE:g}x).",
        )
        lift = f"Money in {money_in.x_income:g}x income: code lifted {level} to {floor}."
        note = f"{note} {lift}" if note else lift
        level = floor
    return config.LEVEL_SCORES[level], lines[: config.MAX_MISMATCHES], note


def _output(facts: list[Fact], judgement: KYCJudgement | None, how: str) -> Stage3KYCOutput:
    by_topic = {}
    for f in facts:
        by_topic.setdefault(f.topic, []).append(f)

    def step(name: str, topics: tuple[str, ...], impact: str) -> ReasoningStep:
        chosen = [f for t in topics for f in by_topic.get(t, [])]
        return ReasoningStep(
            step=name,
            finding=" ".join(f"{f.fact_id}: {f.text}" for f in chosen) or "n/a",
            impact=impact,
        )

    trail = [
        step(
            "Income vs turnover",
            ("profile", "income"),
            "Compared with the declared profile (KYC-002, STRUCT-003).",
        ),
        step(
            "Cash and foreign money",
            ("cash", "foreign"),
            "Checked against what the profile allows.",
        ),
        step(
            "Pass-through",
            ("pass_through",),
            "A profile must explain fast in-and-out money (MULE-001).",
        ),
        ReasoningStep(
            step="How it ran",
            finding=how,
            impact="Code computed every number; the model judged only.",
        ),
    ]
    if judgement is None:
        return Stage3KYCOutput(confidence=0.0, reasoning_trail=trail)

    confidence, mismatches, note = score(judgement, facts)
    model_steps = [
        ReasoningStep(step="Model judgement", finding=n, impact="The model's own reasoning.")
        for n in judgement.notes[: config.MAX_MODEL_STEPS]
    ]
    final = ReasoningStep(
        step="Level to score",
        finding=f"Model level {judgement.level}; {len(mismatches)} mismatch(es) kept."
        + (f" {note}" if note else ""),
        impact=f"Code maps the level to {confidence:g}.",
    )
    return Stage3KYCOutput(
        profile_behavior_mismatches=mismatches,
        confidence=confidence,
        reasoning_trail=(trail + model_steps)[:7] + [final],
    )
