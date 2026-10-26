"""Stage 3 with a fake model: level -> score, the fact-id guard, the income floor, the cache
and the modes. pytest never needs Ollama."""

import pytest

from src.llm import LLMStageFailure
from src.llm.cache import LLMMode, ResponseCache
from src.rulebook import RulebookLookup
from src.schemas import KYCProfile, TransactionType
from src.stages import stage3_kyc as s3
from src.stages.stage0_triage import TriageContext
from src.stages.stage3_kyc.prompt import KYCJudgement, Mismatch
from tests.factories import bank, txn


def _ctx(mode=LLMMode.LIVE, amount=400_000):
    """One cash deposit against ₹6,00,000 income: ₹4 lakh is 2.7x a year, ₹9 lakh is 6.1x."""
    rows = [txn(amount=amount, type=TransactionType.CASH_DEPOSIT)]
    profile = KYCProfile(customer_id="CUST-T001", narrative="Teacher. No cash deposits.")
    b = bank(transactions=rows, kyc_profiles=[profile])
    return TriageContext.build(b, RulebookLookup.load(), llm_mode=mode)


class FakeModel:
    def __init__(self, judgement):
        self.judgement, self.calls, self.prompts = judgement, 0, []

    def __call__(self, prompt, schema, stage):
        self.calls += 1
        self.prompts.append(prompt)
        return self.judgement


def _judgement(level, fact_ids=("F6",)):
    return KYCJudgement(
        level=level,
        mismatches=[
            Mismatch(fact_id=f, profile_claim="No cash deposits", why="cash came in")
            for f in fact_ids
        ],
        notes=["Cash contradicts the profile."],
    )


def test_level_maps_to_a_fixed_score(tmp_path):
    ctx = _ctx()
    for level, score in {
        "NONE": 0.0,
        "LOW": 0.0,
        "MODERATE": 0.0,
        "HIGH": 0.65,
        "SEVERE": 0.85,
    }.items():
        cache = ResponseCache("kyc", root=tmp_path / level)
        out = s3.run(
            "ACC-T001", ctx, ctx.review_date, cache=cache, model_call=FakeModel(_judgement(level))
        )
        assert out.confidence == score


def test_prompt_carries_code_computed_facts(tmp_path):
    ctx, model = _ctx(), FakeModel(_judgement("LOW"))
    s3.run(
        "ACC-T001",
        ctx,
        ctx.review_date,
        cache=ResponseCache("kyc", root=tmp_path),
        model_call=model,
    )
    assert "No cash deposits." in model.prompts[0]
    assert "Cash deposits: 1, total ₹4,00,000" in model.prompts[0]


def test_guard_drops_unknown_fact_ids_and_lowers_the_level(tmp_path):
    ctx = _ctx()
    out = s3.run(
        "ACC-T001",
        ctx,
        ctx.review_date,
        cache=ResponseCache("kyc", root=tmp_path),
        model_call=FakeModel(_judgement("SEVERE", fact_ids=("F99",))),
    )
    assert out.confidence == 0.0 and not out.profile_behavior_mismatches
    assert "lowered" in out.reasoning_trail[-1].finding


def test_income_floor_lifts_a_low_level_to_high(tmp_path):
    ctx = _ctx(amount=900_000)
    out = s3.run(
        "ACC-T001",
        ctx,
        ctx.review_date,
        cache=ResponseCache("kyc", root=tmp_path),
        model_call=FakeModel(_judgement("MODERATE")),
    )
    assert out.confidence == 0.65
    assert "6.1x" in out.profile_behavior_mismatches[0]
    assert "lifted MODERATE to HIGH" in out.reasoning_trail[-1].finding


def test_income_floor_keeps_a_higher_level(tmp_path):
    ctx = _ctx(amount=900_000)
    out = s3.run(
        "ACC-T001",
        ctx,
        ctx.review_date,
        cache=ResponseCache("kyc", root=tmp_path),
        model_call=FakeModel(_judgement("SEVERE")),
    )
    assert out.confidence == 0.85 and "lifted" not in out.reasoning_trail[-1].finding


def test_a_saved_answer_is_reused_without_a_call(tmp_path):
    ctx, cache = _ctx(), ResponseCache("kyc", root=tmp_path)
    first = FakeModel(_judgement("HIGH"))
    s3.run("ACC-T001", ctx, ctx.review_date, cache=cache, model_call=first)
    again = FakeModel(_judgement("NONE"))
    out = s3.run("ACC-T001", ctx, ctx.review_date, cache=cache, model_call=again)
    assert again.calls == 0 and out.confidence == 0.65


def test_cached_mode_miss_fails_rather_than_guess(tmp_path):
    ctx = _ctx(LLMMode.CACHED)
    with pytest.raises(LLMStageFailure):
        s3.run("ACC-T001", ctx, ctx.review_date, cache=ResponseCache("kyc", root=tmp_path))


def test_off_mode_is_a_neutral_stand_in():
    ctx = _ctx(LLMMode.OFF)
    out = s3.run("ACC-T001", ctx, ctx.review_date)
    assert out.confidence == 0.0 and "Model off" in out.reasoning_trail[3].finding
