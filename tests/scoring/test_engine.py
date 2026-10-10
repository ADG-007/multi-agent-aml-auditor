"""Scoring engine boundaries (MASTER_PLAN.md Phase 2 unit-test list, plus the kickoff's Q8)."""

from itertools import combinations

import pytest

from src.schemas import STAGE_WEIGHTS, Disposition, StageName, StageStatus
from src.scoring import StageResult, corroboration_passes, score_case


S = StageName
D = Disposition
TP, KYC, AM, G = S.TRANSACTION_PATTERN, S.KYC, S.ADVERSE_MEDIA, S.GRAPH


def ran(**confidences):
    names = {"tp": TP, "kyc": KYC, "am": AM, "g": G}
    return {names[k]: StageResult(StageStatus.RAN, v) for k, v in confidences.items()}


FAILED = StageResult(StageStatus.FAILED)
TIMED_OUT = StageResult(StageStatus.TIMED_OUT)


# --- veto -----------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ["am", "g"])
def test_veto_boundary(stage):
    below = score_case(ran(**{stage: 0.89}))
    assert below.veto_stage is None and below.weighted_score == 0.89  # averaged, not vetoed
    result = score_case(ran(**{stage: 0.90}))
    assert result.disposition == D.SUSPICIOUS_ESCALATE
    assert result.confidence == 0.90
    assert result.weighted_score is None


def test_veto_ignores_the_average():
    result = score_case(ran(g=0.95, tp=0.0, kyc=0.0, am=0.0))
    assert result.disposition == D.SUSPICIOUS_ESCALATE
    assert result.confidence == 0.95
    assert result.veto_stage == G


def test_double_veto_takes_the_max():
    result = score_case(ran(am=0.92, g=0.97))
    assert result.confidence == 0.97 and result.veto_stage == G
    assert score_case(ran(am=0.99, g=0.91)).veto_stage == AM


def test_kyc_and_tp_cannot_veto():
    assert score_case(ran(kyc=0.95)).veto_stage is None
    assert score_case(ran(tp=0.95)).veto_stage is None


def test_a_failed_stage_cannot_veto():
    result = score_case({G: FAILED, **ran(tp=0.1)})
    assert result.veto_stage is None
    assert result.disposition == D.MONITOR and result.evidence_incomplete


# --- thresholds -----------------------------------------------------------------------


def test_clear_monitor_boundary():
    assert score_case(ran(tp=0.34)).disposition == D.CLEAR
    assert score_case(ran(tp=0.35)).disposition == D.MONITOR


def test_monitor_escalate_boundary():
    assert score_case(ran(tp=0.59, kyc=0.59)).disposition == D.MONITOR
    assert score_case(ran(tp=0.60, kyc=0.60)).disposition == D.SUSPICIOUS_ESCALATE


# --- corroboration gate ---------------------------------------------------------------


def test_two_stages_at_exactly_040_pass_the_gate():
    result = score_case(ran(g=0.74, kyc=0.40))  # 0.604, nothing at 0.75
    assert result.weighted_score == 0.604
    assert result.disposition == D.SUSPICIOUS_ESCALATE


def test_one_stage_just_under_040_fails_the_gate():
    result = score_case(ran(g=0.74, kyc=0.3999))
    assert result.weighted_score >= 0.60
    assert result.disposition == D.MONITOR and result.single_signal_capped


def test_solo_branch_boundary():
    capped = score_case(ran(kyc=0.74))
    assert capped.disposition == D.MONITOR and capped.single_signal_capped
    assert score_case(ran(kyc=0.75)).disposition == D.SUSPICIOUS_ESCALATE


def test_one_stage_at_one_with_others_at_zero_passes_the_gate():
    assert corroboration_passes({TP: 1.0, KYC: 0.0, AM: 0.0, G: 0.0})


def test_adverse_news_alone_is_capped():
    result = score_case(ran(am=0.70))
    assert result.disposition == D.MONITOR and result.single_signal_capped


# --- degraded floor -------------------------------------------------------------------


def test_degraded_floor_lifts_clear_to_monitor():
    result = score_case({KYC: FAILED, **ran(tp=0.10, g=0.05)})
    assert result.weighted_score == 0.0727
    assert result.disposition == D.MONITOR
    assert result.evidence_incomplete and result.degraded_stages == [KYC]


def test_degraded_floor_does_not_cap_the_top():
    result = score_case({KYC: TIMED_OUT, **ran(tp=0.9, g=0.8)})
    assert result.disposition == D.SUSPICIOUS_ESCALATE
    assert result.evidence_incomplete and result.degraded_stages == [KYC]


def test_not_routed_is_not_degraded():
    result = score_case({KYC: StageResult(StageStatus.NOT_ROUTED), **ran(tp=0.1)})
    assert result.disposition == D.CLEAR and not result.evidence_incomplete


# --- edge cases settled at kickoff (Q8) -----------------------------------------------


def test_no_weighted_stage_is_clear_at_zero():
    for results in ({}, {TP: StageResult(StageStatus.NOT_ROUTED)}):
        result = score_case(results)
        assert (result.disposition, result.confidence) == (D.CLEAR, 0.0)


def test_every_routed_stage_failed_is_monitor_at_zero():
    result = score_case({TP: FAILED, KYC: TIMED_OUT})
    assert (result.disposition, result.confidence) == (D.MONITOR, 0.0)
    assert result.evidence_incomplete and result.degraded_stages == [TP, KYC]


def test_rounding_trap_two_stages_at_035():
    raw = (0.25 * 0.35 + 0.20 * 0.35) / 0.45
    assert raw < 0.35  # 0.3499999999999999 in binary floating point
    result = score_case(ran(tp=0.35, kyc=0.35))
    assert result.weighted_score == 0.35 and result.disposition == D.MONITOR


# --- renormalisation ------------------------------------------------------------------


SUBSETS = [c for n in range(1, 5) for c in combinations((TP, KYC, AM, G), n)]


@pytest.mark.parametrize("subset", SUBSETS, ids=lambda s: "+".join(x.value for x in s))
def test_renormalisation_for_every_routed_subset(subset):
    values = {TP: 0.11, KYC: 0.23, AM: 0.37, G: 0.29}  # all below veto and solo bars
    results = {s: StageResult(StageStatus.RAN, values[s]) for s in subset}
    weighted = sum(STAGE_WEIGHTS[s] * values[s] for s in subset)
    expected = weighted / sum(STAGE_WEIGHTS[s] for s in subset)
    assert score_case(results).weighted_score == round(expected, 4)


@pytest.mark.parametrize("subset", SUBSETS, ids=lambda s: "+".join(x.value for x in s))
def test_equal_confidences_average_to_themselves(subset):
    results = {s: StageResult(StageStatus.RAN, 0.5) for s in subset}
    assert score_case(results).weighted_score == 0.5


def test_worked_examples_from_scoring_md():
    assert score_case(ran(tp=0.9, kyc=0.7, am=0.2, g=0.8)).weighted_score == 0.655
    assert score_case(ran(am=0.7, kyc=0.1)).weighted_score == 0.4333


# --- guards ---------------------------------------------------------------------------


def test_vision_is_never_a_score_input():
    with pytest.raises(ValueError):
        score_case({S.VISION: StageResult(StageStatus.RAN, 0.95)})


@pytest.mark.parametrize("bad", [None, -0.01, 1.01, float("nan")])
def test_ran_needs_a_valid_confidence(bad):
    with pytest.raises(ValueError):
        StageResult(StageStatus.RAN, bad)


def test_non_ran_stage_takes_no_confidence():
    with pytest.raises(ValueError):
        StageResult(StageStatus.FAILED, 0.5)


def test_every_result_explains_itself():
    for results in ({}, ran(tp=0.2), ran(g=0.95), {TP: FAILED}, ran(kyc=0.7)):
        assert score_case(results).reasoning_trail
