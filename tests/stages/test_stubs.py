"""Stubs: schema-valid outputs, planned values, and a pointer to their replacing phase."""

import pytest

from src.schemas import Disposition, StageName, StageStatus
from src.scoring import StageResult
from src.stages.stubs import (
    REPLACED_IN,
    StubPlan,
    run_auditor_stub,
    run_rulebook_stub,
    run_stub,
)


EVIDENCE_STAGES = [
    StageName.VISION,
    StageName.TRANSACTION_PATTERN,
    StageName.KYC,
    StageName.ADVERSE_MEDIA,
    StageName.GRAPH,
]


@pytest.mark.parametrize("stage", EVIDENCE_STAGES)
def test_default_stub_ran_at_zero_and_names_its_phase(stage):
    outcome = run_stub(stage, StubPlan())
    assert outcome.status == StageStatus.RAN
    trail = outcome.output.reasoning_trail
    assert REPLACED_IN[stage] in trail[0].impact


@pytest.mark.parametrize("stage", EVIDENCE_STAGES)
def test_planned_confidence_is_reported(stage):
    plan = StubPlan({stage: StageResult(StageStatus.RAN, 0.42)})
    output = run_stub(stage, plan).output
    value = output.forgery_confidence if stage == StageName.VISION else output.confidence
    assert value == 0.42


@pytest.mark.parametrize("status", [StageStatus.FAILED, StageStatus.TIMED_OUT])
def test_planned_failure_has_no_output(status):
    outcome = run_stub(StageName.KYC, StubPlan({StageName.KYC: StageResult(status)}))
    assert outcome.status == status and outcome.output is None


def test_rulebook_and_auditor_stubs():
    assert run_rulebook_stub(StubPlan()).output.retrieved_clause_ids == []
    escalated = run_auditor_stub(Disposition.SUSPICIOUS_ESCALATE, 0.95, StubPlan()).output
    assert escalated.str_narrative is not None
    cleared = run_auditor_stub(Disposition.CLEAR, 0.0, StubPlan()).output
    assert cleared.str_narrative is None


def test_rulebook_and_auditor_are_not_evidence_stages():
    with pytest.raises(ValueError):
        run_stub(StageName.RULEBOOK, StubPlan())
