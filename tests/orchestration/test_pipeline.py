"""The LangGraph pipeline: same verdict as the fallback runner, and a real pause on the ID
check that waits for the admin."""

from src.orchestration.case_runner import run_case
from src.orchestration.pipeline import Paused, Pipeline
from src.orchestration.run_state import RunState
from src.rulebook import RulebookLookup
from src.schemas import (
    Disposition,
    PipelineStatus,
    StageName,
    StageStatus,
    TransactionType,
    TriageEvent,
    TriageEventKind,
)
from src.scoring import StageResult
from src.stages.stage0_triage import TriageContext
from src.stages.stubs import StubPlan
from tests.factories import bank, txn


S = StageName


def _setup():
    rows = [
        txn(
            amount=950_000, type=TransactionType.CASH_DEPOSIT, timestamp=f"2026-08-{d:02d}T10:00:00"
        )
        for d in range(1, 7)
    ]
    return TriageContext.build(bank(transactions=rows), RulebookLookup.load())


def _review():
    return TriageEvent(kind=TriageEventKind.PERIODIC_REVIEW, account_id="ACC-T001")


def _forged(forgery=0.95):
    return StubPlan({S.VISION: StageResult(StageStatus.RAN, forgery)})


def test_graph_and_fallback_agree():
    ctx = _setup()
    plan = _forged(0.0)
    graph = Pipeline(ctx, plan=plan).run(_review())
    plain = run_case(_review(), ctx, RunState(), plan)
    assert graph.disposition == plain.disposition
    assert graph.confidence == plain.confidence
    assert graph.stage_status == plain.stage_status


def test_a_forged_id_pauses_until_the_admin_decides():
    ctx = _setup()
    p = Pipeline(ctx, plan=_forged())
    paused = p.run(_review())
    assert isinstance(paused, Paused)
    assert paused.evidence["forgery_confidence"] == 0.95
    assert p.state.history("ACC-T001") == ()  # nothing is handed over while paused

    report = p.resume(paused, "approve")
    assert report.pipeline_status == PipelineStatus.COMPLETED
    assert "APPROVE" in report.forgery_notice
    # The ID check never touches the score: the verdict is Stage 2's alone.
    assert report.disposition != Disposition.BLOCKED_BY_ADMIN_REVIEW


def test_block_gives_the_blocked_disposition():
    ctx = _setup()
    p = Pipeline(ctx, plan=_forged())
    report = p.resume(p.run(_review()), "block")
    assert report.pipeline_status == PipelineStatus.BLOCKED_BY_ADMIN
    assert report.disposition == Disposition.BLOCKED_BY_ADMIN_REVIEW
    assert report.stage_status[S.AUDITOR.value] == StageStatus.RAN


def test_forgery_below_090_does_not_pause():
    assert not isinstance(Pipeline(_setup(), plan=_forged(0.89)).run(_review()), Paused)


def test_a_stage_that_raises_is_failed_not_a_crash(monkeypatch):
    from src.orchestration import steps

    def boom(*args):
        raise RuntimeError("disk on fire")

    monkeypatch.setitem(steps.REAL_STAGES, S.TRANSACTION_PATTERN, boom)
    report = Pipeline(_setup(), plan=_forged(0.0)).run(_review())
    assert report.stage_status[S.TRANSACTION_PATTERN.value] == StageStatus.FAILED
    assert report.disposition != Disposition.CLEAR  # the degraded floor
    assert "disk on fire" in report.investigation_narrative


def test_each_node_is_reported_as_it_finishes():
    seen = []
    report = Pipeline(_setup(), plan=_forged(0.0)).run(
        _review(), on_node=lambda node, work, seconds: seen.append((node, seconds >= 0))
    )
    assert [n for n, _ in seen] == [
        "triage",
        "vision",
        "transaction_pattern",
        "kyc",
        "adverse_media",
        "graph",
        "judge",
        "admin_gate",
        "rulebook",
        "writer",
    ]
    assert all(ok for _, ok in seen) and report.disposition is not None


def test_a_paused_case_can_be_previewed_and_then_finished():
    seen = []
    p = Pipeline(_setup(), plan=_forged())
    paused = p.run(_review(), on_node=lambda node, work, seconds: seen.append(node))
    assert seen[-1] == "judge"  # the gate stopped before admin_gate returned
    preview = p.preview(paused)
    assert preview.pipeline_status == PipelineStatus.PAUSED_PENDING_ADMIN_REVIEW
    assert preview.transaction_pattern_findings is not None and preview.judge_trail
    assert p.state.history("ACC-T001") == ()  # a preview hands nothing over

    report = p.resume(paused, "approve", on_node=lambda node, work, seconds: seen.append(node))
    assert seen[-3:] == ["admin_gate", "rulebook", "writer"]
    assert report.disposition == preview.disposition
