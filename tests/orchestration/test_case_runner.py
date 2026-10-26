"""End-to-end paths through the temporary case runner, driven by planned stub values."""

from src.orchestration.case_runner import run_case
from src.orchestration.run_state import RunState
from src.rulebook import RulebookLookup
from src.schemas import (
    Disposition,
    StageName,
    StageStatus,
    TransactionDirection,
    TransactionType,
    TriageEvent,
    TriageEventKind,
    TriageRule,
)
from src.scoring import StageResult
from src.stages.stage0_triage import TriageContext
from src.stages.stubs import StubPlan
from tests.factories import account, bank, customer, txn


S = StageName


def _ctx(b):
    return TriageContext.build(b, RulebookLookup.load())


def _event(t):
    return TriageEvent(kind=TriageEventKind.TRANSACTION, account_id=t.account_id, transaction=t)


def _plan(**confidences):
    names = {"tp": S.TRANSACTION_PATTERN, "kyc": S.KYC, "am": S.ADVERSE_MEDIA, "g": S.GRAPH}
    return StubPlan({names[k]: StageResult(StageStatus.RAN, v) for k, v in confidences.items()})


def _structuring_txn(**kw):
    return txn(amount=950_000, type=TransactionType.CASH_DEPOSIT, **kw)


def test_benign_transaction_gets_a_thin_clear_report():
    t = txn()
    b = bank(transactions=[t])
    report = run_case(_event(t), _ctx(b), RunState())
    assert report.disposition == Disposition.CLEAR and report.confidence == 0.0
    assert report.stage_status[S.AUDITOR.value] == StageStatus.RAN
    assert report.stage_status[S.GRAPH.value] == StageStatus.NOT_ROUTED
    assert report.triggering_transaction_id == t.transaction_id


def test_routed_stages_run_and_unrouted_stay_empty():
    t = _structuring_txn()
    b = bank(transactions=[t])
    report = run_case(_event(t), _ctx(b), RunState(), _plan(tp=0.5, am=0.1, g=0.4))
    assert report.transaction_pattern_findings.confidence == 0.5
    assert report.kyc_findings is None
    assert report.stage_status[S.KYC.value] == StageStatus.NOT_ROUTED
    # (0.25*0.5 + 0.25*0.1 + 0.30*0.4) / 0.80 = 0.3375, just under the 0.35 line, but the
    # The signal floor lifts it: Transaction Pattern 0.5 and Graph 0.4 are >= 0.40.
    assert report.confidence == 0.3375
    assert report.disposition == Disposition.MONITOR


def test_scoring_matches_the_engine_through_the_runner():
    t = _structuring_txn()
    b = bank(transactions=[t])
    report = run_case(_event(t), _ctx(b), RunState(), _plan(tp=0.9, am=0.2, g=0.8))
    # (0.25*0.9 + 0.25*0.2 + 0.30*0.8) / 0.80 is 0.64375 on paper but 0.6437499999999999 in
    # binary floating point, so 4-decimal rounding gives 0.6437. TP and Graph are >= 0.75.
    assert report.confidence == 0.6437
    assert report.disposition == Disposition.SUSPICIOUS_ESCALATE
    assert report.str_narrative is not None
    assert "Corroboration gate" in report.investigation_narrative


def test_a_failed_stub_floors_the_case():
    t = _structuring_txn()
    b = bank(transactions=[t])
    plan = StubPlan({S.GRAPH: StageResult(StageStatus.FAILED)})
    report = run_case(_event(t), _ctx(b), RunState(), plan)
    assert report.disposition == Disposition.MONITOR
    assert report.evidence_incomplete and report.degraded_stages == [S.GRAPH.value]


def test_quiet_transaction_on_a_flagged_account_opens_no_case():
    first = _structuring_txn()
    b = bank(transactions=[first])
    ctx, state = _ctx(b), RunState()
    first_report = run_case(_event(first), ctx, state, _plan(tp=0.5, am=0.5, g=0.5))
    assert first_report.disposition == Disposition.MONITOR

    quiet = txn(amount=300, timestamp="2026-08-02T10:00:00")
    b.add_transaction(quiet)
    assert run_case(_event(quiet), ctx, state) is None
    assert len(state.history("ACC-T001")) == 1


def test_reinvestigation_reruns_every_stage_as_a_new_case():
    first = _structuring_txn()
    b = bank(transactions=[first])
    ctx, state = _ctx(b), RunState()
    first_report = run_case(_event(first), ctx, state, _plan(tp=0.5, am=0.5, g=0.5))
    assert first_report.disposition == Disposition.MONITOR

    wire = txn(amount=700_000, type=TransactionType.SWIFT, timestamp="2026-08-05T10:00:00")
    b.add_transaction(wire)
    report = run_case(_event(wire), ctx, state, _plan(tp=0.5))
    assert report.case_id == "CASE-ACC-T001-2"
    assert TriageRule.REINVESTIGATION in [f.rule for f in report.triage_findings.fired_rules]
    for stage in (S.TRANSACTION_PATTERN, S.KYC, S.ADVERSE_MEDIA, S.GRAPH):
        assert report.stage_status[stage.value] == StageStatus.RAN
    # Vision is routed too; the test bank has no ID card image, so it fails and degrades.
    assert report.stage_status[S.VISION.value] == StageStatus.FAILED


def test_escalation_reroutes_a_ring_mate():
    b = bank(
        accounts=[account(), account(account_id="ACC-T002", customer_id="CUST-T002")],
        customers=[
            customer(),
            customer(customer_id="CUST-T002", name="Ravi Nair", address="9 Beach Rd, Goa 403001"),
        ],
        transactions=[_structuring_txn(device_id="DEV-SHARED")],
    )
    ctx, state = _ctx(b), RunState()
    first = b.transactions("ACC-T001")[0]
    escalated = run_case(_event(first), ctx, state, _plan(g=0.95))
    assert escalated.disposition == Disposition.SUSPICIOUS_ESCALATE
    assert "DEV-SHARED" in state.known_bad_devices

    mate = txn(
        account_id="ACC-T002",
        device_id="DEV-SHARED",
        amount=800,
        direction=TransactionDirection.DEBIT,
    )
    b.add_transaction(mate)
    report = run_case(_event(mate), ctx, state)
    assert [f.rule for f in report.triage_findings.fired_rules] == [TriageRule.KNOWN_BAD_IDENTIFIER]
    assert report.stage_status[S.GRAPH.value] == StageStatus.RAN


def test_review_case_records_the_earliest_alert_or_none():
    rows = [
        txn(timestamp="2026-07-01T10:00:00"),
        _structuring_txn(timestamp="2026-07-15T10:00:00"),
        _structuring_txn(timestamp="2026-07-20T10:00:00"),
    ]
    b = bank(transactions=rows)
    review = TriageEvent(kind=TriageEventKind.PERIODIC_REVIEW, account_id="ACC-T001")
    report = run_case(review, _ctx(b), RunState())
    assert report.triggering_transaction_id == rows[1].transaction_id

    quiet = bank(transactions=[txn()])
    report = run_case(review, _ctx(quiet), RunState())
    assert report.triggering_transaction_id is None
    assert report.stage_status[S.KYC.value] == StageStatus.RAN


# --------------------------------------------------------------------------------------
# Real stages (Stage 2)
# --------------------------------------------------------------------------------------


def _deposits(days):
    return [_structuring_txn(timestamp=f"2026-08-{d:02d}T10:00:00") for d in days]


def test_a_routed_review_case_gets_the_real_stage2():
    b = bank(transactions=_deposits(range(1, 7)))
    review = TriageEvent(kind=TriageEventKind.PERIODIC_REVIEW, account_id="ACC-T001")
    report = run_case(review, _ctx(b), RunState())
    findings = report.transaction_pattern_findings
    assert report.stage_status[S.TRANSACTION_PATTERN.value] == StageStatus.RAN
    assert findings.velocity_summary != "STUB"
    assert "stub" not in findings.reasoning_trail[0].step
    assert findings.confidence == 0.8  # six in-band cash deposits: k = 6
    assert findings.structuring_indicators


def test_a_live_event_reads_the_frame_ending_at_its_own_time():
    rows = _deposits(range(1, 7))
    b = bank(transactions=rows)
    report = run_case(_event(rows[2]), _ctx(b), RunState())
    # The frame ends at the third deposit, so the later three are not seen: k = 3.
    assert report.transaction_pattern_findings.confidence == 0.4


def test_a_stub_plan_entry_still_forces_the_stub():
    b = bank(transactions=_deposits(range(1, 7)))
    review = TriageEvent(kind=TriageEventKind.PERIODIC_REVIEW, account_id="ACC-T001")
    plan = StubPlan({S.TRANSACTION_PATTERN: StageResult(StageStatus.FAILED)})
    report = run_case(review, _ctx(b), RunState(), plan)
    assert report.stage_status[S.TRANSACTION_PATTERN.value] == StageStatus.FAILED
    assert report.transaction_pattern_findings is None
    assert S.TRANSACTION_PATTERN.value in report.degraded_stages
