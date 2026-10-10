"""RunState: case sequencing, standing, and escalation-only known-bad listing."""

import pytest

from src.orchestration.run_state import RunState
from src.schemas import Disposition, InvestigationReport, PipelineStatus, StageName, StageStatus
from tests.factories import bank, txn


def _report(case_id: str, disposition: Disposition, account_id: str = "ACC-T001"):
    return InvestigationReport(
        case_id=case_id,
        target_account=account_id,
        pipeline_status=PipelineStatus.COMPLETED,
        stage_status={
            StageName.GRAPH.value: StageStatus.RAN,
            StageName.KYC.value: StageStatus.FAILED,
        },
        confidence=0.5,
        disposition=disposition,
    )


@pytest.fixture
def small_bank():
    return bank(
        transactions=[
            txn(device_id="DEV-1", ip_address="10.0.0.1"),
            txn(device_id="DEV-2", ip_address=None),
            txn(),
        ]
    )


def test_case_ids_run_in_sequence_and_reject_gaps(small_bank):
    state = RunState()
    assert state.next_case_id("ACC-T001") == "CASE-ACC-T001-1"
    state.record_case(_report("CASE-ACC-T001-1", Disposition.CLEAR), small_bank)
    assert state.next_case_id("ACC-T001") == "CASE-ACC-T001-2"
    with pytest.raises(ValueError):
        state.record_case(_report("CASE-ACC-T001-5", Disposition.CLEAR), small_bank)


def test_record_keeps_only_stages_that_ran(small_bank):
    record = RunState().record_case(_report("CASE-ACC-T001-1", Disposition.CLEAR), small_bank)
    assert record.stages_ran == frozenset({StageName.GRAPH})


def test_monitor_flags_but_lists_nothing(small_bank):
    state = RunState()
    state.record_case(_report("CASE-ACC-T001-1", Disposition.MONITOR), small_bank)
    assert state.is_flagged("ACC-T001")
    assert not state.known_bad_devices and not state.known_bad_ips
    assert not state.known_bad_addresses


def test_escalation_lists_devices_ips_and_address(small_bank):
    state = RunState()
    state.record_case(_report("CASE-ACC-T001-1", Disposition.SUSPICIOUS_ESCALATE), small_bank)
    assert set(state.known_bad_devices) == {"DEV-1", "DEV-2"}
    assert set(state.known_bad_ips) == {"10.0.0.1"}
    assert set(state.known_bad_addresses) == {small_bank.customer_for("ACC-T001").address}
    assert state.known_bad_devices["DEV-1"] == "ACC-T001"


def test_latest_case_decides_standing_and_list_never_shrinks(small_bank):
    state = RunState()
    state.record_case(_report("CASE-ACC-T001-1", Disposition.SUSPICIOUS_ESCALATE), small_bank)
    state.record_case(_report("CASE-ACC-T001-2", Disposition.CLEAR), small_bank)
    assert state.standing("ACC-T001") == Disposition.CLEAR
    assert not state.is_flagged("ACC-T001")
    assert "DEV-1" in state.known_bad_devices
