"""Phase 2 schema changes: TriageEvent, FiredRule, Stage0TriageOutput, optional trigger."""

import pytest
from pydantic import ValidationError

from src.schemas import (
    Disposition,
    FiredRule,
    InvestigationReport,
    PipelineStatus,
    Stage0TriageOutput,
    StageName,
    TriageEvent,
    TriageEventKind,
    TriageRule,
)
from tests.factories import txn


def test_transaction_event_requires_its_transaction():
    t = txn(account_id="ACC-T001")
    event = TriageEvent(kind=TriageEventKind.TRANSACTION, account_id="ACC-T001", transaction=t)
    assert event.transaction is t

    with pytest.raises(ValidationError):
        TriageEvent(kind=TriageEventKind.TRANSACTION, account_id="ACC-T001")


def test_review_and_kyc_events_take_no_transaction():
    TriageEvent(kind=TriageEventKind.PERIODIC_REVIEW, account_id="ACC-T001")
    TriageEvent(kind=TriageEventKind.KYC_SUBMITTED, account_id="ACC-T001")
    with pytest.raises(ValidationError):
        TriageEvent(
            kind=TriageEventKind.PERIODIC_REVIEW, account_id="ACC-T001", transaction=txn()
        )


def test_event_rejects_another_accounts_transaction():
    with pytest.raises(ValidationError):
        TriageEvent(
            kind=TriageEventKind.TRANSACTION,
            account_id="ACC-T001",
            transaction=txn(account_id="ACC-T999"),
        )


def test_triage_output_uses_stage_names_and_parallel_lists():
    out = Stage0TriageOutput(
        routed_stages=[StageName.GRAPH, StageName.AUDITOR],
        trigger_reasons=["matched a listed device"],
        fired_rules=[FiredRule(rule=TriageRule.KNOWN_BAD_IDENTIFIER, detail="DEV-1")],
    )
    assert out.routed_stages == [StageName.GRAPH, StageName.AUDITOR]

    with pytest.raises(ValidationError):
        Stage0TriageOutput(routed_stages=["Grpah"])

    with pytest.raises(ValidationError):
        Stage0TriageOutput(trigger_reasons=["a reason with no rule"])


def test_report_allows_a_review_with_no_triggering_transaction():
    report = InvestigationReport(
        case_id="CASE-ACC-T001-1",
        target_account="ACC-T001",
        pipeline_status=PipelineStatus.COMPLETED,
        confidence=0.0,
        disposition=Disposition.CLEAR,
    )
    assert report.triggering_transaction_id is None
