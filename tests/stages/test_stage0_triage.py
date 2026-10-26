"""Stage 0 as a whole: routes per event kind, re-investigation and the known-bad loop."""

import pytest

from src.bank import BankData
from src.orchestration.run_state import RunState
from src.rulebook import RulebookLookup
from src.schemas import (
    Disposition,
    InvestigationReport,
    PipelineStatus,
    StageName,
    TransactionType,
    TriageEvent,
    TriageEventKind,
    TriageRule,
)
from src.stages.stage0_triage import TriageContext, first_alert, triage
from tests.factories import account, bank, customer, media_entry, txn


S = StageName


def _ctx(b):
    return TriageContext.build(b, RulebookLookup.load())


def _txn_event(t):
    return TriageEvent(kind=TriageEventKind.TRANSACTION, account_id=t.account_id, transaction=t)


def _review(account_id="ACC-T001"):
    return TriageEvent(kind=TriageEventKind.PERIODIC_REVIEW, account_id=account_id)


def _record(state, b, account_id, disposition):
    state.record_case(
        InvestigationReport(
            case_id=state.next_case_id(account_id),
            target_account=account_id,
            pipeline_status=PipelineStatus.COMPLETED,
            confidence=0.5,
            disposition=disposition,
        ),
        b,
    )


def test_benign_transaction_routes_only_to_auditor():
    t = txn()
    b = bank(transactions=[t])
    out = triage(_txn_event(t), _ctx(b), RunState())
    assert out.routed_stages == [S.AUDITOR]
    assert out.fired_rules == [] and out.trigger_reasons == []


def test_cash_structuring_route():
    t = txn(amount=950_000, type=TransactionType.CASH_DEPOSIT)
    b = bank(transactions=[t])
    out = triage(_txn_event(t), _ctx(b), RunState())
    assert out.routed_stages == [
        S.TRANSACTION_PATTERN,
        S.ADVERSE_MEDIA,
        S.GRAPH,
        S.RULEBOOK,
        S.AUDITOR,
    ]


def test_kyc_submission_routes_kyc_vision_and_graph_plus_rulebook():
    # A KYC event also checks links (Stage 5), so a quiet ring member is still linked.
    b = bank()
    event = TriageEvent(kind=TriageEventKind.KYC_SUBMITTED, account_id="ACC-T001")
    out = triage(event, _ctx(b), RunState())
    assert out.routed_stages == [S.VISION, S.KYC, S.GRAPH, S.RULEBOOK, S.AUDITOR]


def test_name_match_alone_routes_adverse_media():
    b = bank(media=[media_entry(entity_name="Asha Verma")])
    t = txn()
    b.add_transaction(t)
    out = triage(_txn_event(t), _ctx(b), RunState())
    assert out.routed_stages == [S.ADVERSE_MEDIA, S.RULEBOOK, S.AUDITOR]


def test_review_unions_every_rule_and_always_reviews_kyc():
    rows = [
        txn(amount=950_000, type=TransactionType.CASH_DEPOSIT, timestamp="2026-07-01T10:00:00"),
        txn(amount=600_000, type=TransactionType.SWIFT, timestamp="2026-08-01T10:00:00"),
    ]
    b = bank(transactions=rows)
    out = triage(_review(), _ctx(b), RunState())
    fired = [f.rule for f in out.fired_rules]
    assert TriageRule.CASH_THRESHOLD in fired and TriageRule.CROSS_BORDER in fired
    assert TriageRule.KYC_EVENT in fired
    assert set(out.routed_stages) == {
        S.VISION,
        S.TRANSACTION_PATTERN,
        S.KYC,
        S.ADVERSE_MEDIA,
        S.GRAPH,
        S.RULEBOOK,
        S.AUDITOR,
    }
    assert first_alert(out, b, "ACC-T001").transaction_id == rows[0].transaction_id


def test_review_with_nothing_but_the_kyc_review_has_no_first_alert():
    b = bank(transactions=[txn()])
    out = triage(_review(), _ctx(b), RunState())
    assert [f.rule for f in out.fired_rules] == [TriageRule.KYC_EVENT]
    assert first_alert(out, b, "ACC-T001") is None


def test_flagged_account_tripping_a_rule_reruns_every_stage():
    b = bank(transactions=[txn()])
    state = RunState()
    _record(state, b, "ACC-T001", Disposition.MONITOR)
    t = txn(amount=600_000, type=TransactionType.SWIFT)
    b.add_transaction(t)
    out = triage(_txn_event(t), _ctx(b), state)
    assert out.fired_rules[-1].rule == TriageRule.REINVESTIGATION
    assert out.routed_stages == [
        S.VISION,
        S.TRANSACTION_PATTERN,
        S.KYC,
        S.ADVERSE_MEDIA,
        S.GRAPH,
        S.RULEBOOK,
        S.AUDITOR,
    ]


def test_flagged_account_with_a_quiet_transaction_trips_nothing():
    b = bank(transactions=[txn()])
    state = RunState()
    _record(state, b, "ACC-T001", Disposition.MONITOR)
    t = txn(amount=500)
    b.add_transaction(t)
    out = triage(_txn_event(t), _ctx(b), state)
    assert out.fired_rules == [] and out.routed_stages == [S.AUDITOR]


def test_unposted_live_transaction_is_screened_with_its_history():
    t = txn(amount=950_000, type=TransactionType.CASH_DEPOSIT)
    b = bank()
    out = triage(_txn_event(t), _ctx(b), RunState())
    assert out.fired_rules[0].transaction_id == t.transaction_id


def test_escalation_reroutes_a_ring_mate_through_the_known_bad_list():
    mate_account = account(account_id="ACC-T002", customer_id="CUST-T002")
    mate_customer = customer(
        customer_id="CUST-T002", name="Ravi Nair", address="44 Hill View, Kochi 682001"
    )
    b = bank(
        accounts=[account(), mate_account],
        customers=[customer(), mate_customer],
        transactions=[txn(device_id="DEV-SHARED")],
    )
    ctx = _ctx(b)
    state = RunState()

    shared = txn(account_id="ACC-T002", device_id="DEV-SHARED", amount=800, direction="DEBIT")
    b.add_transaction(shared)
    assert triage(_txn_event(shared), ctx, state).fired_rules == []

    _record(state, b, "ACC-T001", Disposition.SUSPICIOUS_ESCALATE)
    out = triage(_txn_event(shared), ctx, state)
    assert [f.rule for f in out.fired_rules] == [TriageRule.KNOWN_BAD_IDENTIFIER]
    assert S.GRAPH in out.routed_stages and S.ADVERSE_MEDIA in out.routed_stages


@pytest.mark.parametrize("kind", list(TriageEventKind))
def test_every_route_ends_with_the_auditor(kind):
    t = txn()
    b = bank(transactions=[t])
    event = TriageEvent(
        kind=kind,
        account_id="ACC-T001",
        transaction=t if kind == TriageEventKind.TRANSACTION else None,
    )
    assert triage(event, _ctx(b), RunState()).routed_stages[-1] == S.AUDITOR


def test_real_bank_review_counts():
    """The routing reach on the committed data."""
    b = BankData.load()
    ctx = _ctx(b)
    state = RunState()
    per_rule: dict[TriageRule, int] = {}
    for account_id in b.account_ids():
        out = triage(_review(account_id), ctx, state)
        for rule in {f.rule for f in out.fired_rules}:
            per_rule[rule] = per_rule.get(rule, 0) + 1
    assert per_rule[TriageRule.KYC_EVENT] == 370
    assert per_rule[TriageRule.CASH_THRESHOLD] == 30
    assert per_rule[TriageRule.CROSS_BORDER] == 30
    assert per_rule[TriageRule.PASS_THROUGH] == 28
    assert per_rule[TriageRule.NAME_MATCH] == 37


def test_context_holds_the_review_date_from_when_it_was_built():
    b = bank(transactions=[txn(timestamp="2026-08-01T10:00:00")])
    ctx = _ctx(b)
    assert ctx.review_date == b.review_date()
    b.add_transaction(txn(timestamp="2026-08-09T10:00:00"))
    assert ctx.review_date.isoformat() == "2026-08-01T10:00:00"
