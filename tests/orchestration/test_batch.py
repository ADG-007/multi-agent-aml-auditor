"""Batch order and re-screening: a flagged account exposes a ring-mate in the next round."""

import copy

from src.orchestration.batch import review, review_order, run_batch
from src.orchestration.pipeline import Pipeline
from src.rulebook import RulebookLookup
from src.schemas import Disposition, SeverityTier, TransactionDirection, TransactionType
from src.stages.stage0_triage import TriageContext
from tests.factories import account, bank, customer, media_entry, txn


def test_review_order_puts_alerts_first():
    rows = [txn(account_id="ACC-T002", amount=950_000, type=TransactionType.CASH_DEPOSIT)]
    b = bank(
        accounts=[account(), account("ACC-T002", "CUST-T002")],
        customers=[customer(), customer("CUST-T002", name="Ravi Nair", address="9 Beach Rd")],
        transactions=rows,
    )
    ctx = TriageContext.build(b, RulebookLookup.load())
    assert review_order(ctx, ["ACC-T001", "ACC-T002"]) == ["ACC-T002", "ACC-T001"]


def test_rescreening_reaches_a_ring_mate():
    # ACC-T002 holds a sanctioned name and escalates; ACC-T001 shares its phone and shows
    # nothing else. Round 2 re-screens ACC-T001 against the filled known-bad list.
    rows = [
        txn(account_id="ACC-T001", device_id="DEV-RING", direction=TransactionDirection.DEBIT),
        txn(account_id="ACC-T002", device_id="DEV-RING", direction=TransactionDirection.DEBIT),
    ]
    b = bank(
        accounts=[account(), account("ACC-T002", "CUST-T002")],
        customers=[
            customer(),
            customer("CUST-T002", name="Vikram Sethi", address="9 Beach Rd", dob="1980-05-01"),
        ],
        transactions=rows,
        media=[media_entry(date_of_birth="1980-05-01")],
    )
    ctx = TriageContext.build(b, RulebookLookup.load())
    result = run_batch(ctx, ["ACC-T001", "ACC-T002"])
    assert result.latest["ACC-T002"].disposition == Disposition.SUSPICIOUS_ESCALATE
    assert len(result.rounds) >= 2 and "ACC-T001" in result.rounds[1]
    mate = result.latest["ACC-T001"]
    assert mate.case_id == "CASE-ACC-T001-2"
    assert mate.graph_findings.connected_flagged_accounts == ["ACC-T002"]
    # One hard link scores 0.60; one stage alone never escalates, so the gate holds MONITOR.
    assert mate.disposition == Disposition.MONITOR and mate.single_signal_capped


def _news_ring():
    # ACC-T002 is in the news (MONITOR, never escalated); ACC-T001 shares its phone only.
    rows = [
        txn(account_id="ACC-T001", device_id="DEV-RING", direction=TransactionDirection.DEBIT),
        txn(account_id="ACC-T002", device_id="DEV-RING", direction=TransactionDirection.DEBIT),
    ]
    b = bank(
        accounts=[account(), account("ACC-T002", "CUST-T002")],
        customers=[customer(), customer("CUST-T002", name="Vikram Sethi", address="9 Beach Rd")],
        transactions=rows,
        media=[media_entry(tier=SeverityTier.ADVERSE_NEWS)],
    )
    return TriageContext.build(b, RulebookLookup.load())


def test_rescreening_follows_a_link_to_a_monitor_account():
    # No escalation, so the known-bad list stays empty. ACC-T001 runs first, before its
    # ring-mate is flagged; round 2 re-runs it because a linked account is now MONITOR.
    result = run_batch(_news_ring(), ["ACC-T001", "ACC-T002"])
    assert result.latest["ACC-T002"].disposition == Disposition.MONITOR
    assert result.rounds[1] == ["ACC-T001"]
    mate = result.latest["ACC-T001"]
    assert mate.case_id == "CASE-ACC-T001-2"
    assert mate.graph_findings.connected_flagged_accounts == ["ACC-T002"]
    assert len(result.rounds) == 2  # nothing new after that: the rounds stop


def test_replaying_a_final_case_gives_the_batch_verdict():
    # The demo page replays an account's final case from the state saved before it.
    ctx = _news_ring()
    result = run_batch(ctx, ["ACC-T001", "ACC-T002"])
    for account_id, final in result.latest.items():
        start = copy.deepcopy(result.before[final.case_id])
        again = Pipeline(ctx, state=start).run(review(account_id))
        assert (again.case_id, again.disposition, again.confidence) == (
            final.case_id,
            final.disposition,
            final.confidence,
        )


def test_every_report_carries_the_writer_trail():
    result = run_batch(_news_ring(), ["ACC-T001", "ACC-T002"])
    for report in result.latest.values():
        steps = [s.step for s in report.auditor_trail]
        assert steps == ["How it ran", "Citation check"]
