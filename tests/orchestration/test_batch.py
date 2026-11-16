"""Batch mode: order, re-screening rounds, the admin queue, the backup phase, saved files."""

import json

from src.agents.vision import claims as cf
from src.orchestration import batch
from src.orchestration.batch import review_order, run_batch
from src.orchestration.steps import Agents
from src.rulebook import RulebookLookup
from src.schemas import (
    Disposition,
    KYCProfile,
    SeverityTier,
    StageStatus,
    TransactionDirection,
    TransactionType,
)
from src.stages.stage0_triage import TriageContext
from tests.factories import account, bank, customer, fake_client, media_entry, txn


def _pair(media, names=("Asha Verma", "Vikram Sethi"), dob="1980-05-01", profiles=None):
    rows = [
        txn(account_id="ACC-T001", device_id="DEV-RING", direction=TransactionDirection.DEBIT),
        txn(account_id="ACC-T002", device_id="DEV-RING", direction=TransactionDirection.DEBIT),
    ]
    b = bank(
        accounts=[account(), account("ACC-T002", "CUST-T002")],
        customers=[
            customer(name=names[0]),
            customer("CUST-T002", name=names[1], address="9 Beach Rd", dob=dob),
        ],
        transactions=rows,
        media=media,
        kyc_profiles=profiles or [],
    )
    return TriageContext.build(b, RulebookLookup.load())


def _news_ring():
    return _pair([media_entry(tier=SeverityTier.ADVERSE_NEWS)])


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
    ctx = _pair([media_entry(date_of_birth="1980-05-01")])
    result = run_batch(ctx, ["ACC-T001", "ACC-T002"], Agents.build())
    assert result.latest["ACC-T002"].disposition == Disposition.SUSPICIOUS_ESCALATE
    assert len(result.rounds) >= 2 and "ACC-T001" in result.rounds[1]
    mate = result.latest["ACC-T001"]
    assert mate.case_id == "CASE-ACC-T001-2"
    assert mate.graph_findings.connected_flagged_accounts == ["ACC-T002"]
    # One hard link scores 0.60; one stage alone never escalates, so the gate holds MONITOR.
    assert mate.disposition == Disposition.MONITOR and mate.single_signal_capped
    assert mate.graph_label == "LINKED_TO_FLAGGED_ACCOUNTS"


def test_rescreening_follows_a_link_to_a_monitor_account():
    # No escalation, so the known-bad list stays empty. ACC-T001 runs first, before its
    # ring-mate is flagged; round 2 re-runs it because a linked account is now MONITOR.
    result = run_batch(_news_ring(), ["ACC-T001", "ACC-T002"], Agents.build())
    assert result.latest["ACC-T002"].disposition == Disposition.MONITOR
    assert result.rounds[1] == ["ACC-T001"]
    mate = result.latest["ACC-T001"]
    assert mate.case_id == "CASE-ACC-T001-2"
    assert mate.graph_findings.connected_flagged_accounts == ["ACC-T002"]
    assert len(result.rounds) == 2  # nothing new after that: the rounds stop


def test_batch_reports_are_drafts_with_every_trail():
    result = run_batch(_news_ring(), ["ACC-T001", "ACC-T002"], Agents.build())
    for report in result.latest.values():
        assert report.is_draft
        assert {"vision", "web_research", "graph", "rag", "auditor"} <= set(report.agent_trails)
        assert report.judge_trail and report.report_outline
        assert report.stage_status["Auditor"] == StageStatus.RAN
    assert len(result.cases) == 3  # ACC-T001 was judged twice


def test_models_grouped_by_phase_and_unloaded_between(tmp_path):
    # Live mode with a fake server: no hit on either account means no model answer is
    # needed; the client is still told to free the GPU at the end of each model phase.
    ctx = _pair([])
    client = fake_client(tmp_path)
    result = run_batch(ctx, ["ACC-T001", "ACC-T002"], Agents.build(client))
    assert [r.disposition for r in result.latest.values()] == [Disposition.CLEAR] * 2
    assert client.fake.chats() == []
    assert list(result.phase_seconds) == [
        "1 Vision + Web Research",
        "2 Judge",
        "3 Admin queue",
        "4 Graph label + RAG + Auditor",
        "5 Backup queue",
        "6 Judge again",
    ]


def _claims(**quotes) -> str:
    return json.dumps({name: quotes.get(name, "") for name in cf.ClaimsForm.model_fields})


def test_backup_phase_settles_a_disagreement_and_the_case_is_judged_again(tmp_path):
    # The two claim reads disagree on "no cash"; the account has cash deposits. The backup
    # model's read settles it, KYC becomes HIGH, and the case goes from CLEAR to MONITOR.
    profile = "Customer stated that no cash deposits would be made into the account."
    rows = [
        txn(amount=30_000, type=TransactionType.CASH_DEPOSIT, timestamp="2026-08-10T10:00:00"),
        txn(amount=30_000, type=TransactionType.CASH_DEPOSIT, timestamp="2026-08-20T10:00:00"),
    ]
    b = bank(
        transactions=rows, kyc_profiles=[KYCProfile(customer_id="CUST-T001", narrative=profile)]
    )
    ctx = TriageContext.build(b, RulebookLookup.load())
    quote = "no cash deposits would be made into the account"
    client = fake_client(tmp_path, _claims(no_cash=quote), _claims(), _claims(no_cash=quote))
    client.fake.replies += ["junk"] * 8  # the Gemma roles fail and fall back
    agents = Agents.build(client, searcher=lambda text: [])
    result = run_batch(ctx, ["ACC-T001"], agents)
    report = result.latest["ACC-T001"]
    assert result.backup_accounts == ["ACC-T001"] == result.changed_by_backup
    assert report.kyc_findings.confidence == 0.65
    assert report.disposition == Disposition.MONITOR
    assert result.warnings and "backup model" in result.warnings[0]
    assert client.fake.chats()[2]["model"] == client.model_for(agents.vision.spec.role.BACKUP)


def test_saved_files(tmp_path):
    result = run_batch(_news_ring(), ["ACC-T001", "ACC-T002"], Agents.build())
    batch.save(result, tmp_path, client_calls=[])
    reports = (tmp_path / "reports.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(reports) == 2 and json.loads(reports[0])["target_account"] == "ACC-T001"
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["dispositions"] == {"MONITOR": 2} and summary["rounds"] == [2, 1]
    assert json.loads((tmp_path / "admin_queue.json").read_text()) == []


def test_call_stats_group_by_agent_and_model():
    calls = [
        dict(agent="vision", model="q", cached=False, ok=True, attempts=1, seconds=2.0),
        dict(agent="vision", model="q", cached=True, ok=True, attempts=2, seconds=4.0),
        dict(agent="vision", model="q", cached=False, ok=False, attempts=2, seconds=9.0),
    ]
    stats = batch.call_stats(calls)["vision (q)"]
    assert (stats["calls"], stats["failed"], stats["from_cache"], stats["retried"]) == (3, 1, 1, 2)
    assert stats["seconds_per_call"] == 3.0
