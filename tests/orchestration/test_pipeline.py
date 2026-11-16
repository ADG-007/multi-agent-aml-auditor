"""The live graph: node order, the admin pause, degraded agents, explained cash."""

import json

import pytest

from src.agents.vision import config as vision_config
from src.agents.vision import ocr
from src.orchestration.case_runner import run_case
from src.orchestration.pipeline import NODES, Paused, Pipeline
from src.orchestration.run_state import RunState
from src.orchestration.steps import Agents
from src.rulebook import RulebookLookup
from src.schemas import (
    Disposition,
    KYCProfile,
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
from tests.factories import bank, fake_client, txn


S = StageName
GOOD_CARD = "Name\nAsha Verma\nDate of Birth\n12/04/1985\n1234 5678 9012\n"
FORGED_CARD = "Name\nAsha Verma\nDate of Birth\n12/04/1981\n1234 5678 9012\n"


def _review(account_id="ACC-T001"):
    return TriageEvent(kind=TriageEventKind.PERIODIC_REVIEW, account_id=account_id)


def _ctx(rows=None, profile=None):
    profiles = [KYCProfile(customer_id="CUST-T001", narrative=profile)] if profile else []
    b = bank(transactions=rows or [txn(timestamp="2026-08-01T10:00:00")], kyc_profiles=profiles)
    return TriageContext.build(b, RulebookLookup.load())


@pytest.fixture
def documents(tmp_path, monkeypatch):
    """One ID card on file; `card["text"]` is what Tesseract reads from it."""
    folder = tmp_path / "documents"
    folder.mkdir()
    manifest = [{"customer_id": "CUST-T001", "document_type": "AADHAAR", "file": "card.png"}]
    (folder / vision_config.MANIFEST_NAME).write_text(json.dumps(manifest))
    (folder / "card.png").write_bytes(b"image")
    card = {"text": GOOD_CARD}
    monkeypatch.setattr(ocr, "tesseract", lambda path: card["text"])
    return folder, card


def _cash_rows(amount=45_000, days=range(1, 21), months=(7, 8)):
    return [
        txn(
            amount=amount,
            type=TransactionType.CASH_DEPOSIT,
            timestamp=f"2026-{month:02d}-{day:02d}T10:00:00",
        )
        for month in months
        for day in days
    ]


def test_nodes_run_in_order_and_the_plain_runner_agrees(documents):
    folder, _ = documents
    ctx = _ctx()
    seen = []
    pipeline = Pipeline(ctx, Agents.build(documents_dir=folder))
    report = pipeline.run(_review(), on_node=lambda name, work, seconds: seen.append(name))
    assert tuple(seen) == NODES
    assert report.disposition == Disposition.CLEAR and report.pipeline_status.value == "COMPLETED"
    assert report.stage_status[S.VISION.value] == StageStatus.RAN
    plain = run_case(_review(), ctx, RunState(), agents=Agents.build(documents_dir=folder))
    assert (plain.disposition, plain.confidence, plain.stage_status) == (
        report.disposition,
        report.confidence,
        report.stage_status,
    )


def test_id_flagged_by_both_reads_pauses_and_resumes(tmp_path, documents):
    folder, card = documents
    card["text"] = FORGED_CARD
    image = '{"name": "Asha Verma", "date_of_birth": "12/04/1981", "id_number": "1234 5678 9012"}'
    client = fake_client(tmp_path, image)
    pipeline = Pipeline(_ctx(), Agents.build(client, documents_dir=folder))
    paused = pipeline.run(_review())
    assert isinstance(paused, Paused)
    assert paused.evidence["forgery_confidence"] == vision_config.BOTH_READS_DISAGREE
    assert paused.evidence["disposition_if_approved"] == "CLEAR"
    preview = pipeline.preview(paused)
    assert preview.pipeline_status == PipelineStatus.PAUSED_PENDING_ADMIN_REVIEW
    assert "paused the case" in preview.forgery_notice

    report = pipeline.resume(paused, "approve")
    # The ID check never scores: approved, the case is what the judge said.
    assert report.disposition == Disposition.CLEAR and report.confidence == 0.0
    assert "Admin decision: APPROVE" in report.forgery_notice
    assert S.VISION.value not in report.judge_inputs


def test_admin_block_ends_the_case_blocked(tmp_path, documents):
    folder, card = documents
    card["text"] = FORGED_CARD
    image = '{"name": "Asha Verma", "date_of_birth": "12/04/1981", "id_number": "1234 5678 9012"}'
    pipeline = Pipeline(_ctx(), Agents.build(fake_client(tmp_path, image), documents_dir=folder))
    report = pipeline.resume(pipeline.run(_review()), "block")
    assert report.disposition == Disposition.BLOCKED_BY_ADMIN_REVIEW
    assert report.pipeline_status == PipelineStatus.BLOCKED_BY_ADMIN


def test_quiet_transaction_on_a_flagged_account_opens_no_case():
    rows = [txn(amount=950_000, type=TransactionType.CASH_DEPOSIT, timestamp="2026-08-01T10:00")]
    ctx = _ctx(rows)
    state = RunState()
    first = TriageEvent(
        kind=TriageEventKind.TRANSACTION, account_id="ACC-T001", transaction=rows[0]
    )
    plan = StubPlan({S.TRANSACTION_PATTERN: StageResult(StageStatus.RAN, 0.8)})
    pipeline = Pipeline(ctx, state=state, plan=plan)
    assert pipeline.run(first).disposition == Disposition.MONITOR
    quiet = txn(timestamp="2026-08-02T10:00:00")
    ctx.bank.add_transaction(quiet)
    event = TriageEvent(kind=TriageEventKind.TRANSACTION, account_id="ACC-T001", transaction=quiet)
    assert pipeline.run(event) is None


def test_a_failed_weighted_agent_floors_the_case_at_monitor(monkeypatch, documents):
    folder, _ = documents
    agents = Agents.build(documents_dir=folder)

    def boom(*args, **kwargs):
        raise RuntimeError("list unavailable")

    monkeypatch.setattr(agents.web, "run", boom)
    rows = [txn(counterparty_name="Vikram Sethi", timestamp="2026-08-01T10:00:00")]
    from tests.factories import media_entry

    b = bank(transactions=rows, media=[media_entry()])
    ctx = TriageContext.build(b, RulebookLookup.load())
    report = Pipeline(ctx, agents).run(_review())
    assert report.stage_status[S.ADVERSE_MEDIA.value] == StageStatus.FAILED
    assert report.disposition == Disposition.MONITOR and report.evidence_incomplete
    assert "list unavailable" in report.investigation_narrative


def test_explained_cash_sets_the_monthly_finding_aside(tmp_path, documents):
    folder, _ = documents
    shop = (
        "Runs a shop. The shop's cash takings are deposited at the branch on most working "
        "days, around Rs 9.20 lakh a month, and paid on to suppliers."
    )
    quote = "cash takings are deposited at the branch on most working days, around Rs 9.20 lakh"
    read = json.dumps(
        {
            name: (quote if name == "cash_takings" else "")
            for name in (
                "no_cash cash_takings no_foreign foreign_dealings no_business "
                "business_receipts single_source one_off_event expected_activity"
            ).split()
        }
    )
    rows = _cash_rows(amount=46_000)  # 9.2 lakh in each of two months: m = 2
    explained = fake_client(tmp_path / "a", read, read)
    explained.fake.replies += ["junk"] * 6
    report = Pipeline(
        _ctx(rows, shop), Agents.build(explained, documents_dir=folder, searcher=lambda t: [])
    ).run(_review())
    assert report.transaction_pattern_findings.confidence == 0.6
    assert report.judge_inputs[S.TRANSACTION_PATTERN.value] == 0.0
    assert report.judge_trail[0].step == "Explained cash"
    assert report.disposition == Disposition.CLEAR

    # The same cash with a profile that declares nothing stays a finding.
    silent = fake_client(tmp_path / "b", read.replace(quote, ""), read.replace(quote, ""))
    silent.fake.replies += ["junk"] * 6
    plain = Pipeline(
        _ctx(rows, "Salaried employee."),
        Agents.build(silent, documents_dir=folder, searcher=lambda t: []),
    ).run(_review())
    assert plain.judge_inputs[S.TRANSACTION_PATTERN.value] == 0.6
    assert plain.disposition == Disposition.MONITOR


def test_sign_off_ends_the_draft(documents):
    folder, _ = documents
    report = Pipeline(_ctx(), Agents.build(documents_dir=folder)).run(_review())
    draft = report.model_copy(update={"is_draft": True})
    assert Pipeline.sign_off(draft).is_draft is False
