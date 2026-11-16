"""The page: its data helpers, and a smoke run of the whole script on a saved batch."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.demo import app
from src.orchestration import batch
from src.orchestration.steps import Agents
from src.rulebook import RulebookLookup
from src.schemas import SeverityTier, TransactionDirection
from src.stages.stage0_triage import TriageContext
from tests.factories import account, bank, customer, media_entry, txn


def _saved_run(folder):
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
    ctx = TriageContext.build(b, RulebookLookup.load())
    result = batch.run_batch(ctx, ["ACC-T001", "ACC-T002"], Agents.build())
    batch.save(result, folder, client_calls=[])
    return result


def test_run_folders_and_rows(tmp_path):
    _saved_run(tmp_path / "run_a" / "live")
    assert app.run_folders(tmp_path) == [tmp_path / "run_a" / "live"]
    reports = app.load_reports(tmp_path / "run_a" / "live")
    rows = app.result_rows(reports)
    assert [r["account"] for r in rows] == ["ACC-T001", "ACC-T002"]
    assert rows[0]["label"] == "LINKED_TO_FLAGGED_ACCOUNTS" and rows[0]["draft"] is True
    assert rows[1]["verdict"].endswith("MONITOR")


def test_page_renders_a_saved_run_without_error(tmp_path, monkeypatch):
    _saved_run(tmp_path / "run_a" / "live")
    monkeypatch.setenv("AML_OUTPUTS", str(tmp_path))
    at = AppTest.from_file(str(Path(app.__file__)), default_timeout=120)
    at.run()
    assert not at.exception
    assert at.title[0].value == "AML Auditor v2"
    assert [m.label for m in at.metric[:4]] == ["Accounts", "Clear", "Monitor", "Escalate"]
    assert at.metric[2].value == "2"


def test_page_renders_with_no_saved_run(tmp_path, monkeypatch):
    monkeypatch.setenv("AML_OUTPUTS", str(tmp_path / "empty"))
    at = AppTest.from_file(str(Path(app.__file__)), default_timeout=120)
    at.run()
    assert not at.exception
