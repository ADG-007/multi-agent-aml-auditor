"""A small saved run and an app over it."""

import json

import pytest
from fastapi.testclient import TestClient

from src.agents.vision import config as vision_config
from src.agents.vision import ocr
from src.api.app import ApiConfig, create_app
from src.orchestration import batch
from src.orchestration.steps import Agents
from src.rulebook import RulebookLookup
from src.schemas import SeverityTier, TransactionDirection, TransactionType
from src.stages.stage0_triage import TriageContext
from tests.factories import account, bank, customer, media_entry, txn


PRESENTER = ("127.0.0.1", 50000)
VISITOR = ("192.168.1.9", 50000)
RUN = "run_a~live"


def small_ctx() -> TriageContext:
    rows = [
        txn(account_id="ACC-T001", device_id="DEV-RING", direction=TransactionDirection.DEBIT),
        txn(account_id="ACC-T002", device_id="DEV-RING", direction=TransactionDirection.DEBIT),
        txn(
            account_id="ACC-T002",
            amount=40_000,
            type=TransactionType.CASH_DEPOSIT,
            direction=TransactionDirection.CREDIT,
            timestamp="2026-08-03T10:00:00",
        ),
        txn(
            account_id="ACC-T002",
            amount=900,
            type=TransactionType.UPI,
            direction=TransactionDirection.CREDIT,
            timestamp="2026-08-04T10:00:00",
        ),
    ]
    b = bank(
        accounts=[account(), account("ACC-T002", "CUST-T002")],
        customers=[customer(), customer("CUST-T002", name="Vikram Sethi", address="9 Beach Rd")],
        transactions=rows,
        media=[media_entry(tier=SeverityTier.ADVERSE_NEWS)],
    )
    return TriageContext.build(b, RulebookLookup.load())


@pytest.fixture
def ctx():
    return small_ctx()


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """A data folder with one ID card on file for ACC-T001."""
    folder = tmp_path / "data" / "documents"
    folder.mkdir(parents=True)
    manifest = [
        {
            "customer_id": "CUST-T001",
            "account_id": "ACC-T001",
            "document_type": "AADHAAR",
            "file": "card.png",
        }
    ]
    (folder / vision_config.MANIFEST_NAME).write_text(json.dumps(manifest))
    (folder / "card.png").write_bytes(b"image")
    monkeypatch.setattr(ocr, "tesseract", lambda path: "")
    return tmp_path / "data"


@pytest.fixture
def config(tmp_path, ctx, data_dir):
    result = batch.run_batch(ctx, ["ACC-T001", "ACC-T002"], Agents.build())
    batch.save(result, tmp_path / "outputs" / "run_a" / "live", client_calls=[])
    return ApiConfig(outputs=tmp_path / "outputs", data_dir=data_dir, static_dir=None)


@pytest.fixture
def make_client(config, ctx):
    def make(client=PRESENTER, online=True, jobs=None):
        app = create_app(config, ctx=ctx, jobs=jobs, probe=lambda: online)
        return TestClient(app, client=client)

    return make
