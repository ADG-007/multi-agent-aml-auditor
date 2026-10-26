"""Stage 1: OCR text -> fields -> compare with the record. A fake reader stands in for
Tesseract; one real-image check runs when the rendered cards exist."""

import json
from pathlib import Path

import pytest

from src.bank import BankData
from src.rulebook import RulebookLookup
from src.stages import stage1_vision as s1
from src.stages.stage0_triage import TriageContext
from tests.factories import bank, customer


AADHAAR = """Name
{name}
Date of Birth
{dob}
Gender
FEMALE
1234 5678 9012
AADHAAR"""

PAN = """INCOME TAX DEPARTMENT
Permanent Account Number
ABCDE1234F
Name
ASHA VERMA
Father's Name
RAVI VERMA
Date of Birth
12/04/1985"""


def _folder(tmp_path):
    manifest = [
        {
            "customer_id": "CUST-T001",
            "account_id": "ACC-T001",
            "document_type": "AADHAAR",
            "file": "a.png",
        },
        {
            "customer_id": "CUST-T001",
            "account_id": "ACC-T001",
            "document_type": "PAN",
            "file": "p.png",
        },
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def _run(tmp_path, aadhaar_text):
    ctx = TriageContext.build(bank(customers=[customer()]), RulebookLookup.load())
    texts = {"a.png": aadhaar_text, "p.png": PAN}
    return s1.run(
        "ACC-T001", ctx, documents_dir=_folder(tmp_path), reader=lambda path: texts[Path(path).name]
    )


def test_matching_documents_score_zero(tmp_path):
    out = _run(tmp_path, AADHAAR.format(name="Asha Verma", dob="12/04/1985"))
    assert out.forgery_confidence == 0.0 and not out.field_discrepancies
    assert out.extracted_fields["AADHAAR.id_number"] == "1234 5678 9012"


def test_a_changed_date_of_birth_pauses(tmp_path):
    out = _run(tmp_path, AADHAAR.format(name="Asha Verma", dob="12/04/1980"))
    assert out.forgery_confidence == 0.95
    assert "AADHAAR dob" in out.field_discrepancies[0]


def test_ocr_noise_in_a_name_is_not_a_forgery(tmp_path):
    out = _run(tmp_path, AADHAAR.format(name="ASHA VER!MA,", dob="12/04/1985"))
    assert out.forgery_confidence == 0.0


def test_an_unread_field_is_a_note_not_a_pause(tmp_path):
    out = _run(tmp_path, "Name\nAsha Verma\n1234 5678 9012")
    assert out.forgery_confidence == 0.30


def test_a_garbled_passport_label_is_still_found():
    text = (
        "PASSPORT\nType / Code\nP/IND\nSumame\nVERMA\nGiven Name(s)\nASHA\nPassport No.\nc1234567"
    )
    fields = s1.agent.extract("PASSPORT", text)
    assert fields["name"] == "ASHA VERMA" and fields["id_number"] == "c1234567"


@pytest.mark.skipif(
    not Path("data/documents/CUST-0004_AADHAAR.png").exists(),
    reason="ID images not rendered (python -m src.generators.id_documents)",
)
def test_real_cards_clean_and_forged():
    ctx = TriageContext.build(BankData.load(), RulebookLookup.load())
    assert s1.run("ACC-0016", ctx).forgery_confidence == 0.0
    assert s1.run("ACC-0004", ctx).forgery_confidence == 0.95
