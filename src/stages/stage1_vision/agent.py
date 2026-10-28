"""Stage 1's entry point: read the ID cards on file and compare them with the record.

    run(account_id, ctx, as_of, transaction) -> Stage1VisionOutput

Tesseract reads each listed card (manifest.json names files only, no forged flag). Name, DOB
and id number are found by their labels. A clear mismatch gives 0.95, which pauses the case.
Never scored. No model calls.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pytesseract
from PIL import Image
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from src.matching import normalise
from src.schemas import (
    FORGERY_PAUSE_THRESHOLD,
    Customer,
    ReasoningStep,
    Stage1VisionOutput,
    Transaction,
)
from src.stages.stage0_triage import TriageContext
from src.stages.stage1_vision import config


MATCH, MISMATCH, UNREAD = "agrees", "DISAGREES", "unread"


@dataclass(frozen=True)
class FieldCheck:
    document: str
    field: str
    read: str | None
    expected: str
    result: str


def run(
    account_id: str,
    ctx: TriageContext,
    as_of: datetime | None = None,
    transaction: Transaction | None = None,
    documents_dir: Path | None = None,
    reader=None,
) -> Stage1VisionOutput:
    """`reader` turns an image path into text; tests pass a fake, the pipeline uses Tesseract."""
    folder = documents_dir or ctx.documents_dir or config.DOCUMENTS_DIR
    read_text = reader or _tesseract
    customer = ctx.bank.customer_for(account_id)
    documents = _documents_for(folder, customer.customer_id)
    if not documents:
        raise FileNotFoundError(f"no ID document on file for {customer.customer_id} in {folder}")

    extracted: dict[str, str] = {}
    checks: list[FieldCheck] = []
    for doc_type, file in documents:
        fields = extract(doc_type, read_text(folder / file))
        extracted.update({f"{doc_type}.{k}": v for k, v in fields.items()})
        checks += compare(doc_type, fields, customer)

    mismatches = [c for c in checks if c.result == MISMATCH]
    unread = [c for c in checks if c.result == UNREAD]
    if mismatches:
        confidence = config.MISMATCH_CONFIDENCE
    elif unread:
        confidence = config.UNREAD_CONFIDENCE
    else:
        confidence = 0.0

    trail = [
        ReasoningStep(
            step=f"OCR {doc_type}",
            finding="; ".join(
                f"{c.field} {c.result}" + (f" (read {c.read!r})" if c.result != MATCH else "")
                for c in checks
                if c.document == doc_type
            ),
            impact="Printed fields compared with the customer record (DOC-001).",
        )
        for doc_type, _ in documents
    ]
    trail.append(
        ReasoningStep(
            step="Forgery confidence",
            finding=f"{len(mismatches)} disagreeing field(s), {len(unread)} unread.",
            impact=_impact(confidence),
        )
    )
    return Stage1VisionOutput(
        document_type="+".join(doc_type for doc_type, _ in documents),
        extracted_fields=extracted,
        field_discrepancies=[
            f"{c.document} {c.field}: printed {c.read!r}, record {c.expected!r}" for c in mismatches
        ],
        forgery_confidence=confidence,
        reasoning_trail=trail,
    )


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------


def _tesseract(path: Path) -> str:
    """Tesseract's text, saved by image hash: the same image always reads the same."""
    saved = config.OCR_CACHE_DIR / f"{hashlib.sha1(path.read_bytes()).hexdigest()}.txt"
    if saved.exists():
        return saved.read_text(encoding="utf-8")
    text = pytesseract.image_to_string(Image.open(path))
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(text, encoding="utf-8")
    return text


def _documents_for(folder: Path, customer_id: str) -> list[tuple[str, str]]:
    with (folder / config.MANIFEST_NAME).open(encoding="utf-8") as f:
        manifest = json.load(f)
    return [(m["document_type"], m["file"]) for m in manifest if m["customer_id"] == customer_id]


def extract(doc_type: str, text: str) -> dict[str, str]:
    """The printed name, date of birth and id number, found by their labels."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    fields: dict[str, str] = {}
    if doc_type == "PASSPORT":
        surname, given = _after(lines, "Surname"), _after(lines, "Given Name(s)")
        if surname or given:
            fields["name"] = f"{given or ''} {surname or ''}".strip()
        number = _after(lines, "Passport No.")
        if number:
            fields["id_number"] = number
        expiry = _after(lines, "Date of Expiry")
        if expiry:
            fields["expiry"] = expiry
    else:
        name = _after(lines, "Name")
        if name:
            fields["name"] = name
        if doc_type == "AADHAAR":
            found = re.search(r"\b\d{4}\s\d{4}\s\d{4}\b", text)
            if found:
                fields["id_number"] = found.group()
        if doc_type == "PAN":
            pan = _after(lines, "Permanent Account Number")
            if pan:
                fields["pan_number"] = pan
    dob = _after(lines, "Date of Birth")
    if dob:
        fields["dob"] = dob
    return fields


def _after(lines: list[str], label: str) -> str | None:
    """The line after the line that reads most like `label` (OCR can garble a label:
    "Sumame" for "Surname"), if any line is close enough."""
    scored = [(fuzz.ratio(line.lower(), label.lower()), -i, i) for i, line in enumerate(lines[:-1])]
    best = max(scored, default=None)
    if best is None or best[0] < config.LABEL_MATCH_MIN:
        return None
    return lines[best[2] + 1]


# --------------------------------------------------------------------------------------
# Comparing
# --------------------------------------------------------------------------------------


def compare(doc_type: str, fields: dict[str, str], customer: Customer) -> list[FieldCheck]:
    checks = [
        _check(doc_type, "name", fields.get("name"), customer.name, _name_agrees),
        _check(doc_type, "dob", fields.get("dob"), customer.dob.strftime("%d/%m/%Y"), _dob_agrees),
    ]
    # A PAN number is printed but not stored on the record, so only the OVD's number is checked.
    if doc_type == customer.id_document_type.value:
        checks.append(
            _check(doc_type, "id_number", fields.get("id_number"), customer.id_number, _id_agrees)
        )
    return checks


def _check(doc_type, field, read, expected, agrees) -> FieldCheck:
    if read is None:
        return FieldCheck(doc_type, field, None, expected, UNREAD)
    verdict = agrees(read, expected)
    result = UNREAD if verdict is None else MATCH if verdict else MISMATCH
    return FieldCheck(doc_type, field, read, expected, result)


def _name_agrees(read: str, expected: str) -> bool:
    """Word order and case do not matter; neither does a stray OCR mark that splits a word
    ("TANMAY!I" for "TANMAYI"), so the names are also compared with spaces removed."""
    a, b = normalise(read), normalise(expected)
    best = max(fuzz.token_sort_ratio(a, b), fuzz.ratio(a.replace(" ", ""), b.replace(" ", "")))
    return best >= config.NAME_MATCH_MIN


def _dob_agrees(read: str, expected: str) -> bool | None:
    found = re.search(r"\d{2}/\d{2}/\d{4}", read)
    if not found:
        return None
    try:
        printed = datetime.strptime(found.group(), "%d/%m/%Y").date()
    except ValueError:
        return None
    return printed == datetime.strptime(expected, "%d/%m/%Y").date()


def _id_agrees(read: str, expected: str) -> bool:
    a = re.sub(r"[^0-9A-Z]", "", read.upper())
    b = re.sub(r"[^0-9A-Z]", "", expected.upper())
    return Levenshtein.distance(a, b) <= config.ID_NUMBER_MAX_EDITS


def _impact(confidence: float) -> str:
    if confidence >= FORGERY_PAUSE_THRESHOLD:
        return (
            "A printed field disagrees with the record: the case pauses for the admin. "
            "Never scored (DOC-002)."
        )
    if confidence > 0:
        return "Some fields could not be read; noted for the analyst, no pause. Never scored."
    return "Every field read agrees with the record. Never scored."


__all__ = ["run", "extract", "compare", "date"]
