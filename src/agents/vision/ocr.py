"""ID card tools: the Tesseract read, the label-based field extraction and the field compare.

manifest.json names files only; it carries no forged flag. Name, date of birth and id number
are found by their printed labels and compared with the customer record.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from src.agents.vision import config
from src.matching import normalise
from src.schemas import Customer


MATCH, MISMATCH, UNREAD = "agrees", "DISAGREES", "unread"

#: The fields compared on each card.
FIELDS = ("name", "dob", "id_number")


@dataclass(frozen=True)
class FieldCheck:
    document: str
    field: str
    read: str | None
    expected: str
    result: str


def tesseract(path: Path) -> str:
    """Tesseract's text, saved by image hash: the same image always reads the same."""
    saved = config.OCR_CACHE_DIR / f"{hashlib.sha1(path.read_bytes()).hexdigest()}.txt"
    if saved.exists():
        return saved.read_text(encoding="utf-8")
    import pytesseract  # noqa: PLC0415 - only needed when a card is read for the first time
    from PIL import Image  # noqa: PLC0415

    text = pytesseract.image_to_string(Image.open(path))
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(text, encoding="utf-8")
    return text


def documents_for(folder: Path, customer_id: str) -> list[tuple[str, str]]:
    """(document type, file name) for every card on file."""
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
            found = re.search(r"(?<!\d)\d{4}[ \t]+\d{4}[ \t]+\d{4}(?!\d)", text)
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
    """The line after the line that reads most like `label`, if any line is close enough."""
    scored = [(fuzz.ratio(line.lower(), label.lower()), -i, i) for i, line in enumerate(lines[:-1])]
    best = max(scored, default=None)
    if best is None or best[0] < config.LABEL_MATCH_MIN:
        return None
    return lines[best[2] + 1]


def compare(doc_type: str, fields: dict[str, str | None], customer: Customer) -> list[FieldCheck]:
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
    if not read:
        return FieldCheck(doc_type, field, None, expected, UNREAD)
    verdict = agrees(read, expected)
    result = UNREAD if verdict is None else MATCH if verdict else MISMATCH
    return FieldCheck(doc_type, field, read, expected, result)


def _name_agrees(read: str, expected: str) -> bool:
    """Word order and case do not matter; neither does a stray mark that splits a word."""
    a, b = normalise(read), normalise(expected)
    best = max(fuzz.token_sort_ratio(a, b), fuzz.ratio(a.replace(" ", ""), b.replace(" ", "")))
    return best >= config.NAME_MATCH_MIN


def _dob_agrees(read: str, expected: str) -> bool | None:
    found = re.search(r"(\d{2})[/.-](\d{2})[/.-](\d{4})", read)
    if not found:
        return None
    try:
        printed = datetime.strptime("/".join(found.groups()), "%d/%m/%Y").date()
    except ValueError:
        return None
    return printed == datetime.strptime(expected, "%d/%m/%Y").date()


def _id_agrees(read: str, expected: str) -> bool:
    a = re.sub(r"[^0-9A-Z]", "", read.upper())
    b = re.sub(r"[^0-9A-Z]", "", expected.upper())
    return Levenshtein.distance(a, b) <= config.ID_NUMBER_MAX_EDITS
