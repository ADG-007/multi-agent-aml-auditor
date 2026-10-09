"""Generator 6 of 9 - rendered identity document images.

Used exclusively by Stage 1, which OCRs these with Tesseract and cross-checks the extracted
fields against the customer record. No vision model is involved anywhere in the pipeline.

Two documents per customer, paired the way a real file is: a domestic customer submits
Aadhaar plus PAN, an NRI submits passport plus PAN - two images per customer.

Layouts follow the real documents' field sets and proportions - CR80 card geometry for
Aadhaar and PAN, and a passport data page. The passport's machine-readable zone is left off:
Tesseract cannot read it without a specialist OCR-B reader, Stage 1 reads the printed fields
instead, and a zone nothing reads is decoration. Following phase_1_execution.md's
calibration, the Aadhaar number carries no Verhoeff digit. Nothing downstream depends on it:
forgery detection works from visual tampering and from field cross-checking, not from
arithmetic on the number.

The forged subset gets real visual tampering - a field re-rendered in the wrong typeface, a
value overlaid slightly out of alignment, a patch of background painted over and rewritten -
so that the image-forensic half of Stage 1's signal has something genuine to find. A
text-only difference would let Stage 1 pass while doing none of the work it claims to.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.generators import config as cfg
from src.generators.identifiers import pan_number
from src.generators.names import INDIAN_MALE_FIRST_NAMES
from src.generators.streams import Streams
from src.schemas import Customer, IdDocumentType, ResidencyStatus

# CR80 card geometry at 300 dpi - the real dimensions of an Aadhaar or PAN card.
CARD_WIDTH, CARD_HEIGHT = 1011, 638

# A passport data page at roughly 200 dpi.
PAGE_WIDTH, PAGE_HEIGHT = 984, 693

#: Font families to try, in order. Tesseract reads clean sans and serif faces well.
FONT_CANDIDATES = {
    "sans": ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Helvetica.ttc"),
    "sans_bold": ("arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"),
    "serif": ("times.ttf", "DejaVuSerif.ttf", "LiberationSerif-Regular.ttf"),
}

FONT_SEARCH_DIRS = (
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/truetype/liberation"),
    Path("/Library/Fonts"),
)

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def load_font(style: str, size: int) -> ImageFont.FreeTypeFont:
    """Resolve a font by style, falling back across platforms.

    OCR accuracy depends directly on this: PIL's built-in bitmap font is tiny and renders
    poorly, so a real TrueType face is worth searching for. If none is found the documents
    still render, but Stage 1's extraction quality will suffer - which is why the
    verification step reports which family was actually used.
    """
    key = (style, size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    for filename in FONT_CANDIDATES[style]:
        for directory in FONT_SEARCH_DIRS:
            candidate = directory / filename
            if candidate.exists():
                font = ImageFont.truetype(str(candidate), size)
                _FONT_CACHE[key] = font
                return font
        try:
            font = ImageFont.truetype(filename, size)
            _FONT_CACHE[key] = font
            return font
        except OSError:
            continue

    font = ImageFont.load_default(size)
    _FONT_CACHE[key] = font
    return font


def resolved_font_family(style: str = "sans") -> str:
    """Which font file actually backs a style, for the verification summary."""
    font = load_font(style, 20)
    return getattr(font, "path", "PIL default bitmap font")


def _photo_placeholder(image: Image.Image, box: tuple[int, int, int, int]) -> None:
    """A neutral silhouette where a photograph would be.

    Deliberately not a synthesised face: the document only needs the photo region to exist
    so the layout is right and OCR is not confused by an empty rectangle.

    Drawn on a tile the size of the frame and pasted in, so the shoulders are cut off at the
    frame's edge the way a cropped photograph is, rather than spilling over the card below.
    """
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    tile = Image.new("RGB", (width, height), (226, 228, 232))
    tile_draw = ImageDraw.Draw(tile)
    head_r = width // 6
    cx = width // 2
    cy = height // 3
    tile_draw.ellipse((cx - head_r, cy - head_r, cx + head_r, cy + head_r), fill=(170, 174, 180))
    body_w = width // 2
    tile_draw.ellipse(
        (cx - body_w, cy + head_r // 2, cx + body_w, height + height // 3),
        fill=(170, 174, 180),
    )
    image.paste(tile, (x0, y0))
    ImageDraw.Draw(image).rectangle(box, outline=(140, 144, 150), width=2)


def _guilloche(draw: ImageDraw.ImageDraw, size: tuple[int, int], colour: tuple[int, int, int]) -> None:
    """A faint repeating background pattern, standing in for a security print.

    Its real job is to give the tampering something to disturb: painting a patch over this
    pattern leaves a visible discontinuity, which is exactly the edge irregularity Stage 1's
    image-forensic heuristic looks for.
    """
    width, height = size
    for offset in range(-height, width, 26):
        draw.line((offset, 0, offset + height, height), fill=colour, width=1)
    for offset in range(0, width + height, 26):
        draw.line((offset, 0, offset - height, height), fill=colour, width=1)


def _field(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    label: str,
    value: str,
    label_font: ImageFont.FreeTypeFont,
    value_font: ImageFont.FreeTypeFont,
    label_colour: tuple[int, int, int] = (90, 94, 100),
    value_colour: tuple[int, int, int] = (20, 22, 26),
) -> None:
    x, y = position
    draw.text((x, y), label, font=label_font, fill=label_colour)
    draw.text((x, y + 22), value, font=value_font, fill=value_colour)


def render_aadhaar(customer: Customer, rng: random.Random) -> Image.Image:
    """The Aadhaar letter's card face: emblem band, photo, particulars, number."""
    image = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), (253, 252, 248))
    draw = ImageDraw.Draw(image)
    _guilloche(draw, (CARD_WIDTH, CARD_HEIGHT), (246, 240, 228))

    # Saffron header band and green footer band, as on the real card.
    draw.rectangle((0, 0, CARD_WIDTH, 96), fill=(255, 153, 51))
    draw.rectangle((0, CARD_HEIGHT - 84, CARD_WIDTH, CARD_HEIGHT), fill=(19, 136, 8))

    header_font = load_font("sans_bold", 30)
    sub_font = load_font("sans", 22)
    label_font = load_font("sans", 19)
    value_font = load_font("sans_bold", 27)

    draw.text((150, 18), "GOVERNMENT OF INDIA", font=header_font, fill=(255, 255, 255))
    draw.text(
        (150, 56),
        "UNIQUE IDENTIFICATION AUTHORITY OF INDIA",
        font=sub_font,
        fill=(255, 255, 255),
    )
    draw.ellipse((40, 14, 120, 82), outline=(255, 255, 255), width=3)

    _photo_placeholder(image, (44, 140, 214, 380))

    _field(draw, (250, 150), "Name", customer.name, label_font, value_font)
    _field(
        draw,
        (250, 228),
        "Date of Birth",
        customer.dob.strftime("%d/%m/%Y"),
        label_font,
        value_font,
    )
    _field(draw, (250, 306), "Gender", customer.gender.value, label_font, value_font)

    address_font = load_font("sans", 18)
    draw.text((250, 384), "Address", font=label_font, fill=(90, 94, 100))
    # Wrapped at the commas between address parts, as the printed card does, never mid-word.
    lines: list[str] = []
    for part in customer.address.split(", "):
        if lines and len(lines[-1]) + 2 + len(part) <= 52:
            lines[-1] += f", {part}"
        else:
            lines.append(part)
    for row, line in enumerate(lines[:3]):
        draw.text((250, 408 + row * 22), line, font=address_font, fill=(40, 42, 46))

    # The Aadhaar number sits in dark type on the light card body, immediately above the
    # green band - which is where the real card prints it, and also the only way Tesseract
    # reads it reliably. An earlier revision set it in white on the green footer; it looked
    # right and OCR could not recover a single digit, which would have left Stage 1 unable to
    # cross-check the one field that identifies the document.
    number_font = load_font("sans_bold", 46)
    draw.text(
        (CARD_WIDTH // 2 - 200, CARD_HEIGHT - 148),
        customer.id_number,
        font=number_font,
        fill=(20, 22, 26),
    )
    draw.text(
        (CARD_WIDTH // 2, CARD_HEIGHT - 66),
        "AADHAAR",
        font=load_font("sans", 24),
        fill=(255, 255, 255),
        anchor="ma",
    )
    return image


def render_pan(customer: Customer, pan: str, rng: random.Random) -> Image.Image:
    """The PAN card face: department header, number, name, father's name, date of birth."""
    image = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), (232, 240, 250))
    draw = ImageDraw.Draw(image)
    _guilloche(draw, (CARD_WIDTH, CARD_HEIGHT), (216, 228, 244))

    draw.rectangle((0, 0, CARD_WIDTH, 92), fill=(21, 72, 138))

    header_font = load_font("sans_bold", 28)
    sub_font = load_font("sans", 21)
    label_font = load_font("sans", 19)
    value_font = load_font("sans_bold", 27)

    draw.text((32, 16), "INCOME TAX DEPARTMENT", font=header_font, fill=(255, 255, 255))
    draw.text((32, 54), "GOVT. OF INDIA", font=sub_font, fill=(214, 226, 244))
    draw.text(
        (CARD_WIDTH - 360, 34),
        "PERMANENT ACCOUNT NUMBER CARD",
        font=load_font("sans", 17),
        fill=(255, 255, 255),
    )

    number_font = load_font("sans_bold", 44)
    draw.text((36, 128), "Permanent Account Number", font=label_font, fill=(90, 94, 100))
    draw.text((36, 154), pan, font=number_font, fill=(16, 40, 86))

    _field(draw, (36, 246), "Name", customer.name.upper(), label_font, value_font)
    _field(
        draw,
        (36, 330),
        "Father's Name",
        # A real PAN card prints the father's full name. He shares the holder's surname.
        f"{rng.choice(INDIAN_MALE_FIRST_NAMES)} {customer.name.split()[-1]}".upper(),
        label_font,
        value_font,
    )
    _field(
        draw,
        (36, 414),
        "Date of Birth",
        customer.dob.strftime("%d/%m/%Y"),
        label_font,
        value_font,
    )

    _photo_placeholder(image, (CARD_WIDTH - 250, 150, CARD_WIDTH - 60, 410))

    draw.line((CARD_WIDTH - 250, CARD_HEIGHT - 96, CARD_WIDTH - 60, CARD_HEIGHT - 96), fill=(90, 94, 100), width=2)
    draw.text(
        (CARD_WIDTH - 250, CARD_HEIGHT - 90),
        "Signature",
        font=load_font("sans", 17),
        fill=(90, 94, 100),
    )
    return image


def render_passport(customer: Customer, rng: random.Random) -> Image.Image:
    """The passport data page's printed particulars."""
    image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (243, 238, 230))
    draw = ImageDraw.Draw(image)
    _guilloche(draw, (PAGE_WIDTH, PAGE_HEIGHT), (234, 226, 214))

    header_font = load_font("serif", 27)
    label_font = load_font("sans", 16)
    value_font = load_font("sans_bold", 23)

    draw.text((32, 22), "REPUBLIC OF INDIA", font=header_font, fill=(60, 22, 22))
    draw.text((32, 58), "PASSPORT", font=load_font("serif", 21), fill=(60, 22, 22))

    _photo_placeholder(image, (32, 110, 250, 390))

    parts = customer.name.upper().split()
    surname = parts[-1]
    given = " ".join(parts[:-1]) or "-"
    expiry = customer.id_expiration_date or (cfg.REFERENCE_DATE + timedelta(days=3650))
    issue = expiry - timedelta(days=3650)

    left = 290
    _field(draw, (left, 110), "Type / Code", "P / IND", label_font, value_font)
    _field(draw, (left + 260, 110), "Passport No.", customer.id_number, label_font, value_font)
    _field(draw, (left, 176), "Surname", surname, label_font, value_font)
    _field(draw, (left, 242), "Given Name(s)", given, label_font, value_font)
    _field(draw, (left, 308), "Nationality", "INDIAN", label_font, value_font)
    _field(
        draw,
        (left + 260, 308),
        "Date of Birth",
        customer.dob.strftime("%d/%m/%Y"),
        label_font,
        value_font,
    )
    _field(draw, (left, 374), "Place of Issue", rng.choice(("MUMBAI", "DELHI", "CHENNAI", "KOCHI", "HYDERABAD")), label_font, value_font)
    _field(draw, (left + 260, 374), "Sex", customer.gender.value[0], label_font, value_font)
    _field(draw, (left, 440), "Date of Issue", issue.strftime("%d/%m/%Y"), label_font, value_font)
    _field(draw, (left + 260, 440), "Date of Expiry", expiry.strftime("%d/%m/%Y"), label_font, value_font)
    return image


#: Where each tamperable value actually sits on each layout: (x, y, width, height).
#: Held as data rather than guessed at, because a patch drawn at the wrong offset leaves the
#: original value legible and adds a second contradictory one beside it. That still produces
#: a discrepancy Stage 1 would catch, but it is not a forgery - a forger who leaves the
#: original date visible has not forged anything.
TAMPER_TARGETS: dict[str, dict[str, tuple[int, int, int, int]]] = {
    "AADHAAR": {
        "name": (248, 168, 480, 40),
        "date_of_birth": (248, 246, 300, 40),
        "id_number": (303, 486, 420, 54),
    },
    "PASSPORT": {
        "name": (288, 194, 420, 38),
        "date_of_birth": (548, 326, 300, 38),
        "id_number": (548, 128, 380, 38),
    },
}

#: Card background behind a patched region, per layout.
TAMPER_BACKGROUND = {"AADHAAR": (253, 252, 248), "PASSPORT": (243, 238, 230)}


def apply_tampering(
    image: Image.Image,
    customer: Customer,
    document_type: str,
    rng: random.Random,
) -> tuple[Image.Image, list[str]]:
    """Visibly alter a rendered document, the way a crude forgery actually looks.

    Three techniques, two or three applied per document:

      patch-and-rewrite  paint an opaque rectangle over a printed value and type a new one
                         into it. The patch interrupts the background pattern, leaving the
                         edge discontinuity an image-forensic check is meant to notice.
      typeface swap      re-render one field in a different family and size from the rest of
                         the card, which is the single most common giveaway on a real
                         forgery.
      misalignment       set a value a few pixels off the baseline its label sits on, so the
                         field no longer lines up with its neighbours.

    The rewritten values also disagree with the customer record, which is what gives Stage
    1's OCR cross-check something to catch independently of the visual signal. Both halves
    are present on purpose: either alone would let one half of the stage look effective
    while doing nothing.
    """
    tampered = image.copy()
    draw = ImageDraw.Draw(tampered)
    applied: list[str] = []

    targets = TAMPER_TARGETS[document_type]
    background = TAMPER_BACKGROUND[document_type]

    def blank(field: str) -> tuple[int, int, int, int]:
        x, y, width, height = targets[field]
        draw.rectangle((x, y, x + width, y + height), fill=background)
        return x, y, width, height

    # The date-of-birth patch always applies, plus one or both of the other two. That
    # guarantees every forged document carries both halves of Stage 1's signal: a field that
    # disagrees with the customer record, and a visual artefact in the image. Leaving the
    # patch optional would produce forgeries detectable only by image forensics, which is
    # realistic but would make a weak forensic heuristic look like a false negative rather
    # than like a weak heuristic. A visual-only variant is worth adding once Phase 5 has
    # measured how well the forensic half actually performs.
    techniques = ["patch"] + rng.sample(["typeface", "misalign"], k=rng.choice((1, 2)))

    for technique in techniques:
        if technique == "patch":
            # Paint out the printed date of birth and type a different one into the gap.
            # The patch interrupts the background pattern, which is the edge discontinuity
            # the image-forensic half of Stage 1 is meant to notice.
            x, y, _width, _height = blank("date_of_birth")
            year = customer.dob.year - rng.choice((3, 5, 7))
            # A 29 February birthday has no counterpart in a non-leap year; the 28th stands in.
            day = min(customer.dob.day, 28) if customer.dob.month == 2 else customer.dob.day
            shifted = customer.dob.replace(year=year, day=day)
            draw.text(
                (x + 2, y + 2),
                shifted.strftime("%d/%m/%Y"),
                font=load_font("serif", 26),
                fill=(24, 24, 28),
            )
            applied.append("patched_date_of_birth")

        elif technique == "typeface":
            # Re-render the name in a serif face on a document set in sans - the single most
            # common giveaway on a real forgery.
            x, y, _width, _height = blank("name")
            draw.text(
                (x + 2, y + 2),
                customer.name.upper(),
                font=load_font("serif", 25),
                fill=(24, 24, 28),
            )
            applied.append("mismatched_typeface_on_name")

        else:
            # Set the identity number a few pixels off the baseline its label sits on.
            x, y, _width, _height = blank("id_number")
            draw.text(
                (x + rng.randint(-24, 24), y + rng.randint(-9, 9)),
                customer.id_number,
                font=load_font("sans_bold", 44 if document_type == "AADHAAR" else 23),
                fill=(20, 22, 26),
            )
            applied.append("misaligned_identity_number")

    return tampered, applied


def generate(
    customers: list[Customer],
    accounts,
    forged_account_ids: set[str],
    seed: int = cfg.SEED,
) -> list[dict]:
    """Render every document and write the manifest.

    The manifest deliberately records only which file belongs to which customer and what
    kind of document it is. It carries no forged flag and no rendered field values: the
    forged flag is ground truth and lives in the evidence ledger, and publishing the
    rendered values would let Stage 1 skip OCR entirely and compare two records instead.

    Each customer's documents draw from a stream keyed by that customer, so rendering can be
    re-run by itself from the committed JSON and produce the same images - the images are
    not committed, only the data they are drawn from - and a new customer never changes an
    existing customer's cards.
    """
    streams = Streams(seed)
    cfg.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    account_by_customer = {a.customer_id: a.account_id for a in accounts}
    manifest: list[dict] = []

    for customer in customers:
        rng = streams.rng("documents", customer.customer_id)
        account_id = account_by_customer[customer.customer_id]
        is_forged_case = account_id in forged_account_ids
        pan = pan_number(rng, customer.name.split()[-1])

        if customer.residency_status == ResidencyStatus.DOMESTIC:
            primary_type = IdDocumentType.AADHAAR
            primary = render_aadhaar(customer, rng)
        else:
            primary_type = IdDocumentType.PASSPORT
            primary = render_passport(customer, rng)

        secondary_type = IdDocumentType.PAN
        secondary = render_pan(customer, pan, rng)

        # Only the primary document is forged. A file where every document is tampered is
        # not how a forgery presents - the point of submitting a forged OVD is that the rest
        # of the file looks ordinary.
        if is_forged_case:
            primary, _applied = apply_tampering(primary, customer, primary_type.value, rng)

        for document_type, image in (
            (primary_type, primary),
            (secondary_type, secondary),
        ):
            filename = f"{customer.customer_id}_{document_type.value}.png"
            image.save(cfg.DOCUMENTS_DIR / filename, optimize=True)
            manifest.append(
                {
                    "customer_id": customer.customer_id,
                    "account_id": account_id,
                    "document_type": document_type.value,
                    "file": filename,
                }
            )

    cfg.DOCUMENT_MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def render_from_committed_data(seed: int = cfg.SEED) -> list[dict]:
    """Re-render every image from the committed JSON, without re-running the other generators.

    Which documents are forged comes from the evidence ledger. That is ground truth, and
    reading it is correct here: this is the generator that planted the forgeries, not a
    pipeline stage.
    """
    from src.schemas import Account, EvidenceLedgerEntry

    def load(path, model):
        return [model.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]

    customers = load(cfg.CUSTOMERS_FILE, Customer)
    accounts = load(cfg.ACCOUNTS_FILE, Account)
    ledger = load(cfg.EVIDENCE_LEDGER_FILE, EvidenceLedgerEntry)
    forged = {entry.account_id for entry in ledger if entry.forged_document_planted}
    return generate(customers, accounts, forged, seed)


if __name__ == "__main__":
    rendered = render_from_committed_data()
    print(f"Rendered {len(rendered)} document images into {cfg.DOCUMENTS_DIR}.")
