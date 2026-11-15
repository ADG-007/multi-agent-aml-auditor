"""The Vision agent's numbers. The ID check never scores; the KYC level is set by Python."""

from pathlib import Path


PROMPT_VERSION: str = "vision-v1"

# --------------------------------------------------------------------------------------
# ID check
# --------------------------------------------------------------------------------------

DOCUMENTS_DIR: Path = Path("data") / "documents"
MANIFEST_NAME: str = "manifest.json"
#: Saved Tesseract text, one file per image hash (git-ignored, rebuilt on demand).
OCR_CACHE_DIR: Path = Path("outputs") / "cache" / "ocr"

#: A printed name agrees at or above this token-sort ratio.
NAME_MATCH_MIN: float = 90.0
#: An id number may differ by this many characters (one misread) and still agree.
ID_NUMBER_MAX_EDITS: int = 1
#: A label line is found when it is this close to the expected label.
LABEL_MATCH_MIN: float = 75.0

#: Both reads disagree with the record on a field: pause for the admin.
BOTH_READS_DISAGREE: float = 0.95
#: One read disagrees and the other could not read the field: a notice, no pause.
ONE_READ_DISAGREES: float = 0.60
#: The reads disagree with each other, or a field stayed unread: noted only.
UNCONFIRMED: float = 0.30

# --------------------------------------------------------------------------------------
# KYC claims and rules (public red flags; the level is Python's, never the model's)
# --------------------------------------------------------------------------------------

#: One moderate concern is written in the trail and carries no score: a weak second signal
#: must not complete the judge's solo branch. Two moderate concerns make HIGH.
LEVEL_SCORES: dict[str, float] = {"NONE": 0.0, "MODERATE": 0.0, "HIGH": 0.65, "SEVERE": 0.85}

#: Money in against the declared base (income, or declared turnover when larger).
INCOME_MODERATE_FROM: float = 1.5
INCOME_HIGH_FROM: float = 3.0

#: "No cash" is contradicted by cash deposits worth at least this share of monthly income.
NO_CASH_MIN_INCOME_SHARE: float = 0.10

#: "No business" / "single source" is contradicted by at least this many different payers.
#: Ordinary accounts in the tuning set have 12 to 54 payers in 90 days.
MANY_PAYERS_FROM: int = 60

#: Declared cash takings explain monthly cash up to this multiple of the declared amount.
CASH_EXPLAINED_WITHIN: float = 1.25

#: A declared one-off event covers an excess up to this multiple of its stated amount.
EVENT_COVERS_WITHIN: float = 1.25

MAX_TRAIL_STEPS: int = 10
