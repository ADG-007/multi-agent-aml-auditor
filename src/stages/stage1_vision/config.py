"""Stage 1's numbers (OCR field check, no image forensics). Never scored."""

from pathlib import Path


DOCUMENTS_DIR: Path = Path("data") / "documents"
MANIFEST_NAME: str = "manifest.json"
#: Saved OCR text, one file per image hash (git-ignored, rebuilt on demand).
OCR_CACHE_DIR: Path = Path("outputs") / "cache" / "ocr"

#: A printed name agrees at or above this token-sort ratio (after normalising OCR noise).
NAME_MATCH_MIN: float = 90.0

#: An id number may differ by this many characters (one OCR misread) and still agree.
ID_NUMBER_MAX_EDITS: int = 1

#: A label line is found when it is this close to the expected label ("Sumame" for "Surname").
LABEL_MATCH_MIN: float = 75.0

#: A clear printed value that disagrees with the record: pause for the admin.
MISMATCH_CONFIDENCE: float = 0.95
#: A field could not be read, and nothing disagrees: noted, no pause.
UNREAD_CONFIDENCE: float = 0.30
