"""Stage 6's numbers (ChromaDB search only, no LLM)."""

from pathlib import Path


CHROMA_DIR: Path = Path("data") / "rulebook" / "chroma"
CLAUSES_FILE: Path = Path("data") / "rulebook" / "clauses.json"  # what the index is built from
COLLECTION_NAME: str = "rulebook"  # the generator's name (src/generators/rulebook.py)
#: Saved search hits, one file per query + index build (git-ignored, rebuilt on demand).
SEARCH_CACHE_DIR: Path = Path("outputs") / "cache" / "rulebook"

#: Clauses kept per query.
TOP_K: int = 3

#: Cosine distance above this is not a fit (good matches ~0.35-0.53, unrelated ~0.96).
MAX_DISTANCE: float = 0.75

#: A weighted stage takes part in the search when its confidence is at least this.
SIGNAL_MIN: float = 0.40

#: Anything shaped like a clause id in the writer's text. INJECTED-n covers session clauses.
CLAUSE_ID_PATTERN: str = r"\b(?:[A-Z]{2,6}-\d{3}|INJECTED-\d+)\b"
