"""The RAG agent's numbers. It never scores."""

from pathlib import Path


PROMPT_VERSION: str = "rag-v1"

CLAUSES_FILE: Path = Path("data") / "rulebook" / "clauses.json"
#: The v2 index, embedded with the local embedding model (git-ignored, rebuilt on demand:
#: python -m src.agents.rag.index).
CHROMA_DIR: Path = Path("data") / "rulebook" / "chroma_v2"
COLLECTION_NAME: str = "rulebook_v2"

#: What the embedding model is told a query is for.
QUERY_INSTRUCTION: str = (
    "Instruct: Given a finding from a bank account review, retrieve the rulebook clause "
    "that governs it\nQuery: "
)

#: Clauses kept per query.
TOP_K: int = 3

#: Cosine distance above this is not a fit. On the tuning set the right clause sits at
#: 0.24-0.47 and unrelated clauses at 0.48 or more.
MAX_DISTANCE: float = 0.50

#: A weighted stage takes part in the search when its confidence is at least this.
SIGNAL_MIN: float = 0.40

#: The model may add this many queries of its own.
MAX_MODEL_QUERIES: int = 2

#: Clauses shown to the model at most, nearest first.
MAX_CLAUSES_IN_PROMPT: int = 8

#: Anything shaped like a clause id. INJECTED-n covers clauses added in a session.
CLAUSE_ID_PATTERN: str = r"\b(?:[A-Z]{2,6}-\d{3}|INJECTED-\d+)\b"
