"""The rulebook index: one vector per clause, from the local embedding model, in ChromaDB.

    python -m src.agents.rag.index      # (re)build data/rulebook/chroma_v2

The embedding model runs on the CPU. Search hits are saved per query and index build in the
model-answer cache, so a replay needs neither the embedding model nor the index.
"""

from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache

import chromadb

from src.agents.rag import config
from src.llm import LLMClient, LLMFailureReason, LLMMode, LLMStageFailure, Role, fingerprint
from src.schemas import RulebookClause


SEARCH_BUCKET = "rag_search"


def load_clauses() -> list[RulebookClause]:
    with config.CLAUSES_FILE.open(encoding="utf-8") as f:
        return [RulebookClause(**c) for c in json.load(f)]


def clause_document(clause: RulebookClause) -> str:
    return f"{clause.title}. {clause.text}"


@lru_cache(maxsize=1)
def index_version() -> str:
    """Changes when the clause text changes, so old saved hits stop matching."""
    return hashlib.sha256(config.CLAUSES_FILE.read_bytes()).hexdigest()[:16]


def build(client: LLMClient) -> int:
    """Embed every clause into a fresh collection. Returns the number of clauses."""
    clauses = load_clauses()
    vectors = client.embed([clause_document(c) for c in clauses])
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    store = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if config.COLLECTION_NAME in [c.name for c in store.list_collections()]:
        store.delete_collection(config.COLLECTION_NAME)
    collection = store.create_collection(
        config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "index_version": index_version()},
    )
    collection.add(
        ids=[c.clause_id for c in clauses],
        embeddings=vectors,
        documents=[clause_document(c) for c in clauses],
        metadatas=[{"category": c.category.value, "title": c.title} for c in clauses],
    )
    _collection.cache_clear()
    return collection.count()


@lru_cache(maxsize=1)
def _collection():
    store = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return store.get_collection(config.COLLECTION_NAME)


def search(client: LLMClient, text: str, k: int = config.TOP_K) -> list[tuple[str, float]]:
    """(clause id, cosine distance) for the k nearest clauses, nearest first."""
    model = client.model_for(Role.EMBED)
    key = fingerprint(model, index_version(), text, [str(k)])
    saved = client.cache.get(SEARCH_BUCKET, key)
    if saved is not None:
        return [(cid, d) for cid, d in saved["hits"]]
    if client.mode != LLMMode.LIVE:
        raise LLMStageFailure(
            LLMFailureReason.CACHE_MISS, "rag", "no saved search for this query", model
        )
    vector = client.embed([config.QUERY_INSTRUCTION + text])[0]
    result = _collection().query(query_embeddings=[vector], n_results=k)
    hits = [
        (cid, round(float(d), 4))
        for cid, d in zip(result["ids"][0], result["distances"][0], strict=True)
    ]
    client.cache.put(SEARCH_BUCKET, key, {"model": model, "query": text, "hits": hits})
    return hits


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    count = build(LLMClient(LLMMode.LIVE))
    print(f"{count} clauses embedded into {config.CHROMA_DIR} ({config.COLLECTION_NAME}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
