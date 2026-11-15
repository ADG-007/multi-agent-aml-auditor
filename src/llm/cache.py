"""Saved model answers: one JSONL file per bucket under data/llm_cache/.

The key is model id + prompt version + the full prompt (+ image hashes, system text and the
answer schema's name). The files of the final runs live in git, so a fresh copy replays a run
with no model installed.

Modes (one per run):
    off     no model call and no cache; agents give their Python-only result (tests).
    cached  a saved answer or nothing: a miss is a failed call, never a silent zero.
    live    a saved answer if there is one, else call the model and save the answer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from enum import Enum
from pathlib import Path


CACHE_ROOT = Path("data") / "llm_cache"


class LLMMode(str, Enum):
    OFF = "off"
    CACHED = "cached"
    LIVE = "live"


def fingerprint(model_id: str, prompt_version: str, prompt: str, extra: Sequence[str] = ()) -> str:
    raw = json.dumps([model_id, prompt_version, prompt, list(extra)], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ResponseCache:
    """Append-only. A bucket is read once and kept in memory."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or CACHE_ROOT
        self._buckets: dict[str, dict[str, dict]] = {}

    def path(self, bucket: str) -> Path:
        return self.root / f"{bucket}.jsonl"

    def _load(self, bucket: str) -> dict[str, dict]:
        if bucket not in self._buckets:
            records: dict[str, dict] = {}
            path = self.path(bucket)
            if path.exists():
                with path.open(encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            records[record["key"]] = record
            self._buckets[bucket] = records
        return self._buckets[bucket]

    def get(self, bucket: str, key: str) -> dict | None:
        return self._load(bucket).get(key)

    def put(self, bucket: str, key: str, record: dict) -> None:
        record = {"key": key, **record}
        self._load(bucket)[key] = record
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path(bucket).open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def count(self, bucket: str) -> int:
        return len(self._load(bucket))
