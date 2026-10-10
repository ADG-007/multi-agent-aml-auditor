"""Deterministic lookup of the numbers the rulebook carries.

Stage 0 and Stage 2 (and the Policy Injector in Phase 12) must move together when a clause
changes, so no threshold that a clause defines is ever written into a stage's code: it is
read from the clause metadata here, at run time. A missing clause or key is an error, never a
silent fallback to a default.

Stage 6's retrieval (ChromaDB) is separate and arrives in Phase 8; this is a plain lookup by
clause id.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from src.schemas import RulebookClause


#: The cash-reporting (CTR) clause: threshold_inr and near_threshold_pct.
STRUCTURING_CLAUSE_ID = "STRUCT-001"
#: The cross-border wire-reporting (CBWTR) clause: threshold_inr.
CROSS_BORDER_CLAUSE_ID = "XB-004"
#: The suspected-mule transaction floor: txn_floor_inr.
MULE_FLOOR_CLAUSE_ID = "MULE-002"


@dataclass(frozen=True)
class StructuringThreshold:
    """The CTR threshold and the structuring band just under it."""

    threshold_inr: float
    near_threshold_pct: float

    @property
    def band_floor_inr(self) -> float:
        """Lower edge of the structuring band, [threshold x pct, threshold)."""
        return self.threshold_inr * self.near_threshold_pct


class RulebookLookup:
    def __init__(self, clauses: Iterable[RulebookClause]) -> None:
        self._clauses = {c.clause_id: c for c in clauses}

    @classmethod
    def load(cls, path: str | Path = "data/rulebook/clauses.json") -> RulebookLookup:
        with Path(path).open(encoding="utf-8") as f:
            return cls(RulebookClause(**r) for r in json.load(f))

    def clause(self, clause_id: str) -> RulebookClause:
        try:
            return self._clauses[clause_id]
        except KeyError:
            raise KeyError(f"rulebook has no clause {clause_id!r}") from None

    def metadata_value(self, clause_id: str, key: str) -> float:
        metadata = self.clause(clause_id).metadata
        if key not in metadata:
            raise KeyError(f"clause {clause_id} carries no {key!r} in its metadata")
        return float(metadata[key])

    def structuring(self) -> StructuringThreshold:
        return StructuringThreshold(
            threshold_inr=self.metadata_value(STRUCTURING_CLAUSE_ID, "threshold_inr"),
            near_threshold_pct=self.metadata_value(STRUCTURING_CLAUSE_ID, "near_threshold_pct"),
        )

    def cross_border_threshold(self) -> float:
        return self.metadata_value(CROSS_BORDER_CLAUSE_ID, "threshold_inr")

    def mule_floor(self) -> float:
        return self.metadata_value(MULE_FLOOR_CLAUSE_ID, "txn_floor_inr")
