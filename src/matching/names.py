"""The one fuzzy name matcher, shared by Stage 0's pre-filter and Stage 4 (Phase 6).

MASTER_PLAN.md requires both stages to use the same matcher, so it lives here rather than in
either stage. Matching is rapidfuzz's token-sort ratio (word order does not matter, so
"Sethi Vikram" matches "Vikram Sethi") against every entry's name and every alias.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rapidfuzz import fuzz, process
from rapidfuzz.utils import default_process

from src.schemas import AdverseMediaEntry, SeverityTier


def normalise(name: str) -> str:
    """Lower-case, punctuation to spaces, runs of whitespace collapsed to one space."""
    return " ".join(default_process(name).split())


@dataclass(frozen=True)
class NameMatch:
    entry_id: str
    entity_name: str
    matched_text: str
    score: float
    tier: SeverityTier

    @property
    def via_alias(self) -> bool:
        return self.matched_text != self.entity_name


class NameIndex:
    """Every entry name and alias, normalised once, with a per-name result cache."""

    def __init__(self, entries: Iterable[AdverseMediaEntry]) -> None:
        self._targets: list[tuple[str, AdverseMediaEntry]] = []
        for entry in entries:
            for text in (entry.entity_name, *entry.aliases):
                self._targets.append((text, entry))
        self._choices = [normalise(text) for text, _ in self._targets]
        self._cache: dict[tuple[str, float], NameMatch | None] = {}

    def best_match(self, name: str, cutoff: float) -> NameMatch | None:
        """The highest-scoring name or alias at or above `cutoff`, or None."""
        key = (name, cutoff)
        if key not in self._cache:
            self._cache[key] = self._search(name, cutoff)
        return self._cache[key]

    def _search(self, name: str, cutoff: float) -> NameMatch | None:
        query = normalise(name)
        if not query:
            return None
        found = process.extractOne(
            query,
            self._choices,
            scorer=fuzz.token_sort_ratio,
            processor=None,
            score_cutoff=cutoff,
        )
        if found is None:
            return None
        _, score, index = found
        text, entry = self._targets[index]
        return NameMatch(
            entry_id=entry.entry_id,
            entity_name=entry.entity_name,
            matched_text=text,
            score=round(score, 2),
            tier=entry.severity_tier,
        )
