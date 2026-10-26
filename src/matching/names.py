"""The one fuzzy name matcher, shared by Stage 0 and Stage 4.

rapidfuzz token-sort ratio against every name and alias, so word order does not matter.
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

    def all_matches(self, name: str, cutoff: float) -> list[NameMatch]:
        """Every entry at or above `cutoff`, best first, so a sanctions entry never hides
        behind a higher-scoring look-alike."""
        query = normalise(name)
        if not query:
            return []
        found = process.extract(
            query,
            self._choices,
            scorer=fuzz.token_sort_ratio,
            processor=None,
            score_cutoff=cutoff,
            limit=None,
        )
        best: dict[str, NameMatch] = {}
        for _, score, index in found:
            text, entry = self._targets[index]
            current = best.get(entry.entry_id)
            if current is None or score > current.score:
                best[entry.entry_id] = NameMatch(
                    entry_id=entry.entry_id,
                    entity_name=entry.entity_name,
                    matched_text=text,
                    score=round(score, 2),
                    tier=entry.severity_tier,
                )
        return sorted(best.values(), key=lambda m: (-m.score, m.entry_id))

    def entry(self, entry_id: str) -> AdverseMediaEntry:
        return next(e for _, e in self._targets if e.entry_id == entry_id)

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
