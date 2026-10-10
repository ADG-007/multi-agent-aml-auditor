"""Rulebook lookup: real values, and errors instead of silent defaults."""

import pytest

from src.rulebook import RulebookLookup
from src.schemas import RulebookCategory, RulebookClause


def test_reads_the_committed_values():
    rulebook = RulebookLookup.load()
    structuring = rulebook.structuring()
    assert structuring.threshold_inr == 1_000_000
    assert structuring.near_threshold_pct == 0.9
    assert structuring.band_floor_inr == 900_000
    assert rulebook.cross_border_threshold() == 500_000
    assert rulebook.mule_floor() == 1_000


def test_a_changed_clause_changes_the_value():
    clause = RulebookClause(
        clause_id="XB-004",
        title="t",
        text="t",
        category=RulebookCategory.CROSS_BORDER,
        sources=[],
        metadata={"threshold_inr": 250_000},
    )
    assert RulebookLookup([clause]).cross_border_threshold() == 250_000


def test_missing_clause_or_key_raises():
    clause = RulebookClause(
        clause_id="STRUCT-001",
        title="t",
        text="t",
        category=RulebookCategory.STRUCTURING,
        sources=[],
        metadata={"threshold_inr": 1_000_000},
    )
    rulebook = RulebookLookup([clause])
    with pytest.raises(KeyError):
        rulebook.structuring()
    with pytest.raises(KeyError):
        rulebook.mule_floor()
