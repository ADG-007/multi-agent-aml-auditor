"""The shared name matcher: aliases, word order and the cutoff boundary."""

from rapidfuzz import fuzz

from src.matching import NameIndex, normalise
from src.schemas import SeverityTier
from tests.factories import media_entry


def test_normalise_ignores_case_and_punctuation():
    assert normalise("  VIKRAM  Sethi, Jr. ") == normalise("vikram sethi jr")


def test_exact_name_and_word_order():
    index = NameIndex([media_entry(entity_name="Vikram Sethi")])
    assert index.best_match("Vikram Sethi", cutoff=85).score == 100
    assert index.best_match("Sethi Vikram", cutoff=85).score == 100


def test_alias_matches_and_is_reported():
    index = NameIndex([media_entry(entity_name="Vikram Sethi", aliases=["Vicky Sethi"])])
    match = index.best_match("Vicky Sethi", cutoff=85)
    assert match.matched_text == "Vicky Sethi"
    assert match.via_alias
    assert match.tier == SeverityTier.CONFIRMED_SANCTIONS


def test_cutoff_is_inclusive_and_exclusive_below():
    index = NameIndex([media_entry(entity_name="Vikram Sethi")])
    probe = "Vikram Sethia"
    score = fuzz.token_sort_ratio(normalise(probe), normalise("Vikram Sethi"))
    assert index.best_match(probe, cutoff=score) is not None
    assert index.best_match(probe, cutoff=score + 0.01) is None


def test_unrelated_name_does_not_match():
    index = NameIndex([media_entry(entity_name="Vikram Sethi")])
    assert index.best_match("Asha Verma", cutoff=85) is None
    assert index.best_match("", cutoff=85) is None
