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
def test_all_matches_lists_each_entry_once_best_first():
    index = NameIndex([
        media_entry(entry_id="AM-T001", entity_name="Vikram Sethi", aliases=["Vicky Sethi"]),
        media_entry(entry_id="AM-T002", entity_name="Vikram Sethia", tier=SeverityTier.ADVERSE_NEWS),
    ])
    matches = index.all_matches("Vikram Sethi", cutoff=70)
    assert [m.entry_id for m in matches] == ["AM-T001", "AM-T002"]
    assert matches[0].score == 100 and not matches[0].via_alias
    assert matches[1].score < 100


def test_all_matches_of_an_empty_name_is_empty():
    index = NameIndex([media_entry()])
    assert index.all_matches("", cutoff=85) == []
