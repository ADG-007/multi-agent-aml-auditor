"""The 100 / 270 split and the grader."""

from collections import Counter

import pytest

from src.bank import BankData
from src.evaluation import grader, split
from src.evaluation.stage2_subset import load_ledger, load_truth
from src.rulebook import RulebookLookup
from src.schemas import EvidenceLedgerEntry, SeverityTier


@pytest.fixture(scope="module")
def truth():
    return load_truth(BankData.load(), RulebookLookup.load())


def test_saved_split_is_what_the_rule_picks(truth):
    assert split.load_split() == split.pick(truth)


def test_split_is_100_and_270_with_no_overlap(truth):
    tuning, held_out = split.tuning_ids(), split.held_out_ids()
    assert len(tuning) == len(set(tuning)) == 100
    assert len(held_out) == 270
    assert not set(tuning) & set(held_out)
    assert set(tuning) | set(held_out) == set(truth)


def test_mix_matches_the_plan(truth):
    groups = Counter(s["group"] for s in split.load_split()["tuning"])
    assert groups == {"ring": 16, "escalate": 5, "monitor": 13, "trap": 22, "clean": 44}
    for s in split.load_split()["tuning"]:
        entry = truth[s["account_id"]].entry
        if s["group"] in ("trap", "clean"):
            assert entry.expected_disposition.value == grader.CLEAR
        if s["group"] == "trap":
            assert entry.decoy_types
        if s["group"] == "clean":
            assert not entry.decoy_types


def test_rings_are_whole_and_one_of_each_type(truth):
    slots = Counter(s["slot"] for s in split.load_split()["tuning"] if s["group"] == "ring")
    assert set(slots.values()) == {4}
    assert sorted(slot.split()[1] for slot in slots) == [
        "INTERNATIONAL",
        "LOCAL",
        "MULE",
        "ROUND_TRIP",
    ]


def test_every_structuring_style_is_in_the_tuning_set(truth):
    styles = {truth[a].entry.structuring_style for a in split.tuning_ids()}
    assert {"naive", "small", "medium", "mixed"} <= styles


def _entry(**kwargs) -> EvidenceLedgerEntry:
    return EvidenceLedgerEntry(account_id="A", customer_id="C", **kwargs)


def test_design_label_leaves_the_forged_channel_out():
    assert grader.design_label(_entry(forged_document_planted=True)) == grader.CLEAR
    both = _entry(forged_document_planted=True, kyc_mismatch_planted=True)
    assert both.expected_disposition.value == grader.ESCALATE
    assert grader.design_label(both) == grader.MONITOR
    sanctions = _entry(adverse_media_hit=SeverityTier.CONFIRMED_SANCTIONS)
    assert grader.design_label(sanctions) == grader.ESCALATE


def test_grade_counts():
    ledger = {
        "A1": _entry(structuring_planted=True, structuring_style="small").model_copy(
            update={"account_id": "A1"}
        ),
        "A2": _entry(kyc_mismatch_planted=True).model_copy(update={"account_id": "A2"}),
        "A3": _entry(decoy_types=["cash_business"]).model_copy(update={"account_id": "A3"}),
        "A4": _entry(forged_document_planted=True).model_copy(update={"account_id": "A4"}),
        "A5": _entry().model_copy(update={"account_id": "A5"}),
    }
    outcomes = [
        grader.Outcome("A1", grader.MONITOR, transaction_pattern=0.6),
        grader.Outcome("A2", grader.CLEAR, kyc=0.0),
        grader.Outcome("A3", grader.MONITOR, transaction_pattern=0.6),
        grader.Outcome("A4", grader.CLEAR, paused=True),
        grader.Outcome("A5", grader.CLEAR),
    ]
    g = grader.grade(outcomes, ledger)
    assert (g.accounts, g.exact_key, g.exact_design) == (5, 2, 3)
    assert (g.caught, g.bad) == (1, 3)
    assert (g.false_alarms, g.clean) == (1, 2)
    assert (g.kyc_found, g.kyc_planted) == (0, 1)
    assert (g.cash_business_alarms, g.cash_business) == (1, 1)
    assert g.structurers_caught["small"] == g.structurers_scored["small"] == 1
    assert (g.fake_ids_paused, g.fake_ids, g.false_pauses) == (1, 1, 0)
    assert "exact match" in grader.render(g, "t")


def test_key_labels_are_49_73_248():
    labels = Counter(e.expected_disposition.value for e in load_ledger().values())
    assert labels == {grader.ESCALATE: 49, grader.MONITOR: 73, grader.CLEAR: 248}
