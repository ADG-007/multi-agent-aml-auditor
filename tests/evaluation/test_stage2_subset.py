"""Stage 2's subset: picked by rule, reproducible, and matching the saved file."""

import pytest

from src.bank import BankData
from src.evaluation import stage2_subset
from src.rulebook import RulebookLookup


@pytest.fixture(scope="module")
def truth():
    return stage2_subset.load_truth(BankData.load(), RulebookLookup.load())


def test_saved_subset_is_what_the_rule_picks(truth):
    picked = stage2_subset.pick(truth)
    saved = stage2_subset.load_subsets()
    assert saved["shared"] == picked["shared"] and saved["stage2"] == picked["stage2"]


def test_every_slot_holds_an_account_that_fits_it(truth):
    picked = stage2_subset.pick(truth)
    fits = dict(stage2_subset.SLOTS)
    ids = [s["account_id"] for s in picked["stage2"]]
    assert len(ids) == len(stage2_subset.SLOTS) == len(set(ids))
    for s in picked["stage2"]:
        assert fits[s["slot"]](truth[s["account_id"]])
    small = next(s["account_id"] for s in picked["stage2"] if s["slot"] == "structurer_small")
    assert truth[small].structuring_only


def test_naive_slots_take_the_lowest_and_highest_k(truth):
    picked = {s["slot"]: s["account_id"] for s in stage2_subset.pick(truth)["stage2"]}
    naive_k = [t.k for t in truth.values() if t.entry.structuring_style == "naive"]
    assert truth[picked["structurer_naive_low_k"]].k == min(naive_k)
    assert truth[picked["structurer_naive_high_k"]].k == max(naive_k)


def test_cash_business_slots_hold_a_merchant_and_a_shop(truth):
    picked = {s["slot"]: s["account_id"] for s in stage2_subset.pick(truth)["stage2"]}
    merchant = truth[picked["cash_business_merchant"]].entry.decoy_types
    shop = truth[picked["cash_business_shop"]].entry.decoy_types
    assert merchant == ["merchant_lookalike", "cash_business"]
    assert shop == ["cash_business"]
    assert "cash_business" not in truth[picked["merchant_lookalike"]].entry.decoy_types


def test_shared_set_is_ring04_and_two_ring12_mules(truth):
    shared = stage2_subset.pick(truth)["shared"]
    assert {truth[a].entry.graph_ring_membership for a in shared[:4]} == {"RING-04"}
    assert [truth[a].role for a in shared[4:]] == ["network_mule", "network_mule"]


def test_mule_roles_split_cleanly(truth):
    roles = [t.role for t in truth.values() if t.entry.pass_through_planted]
    assert sorted(set(roles)) == ["complicit_merchant", "network_mule", "solo_mule"]
    assert (roles.count("network_mule"), roles.count("solo_mule")) == (8, 2)
    assert roles.count("complicit_merchant") == 2
