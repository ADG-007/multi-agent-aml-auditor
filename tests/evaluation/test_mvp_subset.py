"""The 50 accounts: the plan's table agrees with the answer key, and the pick is stable."""

from collections import Counter

from src.evaluation.mvp_subset import build, load_ledger


def test_plan_table_matches_the_answer_key_and_tallies():
    subset, problems = build(load_ledger())
    assert problems == []
    rows = subset["accounts"]
    assert len(rows) == 50 and len({r["account_id"] for r in rows}) == 50
    assert Counter(r["kind"] for r in rows) == {"bad": 20, "trap": 9, "clean": 21}
    assert Counter(r["expected"] for r in rows) == {
        "SUSPICIOUS-ESCALATE": 9,
        "MONITOR": 11,
        "CLEAR": 30,
    }


def test_mvp_answers_keep_the_ledger_label_and_say_why():
    rows = {r["account_id"]: r for r in build(load_ledger())[0]["accounts"]}
    for account_id in ("ACC-0008", "ACC-0121"):
        row = rows[account_id]
        assert row["expected"] == "MONITOR" and row["ledger"] == "SUSPICIOUS-ESCALATE"
        assert "never scored" in row["why"]
    assert sum("ledger" in r for r in rows.values()) == 2


def test_clean_pick_is_reproducible():
    assert build(load_ledger())[0] == build(load_ledger())[0]
