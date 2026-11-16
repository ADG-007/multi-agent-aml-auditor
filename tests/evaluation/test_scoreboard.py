"""The scoreboard file the demo site reads: numbers, confusion, misses, show cases."""

from types import SimpleNamespace

from src.evaluation import grader
from src.evaluation.run_eval import build_scoreboard, show_case_title
from src.schemas import Disposition, EvidenceLedgerEntry, SeverityTier


def _entry(account_id="ACC-T001", **kwargs) -> EvidenceLedgerEntry:
    return EvidenceLedgerEntry(account_id=account_id, customer_id="CUST-T001", **kwargs)


def _report(disposition: Disposition, confidence: float = 0.5):
    return SimpleNamespace(disposition=disposition, confidence=confidence)


def _board():
    ledger = {
        "ACC-T001": _entry("ACC-T001", structuring_planted=True, kyc_mismatch_planted=True),
        "ACC-T002": _entry("ACC-T002", kyc_mismatch_planted=True),
        "ACC-T003": _entry("ACC-T003"),
    }
    got = {
        "ACC-T001": Disposition.MONITOR,
        "ACC-T002": Disposition.MONITOR,
        "ACC-T003": Disposition.CLEAR,
    }
    latest = {a: _report(d) for a, d in got.items()}
    grade = grader.grade(
        (grader.Outcome(account_id=a, disposition=d.value) for a, d in got.items()), ledger
    )
    return build_scoreboard(grade, latest, ledger, ("ACC-T001", "ACC-T404"), "tuning", "off")


def test_confusion_sums_to_accounts():
    board = _board()
    assert sum(n for row in board["confusion"].values() for n in row.values()) == 3
    assert board["accounts"] == board["metrics"]["accounts"] == 3


def test_misses_are_key_versus_got():
    assert _board()["misses"] == [
        {
            "account": "ACC-T001",
            "key": "SUSPICIOUS-ESCALATE",
            "design": "SUSPICIOUS-ESCALATE",
            "got": "MONITOR",
        }
    ]


def test_show_cases_skip_accounts_outside_the_run():
    cases = _board()["show_cases"]
    assert [c["account"] for c in cases] == ["ACC-T001"]
    assert cases[0]["got"] == "MONITOR" and cases[0]["confidence"] == 0.5


def test_show_case_titles_come_from_ledger_fields():
    assert show_case_title(_entry(structuring_style="small")) == "Monthly-cap structurer (small)"
    assert show_case_title(_entry(decoy_types=["cash_business"])).startswith("Cash-business trap")
    assert show_case_title(_entry(forged_document_planted=True)) == "Forged ID card"
    assert show_case_title(_entry(graph_ring_membership="RING-1")) == "Linked to a flagged ring"
    assert (
        show_case_title(_entry(adverse_media_hit=SeverityTier.CONFIRMED_SANCTIONS))
        == "Sanctions or adverse-media hit"
    )
    assert show_case_title(_entry()) == "Clean account"
