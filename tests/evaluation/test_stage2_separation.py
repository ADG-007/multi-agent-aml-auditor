"""Stage 2 separation hard checks pass on the saved subset."""

from src.bank import BankData
from src.evaluation import stage2_separation
from src.evaluation.stage2_subset import load_subsets, load_truth
from src.rulebook import RulebookLookup
from src.stages.stage0_triage import TriageContext


def test_separation_hard_checks_pass():
    bank, rulebook = BankData.load(), RulebookLookup.load()
    subsets = load_subsets()
    accounts = {s["account_id"] for s in subsets["stage2"]} | set(subsets["shared"])
    result = stage2_separation.measure(
        TriageContext.build(bank, rulebook), subsets, load_truth(bank, rulebook, accounts)
    )
    assert result.failures == []
    assert any(line.startswith("  PASS  structurers") for line in result.lines)
