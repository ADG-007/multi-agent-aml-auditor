"""Coverage hard checks pass, and pipeline code never touches the answer key."""

import re
from pathlib import Path

from src.bank import BankData
from src.evaluation.stage0_coverage import load_ledger, measure


#: Packages that make up the pipeline. They may never read ground truth or the corpus's
#: ground-truth link. src/bank names `_ground_truth` only to refuse it, src/generators writes
#: it, src/evaluation reads it, and src/schemas defines the models - all excluded.
PIPELINE_PACKAGES = ["stages", "scoring", "orchestration", "matching", "rulebook", "llm"]
FORBIDDEN = re.compile(r"_ground_truth|evidence_ledger|fraud_rings|linked_actor_id|src\.evaluation")


def test_coverage_hard_checks_pass():
    coverage = measure(BankData.load(), load_ledger())
    assert coverage.failures == []


def test_pipeline_code_never_references_ground_truth():
    offenders = []
    for package in PIPELINE_PACKAGES:
        for path in Path("src", package).rglob("*.py"):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if FORBIDDEN.search(line):
                    offenders.append(f"{path}:{number}: {line.strip()}")
    assert offenders == []


def test_bank_module_reads_no_ground_truth_and_blanks_the_link():
    source = Path("src/bank/data_store.py").read_text(encoding="utf-8")
    assert "evidence_ledger" not in source and "fraud_rings" not in source
    assert '"linked_actor_id": None' in source
