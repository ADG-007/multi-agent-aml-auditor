"""Amendment 4 data checks: label audit (V4) and leak test (V5) on the committed data."""

from src.evaluation import label_audit, leak_test
from src.schemas import Disposition, EvidenceLedgerEntry


def test_label_audit_passes():
    result = label_audit.audit()
    assert result.failures == []
    assert any("labels 49 / 73 / 248" in line for line in result.lines)
    assert any("14 labels the scoring design cannot reach" in line for line in result.lines)


def test_forgery_alone_cannot_be_scored_up_to_its_label():
    entry = EvidenceLedgerEntry(
        account_id="ACC-T001", customer_id="CUST-T001", forged_document_planted=True
    )
    assert entry.expected_disposition == Disposition.MONITOR
    assert label_audit.label_without_forgery(entry) == Disposition.CLEAR


def test_no_pipeline_module_reads_the_answer_key():
    assert leak_test.static_findings() == []
    opened, hits = leak_test.runtime_findings(step=25)
    assert opened > 0 and hits == []


def test_static_check_would_catch_a_leak():
    source = (
        '"""Docstrings may name data/_ground_truth/."""\n'
        "from src.evaluation import stage0_coverage\n"
        'PATH = "data/_ground_truth/evidence_ledger.json"\n'
    )
    findings = leak_test.scan("stages/fake.py", source)
    assert len(findings) == 3  # the import, and two forbidden names in the path string
    assert all(":1 " not in f for f in findings)  # the docstring is allowed
