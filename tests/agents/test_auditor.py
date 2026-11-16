"""The Auditor: outline from picked ids, text checked against the outline, template fallback."""

import json

from src.agents.auditor.agent import (
    AuditorAgent,
    CaseBrief,
    ReportText,
    check_text,
    numbers_in,
    template,
)
from src.llm import LLMMode
from src.schemas import Disposition, StageStatus
from tests.factories import fake_client


def brief(**kwargs) -> CaseBrief:
    base = dict(
        case_id="CASE-ACC-0044-1",
        account_id="ACC-0044",
        disposition=Disposition.SUSPICIOUS_ESCALATE,
        confidence=0.6222,
        scoring_steps=["Weighted average: 0.6222 -> Renormalised."],
        stage_status={"TransactionPattern": "RAN", "KYC": "RAN"},
        findings={
            "TransactionPattern": ["confidence 0.6", "2 months of cash at 9.75 lakh"],
            "KYC": ["confidence 0.65", "profile says no cash, 33 cash deposits"],
        },
        clauses={"STRUCT-002": ("Monthly cash aggregation", "connected cash transactions")},
        rechecks_possible=("KYC_CLAIMS",),
    )
    return CaseBrief(**(base | kwargs))


def outline(points, recheck="NONE") -> str:
    return json.dumps(
        {
            "points": [{"fact_ids": f, "clause_ids": c} for f, c in points],
            "recheck": recheck,
            "recheck_reason": "the claim looks wrong",
        }
    )


def text(narrative="Cash of 9.75 lakh in 2 months.", conclusion="See STRUCT-002.") -> str:
    return json.dumps(
        {
            "executive_summary": "Escalate at confidence 0.6222.",
            "investigation_narrative": narrative,
            "compliance_conclusion": conclusion,
            "str_narrative": "STR draft: 33 cash deposits.",
        }
    )


PICK = outline([(["F4", "F3"], ["STRUCT-002"]), (["F6", "F99"], ["FAKE-001"])])


def test_numbers_are_compared_in_one_normal_form():
    assert numbers_in("₹9,75,000 and 0.60, 08 March, ACC-0044, STRUCT-002") == {
        "975000",
        "0.6",
        "8",
    }


def test_outline_is_rendered_by_python_from_the_picked_facts(tmp_path):
    result = AuditorAgent(fake_client(tmp_path, PICK, text())).run(brief())
    assert result.outline[0] == "Verdict: SUSPICIOUS-ESCALATE at confidence 0.6222."
    assert result.outline[1] == (
        "- TransactionPattern: 2 months of cash at 9.75 lakh TransactionPattern: confidence 0.6 "
        "[STRUCT-002: Monthly cash aggregation]"
    )
    assert result.outline[2] == "- KYC: profile says no cash, 33 cash deposits"
    assert "Unknown ids dropped: F99, FAKE-001" in result.run.trail[-3].finding
    assert not result.used_template and result.status == StageStatus.RAN
    assert result.draft.str_narrative.startswith("STR draft")


def test_a_number_outside_the_outline_gets_one_rewrite_then_the_template(tmp_path):
    bad = text(narrative="Cash of 9.75 lakh, about 97% of the limit.")
    fixed = AuditorAgent(fake_client(tmp_path / "a", PICK, bad, text())).run(brief())
    assert not fixed.used_template
    assert "numbers not in the outline: 97" in fixed.run.trail[-3].finding
    failed = AuditorAgent(fake_client(tmp_path / "b", PICK, bad, bad)).run(brief())
    assert failed.used_template
    assert failed.draft.executive_summary == template(brief()).executive_summary


def test_a_clause_outside_the_outline_is_caught_and_never_printed(tmp_path):
    bad = text(conclusion="Breaches STRUCT-002 and KYC-002.")
    result = AuditorAgent(fake_client(tmp_path, PICK, bad, bad)).run(brief())
    assert result.used_template
    assert "KYC-002" not in result.draft.compliance_conclusion
    draft = ReportText.model_validate_json(bad)
    assert "clause ids not in the outline: KYC-002" in check_text(
        draft, ["Verdict 0.6222", "- 9.75 lakh, 2 months, 33 [STRUCT-002: x]"], ["STRUCT-002"]
    )


def test_clear_case_with_no_finding_uses_the_template_and_no_model(tmp_path):
    client = fake_client(tmp_path)
    quiet = brief(
        disposition=Disposition.CLEAR, confidence=0.0, findings={}, clauses={}, scoring_steps=[]
    )
    result = AuditorAgent(client).run(quiet)
    assert client.fake.requests == []
    assert result.used_template and result.status == StageStatus.RAN
    assert result.draft.str_narrative is None


def test_models_off_or_failed_still_gives_a_report(tmp_path):
    off = AuditorAgent(fake_client(tmp_path, mode=LLMMode.OFF)).run(brief())
    assert off.used_template and off.status == StageStatus.RAN
    assert off.draft.str_narrative.startswith("Draft STR")
    failed = AuditorAgent(fake_client(tmp_path, "junk", "junk", "junk", "junk")).run(brief())
    assert failed.used_template and failed.status == StageStatus.FAILED
    assert failed.outline[0].startswith("Verdict")


def test_str_only_on_escalate(tmp_path):
    monitor = brief(disposition=Disposition.MONITOR)
    result = AuditorAgent(fake_client(tmp_path, PICK, text())).run(monitor)
    assert result.draft.str_narrative is None


def test_recheck_is_asked_only_when_possible_and_allowed(tmp_path):
    asked = outline([(["F4", "F6"], ["STRUCT-002"])], recheck="KYC_CLAIMS")
    yes = AuditorAgent(fake_client(tmp_path / "a", asked, text())).run(brief())
    assert yes.recheck == "KYC_CLAIMS"
    assert [j.kind for j in yes.run.queued] == ["Re-check KYC_CLAIMS"]
    other = outline([(["F4", "F6"], ["STRUCT-002"])], recheck="SAME_PERSON")
    no = AuditorAgent(fake_client(tmp_path / "b", other, text())).run(brief())
    assert no.recheck is None and not no.run.queued
    second = AuditorAgent(fake_client(tmp_path / "c", asked, text())).run(
        brief(), allow_recheck=False
    )
    assert second.recheck is None


def test_unusable_outline_falls_back_to_the_python_outline(tmp_path):
    empty = outline([(["F99"], [])])
    result = AuditorAgent(fake_client(tmp_path, empty, text())).run(brief())
    assert len(result.outline) == 7  # verdict + 5 facts + 1 clause
    assert result.outline[-1] == "- Clause [STRUCT-002: Monthly cash aggregation]"


def test_a_clause_the_rag_agent_did_not_return_never_reaches_the_facts(tmp_path):
    findings = {"TransactionPattern": ["cash just under the monthly report (MULE-002)"]}
    result = AuditorAgent(fake_client(tmp_path, mode=LLMMode.OFF)).run(brief(findings=findings))
    joined = "\n".join(result.outline)
    assert "MULE-002" not in joined and "monthly report" in joined
    assert "STRUCT-002" in joined
