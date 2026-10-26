"""Stage 7 with a fake model: invented citations are removed, STR only on ESCALATE, and a
failure degrades to the template text."""

from src.llm.cache import LLMMode, ResponseCache
from src.schemas import Disposition, StageStatus
from src.stages.stage7_auditor import CaseBrief, run
from src.stages.stage7_auditor.agent import AuditorDraft


def _brief(disposition=Disposition.MONITOR):
    return CaseBrief(
        case_id="CASE-ACC-T001-1",
        account_id="ACC-T001",
        disposition=disposition,
        confidence=0.45,
        scoring_steps=["Weighted average: 0.45"],
        stage_status={"KYC": "RAN"},
        stage_findings={"KYC": ["confidence 0.45"]},
        clauses={"KYC-002": "Inconsistency"},
    )


def _model(**text):
    draft = AuditorDraft(
        executive_summary=text.get("summary", "Summary."),
        investigation_narrative="Story.",
        compliance_conclusion=text.get("conclusion", "Per KYC-002 and FAKE-001."),
        str_narrative=text.get("str", "An STR."),
    )
    return lambda prompt, schema, stage: draft


def test_invented_clause_ids_are_removed(tmp_path):
    out = run(_brief(), LLMMode.LIVE, cache=ResponseCache("a", root=tmp_path), model_call=_model())
    assert out.removed_citations == ["FAKE-001"]
    assert "FAKE-001" not in out.draft.compliance_conclusion
    assert "KYC-002" in out.draft.compliance_conclusion


def test_str_only_on_escalation(tmp_path):
    cache = ResponseCache("a", root=tmp_path)
    assert run(_brief(), LLMMode.LIVE, cache=cache, model_call=_model()).draft.str_narrative is None
    esc = run(
        _brief(Disposition.SUSPICIOUS_ESCALATE), LLMMode.LIVE, cache=cache, model_call=_model()
    )
    assert esc.draft.str_narrative == "An STR."


def test_a_failed_call_degrades_to_the_template(tmp_path):
    def broken(prompt, schema, stage):
        raise RuntimeError("quota")

    out = run(_brief(), LLMMode.LIVE, cache=ResponseCache("a", root=tmp_path), model_call=broken)
    assert out.status == StageStatus.FAILED
    assert "MONITOR" in out.draft.executive_summary


def test_off_mode_writes_the_template():
    out = run(_brief(), LLMMode.OFF)
    assert out.status == StageStatus.RAN and "KYC-002" in out.draft.compliance_conclusion
