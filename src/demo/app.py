"""The demo page.

    streamlit run src/demo/app.py

Four tabs:

- Results: a saved batch run (verdict, judge inputs, label, ID pause for every account).
- Case: one account's report: the verdict and how the judge reached it, the five agents'
  reasoning trails, the clauses, the outline, the text, and the analyst's sign-off.
- Admin queue: the cases the ID check paused, with approve / block.
- Live run: one account through the live graph, node by node, with the admin pause.

The page reads run folders written by the batch (outputs/.../reports.jsonl) and the bank's
own data. It never reads the answer key. Every model is local; on this 4 GB laptop a new
answer takes seconds to tens of seconds (performance reduced by hardware limits).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from src.bank import BankData
from src.llm import LLMClient, LLMMode
from src.orchestration import batch
from src.orchestration.pipeline import Paused, Pipeline
from src.orchestration.run_state import RunState
from src.orchestration.steps import Agents
from src.rulebook import RulebookLookup
from src.schemas import FORGERY_PAUSE_THRESHOLD, InvestigationReport, ReasoningStep
from src.stages.stage0_triage import TriageContext


#: Where saved runs are read from (AML_OUTPUTS overrides it; the tests use that).
OUTPUTS = Path(os.environ.get("AML_OUTPUTS", "outputs"))
AGENTS = ("vision", "web_research", "graph", "rag", "auditor")
AGENT_TITLES = {
    "vision": "Vision (ID check + KYC claims, Qwen)",
    "web_research": "Web Research (sanctions and news, Qwen)",
    "graph": "Graph (patterns, links, label, Gemma)",
    "rag": "RAG (rulebook clauses, Gemma + embedding)",
    "auditor": "Auditor (outline + report text, Gemma)",
}
BADGE = {
    "CLEAR": "🟢 CLEAR",
    "MONITOR": "🟠 MONITOR",
    "SUSPICIOUS-ESCALATE": "🔴 ESCALATE",
    "BLOCKED_BY_ADMIN_REVIEW": "⛔ BLOCKED",
}


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def run_folders(root: Path = OUTPUTS) -> list[Path]:
    """Every folder under outputs/ that holds a saved batch, newest first."""
    if not root.exists():
        return []
    found = [p.parent for p in root.rglob("reports.jsonl")]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def load_reports(folder: Path) -> dict[str, InvestigationReport]:
    reports = {}
    with (folder / "reports.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                report = InvestigationReport.model_validate_json(line)
                reports[report.target_account] = report
    return reports


def load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def result_rows(reports: dict[str, InvestigationReport]) -> list[dict]:
    rows = []
    for account_id in sorted(reports):
        r = reports[account_id]
        vision = r.vision_findings
        rows.append(
            {
                "account": account_id,
                "verdict": BADGE.get(r.disposition.value, r.disposition.value),
                "confidence": r.confidence,
                **{k: v for k, v in sorted(r.judge_inputs.items())},
                "label": r.graph_label or "",
                "ID pause": bool(vision and vision.forgery_confidence >= FORGERY_PAUSE_THRESHOLD),
                "clauses": ", ".join(
                    r.rulebook_findings.retrieved_clause_ids if r.rulebook_findings else []
                ),
                "draft": r.is_draft,
            }
        )
    return rows


@st.cache_resource(show_spinner="Loading the bank...")
def context() -> TriageContext:
    return TriageContext.build(BankData.load(), RulebookLookup.load())


# --------------------------------------------------------------------------------------
# Pieces
# --------------------------------------------------------------------------------------


def show_trail(steps: list[ReasoningStep]) -> None:
    if not steps:
        st.caption("No steps.")
        return
    st.table([{"step": s.step, "finding": s.finding, "impact": s.impact} for s in steps])


def show_report(report: InvestigationReport, key: str) -> None:
    signed = st.session_state.setdefault("signed", set())
    draft = report.is_draft and report.case_id not in signed
    top = st.columns(4)
    top[0].metric("Verdict", BADGE.get(report.disposition.value, report.disposition.value))
    top[1].metric("Confidence", f"{report.confidence:.2f}")
    top[2].metric("Status", "DRAFT" if draft else "SIGNED OFF")
    top[3].metric("Pattern label", report.graph_label or "none")
    if report.forgery_notice:
        st.warning(report.forgery_notice)
    flags = [
        name
        for name, on in (
            ("evidence incomplete", report.evidence_incomplete),
            ("single signal held at MONITOR", report.single_signal_capped),
            ("policy gap", report.policy_gap_identified),
        )
        if on
    ]
    if flags:
        st.info("Flags: " + ", ".join(flags))

    st.subheader("Report")
    st.markdown(f"**Summary.** {report.executive_summary}")
    st.markdown(f"**Narrative.** {report.investigation_narrative}")
    st.markdown(f"**Conclusion.** {report.compliance_conclusion}")
    if report.str_narrative:
        st.markdown(f"**STR draft.** {report.str_narrative}")
    if draft and st.button("Sign off as analyst", key=f"sign-{key}-{report.case_id}"):
        signed.add(report.case_id)
        st.rerun()

    st.subheader("How the judge reached the verdict (Python, no model)")
    st.caption(
        "Judge inputs: "
        + (", ".join(f"{k} {v:g}" for k, v in report.judge_inputs.items()) or "none")
        + ". Stages: "
        + ", ".join(f"{k} {v.value}" for k, v in report.stage_status.items())
    )
    show_trail(report.judge_trail)

    st.subheader("The five agents")
    for name in AGENTS:
        with st.expander(AGENT_TITLES[name]):
            show_trail(report.agent_trails.get(name, []))

    with st.expander("Outline the text was written from"):
        st.caption("Every number and clause id in the report text must appear here.")
        for line in report.report_outline:
            st.markdown(line if line.startswith("-") else f"**{line}**")
    with st.expander("Raw report (JSON)"):
        st.json(json.loads(report.model_dump_json()))


# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------


def tab_results(folder: Path | None, reports: dict[str, InvestigationReport]) -> None:
    if folder is None:
        st.info("No saved run yet. Run a batch first (see the README).")
        return
    summary = load_json(folder / "summary.json", {})
    counts = summary.get("dispositions", {})
    cols = st.columns(5)
    cols[0].metric("Accounts", len(reports))
    cols[1].metric("Clear", counts.get("CLEAR", 0))
    cols[2].metric("Monitor", counts.get("MONITOR", 0))
    cols[3].metric("Escalate", counts.get("SUSPICIOUS-ESCALATE", 0))
    cols[4].metric("Sent to the 9B", f"{summary.get('backup_share', 0):.0%}")
    for warning in summary.get("warnings", []):
        st.warning(warning)
    rows = result_rows(reports)
    wanted = st.multiselect(
        "Verdicts",
        sorted({r["verdict"] for r in rows}),
        default=sorted({r["verdict"] for r in rows}),
    )
    st.dataframe([r for r in rows if r["verdict"] in wanted], width="stretch", height=420)
    grade = folder / "grade.txt"
    if grade.exists():
        with st.expander("Grade of this run (written by the evaluation script)"):
            st.code(grade.read_text(encoding="utf-8"))
    if summary.get("model_calls"):
        with st.expander("Model calls and seconds per call"):
            st.table([{"agent (model)": k, **v} for k, v in sorted(summary["model_calls"].items())])


def tab_case(reports: dict[str, InvestigationReport]) -> None:
    if not reports:
        st.info("No saved run to read a case from.")
        return
    flagged = [a for a in sorted(reports) if reports[a].disposition.value != "CLEAR"]
    only = st.checkbox("Flagged accounts only", value=True)
    choices = flagged if only and flagged else sorted(reports)
    account_id = st.selectbox("Account", choices)
    show_report(reports[account_id], "saved")


def tab_admin(folder: Path | None, reports: dict[str, InvestigationReport]) -> None:
    queue = load_json(folder / "admin_queue.json", []) if folder else []
    if not queue:
        st.info("No case is waiting for the admin in this run.")
        return
    decisions_path = folder / "admin_decisions.json"
    decisions = load_json(decisions_path, {})
    st.caption(
        "The ID check never changes a score. It pauses the case until the admin decides. "
        "The batch graded these with the default decision, approve."
    )
    for item in queue:
        account_id = item["account_id"]
        with st.container(border=True):
            st.markdown(
                f"**{account_id}** ({item['case_id']}): ID check "
                f"{item['forgery_confidence']:g}; verdict if approved "
                f"{item['disposition_if_approved']}"
            )
            for line in item["field_discrepancies"]:
                st.write(f"- {line}")
            decided = decisions.get(account_id, item.get("decision"))
            cols = st.columns(3)
            for column, choice in zip(cols[:2], ("approve", "block"), strict=True):
                if column.button(choice.capitalize(), key=f"{choice}-{account_id}"):
                    decisions[account_id] = choice
                    decisions_path.write_text(json.dumps(decisions, indent=2), encoding="utf-8")
                    st.rerun()
            cols[2].markdown(f"Decision: **{decided or 'waiting'}**")


def tab_live(folder: Path | None) -> None:
    ctx = context()
    cols = st.columns([2, 1, 1])
    account_id = cols[0].selectbox("Account to review", ctx.bank.account_ids(), key="live-account")
    mode = cols[1].selectbox("Models", ("cached", "live", "off"), key="live-mode")
    cols[2].caption("cached: saved answers only. live: calls the local models. off: Python only.")
    after = folder is not None and st.checkbox(
        "Review after the saved run (the other accounts' verdicts are known)",
        value=True,
        help="Links to flagged accounts only score when those accounts were reviewed first. "
        "Unticked, the account is reviewed alone, as the first case of a new run.",
    )
    session = st.session_state
    if st.button("Run the case", type="primary"):
        client = LLMClient(LLMMode(mode), audit_path=OUTPUTS / "audit" / "llm_calls.jsonl")
        state = None
        if after:
            state = RunState.from_cases(batch.load_cases(folder), ctx.bank, skip=account_id)
        pipeline = Pipeline(ctx, Agents.build(client), state=state)
        session["live"] = {"pipeline": pipeline, "log": [], "result": None}
        _drive(lambda on_node: pipeline.run(batch.review(account_id), on_node=on_node))
    live = session.get("live")
    if not live:
        return
    st.table(live["log"])
    result = live["result"]
    if isinstance(result, Paused):
        st.error("Paused for the admin: the ID check flagged the card with both reads.")
        for line in result.evidence["field_discrepancies"]:
            st.write(f"- {line}")
        st.caption(f"Verdict if approved: {result.evidence['disposition_if_approved']}")
        cols = st.columns(2)
        for column, choice in zip(cols, ("approve", "block"), strict=True):
            if column.button(choice.capitalize(), key=f"live-{choice}"):
                _drive(
                    lambda on_node, c=choice, p=live["pipeline"], r=result: p.resume(
                        r, c, on_node=on_node
                    )
                )
                st.rerun()
    elif isinstance(result, InvestigationReport):
        show_report(result, "live")
    elif result is None and live["log"]:
        st.info("A quiet transaction on a flagged account: no new case.")


def _drive(call) -> None:
    """Run or resume the live graph, logging each node as it finishes."""
    live = st.session_state["live"]
    with st.status("Running the live graph...", expanded=True) as status:

        def on_node(name, work, seconds) -> None:
            note = ""
            if name == "judge" and work.score is not None:
                note = f"{work.score.disposition.value} {work.score.confidence:g}"
            live["log"].append({"node": name, "seconds": round(seconds, 1), "result": note})
            st.write(f"{name} ({seconds:.1f} s) {note}")

        live["result"] = call(on_node)
        status.update(label="Done", state="complete")


def main() -> None:
    st.set_page_config(page_title="AML Auditor v2", layout="wide")
    st.title("AML Auditor v2")
    st.caption(
        "Five agents, every model local on Ollama. Python sets every score; the models "
        "read, label and write. A report is a recommendation for the lead analyst."
    )
    folders = run_folders()
    folder = None
    if folders:
        folder = st.sidebar.selectbox("Saved run", folders, format_func=lambda p: str(p))
    reports = load_reports(folder) if folder else {}
    results, case, admin, live = st.tabs(["Results", "Case", "Admin queue", "Live run"])
    with results:
        tab_results(folder, reports)
    with case:
        tab_case(reports)
    with admin:
        tab_admin(folder, reports)
    with live:
        tab_live(folder)


if __name__ == "__main__":
    main()
