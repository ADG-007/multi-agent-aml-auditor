"""The Streamlit page.

    streamlit run src/demo/app.py

Runs the 50-account batch once, then "Run case" replays an account's final case through the
LangGraph pipeline (same state, same verdict as the CLI) with the writer on. A paused case
waits for Approve or Block.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import replace
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # `streamlit run` from the root

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.bank import BankData  # noqa: E402
from src.evaluation.mvp_report import metrics, rows  # noqa: E402
from src.evaluation.mvp_subset import load_subset  # noqa: E402
from src.llm.cache import LLMMode  # noqa: E402
from src.orchestration.batch import review, run_batch  # noqa: E402
from src.orchestration.pipeline import Paused, Pipeline  # noqa: E402
from src.rulebook import RulebookLookup  # noqa: E402
from src.schemas import InvestigationReport  # noqa: E402
from src.stages.stage0_triage import TriageContext  # noqa: E402
from src.stages.stage3_kyc.agent import stale_warning  # noqa: E402


COLOURS = {"CLEAR": "green", "MONITOR": "orange", "SUSPICIOUS-ESCALATE": "red"}

st.set_page_config(page_title="AML Auditor", layout="wide")


@st.cache_resource(show_spinner="Running the 50-account batch (about half a minute)...")
def warm_run(kyc_mode: str):
    ctx = TriageContext.build(BankData.load(), RulebookLookup.load(), llm_mode=LLMMode(kyc_mode))
    subset = load_subset()
    result = run_batch(ctx, [r["account_id"] for r in subset["accounts"]])
    return ctx, subset, result


def main() -> None:
    st.title("AML Auditor")
    with st.sidebar:
        kyc_mode = st.selectbox(
            "Stage 3 (KYC) model",
            ["cached", "live", "off"],
            index=0,
            help="cached = saved 8B answers; live = call Ollama on a miss",
        )
        writer_mode = st.selectbox(
            "Stage 7 (writer) model",
            ["live", "cached", "off"],
            index=0,
            help="live = free Gemini; off = template text",
        )
        st.caption(
            "Stage 3 runs a local 8B model on a 4 GB laptop: performance is reduced by "
            "hardware limits."
        )

    ctx, subset, warm = warm_run(kyc_mode)
    if kyc_mode == "cached" and (warning := stale_warning(len(subset["accounts"]))):
        st.warning(warning)
    tab_case, tab_run = st.tabs(["Run a case", "50-account run"])

    with tab_case:
        case_tab(ctx, subset, warm, LLMMode(writer_mode))
    with tab_run:
        run_tab(subset, warm)


def case_tab(ctx, subset, warm, writer_mode: LLMMode) -> None:
    show = {v: k for k, v in subset["show_cases"].items()}
    labels = {
        r["account_id"]: f"{r['account_id']} · {r['group']}"
        + (f" · show case {show[r['account_id']]}" if r["account_id"] in show else "")
        for r in subset["accounts"]
    }
    order = [*subset["show_cases"].values()] + [a for a in labels if a not in show]
    account_id = st.selectbox("Account", order, format_func=labels.get)

    if st.session_state.get("account") != account_id:  # a new pick clears the old report
        st.session_state.pop("outcome", None)
    if st.button("Run case", type="primary"):
        start = copy.deepcopy(warm.before[warm.latest[account_id].case_id])
        pipeline = Pipeline(replace(ctx, writer_mode=writer_mode), state=start)
        with st.spinner("Running the 8 stages..."):
            outcome = pipeline.run(review(account_id))
        st.session_state["pipeline"] = pipeline
        st.session_state["outcome"] = outcome
        st.session_state["account"] = account_id

    outcome = st.session_state.get("outcome")
    if outcome is None:
        st.info("Pick an account and press Run case.")
        return
    if isinstance(outcome, Paused):
        paused_panel(outcome)
        return
    report_panel(outcome, subset)


def paused_panel(paused: Paused) -> None:
    ev = paused.evidence
    st.warning(
        f"⏸️ {paused.account_id} is PAUSED for the admin: the ID check found "
        f"forgery confidence {ev['forgery_confidence']:.2f}."
    )
    for d in ev["field_discrepancies"]:
        st.write(f"- {d}")
    st.write(
        f"Other stages are done. If approved, the judge's verdict is "
        f"**{ev['disposition_if_approved']}** at {ev['score_if_approved']:.4f} "
        f"(the ID check never changes the score)."
    )
    st.json(ev["stage_status"])
    left, right = st.columns(2)
    decision = None
    if left.button("✅ Approve", width="stretch"):
        decision = "approve"
    if right.button("⛔ Block", width="stretch"):
        decision = "block"
    if decision:
        with st.spinner("Resuming..."):
            st.session_state["outcome"] = st.session_state["pipeline"].resume(paused, decision)
        st.rerun()


def report_panel(r: InvestigationReport, subset: dict) -> None:
    expected = {x["account_id"]: x for x in subset["accounts"]}.get(r.target_account)
    colour = COLOURS.get(r.disposition.value, "gray")
    st.markdown(f"## :{colour}[{r.disposition.value}] · confidence {r.confidence:.4f}")
    st.caption(
        f"{r.case_id} · {r.pipeline_status.value}"
        + (f" · answer key: {expected['expected']} ({expected['group']})" if expected else "")
    )
    if r.forgery_notice:
        st.info(r.forgery_notice)
    flags = [
        n
        for n in ("evidence_incomplete", "single_signal_capped", "policy_gap_identified")
        if getattr(r, n)
    ]
    if flags:
        st.warning("Flags: " + ", ".join(flags))

    stage_rows = []
    for name, field in [
        ("Triage", "triage_findings"),
        ("Vision", "vision_findings"),
        ("TransactionPattern", "transaction_pattern_findings"),
        ("KYC", "kyc_findings"),
        ("AdverseMedia", "adverse_media_findings"),
        ("Graph", "graph_findings"),
        ("Rulebook", "rulebook_findings"),
        ("Auditor", None),
    ]:
        f = getattr(r, field) if field else None
        score = getattr(f, "confidence", getattr(f, "forgery_confidence", None))
        stage_rows.append(
            {
                "Stage": name,
                "Status": r.stage_status.get(name).value if r.stage_status.get(name) else "-",
                "Score": None if score is None else round(score, 4),
            }
        )
    st.dataframe(pd.DataFrame(stage_rows), hide_index=True, width="content")

    st.subheader("Summary")
    st.write(r.executive_summary)
    st.subheader("Compliance conclusion")
    st.write(r.compliance_conclusion)
    if r.rulebook_findings:
        st.caption(
            "Citable clauses: " + (", ".join(r.rulebook_findings.retrieved_clause_ids) or "none")
        )
    if r.str_narrative:
        st.subheader("STR draft")
        st.write(r.str_narrative)
    with st.expander("Narrative"):
        st.write(r.investigation_narrative)
    with st.expander("Why the case was opened"):
        for reason in r.triage_findings.trigger_reasons:
            st.write(f"- {reason}")
    for title, f in [
        ("ID check", r.vision_findings),
        ("Money moves", r.transaction_pattern_findings),
        ("Profile", r.kyc_findings),
        ("Blacklist", r.adverse_media_findings),
        ("Links", r.graph_findings),
        ("Rules", r.rulebook_findings),
    ]:
        if f is None:
            continue
        with st.expander(f"Reasoning trail · {title}"):
            for s in f.reasoning_trail:
                st.markdown(f"**{s.step}** — {s.finding}  \n_{s.impact}_")
    if r.auditor_trail:
        with st.expander("Reasoning trail · Writer"):
            for s in r.auditor_trail:
                st.markdown(f"**{s.step}** — {s.finding}  \n_{s.impact}_")


def run_tab(subset: dict, warm) -> None:
    table = rows(subset, warm.latest)
    st.write(
        f"{len(warm.order)} accounts, {len(warm.cases)} cases, {len(warm.rounds)} round(s). "
        "Writer off for this table; the verdict never depends on it."
    )
    st.table(pd.DataFrame([metrics(table)]).T.rename(columns={0: "value"}))
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Account": t.account_id,
                    "Kind": t.kind,
                    "Group": t.group,
                    "Expected": t.expected,
                    "Got": t.got,
                    "Right": "✅" if t.right else "❌",
                    "Score": round(t.confidence, 4),
                    **t.stages,
                    "ID": t.forgery,
                }
                for t in table
            ]
        ),
        hide_index=True,
        width="stretch",
    )


main()
