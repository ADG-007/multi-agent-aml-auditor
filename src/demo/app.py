"""The Streamlit page.

    streamlit run src/demo/app.py

Four sections, picked in the sidebar:
- Start here: what the tool does and how to use the page.
- How well it did: one data set's verdicts against its answer key.
- Investigate a case: one account's verdict, why the judge reached it, the evidence, the report.
- Test with new accounts: make accounts (a ready pattern, or built to your own settings), set
  the answer, run them on the bank.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import os
import shutil
import sys
import time
import urllib.request
import zlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # `streamlit run` from the root

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.bank import BankData  # noqa: E402
from src.evaluation.mvp_report import SHORT, Row, confusion, rows  # noqa: E402
from src.evaluation.mvp_subset import load_subset  # noqa: E402
from src.llm import get_stage_config  # noqa: E402
from src.llm.cache import LLMMode, ResponseCache  # noqa: E402
from src.newdata import (  # noqa: E402
    KINDS,
    MIX_SIZE,
    VERDICTS,
    AccountSpec,
    Pack,
    free_folder,
    list_packs,
    listed_people,
    make_custom_pack,
    make_pack,
    mix_title,
    profile_text,
    run_pack,
)
from src.orchestration.batch import BatchResult, review, run_batch  # noqa: E402
from src.orchestration.pipeline import NODE, OnNode, Paused, Pipeline  # noqa: E402
from src.orchestration.steps import FINDINGS_FIELD, CaseWork  # noqa: E402
from src.rulebook import RulebookLookup  # noqa: E402
from src.schemas import (  # noqa: E402
    AUTO_VETO_THRESHOLD,
    CLEAR_BELOW,
    CORROBORATION_MIN_STAGES,
    CORROBORATION_STAGE_MIN,
    ESCALATE_FROM,
    SIGNAL_FLOOR_MIN,
    SOLO_SIGNAL_MIN,
    SOLO_SIGNAL_MIN_RAN,
    STAGE_WEIGHTS,
    InvestigationReport,
    PipelineStatus,
    ReasoningStep,
    StageName,
)
from src.stages.stage0_triage import TriageContext  # noqa: E402
from src.stages.stage1_vision import config as vision_config  # noqa: E402
from src.stages.stage3_kyc import config as kyc_config  # noqa: E402
from src.stages.stage3_kyc.agent import stale_warning  # noqa: E402
from src.stages.stage7_auditor import config as auditor_config  # noqa: E402


SAMPLE = "sample"
SAMPLE_TITLE = "Bank sample: 50 known accounts"
SECONDS_PER_AI_ANSWER = 30  # the local 8B on this 4 GB laptop
SECONDS_PER_REPORT = 30  # one Gemini call plus the free-tier gap between calls
PROMPT_VERSIONS = {"kyc": kyc_config.PROMPT_VERSION, "auditor": auditor_config.PROMPT_VERSION}

#: colour, icon, what the bank does next
VERDICT = {
    "CLEAR": ("#1e8e3e", "✅", "No action needed"),
    "MONITOR": ("#e37400", "👁️", "Keep the account under watch"),
    "SUSPICIOUS-ESCALATE": ("#d93025", "🚨", "Escalate and file a Suspicious Transaction Report"),
    "BLOCKED_BY_ADMIN_REVIEW": ("#5f6368", "⛔", "Blocked by the admin after the ID check"),
}
SAY = {
    "CLEAR": "✅ Clear",
    "MONITOR": "👁️ Monitor",
    "ESCALATE": "🚨 Escalate",
    "BLOCKED": "⛔ Blocked",
}
SAY_FULL = {"CLEAR": "✅ Clear", "MONITOR": "👁️ Monitor", "SUSPICIOUS-ESCALATE": "🚨 Escalate"}
STAGES = {  # weighted stage -> (label, report field)
    StageName.TRANSACTION_PATTERN: ("Money movements", "transaction_pattern_findings"),
    StageName.KYC: ("Customer profile (AI)", "kyc_findings"),
    StageName.ADVERSE_MEDIA: ("Watch lists", "adverse_media_findings"),
    StageName.GRAPH: ("Links to flagged accounts", "graph_findings"),
}
EVIDENCE = {  # tab -> (report field, what to list, what the stage looks for)
    "Money movements": (
        "transaction_pattern_findings",
        ("structuring_indicators", "pass_through_indicators"),
        "Cash just under the ₹10 lakh report line, and money that comes in and leaves fast.",
    ),
    "Customer profile": (
        "kyc_findings",
        ("profile_behavior_mismatches",),
        "Does the customer's declared job, income and account use explain this money?",
    ),
    "Watch lists": (
        "adverse_media_findings",
        ("matched_entries",),
        "Is the holder or a payee on a sanctions or bad-news list?",
    ),
    "Links": (
        "graph_findings",
        ("shared_identifiers", "connected_flagged_accounts"),
        "Does the account share a phone, IP address or home with an account already flagged?",
    ),
    "ID card": (
        "vision_findings",
        ("field_discrepancies",),
        "Do the name, birth date and number printed on the ID card match the record?",
    ),
    "Rules": (
        "rulebook_findings",
        ("retrieved_clause_ids", "policy_violations"),
        "Which rulebook clauses fit the findings? The report may cite only these.",
    ),
}
KYC_MODES = {
    "cached": "Saved AI answers (instant)",
    "live": "AI profile check, saved answers reused",
    "off": "No profile check (fastest)",
}
WRITER_MODES = {"live": "Gemini AI, live", "cached": "Gemini, saved only", "off": "Plain template"}
#: LangGraph node -> what the live run calls it
NODE_LABELS = {
    "triage": "Triage: which checks run",
    "vision": "ID card (OCR)",
    "transaction_pattern": "Money movements",
    "kyc": "Customer profile (AI)",
    "adverse_media": "Watch lists",
    "graph": "Links to flagged accounts",
    "judge": "Judge (fixed rules)",
    "admin_gate": "Admin gate",
    "rulebook": "Rulebook search",
    "writer": "Report writer (AI)",
}
STAGE_OF_NODE = {node: stage for stage, node in NODE.items()}
NOT_RUN = {"NOT_ROUTED": "not needed", "FAILED": "failed", "TIMED_OUT": "timed out"}
WAITING = "⏸️ waiting for your decision"

st.set_page_config(page_title="AML Auditor", page_icon="🛡️", layout="wide")


@dataclass
class Run:
    key: str
    title: str
    ctx: TriageContext | None  # None for a run loaded from its saved results
    answers: dict
    result: BatchResult
    seconds: float
    kyc: str
    saved_at: str | None = None  # when a loaded run finished
    case_count: int | None = None

    @property
    def table(self) -> list[Row]:
        return rows(self.answers, self.result.latest)

    @property
    def cases(self) -> int:
        return len(self.result.cases) if self.case_count is None else self.case_count


# --------------------------------------------------------------------------------------
# Cached work: loaded or run once per server, shared by every session
# --------------------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def rulebook() -> RulebookLookup:
    return RulebookLookup.load()


@st.cache_resource(show_spinner=False)
def committed_bank() -> BankData:
    """Read only: new accounts draw names, dates and the sanctions list from it."""
    return BankData.load()


@st.cache_resource(
    show_spinner="Checking the 50 sample accounts (seconds; the very first run takes ~40 s)..."
)
def sample_run(kyc: str) -> Run:
    started = time.perf_counter()
    ctx = TriageContext.build(BankData.load(), rulebook(), llm_mode=LLMMode(kyc))
    subset = load_subset()
    result = run_batch(ctx, [r["account_id"] for r in subset["accounts"]])
    return Run(SAMPLE, SAMPLE_TITLE, ctx, subset, result, time.perf_counter() - started, kyc)


@st.cache_resource(show_spinner=False)
def pack_runs() -> dict[tuple[str, str], Run]:
    """Finished runs of new accounts, keyed by (folder, KYC mode)."""
    return {}


@st.cache_resource(show_spinner=False)
def known_packs() -> dict[tuple[Path, int], Pack]:
    """Pack objects by (folder, answer-key time), so each answer key is read once."""
    return {}


def packs_now() -> dict[str, Pack]:
    known = known_packs()
    out = {}
    for p in list_packs():
        stamp = (p.path / "_ground_truth" / "answers.json").stat().st_mtime_ns
        out[p.name] = known.setdefault((p.path, stamp), p)
    return out


def saved_answers(pack: Pack, stage: str) -> int:
    """Saved answers of one model stage ("kyc" or "auditor") in the pack's own folder."""
    folder = ResponseCache(stage, root=pack.cache_root).folder
    stamp = folder.stat().st_mtime_ns if folder.exists() else 0
    return _saved(str(folder), stamp, PROMPT_VERSIONS[stage])


@st.cache_data(show_spinner=False)
def _saved(folder: str, stamp: int, version: str) -> int:
    """Counted again only when the folder changes (a new answer is a new file)."""
    return ResponseCache(Path(folder).name, root=Path(folder).parent).count(version)


def saved_kyc(pack: Pack) -> int:
    return saved_answers(pack, "kyc")


def still_needed(pack: Pack) -> tuple[int, int]:
    """New 8B answers and new Gemini reports a run of this pack would ask for."""
    n = len(pack.account_ids)
    return max(n - saved_kyc(pack), 0), max(n - saved_answers(pack, "auditor"), 0)


def models() -> tuple[str, str]:
    """The model IDs Stage 3 and Stage 7 use now (an .env override included)."""
    kyc = get_stage_config("KYC").model_id or "unset"
    return kyc, get_stage_config("Auditor").model_id or "unset"


def model_problems() -> list[str]:
    """Why the local 8B or Gemini cannot answer now. Checked before new accounts run."""
    kyc_model, _ = models()
    problems = []
    host = os.environ.get("OLLAMA_HOST", "localhost:11434")
    host = host if host.startswith("http") else f"http://{host}"
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as reply:
            pulled = {m["name"] for m in json.load(reply).get("models", [])}
        if kyc_model not in pulled:
            problems.append(
                f"The local model `{kyc_model}` is not pulled. Run `ollama pull {kyc_model}`."
            )
    except (OSError, ValueError):
        problems.append(
            f"The local model server (Ollama) is not answering at {host}. "
            "Start it with `ollama serve`."
        )
    if not os.environ.get("GOOGLE_API_KEY"):
        problems.append("No Gemini key: put GOOGLE_API_KEY in the .env file.")
    return problems


def save_run(pack: Pack, run: Run) -> None:
    """Writes the run's summary beside its reports (outputs/run.jsonl), so the results open
    later without running the pipeline again."""
    kyc_model, writer_model = models()
    summary = {
        "finished": time.strftime("%d %b %Y %H:%M"),
        "seconds": round(run.seconds, 1),
        "cases": len(run.result.cases),
        "rounds": run.result.rounds,
        "order": run.result.order,
        "kyc_model": kyc_model,
        "writer_model": writer_model,
    }
    out = pack.path / "outputs" / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def saved_run(pack: Pack) -> Run | None:
    """The pack's last run, read from its saved reports. None if it was never run here."""
    folder = pack.path / "outputs"
    reports = folder / "run.jsonl"
    if not reports.exists():
        return None
    latest = {}
    for line in reports.read_text(encoding="utf-8").splitlines():
        if line.strip():
            report = InvestigationReport.model_validate_json(line)
            latest[report.target_account] = report
    if set(latest) != set(pack.account_ids):
        return None
    summary = {  # a run saved before summaries existed: what the reports alone can tell
        "finished": time.strftime("%d %b %Y %H:%M", time.localtime(reports.stat().st_mtime)),
        "seconds": 0.0,
        "cases": len(latest),
        "rounds": [list(latest)],
        "order": list(latest),
    }
    if (folder / "summary.json").exists():
        summary |= json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    result = BatchResult(order=summary["order"], latest=latest, rounds=summary["rounds"])
    return Run(
        pack.name,
        pack.title,
        None,
        pack.answers,
        result,
        summary["seconds"],
        "live",
        saved_at=summary["finished"],
        case_count=summary["cases"],
    )


def pack_run(pack: Pack) -> Run | None:
    """This session's run of the pack, else its saved run. Never runs the pipeline."""
    return pack_runs().get((pack.name, "live")) or saved_run(pack)


def run_pack_now(pack: Pack) -> Run | None:
    """Runs the pack through the whole pipeline: the local 8B for the profile, Gemini for each
    report. Saved answers are reused; None when a model cannot be reached."""
    new_kyc, new_reports = still_needed(pack)
    if (new_kyc or new_reports) and (problems := model_problems()):
        for problem in problems:
            st.error(problem)
        return None
    kyc_model, writer_model = models()
    label = (
        f"Checking {len(pack.account_ids)} accounts of “{pack.title}” "
        f"with {kyc_model} and {writer_model}"
    )
    if new_kyc or new_reports:
        label += (
            f": {new_kyc} new profile check(s), {new_reports} new report(s), "
            f"about {minutes(new_kyc, new_reports)}"
        )
    started = time.perf_counter()
    with st.status(label + "...", expanded=True) as status:
        ctx, result = run_pack(
            pack, rulebook(), kyc=LLMMode.LIVE, writer=LLMMode.LIVE, progress=status.write
        )
        seconds = time.perf_counter() - started
        status.update(label=f"Done in {seconds:.0f} s", state="complete", expanded=False)
    run = Run(pack.name, pack.title, ctx, pack.answers, result, seconds, "live")
    pack_runs()[(pack.name, "live")] = run
    save_run(pack, run)
    return run


def minutes(ai_answers: int, reports: int = 0) -> str:
    s = ai_answers * SECONDS_PER_AI_ANSWER + reports * SECONDS_PER_REPORT
    return f"{s} s" if s < 90 else f"{round(s / 60)} min"


def full_run_note(accounts: int) -> None:
    kyc_model, writer_model = models()
    st.caption(
        f"Each new account runs the whole pipeline: the local **{kyc_model}** (Ollama) checks "
        f"the profile and **{writer_model}** (Gemini) writes the report. About "
        f"{minutes(accounts, accounts)} for {accounts} account{'s' * (accounts > 1)} "
        "(performance reduced by hardware limits)."
    )


# --------------------------------------------------------------------------------------
# Navigation and the shared data-set picker
# --------------------------------------------------------------------------------------


PAGES: dict = {}


def main() -> None:
    PAGES.update(
        home=st.Page(home, title="Start here", icon="🏠", url_path="start", default=True),
        results=st.Page(results, title="How well it did", icon="📊", url_path="results"),
        case=st.Page(investigate, title="Investigate a case", icon="🔎", url_path="case"),
        new=st.Page(new_data, title="Test with new accounts", icon="🧪", url_path="new"),
    )
    nav = st.navigation(list(PAGES.values()))
    if "next_dataset" in st.session_state:  # set before a page switch, applied before widgets
        st.session_state["dataset"] = st.session_state.pop("next_dataset")
    with st.sidebar:
        st.markdown("### Data set")
        packs = packs_now()
        titles = {SAMPLE: SAMPLE_TITLE, **{k: p.title for k, p in packs.items()}}
        if st.session_state.get("dataset") not in titles:
            st.session_state["dataset"] = SAMPLE
        key = st.selectbox(
            "Data set",
            list(titles),
            format_func=titles.get,
            key="dataset",
            label_visibility="collapsed",
            help="The 50 sample accounts, or a batch you made under Test with new accounts.",
        )
        if key == SAMPLE:
            st.selectbox(
                "Customer profile check",
                ["cached", "live", "off"],
                format_func=KYC_MODES.get,
                key="kyc-sample",
                help="Stage 3 asks a local AI model whether the customer's profile explains "
                f"the money. A new answer takes about {SECONDS_PER_AI_ANSWER} s on this laptop "
                "(performance reduced by hardware limits). Saved answers make runs repeatable.",
            )
        else:
            kyc_model, writer_model = models()
            st.caption(
                f"New accounts run the whole pipeline: **{kyc_model}** (local) checks the "
                f"profile and **{writer_model}** writes each report. Saved answers are reused."
            )
        st.session_state["packs"] = packs
    nav.run()


def current_run(wait: bool = True) -> Run | None:
    """The chosen data set's run. A sample run, or a batch whose answers are all saved, runs
    at once; a batch that needs new AI answers waits for a button."""
    key = st.session_state["dataset"]
    if key == SAMPLE:
        kyc = st.session_state.get("kyc-sample", "cached")
        run = sample_run(kyc)
        if kyc == "cached" and (warning := stale_warning(len(run.answers["accounts"]))):
            st.warning(warning)
        return run
    pack = st.session_state["packs"][key]
    run = pack_run(pack)
    if run is None and wait:
        st.info(f"“{pack.title}” has not been run through the pipeline yet.")
        st.dataframe(planted(pack), hide_index=True, width="stretch")
        full_run_note(len(pack.account_ids))
        if st.button("▶ Run through the pipeline", type="primary"):
            run = run_pack_now(pack)
    return run


def purpose(text: str) -> None:
    st.markdown(
        f"<div style='font-size:1.05rem;opacity:.85;margin:-6px 0 14px'>{text}</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------------------
# Start here
# --------------------------------------------------------------------------------------


def home() -> None:
    st.title("🛡️ AML Auditor")
    purpose(
        "An assistant for anti-money-laundering analysts. It checks a flagged bank account "
        "the way an analyst would, scores the risk with fixed rules, and drafts the report. "
        "A person always makes the final call."
    )
    cols = st.columns(3)
    for col, (head, body) in zip(
        cols,
        (
            (
                "1 · Gather the evidence",
                "Seven checks look at money movements, the customer's profile, watch lists, "
                "links to other accounts, the ID card and the rulebook.",
            ),
            (
                "2 · Score with fixed rules",
                "A judge written in plain Python turns the checks into one score and a "
                "verdict: Clear, Monitor or Escalate. No AI model does the scoring.",
            ),
            (
                "3 · Explain the decision",
                "Every check shows its reasons. A writer drafts the report and may cite only "
                "the rulebook clauses that were found.",
            ),
        ),
        strict=True,
    ):
        with col.container(border=True):
            st.markdown(f"**{head}**\n\n{body}")

    st.subheader("Use this page in three steps")
    a, b, c = st.columns(3)
    with a.container(border=True):
        st.page_link(PAGES["results"], label="**How well it did**", icon="📊")
        st.caption("Every account has a known answer. See how many the system got right.")
    with b.container(border=True):
        st.page_link(PAGES["case"], label="**Investigate a case**", icon="🔎")
        st.caption("Open one account: the verdict, why, the evidence and the report.")
    with c.container(border=True):
        st.page_link(PAGES["new"], label="**Test with new accounts**", icon="🧪")
        st.caption("Make your own accounts, set the right answer, and see if it is caught.")

    st.subheader("How one case flows")
    st.graphviz_chart(
        """digraph {
        rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2f7",
        fontname="Helvetica", fontsize=11]; edge [color="#888888"];
        T [label="Alert or\\nreview"]; S [label="7 evidence\\nchecks"];
        J [label="Judge\\n(fixed rules)", fillcolor="#fde8c8"];
        A [label="ID card fake?\\nadmin decides", fillcolor="#f3d6d6"];
        R [label="Rulebook\\nsearch"]; W [label="Report\\nwriter (AI)"];
        T -> S -> J -> A -> R -> W }""",
        width="stretch",
    )
    judge_rules(expanded=False)
    st.caption(
        "All data is synthetic. The customer-profile AI runs on this 4 GB laptop: about "
        f"{SECONDS_PER_AI_ANSWER} s per new answer (performance reduced by hardware limits)."
    )


def judge_rules(expanded: bool) -> None:
    with st.expander("How the judge decides", expanded=expanded):
        weights = ", ".join(f"{STAGES[s][0]} {w:g}" for s, w in STAGE_WEIGHTS.items())
        st.markdown(
            f"1. **Veto**: a watch-list or link score of {AUTO_VETO_THRESHOLD:g} or more "
            "escalates at once.\n"
            f"2. **Average** of the checks that found something. Weights: {weights}.\n"
            f"3. **Bands**: below {CLEAR_BELOW:g} is Clear, below {ESCALATE_FROM:g} is Monitor, "
            "above that is Escalate if the gate passes.\n"
            f"4. **Gate**: {CORROBORATION_MIN_STAGES} checks at {CORROBORATION_STAGE_MIN:g} or "
            f"more, or one at {SOLO_SIGNAL_MIN:g} or more when {SOLO_SIGNAL_MIN_RAN} or more "
            "checks found something.\n"
            f"5. **Floors**: a check at {SIGNAL_FLOOR_MIN:g} or more, or a check that failed, "
            "is never Clear.\n"
            "6. The ID card check never changes the score. A suspected fake pauses the case "
            "for an admin."
        )


# --------------------------------------------------------------------------------------
# How well it did
# --------------------------------------------------------------------------------------


def results() -> None:
    st.title("📊 How well it did")
    purpose(
        "Each account has a known right answer, written before the run. This page compares "
        "the system's verdict with it. Click a row to open that case."
    )
    run = current_run()
    if run is None:
        return
    table = run.table
    n, right = len(table), sum(t.right for t in table)
    bad = [t for t in table if t.kind == "bad"]
    flagged = [t for t in table if t.got != "CLEAR"]
    esc = [t for t in table if t.expected == "ESCALATE"]
    innocent = [t for t in table if t.kind in ("clean", "trap")]

    st.markdown(f"**{run.title}**")
    cards = (
        ("Right verdicts", f"{right} of {n}", "The verdict equals the known answer."),
        (
            "Bad accounts caught",
            f"{sum(t.got != 'CLEAR' for t in bad)} of {len(bad)}",
            "Bad accounts given Monitor or Escalate.",
        ),
        (
            "Flags that were right",
            f"{sum(t.kind == 'bad' for t in flagged)} of {len(flagged)}",
            "Of the accounts it flagged, how many really are bad.",
        ),
        (
            "Escalations caught",
            f"{sum(t.got == 'ESCALATE' for t in esc)} of {len(esc)}",
            "Accounts whose right answer is Escalate.",
        ),
        (
            "False alarms",
            f"{sum(t.got != 'CLEAR' for t in innocent)} of {len(innocent)}",
            "Innocent accounts, including look-alike traps, that it flagged.",
        ),
    )
    for col, (label, value, help_text) in zip(st.columns(len(cards)), cards, strict=True):
        col.metric(label, value, border=True, help=help_text)
    counts = (
        f"{len(run.result.order)} accounts, {run.cases} cases, "
        f"{len(run.result.rounds)} round(s), {run.seconds:.1f} s."
    )
    if run.saved_at:
        counts = f"Saved results of the run on {run.saved_at} (not run again). " + counts
    if run.key == SAMPLE:
        st.caption(
            f"{counts} Profile check: {KYC_MODES[run.kyc].lower()}. The report writer is off "
            "for this table (replay a case for its report); the verdict never depends on it. "
            "A paused ID check is approved."
        )
    else:
        kyc_model, writer_model = models()
        st.caption(
            f"{counts} Profile check: {kyc_model} on Ollama. Reports: {writer_model}. Open a "
            "case to read its report. A paused ID check is approved."
        )
        model_failures(run)

    left, right_col = st.columns([3, 1])
    with left:
        show = st.segmented_control(
            "Show",
            ["All", "Wrong only", "Flagged only"],
            default="All",
            label_visibility="collapsed",
        )
        shown = [
            t
            for t in table
            if show in (None, "All")
            or (show == "Wrong only" and not t.right)
            or (show == "Flagged only" and t.got != "CLEAR")
        ]
        picked = st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Account": t.account_id,
                        "What it is": t.group,
                        "Right answer": SAY.get(t.expected, t.expected),
                        "System said": SAY.get(t.got, t.got),
                        "": "✓" if t.right else "✗",
                        "Score": t.confidence,
                        "Money": t.stages["TP"],
                        "Profile": t.stages["KYC"],
                        "Lists": t.stages["AM"],
                        "Links": t.stages["G"],
                        "ID card": t.forgery,
                    }
                    for t in shown
                ]
            ),
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "": st.column_config.TextColumn("", width="small"),
            },
            hide_index=True,
            width="stretch",
            height=min(38 + 35 * len(shown), 520),
            on_select="rerun",
            selection_mode="single-row",
            key=f"table-{run.key}-{show}",
        )
        st.caption(
            "Check columns: each check's score, 0 to 1. A dash means the check did not run. "
            "The ID card score never counts toward the verdict."
        )
        if picked.selection.rows:
            account_id = shown[picked.selection.rows[0]].account_id
            if st.button(f"🔎 Open {account_id} →", type="primary"):
                st.session_state[f"acct-{run.key}"] = account_id
                st.switch_page(PAGES["case"])
    with right_col:
        st.markdown("**Right answer vs what it said**")
        c = confusion(table)
        labels = ["CLEAR", "MONITOR", "ESCALATE", "BLOCKED"]
        st.dataframe(
            pd.DataFrame(
                {SAY[g]: [c[(e, g)] for e in labels[:3]] for g in labels},
                index=[SAY[e] for e in labels[:3]],
            ),
            width="stretch",
        )
        st.caption("Rows: the right answer. Columns: what the system said. The diagonal is right.")
        judge_rules(expanded=False)


def model_failures(run: Run) -> None:
    """Warns when the 8B or Gemini did not answer for some accounts of a run."""
    reports = [run.result.latest[a] for a in run.result.order]
    no_kyc = [r.target_account for r in reports if StageName.KYC.value in r.degraded_stages]
    no_report = [
        r.target_account
        for r in reports
        if r.auditor_trail and r.auditor_trail[0].finding.startswith("Writer failed")
    ]
    if no_kyc:
        st.warning(
            f"The local 8B did not answer for {len(no_kyc)} account(s) ({', '.join(no_kyc)}), "
            "so their profile check FAILED and the judge held them at Monitor at least. Check "
            "that Ollama is running, then delete the batch and make it again."
        )
    if no_report:
        st.warning(
            f"Gemini did not answer for {len(no_report)} account(s) ({', '.join(no_report)}): "
            "they show template text. Open the case to see the error."
        )


# --------------------------------------------------------------------------------------
# Investigate a case
# --------------------------------------------------------------------------------------


def investigate() -> None:
    st.title("🔎 Investigate a case")
    purpose(
        "Pick an account to see its verdict, why the judge reached it, the evidence behind "
        "it, and the drafted report."
    )
    run = current_run()
    if run is None:
        return
    answers = {a["account_id"]: a for a in run.answers["accounts"]}
    show = {v: k for k, v in run.answers.get("show_cases", {}).items()}
    table = {t.account_id: t for t in run.table}

    def label(a: str) -> str:
        t = table[a]
        tag = f"  ·  demo case {show[a]}" if a in show else ""
        mark = "✓" if t.right else "✗ wrong"
        return f"{a} · {t.group} · {SAY.get(t.got, t.got)} {mark}{tag}"

    order = [*show] + sorted(a for a in answers if a not in show)
    pick, writer_col, kyc_col, go_col = st.columns([3, 1.3, 1.5, 1.2], vertical_alignment="bottom")
    account_id = pick.selectbox("Account", order, format_func=label, key=f"acct-{run.key}")
    writer = writer_col.selectbox(
        "Report writer",
        list(WRITER_MODES),
        format_func=WRITER_MODES.get,
        help="Used only when you replay the case. The verdict never depends on the writer.",
    )
    kyc_modes = list(KYC_MODES)
    kyc = kyc_col.selectbox(
        "Profile check (replay)",
        kyc_modes,
        index=kyc_modes.index(run.kyc),
        format_func=KYC_MODES.get,
        key=f"replay-kyc-{run.key}",
        help="Saved answers replay the batch exactly. The live AI check calls the local 8B "
        f"for any answer not saved yet (about {SECONDS_PER_AI_ANSWER} s each).",
    )
    case = run.result.latest[account_id]
    key = f"replay-{run.key}-{run.kyc}-{account_id}-{writer}-{kyc}"
    can_replay = run.ctx is not None and case.case_id in run.result.before
    if go_col.button("▶ Replay live", type="primary", width="stretch", disabled=not can_replay):
        st.session_state[key] = replay(run, case, writer, kyc)
    if can_replay:
        st.caption(
            "Below is the case as the batch saw it, shown at once. **Replay live** runs it "
            "again through the LangGraph pipeline from the same starting point, shows each "
            "step as it finishes, writes the report, and stops at a fake ID card for your "
            "decision."
        )
    else:
        st.caption(
            f"Saved result of the run on {run.saved_at}, with its saved report: shown as it "
            "was, not run again. Replay works on a batch run in this session."
        )

    expected = answers.get(account_id)
    rb = rulebook()
    done: Replay | None = st.session_state.get(key)
    if done is None:
        report_view(case, expected, rb, "from the batch")
        return

    if st.button("↩ Back to the batch view", key=f"{key}-back"):
        del st.session_state[key]
        st.rerun()
    seconds = sum(r["Seconds"] for r in done.log)
    with st.expander(f"Live run, step by step ({seconds:.1f} s)"):
        st.dataframe(pd.DataFrame(done.log), hide_index=True, width="stretch")
    if isinstance(done.outcome, Paused):
        paused_panel(done, key)
        st.subheader("What the checks found so far")
        report_view(
            done.pipeline.preview(done.outcome), expected, rb, "live replay · waiting for you"
        )
    elif done.outcome is None:
        st.info("The replay opened no case: triage found nothing to review. Batch case shown.")
        report_view(case, expected, rb, "from the batch")
    else:
        report_view(done.outcome, expected, rb, "live replay")


@dataclass
class Replay:
    """One live replay: the pipeline (it holds a paused case), its outcome and its step log."""

    pipeline: Pipeline
    outcome: InvestigationReport | Paused | None
    log: list[dict]


def replay(run: Run, case: InvestigationReport, writer: str, kyc: str) -> Replay:
    """Run the case again from the state before it, showing each node as it finishes."""
    start = copy.deepcopy(run.result.before[case.case_id])
    ctx = replace(run.ctx, llm_mode=LLMMode(kyc), writer_mode=LLMMode(writer))
    done = Replay(Pipeline(ctx, state=start), None, [])
    with st.status(f"Replaying {case.target_account}...", expanded=True) as status:
        done.outcome = done.pipeline.run(
            review(case.target_account), on_node=live_step(done.log, status)
        )
        finish(done, status)
    return done


def live_step(log: list[dict], status) -> OnNode:
    def on_node(node: str, work: CaseWork, seconds: float) -> None:
        row = node_row(node, work, seconds)
        log.append(row)
        how = f" · _{row['How it ran']}_" if row["How it ran"] else ""
        status.write(f"✅ **{row['Step']}**: {row['Result']} · {seconds:.2f} s{how}")

    return on_node


def finish(done: Replay, status) -> None:
    seconds = sum(r["Seconds"] for r in done.log)
    if isinstance(done.outcome, Paused):
        row = {
            "Step": NODE_LABELS["admin_gate"],
            "Result": WAITING,
            "Seconds": 0.0,
            "How it ran": "LangGraph interrupt(): the case is held in memory until you decide.",
        }
        done.log.append(row)
        status.write(f"⏸️ **{row['Step']}**: {row['Result']}")
        status.update(label=f"Paused for the admin after {seconds:.1f} s", expanded=False)
    else:
        status.update(label=f"Replay done in {seconds:.1f} s", state="complete", expanded=False)


def node_row(node: str, work: CaseWork, seconds: float) -> dict:
    """One line of the live run: what the node found and how it ran (saved or new)."""
    result, how = "-", ""
    if node == "triage" and work.triage_out:
        result = f"{len(work.triage_out.fired_rules)} alert(s)"
        how = "runs: " + ", ".join(s.value for s in work.triage_out.routed_stages)
    elif node in STAGE_OF_NODE:
        stage = STAGE_OF_NODE[node]
        f = work.findings.get(FINDINGS_FIELD[stage])
        status = work.stage_status.get(stage.value)
        if f is None:
            result = NOT_RUN.get(status.value, status.value) if status else "-"
            how = work.errors.get(stage.value, "")
        elif stage == StageName.VISION:
            result = f"forgery {f.forgery_confidence:.2f} (never scored)"
        else:
            result = f"score {f.confidence:.2f}"
            how = next((s.finding for s in f.reasoning_trail if s.step == "How it ran"), "")
    elif node == "judge" and work.score:
        d = work.score.disposition.value
        result = f"{SAY_FULL.get(d, d)} at {work.score.confidence:.2f}"
        how = work.score.reasoning_trail[-1].finding if work.score.reasoning_trail else ""
    elif node == "admin_gate":
        decision = work.admin_decision
        result = f"admin chose {decision.upper()}" if decision else "no pause needed"
    elif node == "rulebook":
        if work.rulebook:
            ids = work.rulebook.output.retrieved_clause_ids
            result = ", ".join(ids) or "policy gap: no clause close enough"
        else:
            status = work.stage_status.get(StageName.RULEBOOK.value)
            result = NOT_RUN.get(status.value, status.value) if status else "-"
    elif node == "writer" and work.auditor:
        result = work.auditor.status.value
        how = work.auditor.trail[0].finding if work.auditor.trail else ""
    return {
        "Step": NODE_LABELS.get(node, node),
        "Result": result,
        "Seconds": round(seconds, 2),
        "How it ran": how,
    }


def paused_panel(done: Replay, key: str) -> None:
    paused = done.outcome
    ev = paused.evidence
    with st.container(border=True):
        st.markdown(
            f"### ⏸️ {paused.account_id} is waiting for your decision\n"
            "The ID card check thinks the card may be fake (score "
            f"**{ev['forgery_confidence']:.2f}**). Every other check is done: their findings "
            "are below."
        )
        for d in ev["field_discrepancies"]:
            st.markdown(f"- {d}")
        disposition = SAY_FULL.get(ev["disposition_if_approved"], ev["disposition_if_approved"])
        st.markdown(
            f"If you approve, the verdict stays **{disposition}** (score "
            f"{ev['score_if_approved']:.2f}): the ID card check never changes the score. If you "
            "block, the account is stopped."
        )
        a, b = st.columns(2)
        for col, decision, text in ((a, "approve", "✅ Approve"), (b, "block", "⛔ Block")):
            if col.button(text, width="stretch", key=f"{key}-{decision}"):
                done.log[:] = [r for r in done.log if r["Result"] != WAITING]
                with st.status("Finishing the case...", expanded=True) as status:
                    done.outcome = done.pipeline.resume(
                        paused, decision, on_node=live_step(done.log, status)
                    )
                    finish(done, status)
                st.rerun()


def report_view(
    r: InvestigationReport, expected: dict | None, rb: RulebookLookup, source: str
) -> None:
    banner(r, expected, source)
    st.markdown(scale(r.confidence), unsafe_allow_html=True)
    notes = [n for n in (r.forgery_notice,) if n]
    if r.evidence_incomplete:
        notes.append("Some evidence is missing: " + ", ".join(r.degraded_stages) + " failed.")
    if r.single_signal_capped:
        notes.append("Held at Monitor: one borderline signal alone did not pass the gate.")
    if r.policy_gap_identified:
        notes.append("Policy gap: no rulebook clause was close enough to cite.")
    for n in notes:
        st.info(n)

    st.subheader("Why this verdict")
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(
            contributions(r),
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "Adds": st.column_config.NumberColumn("Adds to the score", format="%.3f"),
            },
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Each check's score and weight. The verdict score is the weighted average of the "
            "checks that found something; a veto skips the average."
        )
        reasons = r.triage_findings.trigger_reasons if r.triage_findings else []
        if reasons:
            with st.expander(f"Why the case was opened ({len(reasons)} alerts)"):
                for reason in reasons:
                    st.markdown(f"- {reason}")
    with right:
        st.markdown("**The judge's steps**")
        steps(r.judge_trail, numbered=True)

    st.subheader("Evidence")
    present = [(t, *v) for t, v in EVIDENCE.items() if getattr(r, v[0]) is not None]
    for tab, (_, field, keys, asks) in zip(st.tabs([p[0] for p in present]), present, strict=True):
        with tab:
            st.caption(asks)
            evidence(getattr(r, field), keys)

    st.subheader("Report")
    if r.pipeline_status == PipelineStatus.PAUSED_PENDING_ADMIN_REVIEW:
        st.caption("The report is written after your decision.")
        return
    how = r.auditor_trail[0].finding if r.auditor_trail else ""
    st.caption(how)
    if how.endswith("template text."):
        escalated = r.str_narrative is not None or r.disposition.value == "SUSPICIOUS-ESCALATE"
        off = how.startswith("Model off") and source == "from the batch"
        st.info(
            "This is plain template text, not the AI report"
            + (": the batch runs with the writer off. " if off else ". ")
            + "Pick **Gemini AI, live** as the report writer and press **▶ Replay live** for "
            + ("the AI report and STR draft." if escalated else "the AI report.")
        )
    with st.container(border=True):
        st.markdown(f"**Summary.** {r.executive_summary}")
        st.markdown(f"**Conclusion.** {r.compliance_conclusion}")
        if r.rulebook_findings and r.rulebook_findings.retrieved_clause_ids:
            cited = "; ".join(
                f"{cid} {clause_title(rb, cid)}" for cid in r.rulebook_findings.retrieved_clause_ids
            )
            st.markdown(f"**Rulebook clauses.** {cited}")
    if r.str_narrative:
        with st.container(border=True):
            st.markdown("**📄 Suspicious Transaction Report (draft)**")
            st.write(r.str_narrative)
    with st.expander("Full narrative"):
        st.write(r.investigation_narrative or "No narrative yet: replay the case with a writer.")
    if r.auditor_trail:
        with st.expander("How the report was written"):
            steps(r.auditor_trail)


def banner(r: InvestigationReport, expected: dict | None, source: str) -> None:
    colour, icon, action = VERDICT.get(r.disposition.value, ("#5f6368", "•", ""))
    said = SAY.get(SHORT[r.disposition.value], r.disposition.value).split(" ", 1)[-1]
    key = ""
    if expected:
        ok = SHORT[expected["expected"]] == SHORT[r.disposition.value]
        right = SAY_FULL.get(expected["expected"], expected["expected"])
        key = (
            f"Right answer: {right} · {'✓ the system got it right' if ok else '✗ wrong'} · "
            f"{html.escape(expected['group'])}"
        )
    st.markdown(
        f"""<div style="border-left:6px solid {colour};background:{colour}1a;
        padding:14px 18px;border-radius:10px;margin:8px 0">
        <div style="font-size:1.7rem;font-weight:700;color:{colour}">{icon} {said}</div>
        <div style="font-size:1.05rem">{action} · score <b>{r.confidence:.2f}</b></div>
        <div style="opacity:.75;font-size:.85rem">{html.escape(r.target_account)} ·
        {html.escape(r.case_id)} · {html.escape(source)}<br>{key}</div></div>""",
        unsafe_allow_html=True,
    )


def scale(score: float) -> str:
    """The score on the judge's bands."""
    c, e = CLEAR_BELOW * 100, ESCALATE_FROM * 100
    at = min(max(score, 0.0), 1.0) * 100
    return f"""<div style="position:relative;margin:10px 0 2px">
      <div style="display:flex;height:12px;border-radius:6px;overflow:hidden">
        <div style="width:{c}%;background:#1e8e3e66"></div>
        <div style="width:{e - c}%;background:#e3740066"></div>
        <div style="flex:1;background:#d9302566"></div></div>
      <div style="position:absolute;left:calc({at}% - 3px);top:-4px;width:6px;height:20px;
        border-radius:3px;background:currentColor"></div></div>
    <div style="display:flex;font-size:.75rem;opacity:.7;margin-bottom:8px">
      <span style="width:{c}%">Clear</span><span style="width:{e - c}%">Monitor from
      {CLEAR_BELOW:g}</span><span>Escalate from {ESCALATE_FROM:g}</span></div>"""


def contributions(r: InvestigationReport) -> pd.DataFrame:
    """Each weighted check: what it found and how much it adds to the weighted average."""
    scores = {}
    for stage, (_, field) in STAGES.items():
        f = getattr(r, field)
        scores[stage] = f.confidence if f is not None else None
    used = sum(STAGE_WEIGHTS[s] for s, c in scores.items() if c)
    veto = any(
        (scores[s] or 0) >= AUTO_VETO_THRESHOLD for s in (StageName.ADVERSE_MEDIA, StageName.GRAPH)
    )
    out = []
    for stage, (label, _) in STAGES.items():
        c = scores[stage]
        status = r.stage_status.get(stage.value)
        state = {"NOT_ROUTED": "not needed", "FAILED": "failed", "TIMED_OUT": "timed out"}.get(
            status.value if status else "", "-"
        )
        if c is not None:
            state = "found something" if c > 0 else "found nothing"
        out.append(
            {
                "Check": label,
                "Result": state,
                "Score": c,
                "Weight": STAGE_WEIGHTS[stage],
                # a veto skips the average, so no check "adds" to it
                "Adds": None if veto else round(STAGE_WEIGHTS[stage] * c / used, 4) if c else 0.0,
            }
        )
    return pd.DataFrame(out)


def evidence(f, keys: tuple[str, ...]) -> None:
    score = getattr(f, "confidence", getattr(f, "forgery_confidence", None))
    if score is not None:
        never = " (never counts toward the verdict)" if hasattr(f, "forgery_confidence") else ""
        st.markdown(f"Score **{score:g}**{never}")
    found = False
    for k in keys:
        values = getattr(f, k, None) or []
        if values:
            found = True
            st.markdown(f"**{k.replace('_', ' ').capitalize()}**")
            for v in values[:8]:
                st.markdown(f"- {v}")
    if not found:
        st.markdown("Nothing found.")
    with st.expander("Step-by-step reasoning"):
        steps(f.reasoning_trail)


def steps(trail: list[ReasoningStep], numbered: bool = False) -> None:
    for i, s in enumerate(trail, 1):
        head = f"{i}. " if numbered else ""
        st.markdown(f"{head}**{s.step}**: {s.finding}  \n→ _{s.impact}_")


def clause_title(rb: RulebookLookup, cid: str) -> str:
    try:
        return rb.clause(cid).title
    except KeyError:
        return ""


# --------------------------------------------------------------------------------------
# Test with new accounts
# --------------------------------------------------------------------------------------


def planted(pack: Pack) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Account": a["account_id"],
                "What it is": a["group"],
                "Right answer": SAY_FULL.get(a["expected"], a["expected"]),
                "What was planted": a.get("what", ""),
            }
            for a in pack.answers["accounts"]
        ]
    )


def new_data() -> None:
    st.title("🧪 Test with new accounts")
    purpose(
        "Make new bank accounts with a pattern you choose, set the right answer before the "
        "run, then see whether the system gets it right. New accounts sit beside the bank's "
        "own data and never change it."
    )
    quick, build = st.tabs(["⚡ Quick batch: ready patterns", "🛠️ Build your own accounts"])
    with quick:
        quick_batch()
    with build:
        build_your_own()
    current_batch("made")
    st.divider()
    your_batches()
    current_batch("list")


def set_current(name: str, where: str) -> None:
    """Shows this batch's panel below the tabs ("made") or below Your batches ("list")."""
    st.session_state["current_pack"], st.session_state["current_from"] = name, where
    st.rerun()


def open_case(pack_name: str, account_id: str) -> None:
    st.session_state["next_dataset"] = pack_name
    st.session_state[f"acct-{pack_name}"] = account_id
    st.switch_page(PAGES["case"])


def open_results(pack_name: str) -> None:
    st.session_state["next_dataset"] = pack_name
    st.switch_page(PAGES["results"])


def current_batch(where: str) -> None:
    """The batch just made or picked: a Run button, then each account's verdict below it."""
    name = st.session_state.get("current_pack")
    packs = st.session_state["packs"]
    if st.session_state.get("current_from") != where or name not in packs:
        return
    pack = packs[name]
    run = pack_run(pack)
    n = len(pack.account_ids)
    with st.container(border=True):
        if run is None:
            st.markdown(f"**{pack.title}** is made: {n} account{'s' * (n > 1)}, not run yet.")
            st.dataframe(planted(pack), hide_index=True, width="stretch")
            full_run_note(n)
            left, right = st.columns(2)
            if left.button(
                "▶ Run through the pipeline", type="primary", key=f"run-{where}", width="stretch"
            ):
                if run_pack_now(pack) is not None:
                    st.rerun()
            if right.button("Close", key=f"close-{where}", width="stretch"):
                del st.session_state["current_pack"]
                st.rerun()
            return
        table = run.table
        when = f" · saved run of {run.saved_at}" if run.saved_at else ""
        st.success(
            f"**{pack.title}**{when}: {sum(t.right for t in table)} of {len(table)} right. "
            "Open an account to see why, the evidence and its report."
        )
        for t in table:
            trail = run.result.latest[t.account_id].auditor_trail
            ai = bool(trail) and trail[0].finding.startswith(("New answer", "Saved answer"))
            a, b, c, d = st.columns([1.4, 3.2, 2.6, 1.2], vertical_alignment="center")
            a.markdown(f"**{t.account_id}**")
            b.markdown(f"{t.group}  \n{'📝 Gemini report' if ai else '📄 template text'}")
            c.markdown(
                f"Right answer {SAY.get(t.expected, t.expected)}  \nSystem said "
                f"{SAY.get(t.got, t.got)} {'✓' if t.right else '✗'}"
            )
            if d.button("🔎 Open", key=f"open-{where}-{t.account_id}", width="stretch"):
                open_case(name, t.account_id)
        left, right = st.columns(2)
        if left.button("📊 All results for this batch", key=f"all-{where}", width="stretch"):
            open_results(name)
        if right.button("Close", key=f"close-{where}", width="stretch"):
            del st.session_state["current_pack"]
            st.rerun()


def quick_batch() -> None:
    st.markdown(
        "Pick a pattern. Every detail (names, amounts, dates) is drawn at random, but the "
        "same **variation number** always makes the same accounts."
    )
    options = ["mix", *KINDS]
    a, b, c = st.columns([3, 1, 1])
    pattern = a.selectbox(
        "Pattern",
        options,
        format_func=lambda k: (
            f"One of each pattern ({MIX_SIZE} accounts)" if k == "mix" else KINDS[k].label
        ),
    )
    ring = pattern == "mule_ring"
    count = b.number_input(
        "Ring members" if ring else "How many",
        min_value=2 if ring else 1,
        max_value=10,
        value=3 if ring else 1,
        disabled=pattern == "mix",
    )
    seed = c.number_input("Variation", 1, 9_999, 1, help="Same number, same accounts.")
    if pattern == "mix":
        st.caption(
            "Two ordinary customers, a declared property sale, and one of every bad pattern."
        )
        accounts = MIX_SIZE
    else:
        k = KINDS[pattern]
        st.caption(f"**Plants:** {k.what}. **Right answer:** {SAY_FULL[k.expected]}.")
        accounts = int(count)
    full_run_note(accounts)
    if st.button("➕ Create batch", type="primary", key="quick-go"):
        title = (
            mix_title(int(seed))
            if pattern == "mix"
            else f"{KINDS[pattern].label} ({accounts} account{'s' * (accounts > 1)}), "
            f"variation {int(seed)}"
        )
        made = next(
            (
                p
                for p in st.session_state["packs"].values()
                if p.title == title and p.answers.get("how") == "sample"
            ),
            None,
        )
        if made is not None:
            st.toast("Already made: showing it (same pattern and variation, same accounts).")
            set_current(made.name, "made")
        folder = free_folder(title)
        try:
            pack = make_pack(
                pattern, accounts, int(seed), folder.name, folder.parent, bank=committed_bank()
            )
        except (FileExistsError, ValueError) as exc:
            st.error(str(exc))
            return
        set_current(pack.name, "made")


def build_your_own() -> None:
    st.markdown(
        "Set each account's details yourself, and say what the right answer should be. Add "
        "as many accounts as you like, then check them together. Anything you leave empty is "
        "drawn at random."
    )
    draft: list[dict] = st.session_state.setdefault("draft", [])
    pattern = st.selectbox(
        "1 · Start from a pattern",
        list(KINDS),
        format_func=lambda k: KINDS[k].label,
        key="build-pattern",
    )
    k = KINDS[pattern]
    st.caption(f"**Plants:** {k.what}. **Usual right answer:** {SAY_FULL[k.expected]}.")

    st.markdown("**2 · The customer**")
    a, b, c = st.columns(3)
    label = a.text_input("Label in the tables", value=k.label, key=f"b-label-{pattern}")
    if pattern == "sanctions":
        people = listed_people(committed_bank())
        listed = b.selectbox("Person on the list", people, key="b-listed")
        name = None
    else:
        listed = None
        name = b.text_input("Name", placeholder="random", key=f"b-name-{pattern}") or None
    occupation = c.text_input("Job", value=k.occupations[0], key=f"b-job-{pattern}")
    low, high = k.income_lakh
    income = st.slider(
        "Declared income (₹ lakh a year)",
        1,
        100,
        (low + high) // 2,
        key=f"b-income-{pattern}",
    )

    settings: dict[str, float] = {}
    count = 1
    if k.knobs or pattern == "mule_ring":
        st.markdown("**3 · The money**")
        cols = st.columns(len(k.knobs) + (pattern == "mule_ring"))
        for col, knob in zip(cols, k.knobs, strict=False):
            mid = (knob.draw[0] + knob.draw[1]) / 2
            whole = knob.step >= 1
            settings[knob.key] = col.number_input(
                knob.label,
                min_value=int(knob.low) if whole else float(knob.low),
                max_value=int(knob.high) if whole else float(knob.high),
                value=int(round(mid)) if whole else round(mid, 2),
                step=int(knob.step) if whole else float(knob.step),
                help=knob.help or None,
                key=f"b-{pattern}-{knob.key}",
            )
        if pattern == "mule_ring":
            count = cols[-1].number_input("Ring members", 2, 10, 3, key="b-ring")

    st.markdown("**4 · The KYC profile the bank holds**")
    filled = profile_text(
        pattern,
        occupation or k.occupations[0],
        int(income) * 100_000,
        name=listed if pattern == "sanctions" else (None if pattern == "mule_ring" else name),
        settings=settings,
    )
    profile = st.text_area(
        "Profile",
        value=filled,
        height=140,
        key=f"b-profile-{pattern}-{zlib.crc32(filled.encode())}",
        label_visibility="collapsed",
        help="Written from the job, income and money above. Edit it freely; changing those "
        "fields writes it again.",
    )
    st.markdown("**5 · The right answer**")
    expected = st.radio(
        "Right answer",
        VERDICTS,
        index=VERDICTS.index(k.expected),
        horizontal=True,
        format_func=SAY_FULL.get,
        key=f"b-expected-{pattern}",
        label_visibility="collapsed",
    )
    if st.button("➕ Add to my batch", key="b-add"):
        draft.append(
            asdict(
                AccountSpec(
                    pattern,
                    count=int(count),
                    label=label or k.label,
                    name=name,
                    occupation=occupation or None,
                    income=int(income) * 100_000,
                    profile=profile.strip() or filled,
                    expected=expected,
                    settings=settings,
                    listed_name=listed,
                )
            )
        )
        st.toast(f"Added “{label}”")

    if not draft:
        st.info("Your batch is empty. Set up an account above and press **Add to my batch**.")
        return
    st.markdown(f"**Your batch: {sum(d['count'] for d in draft)} account(s)**")
    for i, d in enumerate(draft):
        row, remove = st.columns([6, 1], vertical_alignment="center")
        size = f" ×{d['count']}" if d["count"] > 1 else ""
        row.markdown(
            f"{i + 1}. **{d['label']}**{size} · {KINDS[d['pattern']].label} · income "
            f"₹{d['income'] / 100_000:g} lakh · right answer {SAY_FULL[d['expected']]}"
        )
        if remove.button("Remove", key=f"b-remove-{i}"):
            draft.pop(i)
            st.rerun()
    title = st.text_input("Name this batch", value="My test accounts", key="b-title")
    full_run_note(sum(d["count"] for d in draft))
    if st.button("➕ Create my batch", type="primary", key="b-go"):
        try:
            pack = make_custom_pack(
                [AccountSpec(**d) for d in draft],
                title or "My test accounts",
                bank=committed_bank(),
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state["draft"] = []
        set_current(pack.name, "made")


def your_batches() -> None:
    packs: dict[str, Pack] = st.session_state["packs"]
    st.subheader("Your batches")
    if not packs:
        st.caption("None yet. Batches you create appear here.")
        return
    st.caption(
        "Click a batch. A batch that has run opens its saved results: nothing is run again."
    )
    names = list(packs)
    runs = {n: pack_run(p) for n, p in packs.items()}
    picked = st.dataframe(
        pd.DataFrame(
            [
                {
                    "Batch": p.title,
                    "Accounts": len(p.account_ids),
                    "Status": (
                        "not run yet"
                        if runs[n] is None
                        else f"run {runs[n].saved_at or 'in this session'}"
                    ),
                    "Right": (
                        "-"
                        if runs[n] is None
                        else f"{sum(r.right for r in runs[n].table)} of {len(p.account_ids)}"
                    ),
                    "Made": time.strftime("%d %b %H:%M", time.localtime(p.path.stat().st_mtime)),
                }
                for n, p in packs.items()
            ]
        ),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key=f"batches-table-{st.session_state.get('batches_v', 0)}",
    )
    if picked.selection.rows:
        name = names[picked.selection.rows[0]]
        left, mid, right = st.columns(3)
        if runs[name] is None:
            if left.button("▶ Run through the pipeline", type="primary", width="stretch"):
                if run_pack_now(packs[name]) is not None:
                    set_current(name, "list")
        else:
            if left.button("Show here", type="primary", width="stretch"):
                set_current(name, "list")
            if mid.button("📊 See its results", width="stretch"):
                open_results(name)
            if right.button("🔎 Open its accounts", width="stretch"):
                open_case(name, packs[name].account_ids[0])
    with st.expander("Delete a batch"):
        name = st.selectbox(
            "Batch", list(packs), format_func=lambda n: packs[n].title, key="del-pick"
        )
        st.caption(
            "Deletes its accounts, transactions, ID images and their OCR text, the saved Llama "
            "and Gemini answers, its answer key and its saved results."
        )
        sure = st.checkbox("Yes, delete it and all its data", key="del-sure")
        if st.button("Delete", disabled=not sure, key="del-go"):
            title = packs[name].title
            delete_pack(packs[name])
            st.toast(f"Deleted “{title}” and its data")
            st.rerun()


def delete_pack(pack: Pack) -> None:
    """Removes the pack folder, the OCR text of its ID images and every copy held in memory."""
    for image in pack.documents_dir.rglob("*.png"):
        digest = hashlib.sha1(image.read_bytes()).hexdigest()
        (vision_config.OCR_CACHE_DIR / f"{digest}.txt").unlink(missing_ok=True)
    shutil.rmtree(pack.path)
    for key in [k for k in pack_runs() if k[0] == pack.name]:
        del pack_runs()[key]
    for key in [k for k in known_packs() if k[0] == pack.path]:
        del known_packs()[key]
    if st.session_state.get("current_pack") == pack.name:
        del st.session_state["current_pack"]
    st.session_state["batches_v"] = st.session_state.get("batches_v", 0) + 1

main()
