"""The Auditor agent: write the report around a verdict that is already decided.

    AuditorAgent(client).run(brief) -> AuditorResult

1. Tools turn the verdict and every agent's result into facts with ids.
2. The model picks an outline: which facts and which clauses go into the report, in order,
   and whether one agent's model answer should be checked again. Python drops unknown ids and
   renders the outline lines from the facts, so every number in the outline is Python's.
3. The model writes the text from the outline.
4. A Python check: every clause id and every number in the text must be in the outline. One
   rewrite is asked for; after that the template text is used.

A CLEAR case with no finding gets the template text and no model call. An STR draft exists
only for SUSPICIOUS-ESCALATE. The Auditor never scores and never changes the verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from src.agents.auditor import config
from src.agents.base import Agent, AgentRun, AgentSpec, known_ids
from src.agents.rag.agent import strip_citations, validate_citations
from src.llm import Role
from src.schemas import Disposition, StageStatus


class ReportText(BaseModel):
    executive_summary: str = Field(description="At most 3 sentences.")
    investigation_narrative: str = Field(description="At most 180 words.")
    compliance_conclusion: str = Field(description="At most 3 sentences, citing outline clauses.")
    str_narrative: str | None = Field(
        default=None, description="Suspicious Transaction Report draft; ESCALATE only."
    )


class OutlinePoint(BaseModel):
    fact_ids: list[str] = Field(description="Fact ids this point reports, e.g. F3.")
    clause_ids: list[str] = Field(default_factory=list, description="Allowed clause ids, or [].")


class OutlinePick(BaseModel):
    points: list[OutlinePoint]
    recheck: Literal["NONE", "KYC_CLAIMS", "SAME_PERSON"] = "NONE"
    recheck_reason: str = ""


@dataclass
class CaseBrief:
    """Everything the Auditor may see. Built by the orchestrator from finished agents."""

    case_id: str
    account_id: str
    disposition: Disposition
    confidence: float
    scoring_steps: list[str]
    stage_status: dict[str, str]
    #: agent or stage name -> its result lines (numbers already computed by Python).
    findings: dict[str, list[str]]
    #: citable clause id -> (title, the quote the RAG agent found, or "").
    clauses: dict[str, tuple[str, str]] = field(default_factory=dict)
    policy_gap: bool = False
    admin_note: str | None = None
    graph_label: str | None = None
    #: Re-checks that are possible on this case (an agent's model answer exists).
    rechecks_possible: tuple[str, ...] = ()

    @property
    def has_finding(self) -> bool:
        return (
            self.disposition != Disposition.CLEAR
            or bool(self.clauses)
            or self.policy_gap
            or bool(self.admin_note)
        )


@dataclass
class AuditorResult:
    draft: ReportText
    status: StageStatus
    outline: list[str]
    removed_citations: list[str]
    recheck: str | None
    used_template: bool
    run: AgentRun


OUTLINE_PROMPT = """Plan the report for one bank account review. The verdict is fixed and \
computed by deterministic code: {verdict}.

Pick the points the report must make, most important first. Each point names the fact ids \
it reports and, where a clause governs it, clause ids from ALLOWED CLAUSES only.
- Use only fact ids and clause ids listed below.
- Cover every agent that found something. Leave out facts that say nothing was found, \
unless nothing was found at all.
- The "Scoring" facts say how the verdict was computed. Use at most two of them, and only \
the ones a reader needs (a veto, a gate, a floor, explained cash).
- At most {max_points} points.
- recheck: {recheck_rule}

FACTS:
{facts}

ALLOWED CLAUSES:
{clauses}
"""

TEXT_PROMPT = """Write the investigation report for one bank account review from the \
OUTLINE below.

The verdict is fixed and was computed by deterministic code. Do not change, question or \
recompute it. Write as a recommendation to the lead analyst, never as a final ruling.

Hard rules:
- Use only what the OUTLINE says. Add no fact of your own.
- Every number you write must be copied exactly as it is written in the OUTLINE. Do not \
round, convert, add up or count anything. If you are unsure of a number, leave it out.
- Cite clause ids only if they appear in the OUTLINE, written exactly as given.

Fields:
- executive_summary: at most 3 sentences: the verdict and the main reason.
- investigation_narrative: at most 180 words, in plain words, following the outline's order. \
Say what was found and why it matters. Do not walk through the scoring formula.
- compliance_conclusion: at most 3 sentences, with the clause ids from the outline.
- str_narrative: {str_rule}
{notes}
OUTLINE:
{outline}
"""


class AuditorAgent(Agent):
    spec = AgentSpec(
        name="auditor",
        role=Role.AUDITOR,
        prompt_version=config.PROMPT_VERSION,
        required_tools=("read_verdict", "read_agent_results"),
        optional_tools=("request_recheck",),
    )

    def run(self, brief: CaseBrief, allow_recheck: bool = True) -> AuditorResult:
        run = self.start(brief.account_id)
        verdict = f"{brief.disposition.value} at confidence {brief.confidence:.4f}"
        run.facts.add("Verdict", f"Verdict: {verdict}.")
        for step in brief.scoring_steps:
            run.facts.add("Verdict", _citable(f"Scoring: {step}", brief))
        self.tool(run, "read_verdict")
        for name, lines in brief.findings.items():
            for line in lines[: config.LINES_PER_AGENT]:
                run.facts.add(name, _citable(f"{name}: {line}", brief))
        if brief.admin_note:
            run.facts.add("Admin", _citable(f"Admin: {brief.admin_note}", brief))
        if brief.policy_gap:
            run.facts.add("Policy", "Policy gap: there is evidence but no rulebook clause fits.")
        self.tool(run, "read_agent_results")

        if not brief.has_finding or self.client.off:
            outline = self._python_outline(run, brief)
            why = (
                "Models are off: template text."
                if brief.has_finding
                else "CLEAR with no finding: template text, no model call."
            )
            run.step("How it ran", why, "The verdict was fixed before writing.")
            return self._finish(run, brief, template(brief), outline, [], None, True)

        # 1. Outline
        possible = brief.rechecks_possible if allow_recheck else ()
        pick = self.ask(
            run,
            OUTLINE_PROMPT.format(
                verdict=verdict,
                max_points=config.MAX_POINTS,
                recheck_rule=_recheck_rule(possible),
                facts=run.facts.render(),
                clauses=_clause_lines(brief) or "- none",
            ),
            OutlinePick,
            "Outline",
            queue_on_failure=False,
        )
        outline, note = self._outline(run, brief, pick)
        recheck = None
        if pick is not None and pick.recheck != "NONE":
            if pick.recheck in possible:
                recheck = pick.recheck
                self.tool(run, "request_recheck")
                run.queue(f"Re-check {recheck}", pick.recheck_reason.strip()[:200] or "asked")
                note += f" Re-check asked: {recheck}."
            else:
                note += f" Re-check {pick.recheck} is not possible on this case: ignored."
        run.step("Outline", note, "Every number in the outline comes from a Python fact.")

        # 2. Text, checked against the outline
        escalate = brief.disposition == Disposition.SUSPICIOUS_ESCALATE
        prompt = TEXT_PROMPT.format(
            str_rule=(
                "a short Suspicious Transaction Report draft (what, when, how much, why "
                "suspicious), from the outline only."
                if escalate
                else "null (only an escalated case gets an STR draft)."
            ),
            notes=_notes(brief),
            outline="\n".join(outline),
        )
        draft, problems = None, ""
        for attempt in range(config.MAX_REWRITES + 1):
            ask = prompt if attempt == 0 else prompt + f"\nYour last text broke a rule: {problems}"
            draft = self.ask(run, ask, ReportText, "Report text", queue_on_failure=False)
            if draft is None:
                break
            problems = check_text(draft, outline, list(brief.clauses))
            if not problems:
                break
            run.step("Outline check", problems, "The text must stay inside the outline.")
        used_template = draft is None or bool(problems)
        if used_template:
            run.step(
                "Report text",
                "No text passed the outline check: template text used.",
                "A report is always produced; the verdict does not depend on it.",
            )
            draft = template(brief)
        return self._finish(run, brief, draft, outline, [], recheck, used_template)

    # ----------------------------------------------------------------------------------

    def _outline(
        self, run: AgentRun, brief: CaseBrief, pick: OutlinePick | None
    ) -> tuple[list[str], str]:
        """The outline lines, rendered by Python from the facts the model picked."""
        if pick is None:
            return self._python_outline(run, brief), "No model outline: the Python outline."
        lines = [run.facts.get("F1").text]
        used: set[str] = set()
        dropped: list[str] = []
        for point in pick.points[: config.MAX_POINTS]:
            facts, unknown = known_ids(point.fact_ids, run.facts)
            clauses = [c.strip().upper() for c in point.clause_ids]
            dropped += unknown + [c for c in clauses if c not in brief.clauses]
            clauses = [c for c in clauses if c in brief.clauses]
            facts = [f for f in facts if f != "F1" and f not in used]
            used.update(facts)
            if not facts:
                continue
            text = " ".join(run.facts.get(f).text for f in facts)
            cite = "".join(f" [{c}: {brief.clauses[c][0]}]" for c in clauses)
            lines.append(f"- {text}{cite}")
        if len(lines) == 1:
            return (
                self._python_outline(run, brief),
                "The model's outline held no usable point: the Python outline.",
            )
        note = f"{len(lines) - 1} point(s) from {len(used)} fact(s)."
        if dropped:
            note += f" Unknown ids dropped: {', '.join(dict.fromkeys(dropped))}."
        return lines, note

    def _python_outline(self, run: AgentRun, brief: CaseBrief) -> list[str]:
        """The outline without a model: the verdict, then every fact, then the clauses."""
        lines = [run.facts.get("F1").text]
        lines += [f"- {f.text}" for f in run.facts if f.fact_id != "F1"]
        lines += [f"- Clause [{cid}: {title}]" for cid, (title, _) in brief.clauses.items()]
        return lines

    def _finish(
        self,
        run: AgentRun,
        brief: CaseBrief,
        draft: ReportText,
        outline: list[str],
        removed: list[str],
        recheck: str | None,
        used_template: bool,
    ) -> AuditorResult:
        draft, removed = _enforce(draft, brief)
        run.step(
            "Citation check",
            f"Allowed: {', '.join(brief.clauses) or 'none'}. Removed: "
            f"{', '.join(removed) or 'none'}.",
            "Only clauses the RAG agent retrieved may be cited.",
        )
        status = StageStatus.FAILED if run.model_failed and used_template else StageStatus.RAN
        return AuditorResult(draft, status, outline, removed, recheck, used_template, run)


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------

_ID_TOKEN = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)*-\d+\b")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> set[str]:
    """Every number in `text`, in one normal form. Ids such as ACC-0044 are not numbers."""
    found = set()
    for raw in _NUMBER.findall(_ID_TOKEN.sub(" ", text)):
        digits = raw.replace(",", "").rstrip(".")
        try:
            value = float(digits)
        except ValueError:
            continue
        found.add(f"{value:.6f}".rstrip("0").rstrip("."))
    return found


def check_text(draft: ReportText, outline: list[str], allowed_clauses: list[str]) -> str:
    """What breaks the outline, or "" when nothing does."""
    text = " ".join(
        part
        for part in (
            draft.executive_summary,
            draft.investigation_narrative,
            draft.compliance_conclusion,
            draft.str_narrative or "",
        )
    )
    source = "\n".join(outline)
    problems = []
    extra = sorted(numbers_in(text) - numbers_in(source), key=float)
    if extra:
        problems.append(f"numbers not in the outline: {', '.join(extra)}")
    in_outline, _ = validate_citations(source, allowed_clauses)
    _, bad = validate_citations(text, in_outline)
    if bad:
        problems.append(f"clause ids not in the outline: {', '.join(bad)}")
    return "; ".join(problems)


def _enforce(draft: ReportText, brief: CaseBrief) -> tuple[ReportText, list[str]]:
    """Strip clause ids the RAG agent did not return; STR only on an escalation."""
    removed: list[str] = []
    fields = {}
    for name in ReportText.model_fields:
        text = getattr(draft, name)
        if text is None:
            fields[name] = None
            continue
        _, invalid = validate_citations(text, brief.clauses)
        removed += [c for c in invalid if c not in removed]
        fields[name] = strip_citations(text, invalid)
    if brief.disposition != Disposition.SUSPICIOUS_ESCALATE:
        fields["str_narrative"] = None
    elif not fields["str_narrative"]:
        fields["str_narrative"] = template(brief).str_narrative
    return ReportText(**fields), removed


def template(brief: CaseBrief) -> ReportText:
    """The deterministic report text used when no model text is available."""
    signals = [
        f"{name}: {'; '.join(lines[:2])}." for name, lines in brief.findings.items() if lines
    ]
    verdict = f"{brief.disposition.value} at confidence {brief.confidence:.2f}"
    cites = ", ".join(brief.clauses)
    conclusion = (
        f"Recommended disposition for the lead analyst: {verdict}."
        + (f" Relevant clauses: {cites}." if cites else "")
        + (
            " The evidence is not covered by an existing policy clause (policy gap)."
            if brief.policy_gap
            else ""
        )
    )
    return ReportText(
        executive_summary=f"Account {brief.account_id}: {verdict}.",
        investigation_narrative=" ".join(signals) or "No agent found an adverse signal.",
        compliance_conclusion=conclusion,
        str_narrative=(
            f"Draft STR for account {brief.account_id}: {verdict}. Grounds: "
            + ("; ".join(signals) or "see the agent findings")
            + "."
            if brief.disposition == Disposition.SUSPICIOUS_ESCALATE
            else None
        ),
    )


def _citable(line: str, brief: CaseBrief) -> str:
    """A fact line without the clause ids the RAG agent did not return.

    The Python checks name the rule they apply. Left in a fact, such an id would reach the
    outline and the model would copy a clause it may not cite."""
    _, uncited = validate_citations(line, brief.clauses)
    return strip_citations(line, uncited) if uncited else line


def _clause_lines(brief: CaseBrief) -> str:
    return "\n".join(
        f"- {cid}: {title}" + (f' ("{quote}")' if quote else "")
        for cid, (title, quote) in brief.clauses.items()
    )


def _recheck_rule(possible: tuple[str, ...]) -> str:
    if not possible:
        return "always NONE for this case."
    return (
        "NONE in almost every case. Ask for one re-check only when a model-read result looks "
        f"wrong against the other facts. Allowed here: {', '.join(possible)}."
    )


def _notes(brief: CaseBrief) -> str:
    notes = []
    if brief.policy_gap:
        notes.append("- Say plainly that there is evidence but no rulebook clause fits it.")
    failed = [s for s, status in brief.stage_status.items() if status in ("FAILED", "TIMED_OUT")]
    if failed:
        notes.append(f"- Say the evidence is incomplete: {', '.join(failed)} did not finish.")
    return ("\n".join(notes) + "\n") if notes else ""
