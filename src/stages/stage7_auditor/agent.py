"""Stage 7's entry point: write the report around a verdict that is already decided.

    run(brief, mode) -> AuditorResult

The verdict arrives fixed; the writer only explains it. Clause ids not from Stage 6 are
removed and noted. STR draft only for ESCALATE. Model off or failed -> template text.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, Field

from src.llm import LLMStageFailure, get_stage_config, get_structured_response
from src.llm.cache import LLMMode, ResponseCache, fingerprint
from src.schemas import Disposition, ReasoningStep, StageStatus
from src.stages.stage6_rulebook.agent import strip_citations, validate_citations
from src.stages.stage7_auditor import config


STAGE = "Auditor"
_last_call = [0.0]


class AuditorDraft(BaseModel):
    executive_summary: str = Field(description="At most 3 sentences.")
    investigation_narrative: str = Field(description="At most 200 words, every stage's story.")
    compliance_conclusion: str = Field(description="At most 3 sentences, citing allowed ids.")
    str_narrative: str | None = Field(
        default=None, description="Suspicious Transaction Report draft; ESCALATE only."
    )


@dataclass
class CaseBrief:
    """Everything the writer may see. Built by the orchestrator from finished stages."""

    case_id: str
    account_id: str
    disposition: Disposition
    confidence: float
    scoring_steps: list[str]
    stage_status: dict[str, str]
    stage_findings: dict[str, list[str]]
    clauses: dict[str, str]  # retrieved id -> title
    policy_gap: bool = False
    admin_note: str | None = None
    stage_errors: dict[str, str] = field(default_factory=dict)


@dataclass
class AuditorResult:
    draft: AuditorDraft
    status: StageStatus
    trail: list[ReasoningStep]
    removed_citations: list[str]


def render(brief: CaseBrief) -> str:
    findings = "\n".join(
        f"- {stage} ({brief.stage_status.get(stage, '?')}): " + " | ".join(lines)
        for stage, lines in brief.stage_findings.items()
    )
    not_run = [s for s, st in brief.stage_status.items() if st != "RAN"]
    clauses = "\n".join(f"- {cid}: {title}" for cid, title in brief.clauses.items()) or "- none"
    escalate = brief.disposition == Disposition.SUSPICIOUS_ESCALATE
    str_rule = (
        "a short Suspicious Transaction Report draft (what, when, how much, why suspicious)"
        if escalate
        else "null (only an escalated case gets an STR draft)"
    )
    extra = []
    if brief.policy_gap:
        extra.append("- POLICY GAP: there is real evidence but no rulebook clause fits. Say so.")
    if brief.admin_note:
        extra.append(f"- ADMIN: {brief.admin_note}")
    if not_run:
        extra.append(
            f"- Stages not run or failed: {', '.join(not_run)}. Say the evidence is incomplete "
            "where it matters."
        )
    notes = "\n".join(extra)
    return f"""Write the investigation report for one bank account review.

FIXED VERDICT (computed by deterministic code; do not change, question or recompute it):
disposition {brief.disposition.value}, confidence {brief.confidence:.4f}.
How the score was reached: {" ".join(brief.scoring_steps)}

Write as a recommendation to the lead analyst, never as a final ruling.
- executive_summary: at most 3 sentences.
- investigation_narrative: at most 200 words; cover every stage below in plain words, \
with the real numbers given.
- compliance_conclusion: at most 3 sentences. Cite clause ids ONLY from ALLOWED CLAUSES, \
written exactly as given. Never cite any other id.
- str_narrative: {str_rule}.
{notes}

CASE {brief.case_id}, account {brief.account_id}

STAGE FINDINGS:
{findings}

ALLOWED CLAUSES:
{clauses}
"""


def run(
    brief: CaseBrief,
    mode: LLMMode,
    cache: ResponseCache | None = None,
    model_call=None,
) -> AuditorResult:
    prompt = render(brief)
    model_id = get_stage_config(STAGE).model_id or "unset"
    key = fingerprint(model_id, config.PROMPT_VERSION, prompt)
    cache = cache or ResponseCache("auditor")
    name = f"{brief.account_id}-{key[:12]}"

    draft, how, status = None, "", StageStatus.RAN
    if mode == LLMMode.OFF and model_call is None:
        how = "Model off for this run: template text."
    else:
        saved = cache.load(name, key)
        if saved is not None:
            draft = AuditorDraft.model_validate(saved["draft"])
            how = f"Saved answer from {model_id}, {config.PROMPT_VERSION}."
        elif mode == LLMMode.CACHED and model_call is None:
            how = "No saved writer answer (cached mode): template text."
            status = StageStatus.FAILED
        else:
            try:
                started = _throttle() if model_call is None else time.perf_counter()
                draft = (model_call or get_structured_response)(prompt, AuditorDraft, STAGE)
                cache.save(
                    name,
                    {
                        "key": key,
                        "case_id": brief.case_id,
                        "model": model_id,
                        "prompt_version": config.PROMPT_VERSION,
                        "seconds": round(time.perf_counter() - started, 1),
                        "made_at": datetime.now().isoformat(timespec="seconds"),
                        "draft": draft.model_dump(),
                    },
                )
                how = f"New answer from {model_id}, {config.PROMPT_VERSION}."
            except (LLMStageFailure, Exception) as exc:  # noqa: BLE001 - degrade, never crash
                how = f"Writer failed ({type(exc).__name__}: {str(exc)[:160]}): template text."
                status = StageStatus.FAILED

    if draft is None:
        draft = template(brief)

    draft, removed = _enforce(draft, brief)
    trail = [
        ReasoningStep(
            step="How it ran",
            finding=how,
            impact="The verdict was fixed before writing; the writer never scores.",
        ),
        ReasoningStep(
            step="Citation check",
            finding=f"Allowed: {', '.join(brief.clauses) or 'none'}. Removed: "
            f"{', '.join(removed) or 'none'}.",
            impact="Only clauses Stage 6 retrieved may be cited.",
        ),
    ]
    return AuditorResult(draft=draft, status=status, trail=trail, removed_citations=removed)


def _enforce(draft: AuditorDraft, brief: CaseBrief) -> tuple[AuditorDraft, list[str]]:
    """Strip clause ids Stage 6 did not return; STR only on an escalation."""
    allowed = list(brief.clauses)
    removed: list[str] = []
    fields = {}
    for name in (
        "executive_summary",
        "investigation_narrative",
        "compliance_conclusion",
        "str_narrative",
    ):
        text = getattr(draft, name)
        if text is None:
            fields[name] = None
            continue
        _, invalid = validate_citations(text, allowed)
        removed += [c for c in invalid if c not in removed]
        fields[name] = strip_citations(text, invalid)
    if brief.disposition != Disposition.SUSPICIOUS_ESCALATE:
        fields["str_narrative"] = None
    elif not fields["str_narrative"]:
        fields["str_narrative"] = template(brief).str_narrative
    return AuditorDraft(**fields), removed


def _throttle() -> float:
    wait = config.MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()
    return time.perf_counter()


def template(brief: CaseBrief) -> AuditorDraft:
    """The deterministic report text used when no model answer is available."""
    signals = [f"{stage}: {lines[0]}" for stage, lines in brief.stage_findings.items() if lines]
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
    return AuditorDraft(
        executive_summary=f"Account {brief.account_id}: {verdict}.",
        investigation_narrative=" ".join(signals) or "No stage found an adverse signal.",
        compliance_conclusion=conclusion,
        str_narrative=(
            f"Draft STR for account {brief.account_id}: {verdict}. Grounds: "
            + ("; ".join(signals) or "see the stage findings")
            + "."
            if brief.disposition == Disposition.SUSPICIOUS_ESCALATE
            else None
        ),
    )
