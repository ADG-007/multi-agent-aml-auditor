"""The risk scoring engine: deterministic Python, never model arithmetic (invariant 1).

Implements MASTER_PLAN.md Section 2.2 plus the edge cases settled at Phase 2's kickoff. The
decision tree, in order:

1. Round every confidence to SCORE_DECIMALS.
2. Veto: AdverseMedia or Graph RAN at or above AUTO_VETO_THRESHOLD -> SUSPICIOUS-ESCALATE at
   the vetoing stage's confidence (the higher one if both veto).
3. No weighted stage RAN: MONITOR at 0.0 if any routed weighted stage failed, else CLEAR at 0.0.
4. Renormalised weighted average over the stages that RAN, rounded. Below CLEAR_BELOW -> CLEAR,
   below ESCALATE_FROM -> MONITOR, otherwise SUSPICIOUS-ESCALATE only if the corroboration
   gate passes, else MONITOR with single_signal_capped.
5. Degraded floor: any FAILED/TIMED_OUT weighted stage lifts a CLEAR to MONITOR and sets
   evidence_incomplete. A floor, never a ceiling.

Vision's forgery confidence is never an input (invariant 2); passing it is an error.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel, Field

from src.schemas import (
    AUTO_VETO_THRESHOLD,
    CLEAR_BELOW,
    CORROBORATION_MIN_STAGES,
    CORROBORATION_STAGE_MIN,
    ESCALATE_FROM,
    SCORE_DECIMALS,
    SOLO_SIGNAL_MIN,
    STAGE_WEIGHTS,
    Disposition,
    ReasoningStep,
    StageName,
    StageStatus,
)


VETO_STAGES: tuple[StageName, ...] = (StageName.ADVERSE_MEDIA, StageName.GRAPH)
DEGRADED_STATUSES = frozenset({StageStatus.FAILED, StageStatus.TIMED_OUT})


@dataclass(frozen=True)
class StageResult:
    """One weighted stage's outcome: a confidence only when it RAN."""

    status: StageStatus
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.status == StageStatus.RAN:
            c = self.confidence
            if c is None or math.isnan(c) or not 0.0 <= c <= 1.0:
                raise ValueError(f"a stage that RAN needs a confidence in [0, 1], got {c!r}")
        elif self.confidence is not None:
            raise ValueError(f"a {self.status.value} stage has no confidence")


class ScoreResult(BaseModel):
    disposition: Disposition
    confidence: float = Field(ge=0.0, le=1.0)
    weighted_score: float | None = Field(
        default=None, description="The renormalised average, when one was computed."
    )
    veto_stage: StageName | None = None
    degraded_stages: list[StageName] = Field(default_factory=list)
    evidence_incomplete: bool = False
    single_signal_capped: bool = False
    reasoning_trail: list[ReasoningStep] = Field(default_factory=list)


def score_case(results: Mapping[StageName, StageResult]) -> ScoreResult:
    """Score one case from its weighted stages' results. Missing stages count as NOT_ROUTED."""
    unknown = [s for s in results if s not in STAGE_WEIGHTS]
    if unknown:
        raise ValueError(
            f"only weighted stages are scored; got {[s.value for s in unknown]} "
            "(Vision's forgery confidence is a separate gate, never a score input)"
        )

    ran = {
        stage: _round(result.confidence)
        for stage in STAGE_WEIGHTS
        if (result := results.get(stage)) and result.status == StageStatus.RAN
    }
    degraded = [
        stage
        for stage in STAGE_WEIGHTS
        if (result := results.get(stage)) and result.status in DEGRADED_STATUSES
    ]
    trail: list[ReasoningStep] = []

    # 2. Veto
    vetoes = {s: c for s, c in ran.items() if s in VETO_STAGES and c >= AUTO_VETO_THRESHOLD}
    if vetoes:
        stage = max(vetoes, key=lambda s: (vetoes[s], STAGE_WEIGHTS[s]))
        listed = ", ".join(f"{s.value} {c:g}" for s, c in vetoes.items())
        trail.append(
            _step(
                "Veto check",
                f"{listed} at or above {AUTO_VETO_THRESHOLD:g}",
                f"Escalated without averaging; confidence is {stage.value}'s {vetoes[stage]:g}.",
            )
        )
        return ScoreResult(
            disposition=Disposition.SUSPICIOUS_ESCALATE,
            confidence=vetoes[stage],
            veto_stage=stage,
            degraded_stages=degraded,
            evidence_incomplete=bool(degraded),
            reasoning_trail=trail + _degraded_note(degraded, floored=False),
        )
    trail.append(
        _step(
            "Veto check",
            f"No AdverseMedia or Graph confidence at or above {AUTO_VETO_THRESHOLD:g}",
            "Scored by weighted average.",
        )
    )

    # 3. Nothing ran
    if not ran:
        if degraded:
            trail.append(
                _step(
                    "Weighted average",
                    "Every routed weighted stage failed or timed out",
                    "No evidence to average; held at MONITOR, never CLEAR on missing evidence.",
                )
            )
            return ScoreResult(
                disposition=Disposition.MONITOR,
                confidence=0.0,
                degraded_stages=degraded,
                evidence_incomplete=True,
                reasoning_trail=trail + _degraded_note(degraded, floored=True),
            )
        trail.append(
            _step("Weighted average", "No weighted stage was routed", "Benign fast path: CLEAR.")
        )
        return ScoreResult(disposition=Disposition.CLEAR, confidence=0.0, reasoning_trail=trail)

    # 4. Weighted average, thresholds, corroboration gate
    score = _round(
        sum(STAGE_WEIGHTS[s] * c for s, c in ran.items()) / sum(STAGE_WEIGHTS[s] for s in ran)
    )
    parts = " + ".join(f"{s.value} {c:g}x{STAGE_WEIGHTS[s]:g}" for s, c in ran.items())
    trail.append(
        _step(
            "Weighted average",
            f"({parts}) / {sum(STAGE_WEIGHTS[s] for s in ran):g} = {score:g}",
            "Renormalised over the stages that ran.",
        )
    )

    capped = False
    if score < CLEAR_BELOW:
        disposition = Disposition.CLEAR
        trail.append(_step("Thresholds", f"{score:g} below {CLEAR_BELOW:g}", "CLEAR."))
    elif score < ESCALATE_FROM:
        disposition = Disposition.MONITOR
        trail.append(
            _step(
                "Thresholds",
                f"{score:g} between {CLEAR_BELOW:g} and {ESCALATE_FROM:g}",
                "MONITOR.",
            )
        )
    elif corroboration_passes(ran):
        disposition = Disposition.SUSPICIOUS_ESCALATE
        trail.append(_gate_step(ran, score, passed=True))
    else:
        disposition = Disposition.MONITOR
        capped = True
        trail.append(_gate_step(ran, score, passed=False))

    # 5. Degraded floor
    floored = bool(degraded) and disposition == Disposition.CLEAR
    if floored:
        disposition = Disposition.MONITOR

    return ScoreResult(
        disposition=disposition,
        confidence=score,
        weighted_score=score,
        degraded_stages=degraded,
        evidence_incomplete=bool(degraded),
        single_signal_capped=capped,
        reasoning_trail=trail + _degraded_note(degraded, floored=floored),
    )


def corroboration_passes(ran: Mapping[StageName, float]) -> bool:
    """>= CORROBORATION_MIN_STAGES stages at >= CORROBORATION_STAGE_MIN, or one at >= SOLO."""
    agreeing = sum(1 for c in ran.values() if c >= CORROBORATION_STAGE_MIN)
    return agreeing >= CORROBORATION_MIN_STAGES or any(c >= SOLO_SIGNAL_MIN for c in ran.values())


def _gate_step(ran: Mapping[StageName, float], score: float, passed: bool) -> ReasoningStep:
    agreeing = [s.value for s, c in ran.items() if c >= CORROBORATION_STAGE_MIN]
    strong = [s.value for s, c in ran.items() if c >= SOLO_SIGNAL_MIN]
    finding = (
        f"{score:g} at or above {ESCALATE_FROM:g}; stages at >= {CORROBORATION_STAGE_MIN:g}: "
        f"{agreeing or 'none'}; at >= {SOLO_SIGNAL_MIN:g}: {strong or 'none'}"
    )
    impact = (
        "Corroboration gate passed: SUSPICIOUS-ESCALATE."
        if passed
        else "Corroboration gate failed: a single borderline signal is held at MONITOR."
    )
    return _step("Corroboration gate", finding, impact)


def _degraded_note(degraded: list[StageName], floored: bool) -> list[ReasoningStep]:
    if not degraded:
        return []
    impact = (
        "Evidence incomplete: lifted to MONITOR, never CLEAR on missing evidence."
        if floored
        else "Evidence incomplete; the floor does not cap the result."
    )
    return [_step("Degraded stages", ", ".join(s.value for s in degraded), impact)]


def _round(value: float) -> float:
    return round(value, SCORE_DECIMALS)


def _step(step: str, finding: str, impact: str) -> ReasoningStep:
    return ReasoningStep(step=step, finding=finding, impact=impact)
