"""The one function every LLM-backed stage calls.

Built in Phase 1 even though Phase 1 never calls an LLM, for the same reason the schemas
are: Phase 10's uniform retry policy and Phase 11's uniform cassette recording can only be
uniform if there is a single call path, and retrofitting one across three independently
built stages is far more expensive than standing it up now.

Library choice is LangChain rather than LiteLLM + Instructor (a recorded amendment to
MASTER_PLAN.md Section 4, made during Phase 1 planning): Phase 10 already commits to
LangGraph, so this consolidates on one framework family instead of maintaining two separate
library ecosystems for orchestration and model calls. LangChain supplies both halves of what
was wanted - a provider-agnostic chat interface via init_chat_model, and Pydantic-validated
structured output via with_structured_output.

The six-step content-policy chain (MASTER_PLAN.md Section 4) lives here:
  1. every cloud prompt carries the synthetic-context preamble
  2. a refusal is detected as its own signal, distinct from a rate limit or a schema failure
  3. retry once with softened phrasing
  4. fail over to the configured fallback provider/model
  5. raise LLMStageFailure(CONTENT_POLICY_REFUSAL) so orchestration marks the stage FAILED
  6. log every occurrence regardless of outcome
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Any, TypeVar

from pydantic import BaseModel

from src.llm.config import (
    SYNTHETIC_CONTEXT_PREAMBLE,
    LLMProvider,
    StageLLMConfig,
    get_stage_config,
)
from src.llm.errors import LLMFailureReason, LLMStageFailure

logger = logging.getLogger(__name__)

TSchema = TypeVar("TSchema", bound=BaseModel)


#: Phrases that indicate a safety classifier declined, rather than the model reasoning badly.
#: Deliberately conservative - a false positive here would retry a perfectly good answer.
_REFUSAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bI (?:cannot|can not|can't|am unable to|won't) (?:assist|help|comply|provide)",
        r"\bI'm not able to (?:assist|help|provide)",
        r"\bthis (?:request|content) violates\b",
        r"\bagainst my (?:safety )?guidelines\b",
        r"\bblocked by (?:the )?safety (?:filter|settings)\b",
    )
)

#: Finish reasons providers use when a safety classifier stopped generation.
_REFUSAL_FINISH_REASONS = frozenset({"safety", "content_filter", "blocked", "recitation"})

#: Phrasing softened on the step-3 retry. The scenario is unchanged - only the vocabulary
#: most likely to trip a consumer safety classifier is neutralised.
_SOFTENING_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("money laundering", "irregular fund movement"),
    ("laundering", "irregular fund movement"),
    ("criminal", "non-compliant"),
    ("fraud ring", "linked account cluster"),
    ("fraudulent", "irregular"),
    ("illicit", "irregular"),
    ("terrorist financing", "prohibited-party funding"),
    ("suspect", "subject"),
)


def looks_like_refusal(text: str | None, finish_reason: str | None = None) -> bool:
    """Whether a response is a safety refusal rather than a real answer.

    Checked before schema validation, because a refusal usually fails schema validation too
    and would otherwise be miscounted as a reasoning error in Phase 11's metrics.
    """
    if finish_reason and finish_reason.lower() in _REFUSAL_FINISH_REASONS:
        return True
    if not text:
        return False
    return any(pattern.search(text) for pattern in _REFUSAL_PATTERNS)


def soften_prompt(prompt: str) -> str:
    """Step 3 of the refusal chain: same scenario, less classifier-triggering vocabulary.

    Whole words only, so "suspected" is left alone rather than becoming "subjected". Applied
    to the stage's own prompt before the preamble is added, never to the preamble itself.
    """
    softened = prompt
    for original, replacement in _SOFTENING_SUBSTITUTIONS:
        pattern = rf"\b{re.escape(original)}\b"
        softened = re.sub(pattern, replacement, softened, flags=re.IGNORECASE)
    return softened


def build_prompt(prompt: str, config: StageLLMConfig) -> str:
    """Step 1: prepend the synthetic-context preamble for cloud providers.

    A local Ollama model carries no consumer safety-filter layer, so the preamble is skipped
    there - it would only consume context budget a small local model does not have.
    """
    if config.provider == LLMProvider.OLLAMA:
        return prompt
    return f"{SYNTHETIC_CONTEXT_PREAMBLE}\n\n{prompt}"


def _build_model(provider: LLMProvider, model_id: str, temperature: float) -> Any:
    """Instantiate a LangChain chat model.

    Imported lazily so importing this module never requires a provider integration package.
    Phase 1 has none installed, and each stage's own phase installs only the one it chose.
    """
    try:
        from langchain.chat_models import init_chat_model
    except ImportError as exc:  # pragma: no cover - only hit without langchain installed
        raise LLMStageFailure(
            LLMFailureReason.PROVIDER_UNAVAILABLE,
            stage_name="<init>",
            message=f"LangChain is not importable: {exc}",
        ) from exc

    return init_chat_model(
        model=model_id,
        model_provider=provider.value,
        temperature=temperature,
    )


def _invoke_once(
    prompt: str,
    schema: type[TSchema],
    provider: LLMProvider,
    model_id: str,
    temperature: float,
    stage_name: str,
) -> TSchema:
    """One attempt against one provider/model, with structured output enforced."""
    model = _build_model(provider, model_id, temperature)
    structured = model.with_structured_output(schema)
    try:
        result = structured.invoke(prompt)
    except Exception as exc:
        text = str(exc)
        if looks_like_refusal(text):
            raise LLMStageFailure(
                LLMFailureReason.CONTENT_POLICY_REFUSAL,
                stage_name,
                text,
                provider.value,
                model_id,
            ) from exc
        if "429" in text or "rate limit" in text.lower():
            raise LLMStageFailure(
                LLMFailureReason.RATE_LIMITED, stage_name, text, provider.value, model_id
            ) from exc
        raise LLMStageFailure(
            LLMFailureReason.SCHEMA_VALIDATION_FAILED,
            stage_name,
            text,
            provider.value,
            model_id,
        ) from exc

    if not isinstance(result, schema):
        raise LLMStageFailure(
            LLMFailureReason.SCHEMA_VALIDATION_FAILED,
            stage_name,
            f"Expected {schema.__name__}, got {type(result).__name__}.",
            provider.value,
            model_id,
        )
    return result


def get_structured_response(
    prompt: str,
    schema: type[TSchema],
    stage_name: str,
) -> TSchema:
    """Call this stage's configured model and return a schema-validated object.

    Raises LLMStageFailure on every failure path, which Phase 10's orchestration maps to
    stage_status = FAILED. It never returns a partially valid result and never returns None:
    a degraded stage floors its case at MONITOR, which is the correct conservative outcome,
    so silently returning something half-formed would defeat the scoring design.
    """
    config = get_stage_config(stage_name)

    if not config.is_decided:
        where = config.decided_in_phase or "that stage's own kickoff"
        raise LLMStageFailure(
            LLMFailureReason.NOT_CONFIGURED,
            stage_name,
            f"No model chosen for {stage_name} yet - it is decided at {where}. {config.notes}",
        )

    provider = config.provider
    model_id = config.model_id
    assert provider is not None and model_id is not None  # narrowed by is_decided
    framed = build_prompt(prompt, config)
    softened = build_prompt(soften_prompt(prompt), config)

    # Steps 2-3: first attempt, then one softened retry if a safety classifier declined.
    try:
        return _invoke_once(framed, schema, provider, model_id, config.temperature, stage_name)
    except LLMStageFailure as first_failure:
        if first_failure.reason != LLMFailureReason.CONTENT_POLICY_REFUSAL:
            raise
        logger.warning(
            "content_policy_refusal stage=%s provider=%s model=%s attempt=initial",
            stage_name,
            provider.value,
            model_id,
        )

    try:
        return _invoke_once(
            softened, schema, provider, model_id, config.temperature, stage_name
        )
    except LLMStageFailure as second_failure:
        if second_failure.reason != LLMFailureReason.CONTENT_POLICY_REFUSAL:
            raise
        logger.warning(
            "content_policy_refusal stage=%s provider=%s model=%s attempt=softened",
            stage_name,
            provider.value,
            model_id,
        )

    # Step 4: fail over. A local Ollama model is the natural fallback - it carries no
    # consumer safety-filter layer, so a refusal loop cannot simply repeat there.
    if config.fallback_provider is None or config.fallback_model_id is None:
        raise LLMStageFailure(
            LLMFailureReason.CONTENT_POLICY_REFUSAL,
            stage_name,
            "Refused twice and no fallback model is configured for this stage.",
            provider.value,
            model_id,
        )

    logger.warning(
        "content_policy_refusal stage=%s failing over to provider=%s model=%s",
        stage_name,
        config.fallback_provider.value,
        config.fallback_model_id,
    )
    # The fallback may be a different provider family, so the prompt is re-framed for it: a
    # local fallback gets no preamble, a cloud one does.
    fallback_config = replace(config, provider=config.fallback_provider)
    return _invoke_once(
        build_prompt(soften_prompt(prompt), fallback_config),
        schema,
        config.fallback_provider,
        config.fallback_model_id,
        config.temperature,
        stage_name,
    )
