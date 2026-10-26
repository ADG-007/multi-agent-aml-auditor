"""The one function every LLM-backed stage calls (LangChain, Pydantic structured output).

Refusal chain: synthetic-data preamble on cloud prompts -> detect a refusal -> one softened
retry -> fallback model -> LLMStageFailure (stage FAILED). Every refusal is logged.
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


#: Safety-refusal phrases. Kept narrow: a false hit would retry a good answer.
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

#: Words softened on the retry. The scenario stays the same.
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
    """Whether a response is a safety refusal. Checked before schema validation."""
    if finish_reason and finish_reason.lower() in _REFUSAL_FINISH_REASONS:
        return True
    if not text:
        return False
    return any(pattern.search(text) for pattern in _REFUSAL_PATTERNS)


def soften_prompt(prompt: str) -> str:
    """Same scenario, softer words (whole words only). Never touches the preamble."""
    softened = prompt
    for original, replacement in _SOFTENING_SUBSTITUTIONS:
        pattern = rf"\b{re.escape(original)}\b"
        softened = re.sub(pattern, replacement, softened, flags=re.IGNORECASE)
    return softened


def build_prompt(prompt: str, config: StageLLMConfig) -> str:
    """Prepend the synthetic-data preamble for cloud models. Local Ollama gets none."""
    if config.provider == LLMProvider.OLLAMA:
        return prompt
    return f"{SYNTHETIC_CONTEXT_PREAMBLE}\n\n{prompt}"


def _build_model(
    provider: LLMProvider, model_id: str, temperature: float, **model_kwargs: Any
) -> Any:
    """A LangChain chat model. Imported lazily, so no provider package is needed to import."""
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
        **model_kwargs,
    )


def _invoke_once(
    prompt: str,
    schema: type[TSchema],
    provider: LLMProvider,
    model_id: str,
    temperature: float,
    stage_name: str,
    model_kwargs: dict | None = None,
) -> TSchema:
    """One attempt against one provider/model, with structured output enforced."""
    model = _build_model(provider, model_id, temperature, **(model_kwargs or {}))
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
    """Call the stage's model and return a validated object.

    Every failure raises LLMStageFailure (stage FAILED, case floored at MONITOR). Never a
    half-valid result, never None.
    """
    config = get_stage_config(stage_name)

    if not config.is_decided:
        raise LLMStageFailure(
            LLMFailureReason.NOT_CONFIGURED,
            stage_name,
            f"No model configured for {stage_name}. {config.notes}",
        )

    provider = config.provider
    model_id = config.model_id
    assert provider is not None and model_id is not None  # narrowed by is_decided
    framed = build_prompt(prompt, config)
    softened = build_prompt(soften_prompt(prompt), config)

    # First attempt, then one softened retry on a refusal.
    try:
        return _invoke_once(
            framed, schema, provider, model_id, config.temperature, stage_name, config.model_kwargs
        )
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
            softened, schema, provider, model_id, config.temperature, stage_name,
            config.model_kwargs,
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

    # Fail over. A local model has no safety filter, so a refusal cannot repeat there.
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
    # Re-frame for the fallback's provider (preamble only for cloud).
    fallback_config = replace(config, provider=config.fallback_provider)
    return _invoke_once(
        build_prompt(soften_prompt(prompt), fallback_config),
        schema,
        config.fallback_provider,
        config.fallback_model_id,
        config.temperature,
        stage_name,
    )
