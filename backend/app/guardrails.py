from __future__ import annotations

import functools

import structlog

from app.config import settings

logger = structlog.get_logger()


@functools.lru_cache(maxsize=1)
def _get_injection_scanner():
    """Lazy-load LLM Guard PromptInjection scanner (singleton)."""
    from llm_guard.input_scanners import PromptInjection

    return PromptInjection(threshold=settings.GUARDRAIL_INJECTION_THRESHOLD)


@functools.lru_cache(maxsize=1)
def _get_hhem_model():
    """Lazy-load HHEM cross-encoder model (singleton)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder("vectara/hallucination_evaluation_model")


def scan_input(query: str) -> dict:
    """Scan user query for prompt injection.

    Args:
        query: The user's question text.

    Returns:
        Dict with keys: blocked (bool), reason (str|None), risk_score (float).
    """
    if not settings.GUARDRAILS_ENABLED:
        return {"blocked": False, "reason": None, "risk_score": 0.0}

    scanner = _get_injection_scanner()
    sanitized, is_valid, risk_score = scanner.scan(query)

    if not is_valid:
        logger.warning(
            "guardrail_input_blocked",
            reason="prompt_injection",
            risk_score=risk_score,
            query_preview=query[:100],
        )
        return {"blocked": True, "reason": "prompt_injection", "risk_score": risk_score}

    return {"blocked": False, "reason": None, "risk_score": risk_score}


def check_faithfulness(
    context: str,
    response: str,
    threshold: float | None = None,
) -> tuple[float, bool]:
    """Score faithfulness of LLM response against retrieved context.

    Uses HHEM 2.1 cross-encoder to predict factual consistency.

    Args:
        context: Concatenated retrieved chunk texts.
        response: The LLM-generated answer.
        threshold: Minimum score to consider faithful. Defaults to settings value.

    Returns:
        Tuple of (score, is_faithful).
    """
    if not settings.GUARDRAILS_ENABLED:
        return 1.0, True

    if threshold is None:
        threshold = settings.GUARDRAIL_HHEM_THRESHOLD

    model = _get_hhem_model()
    score = float(model.predict([(context, response)])[0])

    is_faithful = score >= threshold

    if not is_faithful:
        logger.warning(
            "guardrail_output_unfaithful",
            hhem_score=score,
            threshold=threshold,
            response_preview=response[:100],
        )

    return score, is_faithful
