from __future__ import annotations

import json

import structlog
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from app.config import settings

logger = structlog.get_logger()

EXPANSION_SYSTEM_PROMPT = """\
Generate 2-3 search query variants for the following question.
Include at least one Greek and one English variant.
Return ONLY a JSON array of strings, no other text.
Example: ["What is the contract term?", "Ποια είναι η διάρκεια της σύμβασης;"]\
"""


def expand_query(question: str) -> list[str]:
    """Generate cross-lingual query variants using the LLM.

    Args:
        question: The original user question.

    Returns:
        List of query variants including the original. Falls back to
        [question] if expansion is disabled or LLM returns invalid output.
    """
    if not settings.QUERY_EXPANSION_ENABLED:
        return [question]

    try:
        generator = OpenAIChatGenerator(
            api_key=Secret.from_token("ollama"),
            model=settings.LLM_MODEL,
            api_base_url=settings.LLM_BASE_URL,
            generation_kwargs={
                "temperature": 0.3,
                "max_tokens": 200,
            },
        )

        messages = [
            ChatMessage.from_system(EXPANSION_SYSTEM_PROMPT),
            ChatMessage.from_user(question),
        ]

        result = generator.run(messages=messages)
        reply_text = result["replies"][0].text.strip()

        variants = json.loads(reply_text)
        if not isinstance(variants, list) or not all(isinstance(v, str) for v in variants):
            logger.warning("expansion_invalid_format", reply=reply_text)
            return [question]

        # Always include the original question
        if question not in variants:
            variants.insert(0, question)

        logger.info("query_expanded", original=question[:50], variant_count=len(variants))
        return variants

    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("query_expansion_failed", error=str(exc), question=question[:50])
        return [question]
    except Exception as exc:
        logger.warning("query_expansion_error", error=str(exc), question=question[:50])
        return [question]
