from __future__ import annotations

import structlog
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from app.config import settings

logger = structlog.get_logger()

SUMMARY_SYSTEM_PROMPT = """\
Generate a 2-3 sentence summary of the following document.
Focus on the document type, subject matter, and key topics.
Return ONLY the summary text, no headers or formatting.\
"""


def generate_summary(text: str, max_chars: int = 3000) -> str:
    """Generate a short summary of a document for contextual chunk prefixing.

    Calls the configured LLM via the OpenAI-compatible API (Ollama in dev,
    vLLM in prod) and requests a 2-3 sentence summary that can be prepended
    to every chunk as a contextual prefix, improving retrieval quality.

    Args:
        text: The full document text (will be truncated to max_chars).
        max_chars: Maximum characters of the document text to send to the LLM.
            Defaults to 3000 to stay within a safe prompt budget.

    Returns:
        A 2-3 sentence summary string, or an empty string on any LLM failure.
    """
    try:
        generator = OpenAIChatGenerator(
            api_key=Secret.from_token("ollama"),
            model=settings.LLM_MODEL,
            api_base_url=settings.LLM_BASE_URL,
            generation_kwargs={"temperature": 0.1, "max_tokens": 150},
        )

        truncated = text[:max_chars]
        messages = [
            ChatMessage.from_system(SUMMARY_SYSTEM_PROMPT),
            ChatMessage.from_user(truncated),
        ]

        result = generator.run(messages=messages)
        summary: str = result["replies"][0].text.strip()
        logger.info("document_summary_generated", length=len(summary))
        return summary

    except Exception as exc:  # noqa: BLE001
        logger.warning("document_summary_failed", error=str(exc))
        return ""
