from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import httpx
import structlog

from app.config import settings
from app.pipelines.query import SYSTEM_PROMPT

logger = structlog.get_logger()


async def stream_llm_response(
    context: str,
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> AsyncGenerator[str, None]:
    """Stream tokens from Ollama/vLLM via OpenAI-compatible SSE API.

    Sends a streaming chat completion request to the configured LLM endpoint
    and yields individual token strings as they arrive.  The function handles
    the ``data: [DONE]`` sentinel that terminates the SSE stream.

    Args:
        context: Formatted retrieved context string (pre-built numbered chunks).
        question: The user's question.
        conversation_history: Optional list of previous messages in the format
            ``[{"role": "user"|"assistant", "content": "..."}]``.  These are
            inserted between the system prompt and the current user turn so the
            model can reference prior conversation turns.

    Yields:
        Individual token strings as they arrive from the LLM.

    Raises:
        No exceptions are raised to the caller; errors are logged and the
        generator terminates early on timeout or HTTP/parse failures.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history)

    user_content = f"Context:\n{context}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_content})

    payload: dict[str, object] = {
        "model": settings.LLM_MODEL,
        "messages": messages,
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "stream": True,
    }

    url = f"{settings.LLM_BASE_URL}/chat/completions"

    try:
        async with (
            httpx.AsyncClient(timeout=settings.LLM_STREAM_TIMEOUT) as client,
            client.stream("POST", url, json=payload) as response,
        ):
                if response.status_code != 200:
                    body = await response.aread()
                    logger.error(
                        "streaming_http_error",
                        status=response.status_code,
                        body=body[:500],
                    )
                    return

                async for raw_line in response.aiter_lines():
                    # aiter_lines() returns str in production httpx; the test
                    # mock yields bytes — normalise to str in both cases.
                    line: str = raw_line.decode() if isinstance(raw_line, bytes) else raw_line
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # strip "data: " prefix
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                    except (json.JSONDecodeError, IndexError, KeyError) as exc:
                        logger.warning(
                            "streaming_parse_error",
                            error=str(exc),
                            line=line[:100],
                        )
                        continue

    except httpx.TimeoutException:
        logger.error("streaming_timeout", timeout=settings.LLM_STREAM_TIMEOUT)
    except Exception as exc:
        logger.error("streaming_error", error=str(exc))
