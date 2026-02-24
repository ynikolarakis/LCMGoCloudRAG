from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.anyio
@patch("app.pipelines.streaming.httpx.AsyncClient")
async def test_stream_llm_response_yields_tokens(mock_client_cls: MagicMock) -> None:
    """stream_llm_response should yield token strings from SSE chunks."""

    async def mock_aiter(*_args, **_kwargs):
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            b'data: {"choices":[{"delta":{"content":" world"}}]}',
            b"data: [DONE]",
        ]
        for line in lines:
            yield line

    mock_response = AsyncMock()
    mock_response.aiter_lines = mock_aiter
    mock_response.status_code = 200
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_client

    from app.pipelines.streaming import stream_llm_response

    tokens: list[str] = []
    async for token in stream_llm_response("context here", "What is this?"):
        tokens.append(token)

    assert "Hello" in tokens
    assert " world" in tokens
