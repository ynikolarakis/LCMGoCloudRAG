from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db_session
from app.main import app


def _make_mock_session() -> AsyncMock:
    """Build an AsyncMock that satisfies the session.add / session.flush contract."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    return session


@pytest.fixture
async def api_client() -> AsyncClient:
    """AsyncClient with the DB session dependency overridden to avoid a real DB."""
    mock_session = _make_mock_session()

    async def _override_get_db_session():  # type: ignore[return]
        yield mock_session

    app.dependency_overrides[get_db_session] = _override_get_db_session
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_db_session, None)


@pytest.mark.anyio
@patch("app.api.v1.query.query_documents")
async def test_query_returns_response_with_citations(
    mock_query: MagicMock,
    api_client: AsyncClient,
) -> None:
    """Query endpoint should return response with model and latency."""
    mock_query.return_value = {
        "answer": "Test answer [Source: test.pdf, Page 1]",
        "citations": [{"source": "test.pdf", "page": 1, "content_preview": "preview"}],
        "model_used": "qwen3:30b-a3b",
        "latency_ms": 500,
        "retrieved_docs": [],
    }

    resp = await api_client.post(
        "/api/v1/query",
        json={"question": "What is in the document?"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Test answer [Source: test.pdf, Page 1]"
    assert data["model_used"] == "qwen3:30b-a3b"
    assert data["latency_ms"] == 500
    assert len(data["citations"]) == 1
