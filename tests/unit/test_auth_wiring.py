from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db_session
from app.main import app


def _make_mock_session() -> AsyncMock:
    """Build an AsyncMock that satisfies session.add / session.flush."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    return session


@pytest.fixture
async def api_client() -> AsyncClient:
    mock_session = _make_mock_session()

    async def _override():  # type: ignore[return]
        yield mock_session

    app.dependency_overrides[get_db_session] = _override
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_db_session, None)


@pytest.mark.anyio
async def test_health_no_auth_required(api_client: AsyncClient) -> None:
    """Health endpoint should NOT require authentication."""
    resp = await api_client.get("/api/v1/health")
    assert resp.status_code != 401
    assert resp.status_code != 403


@pytest.mark.anyio
@patch("app.api.v1.query.query_documents")
async def test_query_accepts_dev_auth(mock_query: MagicMock, api_client: AsyncClient) -> None:
    """Query endpoint in dev mode should accept requests (dev bypass auth)."""
    mock_query.return_value = {
        "answer": "Test answer",
        "citations": [],
        "model_used": "test",
        "latency_ms": 100,
        "retrieved_docs": [],
    }
    resp = await api_client.post(
        "/api/v1/query",
        json={"question": "test question"},
    )
    assert resp.status_code != 401
    assert resp.status_code != 403
