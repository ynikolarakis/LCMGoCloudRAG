from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db_session
from app.main import app


def _make_mock_session() -> AsyncMock:
    """Build a mock async database session that returns empty results."""
    session = AsyncMock()
    session.add = MagicMock()

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_result.scalars.return_value.all.return_value = []
    mock_result.scalar.return_value = 0
    session.execute = AsyncMock(return_value=mock_result)

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
async def test_audit_logs_returns_paginated_results(api_client: AsyncClient) -> None:
    """GET /api/v1/admin/audit-logs should return paginated audit log entries."""
    resp = await api_client.get("/api/v1/admin/audit-logs")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data


@pytest.mark.anyio
async def test_audit_logs_accepts_action_filter(api_client: AsyncClient) -> None:
    """GET /api/v1/admin/audit-logs?action=query_submitted should filter by action."""
    resp = await api_client.get("/api/v1/admin/audit-logs", params={"action": "query_submitted"})
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data


@pytest.mark.anyio
async def test_audit_logs_accepts_pagination_params(api_client: AsyncClient) -> None:
    """GET /api/v1/admin/audit-logs should accept page and page_size params."""
    resp = await api_client.get("/api/v1/admin/audit-logs", params={"page": 2, "page_size": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["page"] == 2
    assert data["page_size"] == 5
