from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def api_client() -> AsyncClient:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.mark.anyio
async def test_health_no_auth_required(api_client: AsyncClient) -> None:
    """Health endpoint should NOT require authentication."""
    resp = await api_client.get("/api/v1/health")
    assert resp.status_code != 401
    assert resp.status_code != 403


@pytest.mark.anyio
async def test_query_accepts_dev_auth(api_client: AsyncClient) -> None:
    """Query endpoint in dev mode should accept requests (dev bypass auth)."""
    resp = await api_client.post(
        "/api/v1/query",
        json={"question": "test question"},
    )
    # Should not be 401/403 (may fail for other reasons like missing Qdrant)
    assert resp.status_code != 401
    assert resp.status_code != 403
