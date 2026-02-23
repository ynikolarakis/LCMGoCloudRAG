from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_health_endpoint_returns_200(api_client: AsyncClient) -> None:
    """Health endpoint should return 200 even if some services are down."""
    response = await api_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "qdrant" in data
    assert "redis" in data
    assert "llm" in data
