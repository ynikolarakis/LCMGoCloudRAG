from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db_session
from app.main import app


def _make_mock_session() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.delete = AsyncMock()

    # Make execute return an empty result by default
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
async def test_list_documents_returns_paginated(api_client: AsyncClient) -> None:
    """GET /documents should return paginated response."""
    resp = await api_client.get("/api/v1/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data


@pytest.mark.anyio
async def test_get_document_not_found(api_client: AsyncClient) -> None:
    """GET /documents/{id} should return 404."""
    fake_id = str(uuid4())
    resp = await api_client.get(f"/api/v1/documents/{fake_id}")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_get_document_status_not_found(api_client: AsyncClient) -> None:
    """GET /documents/{id}/status should return 404."""
    fake_id = str(uuid4())
    resp = await api_client.get(f"/api/v1/documents/{fake_id}/status")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_delete_document_not_found(api_client: AsyncClient) -> None:
    """DELETE /documents/{id} should return 404."""
    fake_id = str(uuid4())
    resp = await api_client.delete(f"/api/v1/documents/{fake_id}")
    assert resp.status_code == 404
