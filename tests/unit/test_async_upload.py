from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db_session
from app.main import app


def _make_mock_session() -> AsyncMock:
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
@patch("app.api.v1.documents.ingest_document_task")
async def test_upload_dispatches_celery_task(
    mock_task: MagicMock,
    api_client: AsyncClient,
) -> None:
    """Upload endpoint should dispatch Celery task, not call ingest_document synchronously."""
    mock_task.apply_async = MagicMock()

    resp = await api_client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")},
    )

    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "queued"
    assert data["message"] == "Document queued for processing."
    mock_task.apply_async.assert_called_once()


@pytest.mark.anyio
@patch("app.api.v1.documents.ingest_document_task")
async def test_upload_does_not_block_on_ingestion(
    mock_task: MagicMock,
    api_client: AsyncClient,
) -> None:
    """Upload should return immediately without waiting for ingestion."""
    mock_task.apply_async = MagicMock()

    resp = await api_client.post(
        "/api/v1/documents/upload",
        files={"file": ("report.pdf", b"%PDF-1.4 test", "application/pdf")},
    )

    assert resp.status_code == 202
    data = resp.json()
    assert data["chunk_count"] is None
