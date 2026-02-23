from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def anyio_backend() -> str:
    """Configure anyio to use the asyncio backend for all async tests."""
    return "asyncio"


@pytest.fixture
async def api_client() -> AsyncClient:
    """Async HTTP client for API integration tests.

    Yields an ``AsyncClient`` wired directly to the FastAPI ASGI app so no
    real network socket is required during testing.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client
