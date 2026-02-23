from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Stub out the `fastembed` package so that fastembed-haystack components can
# be imported without requiring the full Rust-compiled fastembed wheel.
# These stubs must be registered before any app modules are imported.
# ---------------------------------------------------------------------------

_fastembed_stub = ModuleType("fastembed")
_fastembed_sparse_stub = ModuleType("fastembed.sparse")
_fastembed_sparse_text_stub = ModuleType("fastembed.sparse.sparse_text_embedding")

_fastembed_stub.TextEmbedding = MagicMock  # type: ignore[attr-defined]
_fastembed_sparse_text_stub.SparseTextEmbedding = MagicMock  # type: ignore[attr-defined]

sys.modules.setdefault("fastembed", _fastembed_stub)
sys.modules.setdefault("fastembed.sparse", _fastembed_sparse_stub)
sys.modules.setdefault("fastembed.sparse.sparse_text_embedding", _fastembed_sparse_text_stub)

from app.main import app  # noqa: E402


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
