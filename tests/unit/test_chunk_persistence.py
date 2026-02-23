from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.pipelines.ingestion import create_document_chunks


@pytest.mark.anyio
async def test_create_document_chunks_returns_chunks() -> None:
    """create_document_chunks should create DocumentChunk rows."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()

    doc_id = uuid4()
    chunks = ["First chunk of text.", "Second chunk of text."]

    result = await create_document_chunks(
        session=session,
        doc_id=doc_id,
        chunks=chunks,
        client_id="default",
    )

    assert len(result) == 2
    assert result[0].chunk_seq == 0
    assert result[1].chunk_seq == 1
    assert result[0].document_id == doc_id
    assert result[0].client_id == "default"
    assert session.add.call_count == 2


@pytest.mark.anyio
async def test_create_document_chunks_empty_list() -> None:
    """create_document_chunks should handle empty list."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()

    result = await create_document_chunks(
        session=session,
        doc_id=uuid4(),
        chunks=[],
        client_id="default",
    )

    assert result == []
    session.add.assert_not_called()


@pytest.mark.anyio
async def test_create_document_chunks_sets_token_count() -> None:
    """Each chunk should have a positive token_count."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()

    result = await create_document_chunks(
        session=session,
        doc_id=uuid4(),
        chunks=["Hello world this is a test chunk with some tokens."],
        client_id="default",
    )

    assert len(result) == 1
    assert result[0].token_count > 0
