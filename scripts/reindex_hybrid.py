"""Re-index all documents with sparse embeddings for hybrid search.

This script:
1. Reads all DocumentChunk rows from PostgreSQL
2. Recreates the Qdrant collection with sparse vector support
3. Re-embeds all chunks with both dense (Qwen3) and sparse (BM25) vectors
4. Writes them to the new collection

Usage:
    cd backend && PYTHONPATH=. python ../scripts/reindex_hybrid.py
"""

from __future__ import annotations

import asyncio

import structlog
from haystack import Document as HaystackDocument
from sqlalchemy import func, select

from app.config import settings
from app.database import async_session_factory
from app.models import DocumentChunk
from app.pipelines.ingestion import _build_ingestion_pipeline, _get_document_store

logger = structlog.get_logger()


async def _load_chunks() -> list[dict]:
    """Load all document chunks from PostgreSQL."""
    async with async_session_factory() as session:
        count_result = await session.execute(select(func.count()).select_from(DocumentChunk))
        total = count_result.scalar() or 0
        logger.info("loading_chunks", total=total)

        result = await session.execute(
            select(DocumentChunk).order_by(DocumentChunk.document_id, DocumentChunk.chunk_seq)
        )
        chunks = result.scalars().all()

        return [
            {
                "id": str(chunk.id),
                "content": chunk.content,
                "document_id": str(chunk.document_id),
                "chunk_seq": chunk.chunk_seq,
                "token_count": chunk.token_count,
                "client_id": chunk.client_id,
            }
            for chunk in chunks
        ]


def _reindex(chunks: list[dict]) -> int:
    """Re-embed and write all chunks to Qdrant with sparse vectors."""
    if not chunks:
        logger.info("no_chunks_to_reindex")
        return 0

    haystack_docs = [
        HaystackDocument(
            id=chunk["id"],
            content=chunk["content"],
            meta={
                "source": "reindexed",
                "page_num": chunk["chunk_seq"] + 1,
                "chunk_seq": chunk["chunk_seq"],
                "token_count": chunk["token_count"],
                "client_id": chunk["client_id"],
                "document_id": chunk["document_id"],
            },
        )
        for chunk in chunks
    ]

    store = _get_document_store()
    pipeline = _build_ingestion_pipeline(store)

    # Process in batches of 32
    batch_size = 32
    total_written = 0

    for i in range(0, len(haystack_docs), batch_size):
        batch = haystack_docs[i : i + batch_size]
        result = pipeline.run({"sparse_embedder": {"documents": batch}})
        written = result.get("writer", {}).get("documents_written", len(batch))
        total_written += written
        logger.info("batch_indexed", batch=i // batch_size + 1, written=written, total=total_written)

    return total_written


def main() -> None:
    logger.info("reindex_started", collection=settings.QDRANT_COLLECTION)

    # 1. Load chunks from PostgreSQL
    chunks = asyncio.run(_load_chunks())
    logger.info("chunks_loaded", count=len(chunks))

    if not chunks:
        logger.info("nothing_to_reindex")
        return

    # 2. Re-embed and write to Qdrant
    total = _reindex(chunks)
    logger.info("reindex_complete", total_written=total)


if __name__ == "__main__":
    main()
