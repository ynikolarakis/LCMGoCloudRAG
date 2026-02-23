from __future__ import annotations

import asyncio
from uuid import UUID

import structlog
from sqlalchemy import select

from app.celery_app import celery_app
from app.config import settings
from app.database import async_session_factory
from app.models import Document, DocumentStatus
from app.pipelines.chunking import chunk_text
from app.pipelines.ingestion import create_document_chunks, ingest_document, parse_document

logger = structlog.get_logger()


def _run_async(coro: object) -> object:
    """Run an async coroutine from a synchronous Celery task context.

    Creates a fresh event loop per invocation so Celery worker threads do not
    share loop state across task executions.

    Args:
        coro: An awaitable coroutine to execute.

    Returns:
        The return value of the coroutine.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)  # type: ignore[arg-type]
    finally:
        loop.close()


async def _do_ingest(doc_id: str, file_path: str, client_id: str) -> dict[str, object]:
    """Async implementation of document ingestion with DB persistence.

    Fetches the Document row, marks it PROCESSING, parses the file with
    Docling, chunks the extracted text, persists DocumentChunk rows, runs
    the Haystack embed-and-store pipeline, then marks the Document COMPLETED.

    On any failure the Document is set to FAILED and the exception is
    re-raised so the Celery task can apply its retry policy.

    Args:
        doc_id: UUID string of the Document row.
        file_path: Absolute filesystem path to the uploaded file.
        client_id: Tenant identifier used for Qdrant collection scoping.

    Returns:
        Dict with keys ``doc_id``, ``status``, and ``chunk_count``.

    Raises:
        ValueError: If the Document row is not found in the database.
        Exception: Any parsing, chunking, embedding, or DB error.
    """
    doc_uuid = UUID(doc_id)

    async with async_session_factory() as session:
        result = await session.execute(select(Document).where(Document.id == doc_uuid))
        doc = result.scalar_one_or_none()
        if not doc:
            raise ValueError(f"Document {doc_id} not found in database")

        try:
            # 1. Mark as processing
            doc.status = DocumentStatus.PROCESSING
            await session.flush()
            logger.info("task_ingestion_started", doc_id=doc_id, file_path=file_path, client_id=client_id)

            # 2. Parse document with Docling
            text, page_count = parse_document(file_path)
            logger.info("task_document_parsed", doc_id=doc_id, page_count=page_count, chars=len(text))

            # 3. Chunk extracted text
            chunks = chunk_text(
                text,
                target_tokens=settings.CHUNK_TARGET_TOKENS,
                max_tokens=settings.CHUNK_MAX_TOKENS,
                min_tokens=settings.CHUNK_MIN_TOKENS,
                overlap_pct=settings.CHUNK_OVERLAP_PCT,
            )
            logger.info("task_document_chunked", doc_id=doc_id, num_chunks=len(chunks))

            # 4. Persist DocumentChunk rows to PostgreSQL
            await create_document_chunks(
                session=session,
                doc_id=doc_uuid,
                chunks=chunks,
                client_id=client_id,
            )

            # 5. Embed chunks and write vectors to Qdrant via Haystack pipeline
            ingest_document(
                file_path=file_path,
                filename=doc.original_filename,
                client_id=client_id,
            )

            # 6. Mark document as completed with final counts
            doc.status = DocumentStatus.COMPLETED
            doc.chunk_count = len(chunks)
            doc.page_count = page_count
            await session.commit()

            logger.info(
                "task_ingestion_complete",
                doc_id=doc_id,
                chunk_count=len(chunks),
                page_count=page_count,
            )

            return {"doc_id": doc_id, "status": "completed", "chunk_count": len(chunks)}

        except Exception as exc:
            doc.status = DocumentStatus.FAILED
            await session.commit()
            logger.exception("task_ingestion_failed", doc_id=doc_id, error=str(exc))
            raise


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def ingest_document_task(self, doc_id: str, file_path: str, client_id: str) -> dict[str, object]:
    """Celery task: parse, chunk, embed, and store a document.

    Executes the full ingestion pipeline asynchronously, updating the
    Document status throughout. Retries up to 3 times on transient
    failures using an exponential backoff schedule (10 s, 60 s, 300 s).

    Args:
        doc_id: UUID string of the Document row to process.
        file_path: Absolute filesystem path to the uploaded file.
        client_id: Tenant identifier for multi-tenancy scoping.

    Returns:
        Dict with keys ``doc_id``, ``status``, and ``chunk_count``.
    """
    try:
        return _run_async(_do_ingest(doc_id, file_path, client_id))  # type: ignore[return-value]
    except Exception as exc:
        retry_delays = [10, 60, 300]
        delay = retry_delays[min(self.request.retries, len(retry_delays) - 1)]
        logger.warning(
            "task_retrying",
            doc_id=doc_id,
            attempt=self.request.retries + 1,
            max_retries=self.max_retries,
            delay=delay,
        )
        raise self.retry(exc=exc, countdown=delay) from exc
