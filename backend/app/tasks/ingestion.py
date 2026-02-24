from __future__ import annotations

import asyncio
from uuid import UUID

import structlog
from sqlalchemy import select

from app.celery_app import celery_app
from app.config import settings
from app.database import async_session_factory
from app.metrics import celery_task_duration, celery_tasks_total, documents_ingested_total
from app.models import Document, DocumentStatus
from app.pipelines.chunking import ChunkResult, chunk_document, chunk_text, count_tokens
from app.pipelines.ingestion import create_document_chunks, ingest_document, parse_document
from app.pipelines.summarizer import generate_summary

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
    Docling, generates a summary, chunks the document with structure-aware
    chunking (falling back to text chunking when pages are unavailable),
    persists DocumentChunk rows, runs the Haystack embed-and-store pipeline,
    then marks the Document COMPLETED.

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
            doc_model, text, page_count = parse_document(file_path)
            logger.info("task_document_parsed", doc_id=doc_id, page_count=page_count, chars=len(text))

            # 3. Generate document summary for contextual prefix
            summary = generate_summary(text)
            doc.summary = summary
            await session.flush()

            # 4. Chunk using structure-aware chunking
            chunk_results = chunk_document(
                doc_model,
                target_tokens=settings.CHUNK_TARGET_TOKENS,
                max_tokens=settings.CHUNK_MAX_TOKENS,
                min_tokens=settings.CHUNK_MIN_TOKENS,
                overlap_pct=settings.CHUNK_OVERLAP_PCT,
            )

            # Fallback to basic chunking when Docling model has no page structure
            if not chunk_results:
                text_chunks = chunk_text(
                    text,
                    target_tokens=settings.CHUNK_TARGET_TOKENS,
                    max_tokens=settings.CHUNK_MAX_TOKENS,
                    min_tokens=settings.CHUNK_MIN_TOKENS,
                    overlap_pct=settings.CHUNK_OVERLAP_PCT,
                )
                chunk_results = [
                    ChunkResult(content=c, page_num=i + 1, token_count=count_tokens(c))
                    for i, c in enumerate(text_chunks)
                ]

            logger.info("task_document_chunked", doc_id=doc_id, num_chunks=len(chunk_results))

            # 5. Persist DocumentChunk rows to PostgreSQL
            await create_document_chunks(
                session=session,
                doc_id=doc_uuid,
                chunks=chunk_results,
                client_id=client_id,
            )

            # 6. Embed chunks and write vectors to Qdrant via Haystack pipeline
            ingest_document(
                file_path=file_path,
                filename=doc.original_filename,
                client_id=client_id,
                summary=summary,
            )

            # 7. Mark document as completed with final counts
            doc.status = DocumentStatus.COMPLETED
            doc.chunk_count = len(chunk_results)
            doc.page_count = page_count
            await session.commit()

            logger.info(
                "task_ingestion_complete",
                doc_id=doc_id,
                chunk_count=len(chunk_results),
                page_count=page_count,
            )

            return {"doc_id": doc_id, "status": "completed", "chunk_count": len(chunk_results)}

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
    import time

    start = time.perf_counter()
    try:
        result = _run_async(_do_ingest(doc_id, file_path, client_id))
        celery_tasks_total.labels(task_name="ingest_document", status="success").inc()
        documents_ingested_total.labels(status="completed").inc()
        return result  # type: ignore[return-value]
    except Exception as exc:
        celery_tasks_total.labels(task_name="ingest_document", status="failure").inc()
        documents_ingested_total.labels(status="failed").inc()
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
    finally:
        celery_task_duration.labels(task_name="ingest_document").observe(time.perf_counter() - start)
