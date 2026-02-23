from __future__ import annotations

import os
import tempfile
from pathlib import Path
from uuid import uuid4

import structlog
from fastapi import APIRouter, File, UploadFile

from app.config import settings
from app.exceptions import ValidationError
from app.pipelines.ingestion import ingest_document
from app.schemas.document import DocumentUploadResponse

logger = structlog.get_logger()
router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Upload a document for ingestion into the RAG pipeline.

    Accepts PDF, DOCX, PPTX files. Parses, chunks, embeds, and stores in Qdrant.
    Phase 1: synchronous processing. Phase 2+ will use Celery for async.

    Args:
        file: The uploaded file (PDF, DOCX, or PPTX).

    Returns:
        DocumentUploadResponse with document ID, status, and chunk count.

    Raises:
        ValidationError: If the filename is missing, file type is unsupported,
            or the file exceeds the maximum allowed size.
    """
    if not file.filename:
        raise ValidationError("Filename is required")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise ValidationError(
            f"Unsupported file type: {file.content_type}. Allowed: PDF, DOCX, PPTX"
        )

    # Read file content
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise ValidationError(f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)}MB")

    # Save to temp file for Docling processing
    doc_id = uuid4()
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Run ingestion pipeline (synchronous in Phase 1)
        haystack_docs = ingest_document(
            file_path=tmp_path,
            filename=file.filename,
            client_id=settings.CLIENT_ID,
        )

        logger.info(
            "document_upload_complete",
            doc_id=str(doc_id),
            filename=file.filename,
            chunk_count=len(haystack_docs),
        )

        return DocumentUploadResponse(
            id=doc_id,
            filename=file.filename,
            status="completed",
            chunk_count=len(haystack_docs),
            message=f"Document processed successfully. {len(haystack_docs)} chunks indexed.",
        )
    finally:
        os.unlink(tmp_path)
