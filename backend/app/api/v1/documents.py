from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit import write_audit_log
from app.auth import get_current_user
from app.config import settings
from app.database import get_db_session
from app.exceptions import ValidationError
from app.models import Document, DocumentStatus, User
from app.models.base import AuditAction
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


def _ensure_upload_dir(client_id: str) -> Path:
    """Create the upload directory for a client if it doesn't exist.

    Args:
        client_id: The tenant identifier used as a subdirectory name.

    Returns:
        The resolved Path to the client upload directory.
    """
    upload_dir = Path(settings.UPLOAD_DIR) / client_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


@router.post("/upload", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> DocumentUploadResponse:
    """Upload a document for ingestion into the RAG pipeline.

    Validates the file, creates a Document row with status=queued, saves the
    file to uploads/{client_id}/{doc_id}{suffix}, runs ingestion synchronously,
    and writes an audit log entry. Returns 202 with document ID and status.

    Args:
        file: The uploaded file (PDF, DOCX, or PPTX).
        current_user: The authenticated user, injected by FastAPI.
        session: The async database session, injected by FastAPI.

    Returns:
        DocumentUploadResponse with document ID, final status, and chunk count.

    Raises:
        ValidationError: If the filename is missing, file type is unsupported,
            or the file exceeds the maximum allowed size.
    """
    if not file.filename:
        raise ValidationError("Filename is required")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise ValidationError(f"Unsupported file type: {file.content_type}. Allowed: PDF, DOCX, PPTX")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise ValidationError(f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)}MB")

    # 1. Create Document row with status=queued
    doc_id = uuid4()
    suffix = Path(file.filename).suffix
    stored_filename = f"{doc_id}{suffix}"

    doc = Document(
        id=doc_id,
        filename=stored_filename,
        original_filename=file.filename,
        file_size=len(content),
        content_type=file.content_type or "application/octet-stream",
        status=DocumentStatus.QUEUED,
        client_id=current_user.client_id,
        uploaded_by=current_user.id,
    )
    session.add(doc)
    await session.flush()

    # 2. Save file to disk at uploads/{client_id}/{doc_id}{suffix}
    upload_dir = _ensure_upload_dir(current_user.client_id)
    file_path = upload_dir / stored_filename
    file_path.write_bytes(content)

    # 3. Write audit log (added to session, committed with the transaction)
    await write_audit_log(
        session=session,
        user_id=current_user.id,
        action=AuditAction.DOCUMENT_UPLOAD,
        resource_type="document",
        resource_id=doc_id,
        details={"filename": file.filename, "file_size": len(content), "content_type": file.content_type},
        client_id=current_user.client_id,
    )

    # 4. Run ingestion synchronously (Celery dispatch in a later task)
    try:
        doc.status = DocumentStatus.PROCESSING
        await session.flush()

        haystack_docs = ingest_document(
            file_path=str(file_path),
            filename=file.filename,
            client_id=current_user.client_id,
        )

        doc.status = DocumentStatus.COMPLETED
        doc.chunk_count = len(haystack_docs)
        await session.flush()

        logger.info(
            "document_upload_complete",
            doc_id=str(doc_id),
            filename=file.filename,
            chunk_count=len(haystack_docs),
            user_id=str(current_user.id),
        )
    except Exception:
        doc.status = DocumentStatus.FAILED
        await session.flush()
        logger.exception("document_ingestion_failed", doc_id=str(doc_id))
        raise

    return DocumentUploadResponse(
        id=doc_id,
        filename=file.filename,
        status=doc.status,
        chunk_count=doc.chunk_count,
        message=f"Document processed successfully. {doc.chunk_count} chunks indexed.",
    )
