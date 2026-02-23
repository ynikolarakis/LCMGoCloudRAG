from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, Depends, File, Response, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit import write_audit_log
from app.auth import RoleChecker, get_current_user
from app.config import settings
from app.database import get_db_session
from app.exceptions import NotFoundError, ValidationError
from app.models import Document, DocumentStatus, User
from app.models.base import AuditAction, UserRole
from app.schemas.document import (
    DocumentDetailResponse,
    DocumentResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
    PaginatedDocumentsResponse,
)
from app.tasks.ingestion import ingest_document_task

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
    file to uploads/{client_id}/{doc_id}{suffix}, dispatches a Celery task for
    async ingestion, and writes an audit log entry. Returns 202 immediately.

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

    # 4. Dispatch Celery task for async ingestion
    ingest_document_task.apply_async(
        args=[str(doc_id), str(file_path), current_user.client_id],
    )

    logger.info(
        "document_upload_queued",
        doc_id=str(doc_id),
        filename=file.filename,
        user_id=str(current_user.id),
    )

    return DocumentUploadResponse(
        id=doc_id,
        filename=file.filename,
        status=doc.status,
        chunk_count=None,
        message="Document queued for processing.",
    )


@router.get("/{doc_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    doc_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> DocumentStatusResponse:
    """Get the processing status of a document.

    Lightweight endpoint for polling during async ingestion.
    Users see own docs; managers see all client docs.

    Args:
        doc_id: The UUID of the document to query.
        current_user: The authenticated user, injected by FastAPI.
        session: The async database session, injected by FastAPI.

    Returns:
        DocumentStatusResponse with current status, chunk count, and timestamps.

    Raises:
        NotFoundError: If the document does not exist or is not accessible
            by the current user.
    """
    doc_uuid = UUID(doc_id)
    stmt = select(Document).where(
        Document.id == doc_uuid,
        Document.client_id == current_user.client_id,
    )

    if current_user.role in (UserRole.USER, UserRole.VIEWER):
        stmt = stmt.where(Document.uploaded_by == current_user.id)

    result = await session.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise NotFoundError(f"Document {doc_id} not found")

    return DocumentStatusResponse(
        id=doc.id,
        status=doc.status,
        chunk_count=doc.chunk_count,
        page_count=doc.page_count,
        error_message=None,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.get("", response_model=PaginatedDocumentsResponse)
async def list_documents(
    page: int = 1,
    page_size: int = 20,
    status: str | None = None,
    language: str | None = None,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> PaginatedDocumentsResponse:
    """List documents with pagination and optional filters.

    Users see own docs; managers/admins see all client docs.
    Sorted by created_at descending.

    Args:
        page: The 1-based page number (default 1).
        page_size: Number of items per page (default 20).
        status: Optional filter by DocumentStatus value.
        language: Optional filter by language code (e.g. "el", "en").
        current_user: The authenticated user, injected by FastAPI.
        session: The async database session, injected by FastAPI.

    Returns:
        PaginatedDocumentsResponse with items, total count, and pagination info.
    """
    stmt = select(Document).where(Document.client_id == current_user.client_id)

    if current_user.role in (UserRole.USER, UserRole.VIEWER):
        stmt = stmt.where(Document.uploaded_by == current_user.id)

    if status:
        stmt = stmt.where(Document.status == status)
    if language:
        stmt = stmt.where(Document.language == language)

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await session.execute(count_stmt)
    total = total_result.scalar() or 0

    stmt = stmt.order_by(Document.created_at.desc())
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)

    result = await session.execute(stmt)
    docs = result.scalars().all()

    return PaginatedDocumentsResponse(
        items=[DocumentResponse.model_validate(d) for d in docs],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{doc_id}", response_model=DocumentDetailResponse)
async def get_document(
    doc_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> DocumentDetailResponse:
    """Get document details by ID with audit logging.

    Args:
        doc_id: The UUID of the document to retrieve.
        current_user: The authenticated user, injected by FastAPI.
        session: The async database session, injected by FastAPI.

    Returns:
        DocumentDetailResponse with full document metadata.

    Raises:
        NotFoundError: If the document does not exist or is not accessible
            by the current user.
    """
    doc_uuid = UUID(doc_id)
    stmt = select(Document).where(
        Document.id == doc_uuid,
        Document.client_id == current_user.client_id,
    )

    if current_user.role in (UserRole.USER, UserRole.VIEWER):
        stmt = stmt.where(Document.uploaded_by == current_user.id)

    result = await session.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise NotFoundError(f"Document {doc_id} not found")

    await write_audit_log(
        session=session,
        user_id=current_user.id,
        action=AuditAction.DOCUMENT_VIEW,
        resource_type="document",
        resource_id=doc.id,
        client_id=current_user.client_id,
    )

    return DocumentDetailResponse.model_validate(doc)


@router.delete("/{doc_id}", status_code=204, dependencies=[Depends(RoleChecker(UserRole.MANAGER))])
async def delete_document(
    doc_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    """Delete a document and all associated data.

    Cascade: Qdrant points -> DocumentChunk rows (FK cascade) -> Document row -> file on disk.
    Requires manager role or higher.

    Args:
        doc_id: The UUID of the document to delete.
        current_user: The authenticated user, injected by FastAPI.
        session: The async database session, injected by FastAPI.

    Returns:
        Empty 204 response on success.

    Raises:
        NotFoundError: If the document does not exist within the current client.
    """
    doc_uuid = UUID(doc_id)
    stmt = select(Document).where(
        Document.id == doc_uuid,
        Document.client_id == current_user.client_id,
    )

    result = await session.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise NotFoundError(f"Document {doc_id} not found")

    # 1. Delete Qdrant points for this document
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

        qdrant = QdrantClient(url=settings.QDRANT_URL, timeout=10)
        qdrant.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="meta.source",
                        match=MatchValue(value=doc.original_filename),
                    ),
                ],
            ),
        )
        qdrant.close()
        logger.info("qdrant_points_deleted", doc_id=doc_id, filename=doc.original_filename)
    except Exception:
        logger.warning("qdrant_delete_failed", doc_id=doc_id, exc_info=True)

    # 2. Delete Document (DocumentChunks cascade via FK ondelete=CASCADE)
    await session.delete(doc)
    await session.flush()

    # 3. Delete file from disk
    file_path = Path(settings.UPLOAD_DIR) / doc.client_id / doc.filename
    if file_path.exists():
        file_path.unlink()
        logger.info("file_deleted", path=str(file_path))

    # 4. Audit log
    await write_audit_log(
        session=session,
        user_id=current_user.id,
        action=AuditAction.DOCUMENT_DELETE,
        resource_type="document",
        resource_id=doc_uuid,
        details={"filename": doc.original_filename, "file_size": doc.file_size},
        client_id=current_user.client_id,
    )

    logger.info(
        "document_deleted",
        doc_id=doc_id,
        filename=doc.original_filename,
        user_id=str(current_user.id),
    )

    return Response(status_code=204)
