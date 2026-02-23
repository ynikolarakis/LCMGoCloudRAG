from __future__ import annotations

import datetime
from uuid import uuid4

from app.models.base import DocumentStatus
from app.schemas.document import (
    DocumentDetailResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
    PaginatedDocumentsResponse,
)


def test_upload_response_accepts_document_status_enum() -> None:
    """DocumentUploadResponse should accept DocumentStatus enum values."""
    resp = DocumentUploadResponse(
        id=uuid4(),
        filename="test.pdf",
        status=DocumentStatus.QUEUED,
        chunk_count=None,
        message="Queued for processing",
    )
    assert resp.status == DocumentStatus.QUEUED


def test_document_status_response_schema() -> None:
    """DocumentStatusResponse should accept valid data."""
    resp = DocumentStatusResponse(
        id=uuid4(),
        status=DocumentStatus.PROCESSING,
        chunk_count=None,
        page_count=None,
        error_message=None,
        created_at=datetime.datetime.now(tz=datetime.UTC),
        updated_at=datetime.datetime.now(tz=datetime.UTC),
    )
    assert resp.status == DocumentStatus.PROCESSING


def test_document_detail_response_schema() -> None:
    """DocumentDetailResponse should include uploaded_by."""
    resp = DocumentDetailResponse(
        id=uuid4(),
        filename="stored.pdf",
        original_filename="test.pdf",
        file_size=1024,
        content_type="application/pdf",
        status=DocumentStatus.COMPLETED,
        client_id="default",
        uploaded_by=uuid4(),
        created_at=datetime.datetime.now(tz=datetime.UTC),
        updated_at=datetime.datetime.now(tz=datetime.UTC),
    )
    assert resp.uploaded_by is not None


def test_paginated_response_schema() -> None:
    """PaginatedDocumentsResponse should have items, total, page, page_size."""
    resp = PaginatedDocumentsResponse(
        items=[],
        total=0,
        page=1,
        page_size=20,
    )
    assert resp.items == []
    assert resp.total == 0
