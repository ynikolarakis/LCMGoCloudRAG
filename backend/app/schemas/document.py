from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.models.base import DocumentStatus


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    status: DocumentStatus
    chunk_count: int | None = None
    message: str


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    original_filename: str
    file_size: int
    content_type: str
    status: DocumentStatus
    language: str | None = None
    page_count: int | None = None
    chunk_count: int | None = None
    client_id: str
    created_at: datetime


class DocumentListRequest(BaseModel):
    page: int = 1
    page_size: int = 20
    status: DocumentStatus | None = None
    language: str | None = None


class DocumentDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    original_filename: str
    file_size: int
    content_type: str
    status: DocumentStatus
    language: str | None = None
    page_count: int | None = None
    chunk_count: int | None = None
    client_id: str
    uploaded_by: UUID | None = None
    created_at: datetime
    updated_at: datetime


class DocumentStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: DocumentStatus
    chunk_count: int | None = None
    page_count: int | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class PaginatedDocumentsResponse(BaseModel):
    items: list[DocumentResponse]
    total: int
    page: int
    page_size: int
