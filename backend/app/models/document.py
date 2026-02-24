from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, DocumentStatus, TimestampMixin, generate_uuid


class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=generate_uuid)
    filename: Mapped[str] = mapped_column(String(500), index=True)
    original_filename: Mapped[str] = mapped_column(String(500))
    file_size: Mapped[int] = mapped_column(Integer)
    content_type: Mapped[str] = mapped_column(String(100))
    s3_key: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    status: Mapped[DocumentStatus] = mapped_column(Enum(DocumentStatus), default=DocumentStatus.QUEUED, index=True)
    language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    client_id: Mapped[str] = mapped_column(String(100), index=True)
    uploaded_by: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    chunks: Mapped[list[DocumentChunk]] = relationship(back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=generate_uuid)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    chunk_seq: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    section_title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    page_num: Mapped[int | None] = mapped_column(Integer, nullable=True)
    language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    token_count: Mapped[int] = mapped_column(Integer)
    embedding_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    client_id: Mapped[str] = mapped_column(String(100), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship(back_populates="chunks")
