from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, generate_uuid


class Query(Base):
    __tablename__ = "queries"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=generate_uuid)
    user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    query_text: Mapped[str] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    client_id: Mapped[str] = mapped_column(String(100), index=True)
    conversation_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    response: Mapped[QueryResponse | None] = relationship(back_populates="query", uselist=False)
    conversation: Mapped[object | None] = relationship("Conversation", back_populates="queries")


class QueryResponse(Base):
    __tablename__ = "query_responses"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=generate_uuid)
    query_id: Mapped[UUID] = mapped_column(ForeignKey("queries.id"), unique=True)
    response_text: Mapped[str] = mapped_column(Text)
    citations: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    faithfulness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_used: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    query: Mapped[Query] = relationship(back_populates="response")
