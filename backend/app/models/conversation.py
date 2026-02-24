from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, generate_uuid


class Conversation(Base):
    """Persistent chat session linking a user to a sequence of queries.

    Each conversation groups a series of Query records under a single session,
    enabling chat history retrieval. Deletion cascades to all child queries.
    """

    __tablename__ = "conversations"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=generate_uuid)
    user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    client_id: Mapped[str] = mapped_column(String(100), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    queries: Mapped[list] = relationship("Query", back_populates="conversation", cascade="all, delete-orphan")
