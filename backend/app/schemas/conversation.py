from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ConversationCreate(BaseModel):
    """Request body for creating a new conversation."""

    title: str | None = None


class ConversationResponse(BaseModel):
    """Response schema for a single conversation."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str | None
    client_id: str
    created_at: datetime
    updated_at: datetime


class ConversationMessageResponse(BaseModel):
    """A single message in a conversation thread (user or assistant turn)."""

    role: str
    content: str
    created_at: datetime


class PaginatedConversationsResponse(BaseModel):
    """Paginated list of conversations."""

    items: list[ConversationResponse]
    total: int
    page: int
    page_size: int
