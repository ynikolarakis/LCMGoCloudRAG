from __future__ import annotations

from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.database import get_db_session
from app.models import Query, QueryResponse, User
from app.models.conversation import Conversation
from app.schemas.conversation import (
    ConversationCreate,
    ConversationMessageResponse,
    ConversationResponse,
    PaginatedConversationsResponse,
)

logger = structlog.get_logger()
router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    body: ConversationCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> ConversationResponse:
    """Create a new conversation for the current user.

    Args:
        body: Optional title for the conversation.
        current_user: The authenticated user from the JWT dependency.
        session: Async database session from the connection pool.

    Returns:
        The newly created conversation record.
    """
    conv = Conversation(
        user_id=current_user.id,
        title=body.title,
        client_id=current_user.client_id,
    )
    session.add(conv)
    await session.flush()
    logger.info("conversation_created", conv_id=str(conv.id), user_id=str(current_user.id))
    return ConversationResponse.model_validate(conv)


@router.get("", response_model=PaginatedConversationsResponse)
async def list_conversations(
    page: int = 1,
    page_size: int = 20,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> PaginatedConversationsResponse:
    """List the current user's conversations, most recent first.

    Args:
        page: 1-based page number.
        page_size: Number of results per page (max 100).
        current_user: The authenticated user from the JWT dependency.
        session: Async database session from the connection pool.

    Returns:
        Paginated list of the user's conversations.
    """
    stmt = select(Conversation).where(
        Conversation.user_id == current_user.id,
        Conversation.client_id == current_user.client_id,
    )

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total: int = (await session.execute(count_stmt)).scalar() or 0

    stmt = stmt.order_by(Conversation.updated_at.desc())
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    result = await session.execute(stmt)
    convs = result.scalars().all()

    return PaginatedConversationsResponse(
        items=[ConversationResponse.model_validate(c) for c in convs],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{conv_id}/messages", response_model=list[ConversationMessageResponse])
async def get_conversation_messages(
    conv_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> list[ConversationMessageResponse]:
    """Return the full message thread for a conversation.

    Messages are interleaved user/assistant turns sorted by creation time.

    Args:
        conv_id: UUID string of the target conversation.
        current_user: The authenticated user from the JWT dependency.
        session: Async database session from the connection pool.

    Raises:
        HTTPException: 404 if the conversation is not found or does not belong to the user.

    Returns:
        Ordered list of user and assistant messages.
    """
    conv_uuid = UUID(conv_id)
    stmt = select(Conversation).where(
        Conversation.id == conv_uuid,
        Conversation.user_id == current_user.id,
    )
    result = await session.execute(stmt)
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail=f"Conversation {conv_id} not found")

    query_stmt = (
        select(Query)
        .options(selectinload(Query.response))
        .where(Query.conversation_id == conv_uuid)
        .order_by(Query.created_at.asc())
    )
    query_result = await session.execute(query_stmt)
    queries = query_result.scalars().all()

    messages: list[ConversationMessageResponse] = []
    for q in queries:
        messages.append(
            ConversationMessageResponse(
                role="user",
                content=q.query_text,
                created_at=q.created_at,
            )
        )
        if q.response:
            messages.append(
                ConversationMessageResponse(
                    role="assistant",
                    content=q.response.response_text,
                    created_at=q.response.created_at,
                )
            )
    return messages


@router.delete("/{conv_id}", status_code=204)
async def delete_conversation(
    conv_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    """Delete a conversation and all its associated queries.

    The cascade is handled by the ORM relationship defined on Conversation.

    Args:
        conv_id: UUID string of the conversation to delete.
        current_user: The authenticated user from the JWT dependency.
        session: Async database session from the connection pool.

    Raises:
        HTTPException: 404 if the conversation is not found or does not belong to the user.

    Returns:
        Empty 204 response on success.
    """
    conv_uuid = UUID(conv_id)
    stmt = select(Conversation).where(
        Conversation.id == conv_uuid,
        Conversation.user_id == current_user.id,
    )
    result = await session.execute(stmt)
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail=f"Conversation {conv_id} not found")

    await session.delete(conv)
    logger.info("conversation_deleted", conv_id=conv_id, user_id=str(current_user.id))
    return Response(status_code=204)
