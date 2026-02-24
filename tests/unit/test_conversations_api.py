from __future__ import annotations


def test_conversation_model_has_required_fields() -> None:
    """Conversation model should have id, user_id, title, client_id, timestamps."""
    from app.models.conversation import Conversation

    columns = {c.name for c in Conversation.__table__.columns}
    assert "id" in columns
    assert "user_id" in columns
    assert "title" in columns
    assert "client_id" in columns
    assert "created_at" in columns
    assert "updated_at" in columns


def test_query_has_conversation_id_column() -> None:
    """Query model should have a conversation_id FK."""
    from app.models.query import Query

    columns = {c.name for c in Query.__table__.columns}
    assert "conversation_id" in columns


def test_conversation_id_fk_references_conversations_table() -> None:
    """conversation_id on Query must reference the conversations table."""
    from app.models.query import Query

    fk_targets = {fk.target_fullname for col in Query.__table__.columns for fk in col.foreign_keys}
    assert "conversations.id" in fk_targets


def test_conversation_router_is_registered() -> None:
    """The conversations router should be included in the v1 router."""
    from app.api.v1.router import api_v1_router

    routes = [r.path for r in api_v1_router.routes]
    assert any("/conversations" in r for r in routes)


def test_conversation_model_is_exported_from_models_package() -> None:
    """Conversation must be importable from the top-level models package."""
    from app.models import Conversation

    assert Conversation.__tablename__ == "conversations"


def test_conversation_schema_create_title_is_optional() -> None:
    """ConversationCreate should allow creation with no title."""
    from app.schemas.conversation import ConversationCreate

    schema = ConversationCreate()
    assert schema.title is None

    schema_with_title = ConversationCreate(title="My chat")
    assert schema_with_title.title == "My chat"


def test_conversation_response_schema_from_attributes() -> None:
    """ConversationResponse should accept ORM model instances via model_validate."""
    import uuid
    from datetime import datetime, timezone

    from app.schemas.conversation import ConversationResponse

    now = datetime.now(tz=timezone.utc)

    class FakeConv:
        id = uuid.uuid4()
        title = "Test"
        client_id = "client-001"
        created_at = now
        updated_at = now

    resp = ConversationResponse.model_validate(FakeConv())
    assert resp.client_id == "client-001"
    assert resp.title == "Test"


def test_paginated_conversations_response_structure() -> None:
    """PaginatedConversationsResponse should carry items, total, page, page_size."""
    from app.schemas.conversation import PaginatedConversationsResponse

    resp = PaginatedConversationsResponse(items=[], total=0, page=1, page_size=20)
    assert resp.total == 0
    assert resp.page == 1
    assert resp.page_size == 20
    assert resp.items == []
