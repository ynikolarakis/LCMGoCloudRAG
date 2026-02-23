from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.audit import write_audit_log
from app.models.base import AuditAction


@pytest.mark.anyio
async def test_write_audit_log_creates_entry() -> None:
    """write_audit_log should add an AuditLog to the session."""
    session = AsyncMock()
    session.add = MagicMock()

    user_id = uuid4()
    resource_id = uuid4()

    await write_audit_log(
        session=session,
        user_id=user_id,
        action=AuditAction.DOCUMENT_UPLOAD,
        resource_type="document",
        resource_id=resource_id,
        details={"filename": "test.pdf"},
        ip_address="127.0.0.1",
        client_id="default",
    )

    session.add.assert_called_once()
    log_entry = session.add.call_args[0][0]
    assert log_entry.user_id == user_id
    assert log_entry.action == AuditAction.DOCUMENT_UPLOAD
    assert log_entry.resource_type == "document"
    assert str(log_entry.resource_id) == str(resource_id)
    assert log_entry.details == {"filename": "test.pdf"}
    assert log_entry.ip_address == "127.0.0.1"
    assert log_entry.client_id == "default"


@pytest.mark.anyio
async def test_write_audit_log_optional_fields() -> None:
    """write_audit_log should handle None optional fields."""
    session = AsyncMock()
    session.add = MagicMock()

    await write_audit_log(
        session=session,
        user_id=None,
        action=AuditAction.QUERY_SUBMITTED,
        resource_type="query",
    )

    session.add.assert_called_once()
    log_entry = session.add.call_args[0][0]
    assert log_entry.user_id is None
    assert log_entry.resource_id is None
    assert log_entry.details is None
    assert log_entry.ip_address is None
    assert log_entry.client_id == "default"
