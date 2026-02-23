from __future__ import annotations

from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AuditLog
from app.models.base import AuditAction

logger = structlog.get_logger()


async def write_audit_log(
    session: AsyncSession,
    user_id: UUID | None,
    action: AuditAction,
    resource_type: str,
    resource_id: UUID | None = None,
    details: dict | None = None,
    ip_address: str | None = None,
    client_id: str = "default",
) -> None:
    """Write an immutable audit log entry.

    Adds an AuditLog row to the session but does NOT commit.
    The caller is responsible for committing the session as part of a
    larger transaction, ensuring atomicity with the primary operation.

    Args:
        session: The async SQLAlchemy session to add the entry to.
        user_id: UUID of the acting user, or None for system/anonymous actions.
        action: The AuditAction enum value describing what happened.
        resource_type: A short string identifying the resource kind (e.g. "document", "query").
        resource_id: Optional UUID of the specific resource being acted upon.
        details: Optional dict of additional context stored as JSONB.
        ip_address: Optional originating IP address of the request.
        client_id: Tenant identifier; defaults to "default".
    """
    entry = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=str(resource_id) if resource_id else None,
        details=details,
        ip_address=ip_address,
        client_id=client_id,
    )
    session.add(entry)
    logger.info(
        "audit_log_written",
        action=action.value,
        resource_type=resource_type,
        resource_id=str(resource_id) if resource_id else None,
        user_id=str(user_id) if user_id else None,
    )
