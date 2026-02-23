from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import RoleChecker
from app.database import get_db_session
from app.models import AuditLog, User
from app.models.base import UserRole
from app.schemas.audit import AuditLogResponse, PaginatedAuditLogsResponse

logger = structlog.get_logger()
router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/audit-logs", response_model=PaginatedAuditLogsResponse)
async def list_audit_logs(
    page: int = 1,
    page_size: int = 20,
    action: str | None = None,
    current_user: User = Depends(RoleChecker(UserRole.ADMIN)),
    session: AsyncSession = Depends(get_db_session),
) -> PaginatedAuditLogsResponse:
    """List audit log entries with pagination and optional action filter.

    Args:
        page: 1-based page number (default 1).
        page_size: Number of entries per page (default 20).
        action: Optional AuditAction string to filter results by action type.
        current_user: Authenticated admin user injected by RoleChecker.
        session: Async database session injected by FastAPI.

    Returns:
        A paginated response containing audit log entries and metadata.
    """
    stmt = select(AuditLog)

    if action:
        stmt = stmt.where(AuditLog.action == action)

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await session.execute(count_stmt)
    total = total_result.scalar() or 0

    stmt = stmt.order_by(AuditLog.created_at.desc())
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)

    result = await session.execute(stmt)
    logs = result.scalars().all()

    logger.info(
        "audit_logs_listed",
        admin_id=str(current_user.id),
        page=page,
        page_size=page_size,
        action_filter=action,
        total=total,
    )

    return PaginatedAuditLogsResponse(
        items=[AuditLogResponse.model_validate(log) for log in logs],
        total=total,
        page=page,
        page_size=page_size,
    )
