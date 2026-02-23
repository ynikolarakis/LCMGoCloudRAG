from __future__ import annotations

from uuid import UUID

import structlog
from fastapi import Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.exceptions import AuthError, ForbiddenError
from app.models import User, UserRole

logger = structlog.get_logger()

# Role hierarchy: ADMIN > MANAGER > USER > VIEWER
_ROLE_HIERARCHY: dict[UserRole, int] = {
    UserRole.ADMIN: 40,
    UserRole.MANAGER: 30,
    UserRole.USER: 20,
    UserRole.VIEWER: 10,
}

# Stable UUID for dev user (deterministic for tests)
DEV_USER_ID = UUID("00000000-0000-0000-0000-000000000001")


async def get_current_user_dev() -> User:
    """Dev-mode auth bypass: return a synthetic admin user without a DB hit.

    Returns:
        A User instance with ADMIN role and the stable dev user ID.
    """
    return User(
        id=DEV_USER_ID,
        email="dev@docintel.local",
        full_name="Dev Admin",
        role=UserRole.ADMIN,
        keycloak_id=None,
        client_id=settings.CLIENT_ID,
        is_active=True,
    )


async def get_current_user_prod(request: Request) -> User:
    """Prod-mode auth: validate JWT from request and return the User from DB.

    Args:
        request: The incoming FastAPI request carrying the Authorization header.

    Raises:
        AuthError: Always — JWT validation is not yet implemented.

    Returns:
        The authenticated User model once JWT validation is implemented.
    """
    raise AuthError("JWT validation not implemented — set ENVIRONMENT=dev")


async def get_current_user(request: Request) -> User:
    """Auth dependency dispatcher: dev bypass or prod JWT validation.

    Selects the appropriate auth strategy based on the ENVIRONMENT setting.
    When ENVIRONMENT=dev, returns a synthetic admin user with no DB hit.
    In all other environments, validates the JWT token from the request.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The authenticated User model.
    """
    if settings.ENVIRONMENT == "dev":
        return await get_current_user_dev()
    return await get_current_user_prod(request)


async def seed_dev_user(session: AsyncSession) -> None:
    """Create the dev-admin user in the database if it does not already exist.

    Safe to call multiple times — performs an existence check before inserting.

    Args:
        session: An open async SQLAlchemy session. The caller is responsible
            for committing or rolling back on error.
    """
    result = await session.execute(select(User).where(User.id == DEV_USER_ID))
    existing = result.scalar_one_or_none()
    if existing:
        logger.info("dev_user_exists", user_id=str(DEV_USER_ID))
        return

    dev_user = User(
        id=DEV_USER_ID,
        email="dev@docintel.local",
        full_name="Dev Admin",
        role=UserRole.ADMIN,
        keycloak_id=None,
        client_id=settings.CLIENT_ID,
        is_active=True,
    )
    session.add(dev_user)
    await session.commit()
    logger.info("dev_user_created", user_id=str(DEV_USER_ID))


class RoleChecker:
    """FastAPI dependency that enforces a minimum role level.

    Use as a dependency on any route that requires role-based access control::

        @router.get("/admin-only")
        async def admin_route(user: User = Depends(RoleChecker(UserRole.ADMIN))):
            ...

    The role hierarchy is: ADMIN(40) > MANAGER(30) > USER(20) > VIEWER(10).
    """

    def __init__(self, minimum_role: UserRole) -> None:
        """Initialise the checker with the minimum required role.

        Args:
            minimum_role: The lowest role level that is permitted access.
        """
        self.minimum_role = minimum_role

    async def __call__(self, user: User = Depends(get_current_user)) -> User:
        """Validate the current user's role against the minimum requirement.

        Args:
            user: The currently authenticated user, injected by FastAPI.

        Raises:
            ForbiddenError: When the user's role level is below the minimum.

        Returns:
            The authenticated user if the role check passes.
        """
        user_level = _ROLE_HIERARCHY.get(user.role, 0)
        required_level = _ROLE_HIERARCHY.get(self.minimum_role, 0)
        if user_level < required_level:
            raise ForbiddenError(
                f"Role '{user.role.value}' insufficient. Requires '{self.minimum_role.value}' or higher."
            )
        return user
