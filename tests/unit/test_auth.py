from __future__ import annotations

from uuid import UUID

import pytest

from app.auth import RoleChecker, get_current_user_dev
from app.exceptions import ForbiddenError
from app.models import User, UserRole


@pytest.mark.anyio
async def test_get_current_user_dev_returns_user() -> None:
    """Dev auth should return a User object with admin role."""
    user = await get_current_user_dev()
    assert isinstance(user, User)
    assert user.email == "dev@docintel.local"
    assert user.role == UserRole.ADMIN
    assert user.client_id == "default"
    assert isinstance(user.id, UUID)


@pytest.mark.anyio
async def test_role_checker_allows_sufficient_role() -> None:
    """RoleChecker should pass when user has sufficient role."""
    checker = RoleChecker(minimum_role=UserRole.USER)
    user = User(
        email="test@test.com",
        full_name="Test",
        role=UserRole.ADMIN,
        client_id="default",
    )
    result = await checker(user)
    assert result == user


@pytest.mark.anyio
async def test_role_checker_blocks_insufficient_role() -> None:
    """RoleChecker should raise ForbiddenError for insufficient role."""
    checker = RoleChecker(minimum_role=UserRole.MANAGER)
    user = User(
        email="test@test.com",
        full_name="Test",
        role=UserRole.VIEWER,
        client_id="default",
    )
    with pytest.raises(ForbiddenError):
        await checker(user)


@pytest.mark.anyio
async def test_role_checker_allows_exact_role() -> None:
    """RoleChecker should pass when user has exactly the minimum role."""
    checker = RoleChecker(minimum_role=UserRole.USER)
    user = User(
        email="test@test.com",
        full_name="Test",
        role=UserRole.USER,
        client_id="default",
    )
    result = await checker(user)
    assert result == user
