from __future__ import annotations

import time
from uuid import UUID

import httpx
import structlog
from fastapi import Depends, Request
from jose import JWTError
from jose import jwt as jose_jwt
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


_jwks_cache: dict[str, tuple[dict, float]] = {}
JWKS_CACHE_TTL = 3600  # 1 hour


def _validate_jwt_claims(claims: dict, expected_issuer: str) -> bool:
    """Validate standard JWT claims (expiration, issuer, subject).

    Args:
        claims: Decoded JWT claims dict.
        expected_issuer: The expected issuer URL.

    Returns:
        True if all claims are valid.

    Raises:
        AuthError: If any claim is invalid.
    """
    exp = claims.get("exp")
    if not exp or exp < time.time():
        raise AuthError("Token has expired")

    iss = claims.get("iss")
    if iss != expected_issuer:
        raise AuthError(f"Invalid issuer: {iss}")

    if not claims.get("sub"):
        raise AuthError("Missing subject claim")

    return True


async def _fetch_jwks(keycloak_url: str, realm: str) -> dict:
    """Fetch JWKS keys from Keycloak with 1-hour cache.

    Args:
        keycloak_url: Base Keycloak URL.
        realm: Keycloak realm name.

    Returns:
        JWKS key set as a dict.
    """
    cache_key = f"{keycloak_url}/realms/{realm}"
    now = time.time()

    if cache_key in _jwks_cache:
        cached_keys, cached_at = _jwks_cache[cache_key]
        if now - cached_at < JWKS_CACHE_TTL:
            return cached_keys

    jwks_url = f"{keycloak_url}/realms/{realm}/protocol/openid-connect/certs"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(jwks_url)
        resp.raise_for_status()
        keys = resp.json()

    _jwks_cache[cache_key] = (keys, now)
    return keys


def _decode_jwt(token: str, jwks: dict, audience: str | None = None) -> dict:
    """Decode and verify a JWT using JWKS keys.

    Args:
        token: Raw JWT string.
        jwks: JWKS key set from Keycloak.
        audience: Expected audience (optional).

    Returns:
        Decoded JWT claims dict.

    Raises:
        AuthError: If token is invalid.
    """
    try:
        unverified_header = jose_jwt.get_unverified_header(token)
    except JWTError as e:
        raise AuthError(f"Invalid token header: {e}") from e

    rsa_key = {}
    for key in jwks.get("keys", []):
        if key.get("kid") == unverified_header.get("kid"):
            rsa_key = key
            break

    if not rsa_key:
        raise AuthError("Unable to find matching key in JWKS")

    try:
        payload = jose_jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=audience,
            options={"verify_aud": audience is not None},
        )
        return payload
    except JWTError as e:
        raise AuthError(f"Token validation failed: {e}") from e


async def get_current_user_prod(request: Request) -> User:
    """Prod-mode auth: validate JWT from Keycloak and return User from DB.

    Extracts Bearer token from Authorization header, validates against
    Keycloak JWKS, and looks up the user by keycloak_id.

    Args:
        request: The incoming FastAPI request carrying the Authorization header.

    Raises:
        AuthError: If the token is missing, invalid, or the user is not found.

    Returns:
        The authenticated User model.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise AuthError("Missing or invalid Authorization header")

    token = auth_header.split(" ", 1)[1]

    jwks = await _fetch_jwks(settings.KEYCLOAK_URL, settings.KEYCLOAK_REALM)
    claims = _decode_jwt(token, jwks)

    expected_issuer = f"{settings.KEYCLOAK_URL}/realms/{settings.KEYCLOAK_REALM}"
    _validate_jwt_claims(claims, expected_issuer=expected_issuer)

    keycloak_id = claims["sub"]

    from app.database import async_session_factory

    async with async_session_factory() as session:
        result = await session.execute(select(User).where(User.keycloak_id == keycloak_id))
        user = result.scalar_one_or_none()
        if not user:
            raise AuthError(f"User with keycloak_id '{keycloak_id}' not found")
        if not user.is_active:
            raise AuthError("User account is deactivated")
        return user


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
