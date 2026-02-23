from __future__ import annotations

import time
from uuid import uuid4

import pytest

from app.auth import _validate_jwt_claims
from app.exceptions import AuthError


def test_validate_jwt_claims_valid() -> None:
    """Valid claims should pass validation."""
    claims = {
        "sub": str(uuid4()),
        "iss": "http://localhost:8080/realms/docintel",
        "exp": int(time.time()) + 3600,
        "realm_access": {"roles": ["user"]},
    }
    result = _validate_jwt_claims(claims, expected_issuer="http://localhost:8080/realms/docintel")
    assert result is True


def test_validate_jwt_claims_expired() -> None:
    """Expired token should fail validation."""
    claims = {
        "sub": str(uuid4()),
        "iss": "http://localhost:8080/realms/docintel",
        "exp": int(time.time()) - 3600,
        "realm_access": {"roles": ["user"]},
    }
    with pytest.raises(AuthError, match="expired"):
        _validate_jwt_claims(claims, expected_issuer="http://localhost:8080/realms/docintel")


def test_validate_jwt_claims_wrong_issuer() -> None:
    """Wrong issuer should fail validation."""
    claims = {
        "sub": str(uuid4()),
        "iss": "http://evil.com/realms/docintel",
        "exp": int(time.time()) + 3600,
        "realm_access": {"roles": ["user"]},
    }
    with pytest.raises(AuthError, match="Invalid issuer"):
        _validate_jwt_claims(claims, expected_issuer="http://localhost:8080/realms/docintel")


def test_validate_jwt_claims_missing_subject() -> None:
    """Missing subject should fail validation."""
    claims = {
        "iss": "http://localhost:8080/realms/docintel",
        "exp": int(time.time()) + 3600,
    }
    with pytest.raises(AuthError, match="Missing subject"):
        _validate_jwt_claims(claims, expected_issuer="http://localhost:8080/realms/docintel")
