from __future__ import annotations

from app.config import Settings


def test_keycloak_settings_have_defaults() -> None:
    """Keycloak settings should have sensible defaults for dev."""
    s = Settings(
        DATABASE_URL="postgresql+asyncpg://x:x@localhost/x",
        REDIS_URL="redis://localhost:6379/0",
    )
    assert s.KEYCLOAK_URL == "http://localhost:8080"
    assert s.KEYCLOAK_REALM == "docintel"
    assert s.KEYCLOAK_CLIENT_ID == "docintel-api"
    assert s.KEYCLOAK_CLIENT_SECRET is None


def test_celery_broker_defaults_to_redis() -> None:
    """Celery broker should default to REDIS_URL."""
    s = Settings(
        DATABASE_URL="postgresql+asyncpg://x:x@localhost/x",
        REDIS_URL="redis://localhost:6379/0",
    )
    assert s.CELERY_BROKER_URL == "redis://localhost:6379/0"
    assert s.CELERY_RESULT_BACKEND == "redis://localhost:6379/0"


def test_upload_dir_default() -> None:
    """UPLOAD_DIR should default to ./uploads."""
    s = Settings(
        DATABASE_URL="postgresql+asyncpg://x:x@localhost/x",
        REDIS_URL="redis://localhost:6379/0",
    )
    assert s.UPLOAD_DIR == "./uploads"
