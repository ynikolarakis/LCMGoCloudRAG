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


def test_sparse_embedding_model_default() -> None:
    """Settings should have a default sparse embedding model."""
    settings = Settings(
        DATABASE_URL="postgresql+asyncpg://x:x@localhost/x",
        REDIS_URL="redis://localhost:6379/0",
    )
    assert settings.SPARSE_EMBEDDING_MODEL == "Qdrant/bm25"


def test_retriever_top_k_default_is_50() -> None:
    """Hybrid retriever should fetch top-50 candidates for reranking."""
    settings = Settings(
        DATABASE_URL="postgresql+asyncpg://x:x@localhost/x",
        REDIS_URL="redis://localhost:6379/0",
    )
    assert settings.RETRIEVER_TOP_K == 50


def test_reranker_settings_have_defaults() -> None:
    """Reranker settings should have sensible defaults."""
    settings = Settings(
        DATABASE_URL="postgresql+asyncpg://x:x@localhost/x",
        REDIS_URL="redis://localhost:6379/0",
    )
    assert settings.RERANKER_MODEL == "BAAI/bge-reranker-v2-m3"
    assert settings.RERANKER_TOP_K == 5
    assert settings.RERANKER_THRESHOLD == 0.3


def test_guardrail_settings_have_defaults() -> None:
    """Guardrail settings should have sensible defaults."""
    settings = Settings(
        DATABASE_URL="postgresql+asyncpg://x:x@localhost/x",
        REDIS_URL="redis://localhost:6379/0",
    )
    assert settings.GUARDRAILS_ENABLED is True
    assert settings.GUARDRAIL_INJECTION_THRESHOLD == 0.5
    assert settings.GUARDRAIL_HHEM_THRESHOLD == 0.5


def test_query_expansion_enabled_by_default() -> None:
    """Query expansion should be enabled by default."""
    from app.config import settings

    assert settings.QUERY_EXPANSION_ENABLED is True
