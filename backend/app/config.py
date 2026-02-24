from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env from project root (parent of backend/)
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), env_file_encoding="utf-8", extra="ignore")

    # App
    ENVIRONMENT: str = "dev"
    LOG_LEVEL: str = "DEBUG"
    SECRET_KEY: SecretStr = SecretStr("dev-secret-key-change-in-production")
    CLIENT_ID: str = "default"

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://ragadmin:devpassword@localhost:5432/docintel"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "default_documents"

    # LLM (Ollama dev / vLLM prod — same OpenAI-compatible API)
    LLM_BASE_URL: str = "http://localhost:11434/v1"
    LLM_MODEL: str = "qwen3:30b-a3b"

    # Embeddings
    EMBEDDING_BASE_URL: str = "http://localhost:11434/v1"
    EMBEDDING_MODEL: str = "qwen3-embedding:0.6b"
    EMBEDDING_DIMENSION: int = 1024

    # Upload
    UPLOAD_DIR: str = "./uploads"

    # RAG
    CHUNK_TARGET_TOKENS: int = 450
    CHUNK_MAX_TOKENS: int = 512
    CHUNK_MIN_TOKENS: int = 200
    CHUNK_OVERLAP_PCT: float = 0.15
    RETRIEVER_TOP_K: int = 50
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024
    LLM_STREAM_TIMEOUT: int = 120  # seconds
    CONVERSATION_CONTEXT_MESSAGES: int = 5  # number of past messages to include

    # Sparse Embedding
    SPARSE_EMBEDDING_MODEL: str = "Qdrant/bm25"

    # Reranker
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_TOP_K: int = 5
    RERANKER_THRESHOLD: float = 0.3

    # Guardrails
    GUARDRAILS_ENABLED: bool = True
    GUARDRAIL_INJECTION_THRESHOLD: float = 0.5
    GUARDRAIL_HHEM_THRESHOLD: float = 0.5

    # Query Expansion
    QUERY_EXPANSION_ENABLED: bool = True

    # Keycloak
    KEYCLOAK_URL: str = "http://localhost:8080"
    KEYCLOAK_REALM: str = "docintel"
    KEYCLOAK_CLIENT_ID: str = "docintel-api"
    KEYCLOAK_CLIENT_SECRET: str | None = None

    # Celery
    CELERY_BROKER_URL: str = ""
    CELERY_RESULT_BACKEND: str = ""

    def model_post_init(self, __context: object) -> None:
        """Set Celery URLs from Redis if not explicitly provided."""
        if not self.CELERY_BROKER_URL:
            object.__setattr__(self, "CELERY_BROKER_URL", self.REDIS_URL)
        if not self.CELERY_RESULT_BACKEND:
            object.__setattr__(self, "CELERY_RESULT_BACKEND", self.REDIS_URL)


settings = Settings()
