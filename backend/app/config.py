from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # App
    ENVIRONMENT: str = "dev"
    LOG_LEVEL: str = "DEBUG"
    SECRET_KEY: SecretStr = SecretStr("dev-secret-key-change-in-production")
    CLIENT_ID: str = "default"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://ragadmin:devpassword@localhost:5432/docintel"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "default_documents"

    # LLM (Ollama dev / vLLM prod — same OpenAI-compatible API)
    LLM_BASE_URL: str = "http://host.docker.internal:11434/v1"
    LLM_MODEL: str = "qwen3:30b-a3b"

    # Embeddings
    EMBEDDING_BASE_URL: str = "http://host.docker.internal:11434/v1"
    EMBEDDING_MODEL: str = "qwen3-embedding:0.6b"
    EMBEDDING_DIMENSION: int = 512

    # Upload
    UPLOAD_DIR: str = "./uploads"

    # RAG
    CHUNK_TARGET_TOKENS: int = 450
    CHUNK_MAX_TOKENS: int = 512
    CHUNK_MIN_TOKENS: int = 200
    CHUNK_OVERLAP_PCT: float = 0.15
    RETRIEVER_TOP_K: int = 5
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024

    # Keycloak
    KEYCLOAK_URL: str = "http://localhost:8080"


settings = Settings()
