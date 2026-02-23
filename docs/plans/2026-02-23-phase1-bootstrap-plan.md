# Phase 1 Bootstrap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bootstrap the LCM DocIntel RAG platform with working project scaffolding, FastAPI backend, Docker dev stack, and a minimal Haystack 2.x RAG pipeline (PDF upload + query with citations).

**Architecture:** FastAPI backend with Haystack 2.x pipelines for ingestion (Docling → chunk → embed → Qdrant) and query (embed → retrieve → generate with citations). Docker Compose for dev services (Qdrant, PostgreSQL, Redis, Keycloak). Ollama on host for LLM + embeddings via OpenAI-compatible API.

**Tech Stack:** Python 3.12, FastAPI, Haystack 2.x, Docling, Qdrant, PostgreSQL (SQLAlchemy 2.0 async), Ollama (Qwen3-30B + Qwen3-Embedding), Docker Compose, Alembic, structlog, tiktoken.

---

## Dependency Graph

```
Tasks 1, 2, 15, 16, 17 → PARALLEL (scaffolding, no dependencies)
Task 3 → after Task 1
Tasks 4, 5, 7 → after Task 3 (can be parallel with each other)
Task 6 → after Task 5
Task 8 → after Task 5
Task 9 → after Tasks 4, 5, 7, 8
Tasks 10, 11 → after Task 4 (pipeline code, independent of FastAPI)
Task 12 → after Tasks 9, 10, 11
Task 13 → after Task 4
Task 14 → after Task 9
Task 18 → after ALL tasks
```

---

### Task 1: Project Scaffolding

**Parallel-safe:** Yes — no dependencies.
**Agent:** Any

**Files:**
- Create: all directories, `.gitignore`, `pyproject.toml`, empty `__init__.py` files

**Step 1: Create directory structure**

```bash
# Backend
mkdir -p backend/app/{api/v1,models,schemas,pipelines,middleware}
mkdir -p backend/workers
mkdir -p backend/alembic/versions

# Frontend (placeholder — Task 15 initializes properly)
mkdir -p frontend

# Terraform
mkdir -p terraform/modules/{vpc,compute,database,storage,security,monitoring}
mkdir -p terraform/environments/dev

# Docker
mkdir -p docker

# Tests
mkdir -p tests/{unit,integration,evaluation}

# Scripts & Docs
mkdir -p scripts
mkdir -p docs/plans
```

**Step 2: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
venv/
*.egg

# Environment
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Node
node_modules/
.next/
out/

# Terraform
.terraform/
*.tfstate
*.tfstate.backup
*.tfvars
!*.tfvars.example

# Docker
docker/data/

# Test
.coverage
htmlcov/
.pytest_cache/

# Uploads
uploads/
temp/
```

**Step 3: Create `pyproject.toml`** (Ruff config)

```toml
[project]
name = "docintel"
version = "0.1.0"
requires-python = ">=3.12"

[tool.ruff]
target-version = "py312"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "SIM", "TCH"]
ignore = ["E501"]

[tool.ruff.lint.isort]
known-first-party = ["app", "workers"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Step 4: Create all `__init__.py` files**

Create empty `__init__.py` in:
- `backend/app/__init__.py`
- `backend/app/api/__init__.py`
- `backend/app/api/v1/__init__.py`
- `backend/app/models/__init__.py`
- `backend/app/schemas/__init__.py`
- `backend/app/pipelines/__init__.py`
- `backend/app/middleware/__init__.py`
- `backend/workers/__init__.py`
- `tests/__init__.py`
- `tests/unit/__init__.py`
- `tests/integration/__init__.py`
- `tests/evaluation/__init__.py`

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: project scaffolding with directory structure"
```

---

### Task 2: Docker Dev Stack

**Parallel-safe:** Yes — no dependencies.
**Agent:** devops

**Files:**
- Create: `docker/docker-compose.dev.yml`
- Create: `.env.example`

**Step 1: Create `docker/docker-compose.dev.yml`**

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:6333/healthz"]
      interval: 10s
      timeout: 5s
      retries: 5

  postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-docintel}
      POSTGRES_USER: ${POSTGRES_USER:-ragadmin}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-devpassword}
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-ragadmin} -d ${POSTGRES_DB:-docintel}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  keycloak:
    image: quay.io/keycloak/keycloak:latest
    ports:
      - "8080:8080"
    environment:
      KC_BOOTSTRAP_ADMIN_USERNAME: ${KEYCLOAK_ADMIN:-admin}
      KC_BOOTSTRAP_ADMIN_PASSWORD: ${KEYCLOAK_ADMIN_PASSWORD:-admin}
      KC_DB: postgres
      KC_DB_URL: jdbc:postgresql://postgres:5432/${POSTGRES_DB:-docintel}
      KC_DB_USERNAME: ${POSTGRES_USER:-ragadmin}
      KC_DB_PASSWORD: ${POSTGRES_PASSWORD:-devpassword}
    command: start-dev
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  qdrant_data:
  postgres_data:
  redis_data:
```

**Step 2: Create `.env.example`**

```env
# === Database ===
POSTGRES_DB=docintel
POSTGRES_USER=ragadmin
POSTGRES_PASSWORD=devpassword
DATABASE_URL=postgresql+asyncpg://ragadmin:devpassword@localhost:5432/docintel

# === Redis ===
REDIS_URL=redis://localhost:6379/0

# === Qdrant ===
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=default_documents

# === LLM (Ollama on Windows host) ===
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_MODEL=qwen3:30b-a3b

# === Embeddings (Ollama on Windows host) ===
EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1
EMBEDDING_MODEL=qwen3-embedding:0.6b
EMBEDDING_DIMENSION=512

# === Keycloak ===
KEYCLOAK_ADMIN=admin
KEYCLOAK_ADMIN_PASSWORD=admin
KEYCLOAK_URL=http://localhost:8080

# === App ===
SECRET_KEY=dev-secret-key-change-in-production
CLIENT_ID=default
ENVIRONMENT=dev
LOG_LEVEL=DEBUG
UPLOAD_DIR=./uploads
```

**Step 3: Commit**

```bash
git add docker/docker-compose.dev.yml .env.example
git commit -m "chore: docker compose dev stack with Qdrant, PostgreSQL, Redis, Keycloak"
```

---

### Task 3: Backend Dependencies

**Depends on:** Task 1
**Agent:** python-backend

**Files:**
- Create: `backend/requirements.txt`

**Step 1: Create `backend/requirements.txt`**

```txt
# Core
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.0
pydantic-settings>=2.0
python-multipart>=0.0.9

# Database
sqlalchemy[asyncio]>=2.0
asyncpg>=0.29.0
alembic>=1.13.0

# Haystack RAG
haystack-ai>=2.8.0
qdrant-haystack>=6.0.0

# LLM / Embeddings (OpenAI-compatible client for Ollama/vLLM)
openai>=1.0.0

# Document Processing
docling>=2.0.0

# Chunking
tiktoken>=0.7.0

# Task Queue (placeholder — Celery async in Phase 2)
celery[redis]>=5.4.0
redis>=5.0.0

# Logging
structlog>=24.0.0

# HTTP Client
httpx>=0.27.0

# Dev / Test
pytest>=8.0
pytest-asyncio>=0.23.0
ruff>=0.8.0
```

**Step 2: Commit**

```bash
git add backend/requirements.txt
git commit -m "chore: backend Python dependencies"
```

---

### Task 4: Backend Config & Logging

**Depends on:** Task 3
**Agent:** python-backend

**Files:**
- Create: `backend/app/config.py`
- Create: `backend/app/logging_config.py`

**Step 1: Create `backend/app/config.py`**

```python
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
```

**Step 2: Create `backend/app/logging_config.py`**

```python
from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(log_level: str = "DEBUG") -> None:
    """Configure structlog for JSON structured logging."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer() if log_level == "DEBUG" else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.DEBUG),
    )
```

**Step 3: Commit**

```bash
git add backend/app/config.py backend/app/logging_config.py
git commit -m "feat: application settings and structlog configuration"
```

---

### Task 5: Database Models & Engine

**Depends on:** Task 3
**Agent:** python-backend

**Files:**
- Create: `backend/app/database.py`
- Create: `backend/app/models/base.py`
- Create: `backend/app/models/user.py`
- Create: `backend/app/models/document.py`
- Create: `backend/app/models/query.py`
- Create: `backend/app/models/audit.py`
- Modify: `backend/app/models/__init__.py`

**Step 1: Create `backend/app/database.py`**

```python
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=5,
    max_overflow=15,
    echo=settings.ENVIRONMENT == "dev",
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

**Step 2: Create `backend/app/models/base.py`**

```python
from __future__ import annotations

import enum
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    VIEWER = "viewer"


class DocumentStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AuditAction(str, enum.Enum):
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_VIEW = "document_view"
    DOCUMENT_DELETE = "document_delete"
    QUERY_SUBMITTED = "query_submitted"
    RESPONSE_GENERATED = "response_generated"
    GUARDRAIL_TRIGGERED = "guardrail_triggered"
    ADMIN_ACTION = "admin_action"


class TimestampMixin:
    """Mixin that adds created_at and updated_at columns."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


def generate_uuid() -> UUID:
    return uuid4()
```

**Step 3: Create `backend/app/models/user.py`**

```python
from __future__ import annotations

from uuid import UUID

from sqlalchemy import Boolean, Enum, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UserRole, generate_uuid


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=generate_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.USER)
    keycloak_id: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    client_id: Mapped[str] = mapped_column(String(100), index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
```

**Step 4: Create `backend/app/models/document.py`**

```python
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, DocumentStatus, TimestampMixin, generate_uuid


class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=generate_uuid)
    filename: Mapped[str] = mapped_column(String(500), index=True)
    original_filename: Mapped[str] = mapped_column(String(500))
    file_size: Mapped[int] = mapped_column(Integer)
    content_type: Mapped[str] = mapped_column(String(100))
    s3_key: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus), default=DocumentStatus.QUEUED, index=True
    )
    language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    client_id: Mapped[str] = mapped_column(String(100), index=True)
    uploaded_by: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    chunks: Mapped[list[DocumentChunk]] = relationship(back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=generate_uuid)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    chunk_seq: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    section_title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    page_num: Mapped[int | None] = mapped_column(Integer, nullable=True)
    language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    token_count: Mapped[int] = mapped_column(Integer)
    embedding_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    client_id: Mapped[str] = mapped_column(String(100), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship(back_populates="chunks")
```

**Step 5: Create `backend/app/models/query.py`**

```python
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, generate_uuid


class Query(Base):
    __tablename__ = "queries"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=generate_uuid)
    user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    query_text: Mapped[str] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    client_id: Mapped[str] = mapped_column(String(100), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    response: Mapped[QueryResponse | None] = relationship(back_populates="query", uselist=False)


class QueryResponse(Base):
    __tablename__ = "query_responses"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=generate_uuid)
    query_id: Mapped[UUID] = mapped_column(ForeignKey("queries.id"), unique=True)
    response_text: Mapped[str] = mapped_column(Text)
    citations: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    faithfulness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_used: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    query: Mapped[Query] = relationship(back_populates="response")
```

**Step 6: Create `backend/app/models/audit.py`**

```python
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Enum, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import AuditAction, Base, generate_uuid


class AuditLog(Base):
    """Immutable audit log — no UPDATE or DELETE operations allowed on this table."""

    __tablename__ = "audit_logs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=generate_uuid)
    user_id: Mapped[UUID | None] = mapped_column(nullable=True)
    action: Mapped[AuditAction] = mapped_column(Enum(AuditAction), index=True)
    resource_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    client_id: Mapped[str] = mapped_column(String(100), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
```

**Step 7: Update `backend/app/models/__init__.py`**

```python
from app.models.audit import AuditLog
from app.models.base import AuditAction, Base, DocumentStatus, UserRole
from app.models.document import Document, DocumentChunk
from app.models.query import Query, QueryResponse
from app.models.user import User

__all__ = [
    "AuditAction",
    "AuditLog",
    "Base",
    "Document",
    "DocumentChunk",
    "DocumentStatus",
    "Query",
    "QueryResponse",
    "User",
    "UserRole",
]
```

**Step 8: Commit**

```bash
git add backend/app/database.py backend/app/models/
git commit -m "feat: SQLAlchemy 2.0 async database engine and ORM models"
```

---

### Task 6: Alembic Setup & First Migration

**Depends on:** Task 5
**Agent:** python-backend

**Files:**
- Create: `backend/alembic.ini`
- Create: `backend/alembic/env.py`
- Create: `backend/alembic/script.py.mako`
- Generate: first migration via `alembic revision --autogenerate`

**Step 1: Create `backend/alembic.ini`**

```ini
[alembic]
script_location = alembic
sqlalchemy.url = postgresql+asyncpg://ragadmin:devpassword@localhost:5432/docintel

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

**Step 2: Create `backend/alembic/env.py`**

```python
from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings
from app.models import Base  # noqa: F401 — imports all models so metadata is populated

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = settings.DATABASE_URL
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode with async engine."""
    connectable = create_async_engine(settings.DATABASE_URL)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
```

**Step 3: Create `backend/alembic/script.py.mako`**

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

**Step 4: Generate first migration**

Run from `backend/` directory (requires PostgreSQL running):

```bash
cd backend
alembic revision --autogenerate -m "initial schema: users, documents, chunks, queries, responses, audit_logs"
```

**Step 5: Apply migration**

```bash
alembic upgrade head
```

**Step 6: Commit**

```bash
git add backend/alembic.ini backend/alembic/
git commit -m "feat: Alembic async setup with initial database migration"
```

---

### Task 7: Exceptions & Middleware

**Depends on:** Task 3
**Agent:** python-backend

**Files:**
- Create: `backend/app/exceptions.py`
- Create: `backend/app/middleware/logging.py`

**Step 1: Create `backend/app/exceptions.py`**

```python
from __future__ import annotations

from fastapi import HTTPException, status


class AppError(HTTPException):
    """Base application error."""

    def __init__(self, detail: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR) -> None:
        super().__init__(status_code=status_code, detail=detail)


class NotFoundError(AppError):
    def __init__(self, detail: str = "Resource not found") -> None:
        super().__init__(detail=detail, status_code=status.HTTP_404_NOT_FOUND)


class ValidationError(AppError):
    def __init__(self, detail: str = "Validation error") -> None:
        super().__init__(detail=detail, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


class AuthError(AppError):
    def __init__(self, detail: str = "Authentication required") -> None:
        super().__init__(detail=detail, status_code=status.HTTP_401_UNAUTHORIZED)


class PermissionError(AppError):
    def __init__(self, detail: str = "Insufficient permissions") -> None:
        super().__init__(detail=detail, status_code=status.HTTP_403_FORBIDDEN)


class ServiceUnavailableError(AppError):
    def __init__(self, detail: str = "Service temporarily unavailable") -> None:
        super().__init__(detail=detail, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
```

**Step 2: Create `backend/app/middleware/logging.py`**

```python
from __future__ import annotations

import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every HTTP request with method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            client_ip=request.client.host if request.client else None,
        )

        return response
```

**Step 3: Commit**

```bash
git add backend/app/exceptions.py backend/app/middleware/logging.py
git commit -m "feat: custom exception hierarchy and request logging middleware"
```

---

### Task 8: Pydantic Schemas

**Depends on:** Task 5
**Agent:** python-backend

**Files:**
- Create: `backend/app/schemas/health.py`
- Create: `backend/app/schemas/document.py`
- Create: `backend/app/schemas/query.py`
- Modify: `backend/app/schemas/__init__.py`

**Step 1: Create `backend/app/schemas/health.py`**

```python
from __future__ import annotations

from pydantic import BaseModel


class ServiceStatus(BaseModel):
    status: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    database: ServiceStatus
    qdrant: ServiceStatus
    redis: ServiceStatus
    llm: ServiceStatus
```

**Step 2: Create `backend/app/schemas/document.py`**

```python
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.models.base import DocumentStatus


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    status: DocumentStatus
    chunk_count: int | None = None
    message: str


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    original_filename: str
    file_size: int
    content_type: str
    status: DocumentStatus
    language: str | None = None
    page_count: int | None = None
    chunk_count: int | None = None
    client_id: str
    created_at: datetime
```

**Step 3: Create `backend/app/schemas/query.py`**

```python
from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    client_id: str = "default"


class Citation(BaseModel):
    source: str
    page: int | None = None
    content_preview: str


class QueryResponseSchema(BaseModel):
    answer: str
    citations: list[Citation]
    model_used: str
    latency_ms: int
```

**Step 4: Update `backend/app/schemas/__init__.py`**

```python
from app.schemas.document import DocumentResponse, DocumentUploadResponse
from app.schemas.health import HealthResponse, ServiceStatus
from app.schemas.query import Citation, QueryRequest, QueryResponseSchema

__all__ = [
    "Citation",
    "DocumentResponse",
    "DocumentUploadResponse",
    "HealthResponse",
    "QueryRequest",
    "QueryResponseSchema",
    "ServiceStatus",
]
```

**Step 5: Commit**

```bash
git add backend/app/schemas/
git commit -m "feat: Pydantic v2 request/response schemas"
```

---

### Task 9: FastAPI App & Health Check

**Depends on:** Tasks 4, 5, 7, 8
**Agent:** python-backend

**Files:**
- Create: `backend/app/api/v1/health.py`
- Create: `backend/app/api/v1/router.py`
- Create: `backend/app/main.py`

**Step 1: Create `backend/app/api/v1/health.py`**

```python
from __future__ import annotations

import httpx
import structlog
from fastapi import APIRouter
from qdrant_client import QdrantClient
from redis.asyncio import Redis
from sqlalchemy import text

from app.config import settings
from app.database import async_session_factory
from app.schemas.health import HealthResponse, ServiceStatus

logger = structlog.get_logger()
router = APIRouter()


async def _check_database() -> ServiceStatus:
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        return ServiceStatus(status="healthy")
    except Exception as e:
        logger.error("health_check_database_failed", error=str(e))
        return ServiceStatus(status="unhealthy", detail=str(e))


async def _check_qdrant() -> ServiceStatus:
    try:
        client = QdrantClient(url=settings.QDRANT_URL, timeout=5)
        client.get_collections()
        client.close()
        return ServiceStatus(status="healthy")
    except Exception as e:
        logger.error("health_check_qdrant_failed", error=str(e))
        return ServiceStatus(status="unhealthy", detail=str(e))


async def _check_redis() -> ServiceStatus:
    try:
        client = Redis.from_url(settings.REDIS_URL, socket_timeout=5)
        await client.ping()
        await client.aclose()
        return ServiceStatus(status="healthy")
    except Exception as e:
        logger.error("health_check_redis_failed", error=str(e))
        return ServiceStatus(status="unhealthy", detail=str(e))


async def _check_llm() -> ServiceStatus:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Strip /v1 suffix for Ollama health check
            base_url = settings.LLM_BASE_URL.rstrip("/v1").rstrip("/")
            resp = await client.get(f"{base_url}/api/tags")
            if resp.status_code == 200:
                return ServiceStatus(status="healthy")
            return ServiceStatus(status="unhealthy", detail=f"HTTP {resp.status_code}")
    except Exception as e:
        logger.error("health_check_llm_failed", error=str(e))
        return ServiceStatus(status="unhealthy", detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check health of all backend services."""
    db = await _check_database()
    qdrant = await _check_qdrant()
    redis_status = await _check_redis()
    llm = await _check_llm()

    all_healthy = all(s.status == "healthy" for s in [db, qdrant, redis_status, llm])

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        database=db,
        qdrant=qdrant,
        redis=redis_status,
        llm=llm,
    )
```

**Step 2: Create `backend/app/api/v1/router.py`**

```python
from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.health import router as health_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(health_router, tags=["health"])
```

**Step 3: Create `backend/app/main.py`**

```python
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_v1_router
from app.config import settings
from app.logging_config import setup_logging
from app.middleware.logging import RequestLoggingMiddleware

setup_logging(settings.LOG_LEVEL)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="LCM DocIntel",
        description="Enterprise Multilingual RAG Platform",
        version="0.1.0",
    )

    # Middleware (order matters — last added is first executed)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(RequestLoggingMiddleware)

    # Routers
    application.include_router(api_v1_router)

    return application


app = create_app()
```

**Step 4: Verify health endpoint starts**

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# In another terminal:
curl http://localhost:8000/api/v1/health
```

Expected: JSON with service statuses (some may be unhealthy if Docker not running yet).

**Step 5: Commit**

```bash
git add backend/app/api/ backend/app/main.py
git commit -m "feat: FastAPI app with health check endpoint and CORS middleware"
```

---

### Task 10: RAG Ingestion Pipeline

**Depends on:** Task 4
**Agent:** rag-specialist

**Files:**
- Create: `backend/app/pipelines/chunking.py`
- Create: `backend/app/pipelines/ingestion.py`

**Step 1: Create `backend/app/pipelines/chunking.py`**

Token-based chunker with overlap, using tiktoken.

```python
from __future__ import annotations

import tiktoken
import structlog

logger = structlog.get_logger()

# Use cl100k_base encoding (GPT-4 / general-purpose)
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(_ENCODING.encode(text))


def chunk_text(
    text: str,
    target_tokens: int = 450,
    max_tokens: int = 512,
    min_tokens: int = 200,
    overlap_pct: float = 0.15,
) -> list[str]:
    """Split text into token-sized chunks with overlap.

    Args:
        text: Input text to chunk.
        target_tokens: Target chunk size in tokens.
        max_tokens: Hard ceiling for chunk size.
        min_tokens: Minimum chunk size; smaller chunks merged with neighbors.
        overlap_pct: Fraction of overlap between consecutive chunks.

    Returns:
        List of text chunks, each within [min_tokens, max_tokens] range.
    """
    if not text.strip():
        return []

    # Split into paragraphs first (respects natural boundaries)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Build chunks by accumulating paragraphs up to target token size
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # If single paragraph exceeds max, split by sentences
        if para_tokens > max_tokens:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            # Split long paragraph by sentences
            sentences = para.replace(". ", ".\n").split("\n")
            for sentence in sentences:
                sent_tokens = count_tokens(sentence)
                if current_tokens + sent_tokens > target_tokens and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                current_chunk.append(sentence)
                current_tokens += sent_tokens
            continue

        if current_tokens + para_tokens > target_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_tokens = 0

        current_chunk.append(para)
        current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    # Merge small trailing chunks with previous
    merged: list[str] = []
    for chunk in chunks:
        if merged and count_tokens(chunk) < min_tokens:
            merged[-1] = merged[-1] + "\n\n" + chunk
        else:
            merged.append(chunk)

    # Add overlap between consecutive chunks
    if len(merged) <= 1 or overlap_pct <= 0:
        return merged

    overlap_tokens = int(target_tokens * overlap_pct)
    overlapped: list[str] = [merged[0]]

    for i in range(1, len(merged)):
        prev_tokens = _ENCODING.encode(merged[i - 1])
        overlap_text = _ENCODING.decode(prev_tokens[-overlap_tokens:]) if len(prev_tokens) > overlap_tokens else ""
        if overlap_text.strip():
            overlapped.append(overlap_text.strip() + "\n\n" + merged[i])
        else:
            overlapped.append(merged[i])

    logger.info("chunking_complete", num_chunks=len(overlapped))
    return overlapped
```

**Step 2: Create `backend/app/pipelines/ingestion.py`**

```python
from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

import structlog
from docling.document_converter import DocumentConverter
from haystack import Document as HaystackDocument
from haystack import Pipeline
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from app.config import settings
from app.pipelines.chunking import chunk_text, count_tokens

logger = structlog.get_logger()


def _get_document_store() -> QdrantDocumentStore:
    """Create a QdrantDocumentStore instance."""
    return QdrantDocumentStore(
        url=settings.QDRANT_URL,
        index=settings.QDRANT_COLLECTION,
        embedding_dim=settings.EMBEDDING_DIMENSION,
        recreate_index=False,
        wait_result_from_api=True,
    )


def _build_ingestion_pipeline(document_store: QdrantDocumentStore) -> Pipeline:
    """Build the Haystack ingestion pipeline: embed → write to Qdrant."""
    pipeline = Pipeline()

    embedder = OpenAIDocumentEmbedder(
        api_key=Secret.from_token("ollama"),
        model=settings.EMBEDDING_MODEL,
        api_base_url=settings.EMBEDDING_BASE_URL,
        meta_fields_to_embed=[],
    )

    writer = DocumentWriter(document_store=document_store)

    pipeline.add_component("embedder", embedder)
    pipeline.add_component("writer", writer)
    pipeline.connect("embedder.documents", "writer.documents")

    return pipeline


def parse_document(file_path: str | Path) -> tuple[str, int]:
    """Parse a document using Docling and return (text, page_count).

    Args:
        file_path: Path to the document file.

    Returns:
        Tuple of (extracted_text, page_count).
    """
    converter = DocumentConverter()
    result = converter.convert(str(file_path))
    text = result.document.export_to_markdown()
    # Estimate page count from Docling metadata
    page_count = len(result.document.pages) if hasattr(result.document, "pages") else 1
    return text, page_count


def ingest_document(
    file_path: str | Path,
    filename: str,
    client_id: str = "default",
) -> list[HaystackDocument]:
    """Full ingestion: parse → chunk → embed → store in Qdrant.

    Args:
        file_path: Path to the uploaded file.
        filename: Original filename for citation metadata.
        client_id: Client identifier for multi-tenancy.

    Returns:
        List of Haystack Documents that were indexed.
    """
    logger.info("ingestion_started", filename=filename, client_id=client_id)

    # 1. Parse document
    text, page_count = parse_document(file_path)
    logger.info("document_parsed", filename=filename, page_count=page_count, chars=len(text))

    # 2. Chunk text
    chunks = chunk_text(
        text,
        target_tokens=settings.CHUNK_TARGET_TOKENS,
        max_tokens=settings.CHUNK_MAX_TOKENS,
        min_tokens=settings.CHUNK_MIN_TOKENS,
        overlap_pct=settings.CHUNK_OVERLAP_PCT,
    )
    logger.info("document_chunked", filename=filename, num_chunks=len(chunks))

    # 3. Create Haystack Documents with metadata
    haystack_docs: list[HaystackDocument] = []
    for seq, chunk_content in enumerate(chunks):
        doc = HaystackDocument(
            id=str(uuid4()),
            content=chunk_content,
            meta={
                "source": filename,
                "page_num": seq + 1,  # Approximate; proper page mapping in Phase 2
                "chunk_seq": seq,
                "token_count": count_tokens(chunk_content),
                "client_id": client_id,
            },
        )
        haystack_docs.append(doc)

    # 4. Embed and write to Qdrant via Haystack pipeline
    store = _get_document_store()
    pipeline = _build_ingestion_pipeline(store)
    result = pipeline.run({"embedder": {"documents": haystack_docs}})
    written_count = result.get("writer", {}).get("documents_written", len(haystack_docs))
    logger.info("ingestion_complete", filename=filename, chunks_written=written_count)

    return haystack_docs
```

**Step 3: Commit**

```bash
git add backend/app/pipelines/chunking.py backend/app/pipelines/ingestion.py
git commit -m "feat: RAG ingestion pipeline — Docling parse, token chunking, Haystack embed+Qdrant"
```

---

### Task 11: RAG Query Pipeline

**Depends on:** Task 4
**Agent:** rag-specialist

**Files:**
- Create: `backend/app/pipelines/query.py`

**Step 1: Create `backend/app/pipelines/query.py`**

```python
from __future__ import annotations

import time

import structlog
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from app.config import settings

logger = structlog.get_logger()

SYSTEM_PROMPT = """\
You are a document research assistant.
You ONLY answer based on the provided context. You MUST cite every claim.

Rules:
1. Every statement must end with [Source: filename, page X]
2. If the context does not contain the answer, say "Cannot find answer in the available documents"
3. Never make claims beyond what the context explicitly states
4. Respond in the same language as the user's question
5. If context is in a different language than the question, translate the answer but keep citations in original language\
"""

USER_TEMPLATE = """\
Context:
{% for doc in documents %}
[{{ loop.index }}] Source: {{ doc.meta.source }}, Page {{ doc.meta.page_num }}
{{ doc.content }}

{% endfor %}
Question: {{ query }}\
"""


def _get_document_store() -> QdrantDocumentStore:
    """Create a QdrantDocumentStore instance."""
    return QdrantDocumentStore(
        url=settings.QDRANT_URL,
        index=settings.QDRANT_COLLECTION,
        embedding_dim=settings.EMBEDDING_DIMENSION,
        recreate_index=False,
    )


def _build_query_pipeline(document_store: QdrantDocumentStore) -> Pipeline:
    """Build the Haystack query pipeline: embed → retrieve → prompt → generate."""
    pipeline = Pipeline()

    # Text embedder for the query
    text_embedder = OpenAITextEmbedder(
        api_key=Secret.from_token("ollama"),
        model=settings.EMBEDDING_MODEL,
        api_base_url=settings.EMBEDDING_BASE_URL,
    )

    # Qdrant retriever
    retriever = QdrantEmbeddingRetriever(
        document_store=document_store,
        top_k=settings.RETRIEVER_TOP_K,
    )

    # Chat prompt builder with system + user template
    messages_template = [
        ChatMessage.from_system(SYSTEM_PROMPT),
        ChatMessage.from_user(USER_TEMPLATE),
    ]
    prompt_builder = ChatPromptBuilder(template=messages_template)

    # LLM generator (OpenAI-compatible — works with Ollama and vLLM)
    llm = OpenAIChatGenerator(
        api_key=Secret.from_token("ollama"),
        model=settings.LLM_MODEL,
        api_base_url=settings.LLM_BASE_URL,
        generation_kwargs={
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
        },
    )

    pipeline.add_component("embedder", text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm)

    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.messages")

    return pipeline


def query_documents(question: str, client_id: str = "default") -> dict:
    """Run the RAG query pipeline.

    Args:
        question: The user's question.
        client_id: Client identifier for multi-tenancy filtering.

    Returns:
        Dict with keys: answer, citations, model_used, latency_ms, retrieved_docs.
    """
    logger.info("query_started", question=question[:100], client_id=client_id)
    start_time = time.perf_counter()

    store = _get_document_store()
    pipeline = _build_query_pipeline(store)

    result = pipeline.run({
        "embedder": {"text": question},
        "prompt_builder": {"query": question},
    })

    latency_ms = round((time.perf_counter() - start_time) * 1000)

    # Extract answer from LLM response
    replies = result.get("llm", {}).get("replies", [])
    answer_text = replies[0].text if replies else "Cannot find answer in the available documents"

    # Extract retrieved documents for citation metadata
    retrieved_docs = result.get("retriever", {}).get("documents", [])
    citations = [
        {
            "source": doc.meta.get("source", "unknown"),
            "page": doc.meta.get("page_num"),
            "content_preview": doc.content[:200] if doc.content else "",
        }
        for doc in retrieved_docs
    ]

    logger.info(
        "query_complete",
        latency_ms=latency_ms,
        num_retrieved=len(retrieved_docs),
        answer_length=len(answer_text),
    )

    return {
        "answer": answer_text,
        "citations": citations,
        "model_used": settings.LLM_MODEL,
        "latency_ms": latency_ms,
        "retrieved_docs": retrieved_docs,
    }
```

**Step 2: Commit**

```bash
git add backend/app/pipelines/query.py
git commit -m "feat: RAG query pipeline — embed, retrieve from Qdrant, generate with citations"
```

---

### Task 12: API Endpoints (Upload + Query)

**Depends on:** Tasks 9, 10, 11
**Agent:** python-backend

**Files:**
- Create: `backend/app/api/v1/documents.py`
- Create: `backend/app/api/v1/query.py`
- Modify: `backend/app/api/v1/router.py`

**Step 1: Create `backend/app/api/v1/documents.py`**

```python
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from uuid import uuid4

import structlog
from fastapi import APIRouter, File, UploadFile

from app.config import settings
from app.exceptions import ValidationError
from app.pipelines.ingestion import ingest_document
from app.schemas.document import DocumentUploadResponse

logger = structlog.get_logger()
router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Upload a document for ingestion into the RAG pipeline.

    Accepts PDF, DOCX, PPTX files. Parses, chunks, embeds, and stores in Qdrant.
    Phase 1: synchronous processing. Phase 2+ will use Celery for async.
    """
    if not file.filename:
        raise ValidationError("Filename is required")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise ValidationError(
            f"Unsupported file type: {file.content_type}. Allowed: PDF, DOCX, PPTX"
        )

    # Read file content
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise ValidationError(f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)}MB")

    # Save to temp file for Docling processing
    doc_id = uuid4()
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Run ingestion pipeline (synchronous in Phase 1)
        haystack_docs = ingest_document(
            file_path=tmp_path,
            filename=file.filename,
            client_id=settings.CLIENT_ID,
        )

        return DocumentUploadResponse(
            id=doc_id,
            filename=file.filename,
            status="completed",
            chunk_count=len(haystack_docs),
            message=f"Document processed successfully. {len(haystack_docs)} chunks indexed.",
        )
    finally:
        os.unlink(tmp_path)
```

**Step 2: Create `backend/app/api/v1/query.py`**

```python
from __future__ import annotations

import structlog
from fastapi import APIRouter

from app.pipelines.query import query_documents
from app.schemas.query import Citation, QueryRequest, QueryResponseSchema

logger = structlog.get_logger()
router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponseSchema)
async def submit_query(request: QueryRequest) -> QueryResponseSchema:
    """Submit a question to the RAG pipeline.

    Embeds the question, retrieves relevant chunks from Qdrant,
    and generates an answer with citations using the LLM.
    """
    result = query_documents(
        question=request.question,
        client_id=request.client_id,
    )

    citations = [
        Citation(
            source=c["source"],
            page=c.get("page"),
            content_preview=c.get("content_preview", ""),
        )
        for c in result["citations"]
    ]

    return QueryResponseSchema(
        answer=result["answer"],
        citations=citations,
        model_used=result["model_used"],
        latency_ms=result["latency_ms"],
    )
```

**Step 3: Update `backend/app/api/v1/router.py`**

```python
from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.documents import router as documents_router
from app.api.v1.health import router as health_router
from app.api.v1.query import router as query_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(health_router, tags=["health"])
api_v1_router.include_router(documents_router)
api_v1_router.include_router(query_router)
```

**Step 4: Commit**

```bash
git add backend/app/api/v1/
git commit -m "feat: document upload and query API endpoints wired to RAG pipelines"
```

---

### Task 13: Celery Placeholder

**Depends on:** Task 4
**Agent:** python-backend

**Files:**
- Create: `backend/workers/celery_app.py`

**Step 1: Create `backend/workers/celery_app.py`**

```python
from __future__ import annotations

from celery import Celery

from app.config import settings

celery_app = Celery(
    "docintel",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(name="tasks.health_check")
def health_check_task() -> dict[str, str]:
    """Simple health check task to verify Celery is running."""
    return {"status": "ok"}
```

**Step 2: Commit**

```bash
git add backend/workers/celery_app.py
git commit -m "chore: Celery app placeholder for async task processing"
```

---

### Task 14: Test Infrastructure

**Depends on:** Task 9
**Agent:** python-backend

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/unit/test_health.py`
- Create: `tests/unit/test_chunking.py`

**Step 1: Create `tests/conftest.py`**

```python
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def api_client():
    """Async HTTP client for API integration tests."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client
```

**Step 2: Create `tests/unit/test_health.py`**

```python
from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_health_endpoint_returns_200(api_client: AsyncClient) -> None:
    """Health endpoint should return 200 even if some services are down."""
    response = await api_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "qdrant" in data
    assert "redis" in data
    assert "llm" in data
```

**Step 3: Create `tests/unit/test_chunking.py`**

```python
from __future__ import annotations

from app.pipelines.chunking import chunk_text, count_tokens


def test_count_tokens_returns_positive_int() -> None:
    result = count_tokens("Hello, world!")
    assert isinstance(result, int)
    assert result > 0


def test_chunk_text_empty_string_returns_empty_list() -> None:
    assert chunk_text("") == []


def test_chunk_text_short_text_returns_single_chunk() -> None:
    text = "This is a short paragraph."
    chunks = chunk_text(text, target_tokens=450, max_tokens=512)
    assert len(chunks) == 1
    assert text in chunks[0]


def test_chunk_text_respects_max_tokens() -> None:
    # Create text with many paragraphs
    text = "\n\n".join([f"Paragraph number {i}. " * 20 for i in range(30)])
    chunks = chunk_text(text, target_tokens=100, max_tokens=150)
    for chunk in chunks:
        tokens = count_tokens(chunk)
        # Allow some tolerance due to overlap
        assert tokens < 300, f"Chunk has {tokens} tokens, expected < 300"


def test_chunk_text_creates_multiple_chunks_for_long_text() -> None:
    text = "\n\n".join([f"This is paragraph {i} with enough text to matter." for i in range(50)])
    chunks = chunk_text(text, target_tokens=50, max_tokens=100)
    assert len(chunks) > 1
```

**Step 4: Run tests**

```bash
cd backend
python -m pytest tests/ -v
```

Expected: All tests pass (health test may show services as "unhealthy" which is fine — test checks structure).

**Step 5: Commit**

```bash
git add tests/
git commit -m "feat: test infrastructure with conftest, health and chunking unit tests"
```

---

### Task 15: Frontend Scaffold

**Parallel-safe:** Yes — no dependencies.
**Agent:** frontend-engineer

**Step 1: Initialize Next.js 15 project**

```bash
cd frontend
npx create-next-app@latest . --typescript --tailwind --eslint --app --src-dir --import-alias "@/*" --use-npm
```

When prompted, accept defaults.

**Step 2: Create placeholder files**

Create `frontend/messages/en.json`:
```json
{
  "common": {
    "appName": "LCM DocIntel",
    "loading": "Loading...",
    "error": "An error occurred"
  }
}
```

Create `frontend/messages/el.json`:
```json
{
  "common": {
    "appName": "LCM DocIntel",
    "loading": "Φόρτωση...",
    "error": "Παρουσιάστηκε σφάλμα"
  }
}
```

**Step 3: Commit**

```bash
git add frontend/
git commit -m "chore: Next.js 15 frontend scaffold with TypeScript and Tailwind"
```

---

### Task 16: Terraform Module Stubs

**Parallel-safe:** Yes — no dependencies.
**Agent:** terraform-infra

**Step 1: Create module stub files**

For each module in `terraform/modules/{vpc,compute,database,storage,security,monitoring}`, create four files:

`main.tf`:
```hcl
# {module_name} module — to be implemented
```

`variables.tf`:
```hcl
variable "client_id" {
  description = "Client identifier for resource naming"
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
}
```

`outputs.tf`:
```hcl
# Outputs for {module_name} module
```

`versions.tf`:
```hcl
terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}
```

**Step 2: Create `terraform/environments/dev/main.tf`**

```hcl
# Dev environment — local development only
# Real AWS infrastructure provisioned in client-specific environments

terraform {
  required_version = ">= 1.7"
}

locals {
  client_id   = "dev"
  environment = "dev"
}
```

**Step 3: Commit**

```bash
git add terraform/
git commit -m "chore: Terraform module stubs for vpc, compute, database, storage, security, monitoring"
```

---

### Task 17: Scripts & Docs

**Parallel-safe:** Yes — no dependencies.
**Agent:** Any

**Files:**
- Create: `scripts/setup-dev.sh`
- Create: `docs/architecture.md`, `docs/api-spec.md`, `docs/rag-pipeline.md`, `docs/security.md`, `docs/evaluation.md`

**Step 1: Create `scripts/setup-dev.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== LCM DocIntel Dev Environment Setup ==="

# 1. Copy .env if not exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo "[OK] Created .env from .env.example"
else
    echo "[SKIP] .env already exists"
fi

# 2. Create uploads directory
mkdir -p uploads
echo "[OK] Created uploads directory"

# 3. Start Docker services
echo "[INFO] Starting Docker services..."
docker compose -f docker/docker-compose.dev.yml up -d
echo "[OK] Docker services started"

# 4. Wait for PostgreSQL
echo "[INFO] Waiting for PostgreSQL..."
until docker compose -f docker/docker-compose.dev.yml exec -T postgres pg_isready -U ragadmin -d docintel 2>/dev/null; do
    sleep 2
done
echo "[OK] PostgreSQL is ready"

# 5. Install Python dependencies
echo "[INFO] Installing Python dependencies..."
cd backend
pip install -r requirements.txt
cd ..
echo "[OK] Python dependencies installed"

# 6. Run database migrations
echo "[INFO] Running database migrations..."
cd backend
alembic upgrade head
cd ..
echo "[OK] Database migrations applied"

# 7. Verify Ollama models
echo "[INFO] Checking Ollama models..."
if command -v ollama &>/dev/null; then
    echo "  Ensure these models are pulled:"
    echo "    ollama pull qwen3:30b-a3b"
    echo "    ollama pull qwen3-embedding:0.6b"
else
    echo "  [WARN] Ollama not found. Install from https://ollama.ai"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Start the backend:  cd backend && uvicorn app.main:app --reload --port 8000"
echo "Start the frontend:  cd frontend && npm run dev"
echo "Health check:       curl http://localhost:8000/api/v1/health"
```

Make executable: `chmod +x scripts/setup-dev.sh`

**Step 2: Create placeholder doc files**

Each doc file gets a minimal header:

`docs/architecture.md`: `# LCM DocIntel — Architecture\n\nTODO: Full architecture documentation.`
`docs/api-spec.md`: `# LCM DocIntel — API Specification\n\nTODO: REST + WebSocket API contracts.`
`docs/rag-pipeline.md`: `# LCM DocIntel — RAG Pipeline Design\n\nTODO: Pipeline architecture and configuration.`
`docs/security.md`: `# LCM DocIntel — Security Architecture\n\nTODO: Security architecture & GDPR compliance.`
`docs/evaluation.md`: `# LCM DocIntel — Evaluation Methodology\n\nTODO: RAGAS evaluation methodology and targets.`

**Step 3: Commit**

```bash
git add scripts/ docs/
git commit -m "chore: dev setup script and placeholder documentation"
```

---

### Task 18: Verification & E2E Test

**Depends on:** ALL previous tasks
**Agent:** Any

**Step 1: Start Docker services**

```bash
docker compose -f docker/docker-compose.dev.yml up -d
```

Wait for all services to be healthy:
```bash
docker compose -f docker/docker-compose.dev.yml ps
```

Expected: All 4 services running and healthy.

**Step 2: Run database migration**

```bash
cd backend && alembic upgrade head
```

**Step 3: Start FastAPI**

```bash
cd backend && uvicorn app.main:app --reload --port 8000
```

**Step 4: Run pytest**

```bash
cd backend && python -m pytest tests/ -v
```

Expected: All tests pass.

**Step 5: Test health endpoint**

```bash
curl -s http://localhost:8000/api/v1/health | python -m json.tool
```

Expected: JSON with all service statuses.

**Step 6: Test upload + query E2E**

Upload a PDF:
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@/path/to/sample.pdf" \
  | python -m json.tool
```

Expected: `{"id": "...", "filename": "sample.pdf", "status": "completed", "chunk_count": N, "message": "..."}`

Query the document:
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}' \
  | python -m json.tool
```

Expected: JSON with `answer` (including `[Source: ...]` citations), `citations` array, `model_used`, `latency_ms`.

**Step 7: Final commit**

```bash
git add -A
git commit -m "chore: Phase 1 bootstrap complete — project scaffolding, backend, Docker, RAG pipeline"
```

---

## Summary

| Task | Description | Agent | Parallel? |
|------|------------|-------|-----------|
| 1 | Project scaffolding | Any | Yes |
| 2 | Docker dev stack | devops | Yes |
| 3 | Backend dependencies | python-backend | After 1 |
| 4 | Config & logging | python-backend | After 3 |
| 5 | Database models & engine | python-backend | After 3 |
| 6 | Alembic setup | python-backend | After 5 |
| 7 | Exceptions & middleware | python-backend | After 3 |
| 8 | Pydantic schemas | python-backend | After 5 |
| 9 | FastAPI app & health | python-backend | After 4,5,7,8 |
| 10 | RAG ingestion pipeline | rag-specialist | After 4 |
| 11 | RAG query pipeline | rag-specialist | After 4 |
| 12 | API endpoints | python-backend | After 9,10,11 |
| 13 | Celery placeholder | python-backend | After 4 |
| 14 | Test infrastructure | python-backend | After 9 |
| 15 | Frontend scaffold | frontend-engineer | Yes |
| 16 | Terraform stubs | terraform-infra | Yes |
| 17 | Scripts & docs | Any | Yes |
| 18 | Verification | Any | After ALL |
