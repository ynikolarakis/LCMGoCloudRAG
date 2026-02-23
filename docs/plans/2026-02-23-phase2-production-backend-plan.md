# Phase 2: Production-Ready Backend — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire up authentication, database persistence, Celery async ingestion, and document CRUD to make the backend production-ready.

**Architecture:** Persistence-first approach with auth dev-bypass. Auth dependency returns seed dev user when `ENVIRONMENT=dev`, enabling DB persistence work without Keycloak. Celery async ingestion replaces synchronous processing. Full document CRUD with cascade delete for GDPR compliance.

**Tech Stack:** FastAPI, SQLAlchemy 2.0 async, Celery + Redis, python-jose (JWT), Keycloak, Alembic, pytest + pytest-asyncio + httpx

---

## Dependency Graph

```
Task 1 (Config) ─────┐
                      ├──► Task 5 (Wire auth into endpoints)
Task 2 (Auth dep) ────┤
                      ├──► Task 6 (Upload DB persistence)
Task 3 (Audit) ───────┤
                      ├──► Task 7 (Query DB persistence)
Task 4 (Celery app) ──┤
                      ├──► Task 8 (Chunk DB persistence)
                      │
                      ├──► Task 9 (Celery ingest task) ──► Task 10 (Async upload + status)
                      │
                      ├──► Task 11 (Document list)
                      ├──► Task 12 (Document detail)
                      └──► Task 13 (Document delete + cascade)

Task 14 (Keycloak JWT) ─── independent
Task 15 (Keycloak Docker) ─ independent
Task 16 (Integration tests) ─ after all others
```

**Parallel-safe tasks:** 1, 2, 3, 4 (foundation), 14, 15 (Keycloak)

---

## Task 1: Add Keycloak + Celery Settings to Config

**Files:**
- Modify: `backend/app/config.py:54-56`
- Test: `tests/unit/test_config.py` (create)

**Step 1: Write the failing test**

Create `tests/unit/test_config.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_config.py -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'KEYCLOAK_REALM'`

**Step 3: Write minimal implementation**

Edit `backend/app/config.py` — add these fields to the `Settings` class after line 56 (`KEYCLOAK_URL`):

```python
    # Keycloak
    KEYCLOAK_URL: str = "http://localhost:8080"
    KEYCLOAK_REALM: str = "docintel"
    KEYCLOAK_CLIENT_ID: str = "docintel-api"
    KEYCLOAK_CLIENT_SECRET: str | None = None

    # Celery
    CELERY_BROKER_URL: str = ""  # defaults to REDIS_URL in model_post_init
    CELERY_RESULT_BACKEND: str = ""  # defaults to REDIS_URL in model_post_init

    def model_post_init(self, __context: object) -> None:
        """Set Celery URLs from Redis if not explicitly provided."""
        if not self.CELERY_BROKER_URL:
            object.__setattr__(self, "CELERY_BROKER_URL", self.REDIS_URL)
        if not self.CELERY_RESULT_BACKEND:
            object.__setattr__(self, "CELERY_RESULT_BACKEND", self.REDIS_URL)
```

**Step 4: Run test to verify it passes**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_config.py -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add tests/unit/test_config.py backend/app/config.py
git commit -m "feat: add Keycloak and Celery settings to config"
```

---

## Task 2: Auth Dependency with Dev Bypass + Seed User

**Files:**
- Create: `backend/app/auth.py`
- Modify: `backend/app/main.py`
- Test: `tests/unit/test_auth.py` (create)

**Step 1: Write the failing tests**

Create `tests/unit/test_auth.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_auth.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.auth'`

**Step 3: Write minimal implementation**

Create `backend/app/auth.py`:

```python
from __future__ import annotations

from uuid import UUID, uuid4

import structlog
from fastapi import Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db_session
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
    """Dev-mode auth bypass: return a synthetic admin user.

    Does NOT hit the database — returns an in-memory User object.
    The real seed_dev_user() ensures this user exists in the DB at startup.
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
    """Prod-mode auth: validate JWT and return User from DB.

    Placeholder — implemented in Task 14 (Keycloak JWT).
    """
    raise AuthError("JWT validation not implemented — set ENVIRONMENT=dev")


async def get_current_user(request: Request) -> User:
    """Auth dependency dispatcher: dev bypass or JWT validation."""
    if settings.ENVIRONMENT == "dev":
        return await get_current_user_dev()
    return await get_current_user_prod(request)


async def seed_dev_user(session: AsyncSession) -> None:
    """Create the dev-admin user in the database if it doesn't exist.

    Called at application startup when ENVIRONMENT=dev.
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

    Usage:
        @router.get("/admin", dependencies=[Depends(RoleChecker(UserRole.ADMIN))])
        async def admin_endpoint(): ...
    """

    def __init__(self, minimum_role: UserRole) -> None:
        self.minimum_role = minimum_role

    async def __call__(self, user: User = Depends(get_current_user)) -> User:
        user_level = _ROLE_HIERARCHY.get(user.role, 0)
        required_level = _ROLE_HIERARCHY.get(self.minimum_role, 0)
        if user_level < required_level:
            raise ForbiddenError(
                f"Role '{user.role.value}' insufficient. Requires '{self.minimum_role.value}' or higher."
            )
        return user
```

Now add the lifespan to `backend/app/main.py` — replace the `create_app` function:

```python
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_v1_router
from app.config import settings
from app.logging_config import setup_logging
from app.middleware.logging import RequestLoggingMiddleware

setup_logging(settings.LOG_LEVEL)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup + shutdown hooks."""
    # Startup
    if settings.ENVIRONMENT == "dev":
        from app.auth import seed_dev_user
        from app.database import async_session_factory

        async with async_session_factory() as session:
            await seed_dev_user(session)

    yield
    # Shutdown (nothing to clean up yet)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="LCM DocIntel",
        description="Enterprise Multilingual RAG Platform",
        version="0.1.0",
        lifespan=lifespan,
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

**Step 4: Run tests to verify they pass**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_auth.py -v`
Expected: 4 PASSED

**Step 5: Commit**

```bash
git add backend/app/auth.py backend/app/main.py tests/unit/test_auth.py
git commit -m "feat: add auth dependency with dev bypass, seed user, and RoleChecker"
```

---

## Task 3: Audit Logging Helper

**Files:**
- Create: `backend/app/audit.py`
- Test: `tests/unit/test_audit.py` (create)

**Step 1: Write the failing test**

Create `tests/unit/test_audit.py`:

```python
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.audit import write_audit_log
from app.models.base import AuditAction


@pytest.mark.anyio
async def test_write_audit_log_creates_entry() -> None:
    """write_audit_log should add an AuditLog to the session."""
    session = AsyncMock()
    session.add = MagicMock()

    user_id = uuid4()
    resource_id = uuid4()

    await write_audit_log(
        session=session,
        user_id=user_id,
        action=AuditAction.DOCUMENT_UPLOAD,
        resource_type="document",
        resource_id=resource_id,
        details={"filename": "test.pdf"},
        ip_address="127.0.0.1",
        client_id="default",
    )

    session.add.assert_called_once()
    log_entry = session.add.call_args[0][0]
    assert log_entry.user_id == user_id
    assert log_entry.action == AuditAction.DOCUMENT_UPLOAD
    assert log_entry.resource_type == "document"
    assert str(log_entry.resource_id) == str(resource_id)
    assert log_entry.details == {"filename": "test.pdf"}
    assert log_entry.ip_address == "127.0.0.1"
    assert log_entry.client_id == "default"


@pytest.mark.anyio
async def test_write_audit_log_optional_fields() -> None:
    """write_audit_log should handle None optional fields."""
    session = AsyncMock()
    session.add = MagicMock()

    await write_audit_log(
        session=session,
        user_id=None,
        action=AuditAction.QUERY_SUBMITTED,
        resource_type="query",
    )

    session.add.assert_called_once()
    log_entry = session.add.call_args[0][0]
    assert log_entry.user_id is None
    assert log_entry.resource_id is None
    assert log_entry.details is None
    assert log_entry.ip_address is None
    assert log_entry.client_id == "default"
```

**Step 2: Run test to verify it fails**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_audit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.audit'`

**Step 3: Write minimal implementation**

Create `backend/app/audit.py`:

```python
from __future__ import annotations

from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AuditLog
from app.models.base import AuditAction

logger = structlog.get_logger()


async def write_audit_log(
    session: AsyncSession,
    user_id: UUID | None,
    action: AuditAction,
    resource_type: str,
    resource_id: UUID | None = None,
    details: dict | None = None,
    ip_address: str | None = None,
    client_id: str = "default",
) -> None:
    """Write an immutable audit log entry.

    This function adds an AuditLog row to the session but does NOT commit.
    The caller is responsible for committing the session (usually as part
    of a larger transaction).

    Args:
        session: Active async database session.
        user_id: ID of the user who triggered the action (None for system actions).
        action: The type of audit action (from AuditAction enum).
        resource_type: Type of resource affected (e.g., "document", "query").
        resource_id: ID of the affected resource.
        details: Optional JSON-serializable dict with extra context.
        ip_address: Client IP address.
        client_id: Tenant identifier.
    """
    entry = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=str(resource_id) if resource_id else None,
        details=details,
        ip_address=ip_address,
        client_id=client_id,
    )
    session.add(entry)
    logger.info(
        "audit_log_written",
        action=action.value,
        resource_type=resource_type,
        resource_id=str(resource_id) if resource_id else None,
        user_id=str(user_id) if user_id else None,
    )
```

**Step 4: Run test to verify it passes**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_audit.py -v`
Expected: 2 PASSED

**Step 5: Commit**

```bash
git add backend/app/audit.py tests/unit/test_audit.py
git commit -m "feat: add audit logging helper"
```

---

## Task 4: Celery App Configuration

**Files:**
- Create: `backend/app/celery_app.py`
- Create: `backend/app/tasks/__init__.py`
- Test: `tests/unit/test_celery_config.py` (create)

**Step 1: Write the failing test**

Create `tests/unit/test_celery_config.py`:

```python
from __future__ import annotations

from app.celery_app import celery_app


def test_celery_app_name() -> None:
    """Celery app should be named 'docintel'."""
    assert celery_app.main == "docintel"


def test_celery_task_serializer_is_json() -> None:
    """Celery should use JSON serializer."""
    assert celery_app.conf.task_serializer == "json"


def test_celery_acks_late() -> None:
    """Celery tasks should ack late for reliability."""
    assert celery_app.conf.task_acks_late is True


def test_celery_reject_on_worker_lost() -> None:
    """Tasks should be rejected (requeued) if worker dies."""
    assert celery_app.conf.task_reject_on_worker_lost is True
```

**Step 2: Run test to verify it fails**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_celery_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.celery_app'`

**Step 3: Write minimal implementation**

Create `backend/app/celery_app.py`:

```python
from __future__ import annotations

from celery import Celery

from app.config import settings

celery_app = Celery(
    "docintel",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    worker_concurrency=2,
    broker_connection_retry_on_startup=True,
)

# Auto-discover tasks in app.tasks package
celery_app.autodiscover_tasks(["app.tasks"])
```

Create `backend/app/tasks/__init__.py`:

```python
from __future__ import annotations
```

**Step 4: Run test to verify it passes**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_celery_config.py -v`
Expected: 4 PASSED

**Step 5: Add celery to dependencies**

Edit `pyproject.toml` — add celery to dependencies (or just ensure it's installed):

Run: `cd E:/LCMGoCloudRAG && pip install celery`

**Step 6: Commit**

```bash
git add backend/app/celery_app.py backend/app/tasks/__init__.py tests/unit/test_celery_config.py
git commit -m "feat: add Celery app configuration"
```

---

## Task 5: Wire Auth into Existing Endpoints

**Files:**
- Modify: `backend/app/api/v1/documents.py`
- Modify: `backend/app/api/v1/query.py`
- Modify: `backend/app/api/v1/health.py`
- Test: `tests/unit/test_auth_wiring.py` (create)

**Step 1: Write the failing test**

Create `tests/unit/test_auth_wiring.py`:

```python
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def api_client() -> AsyncClient:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.mark.anyio
async def test_health_no_auth_required(api_client: AsyncClient) -> None:
    """Health endpoint should NOT require authentication."""
    resp = await api_client.get("/api/v1/health")
    # Should not be 401/403 (may fail health checks but that's OK)
    assert resp.status_code != 401
    assert resp.status_code != 403


@pytest.mark.anyio
async def test_query_returns_user_info_in_dev(api_client: AsyncClient) -> None:
    """Query endpoint in dev mode should accept requests (dev bypass auth)."""
    # We only test that auth doesn't block — the actual query may fail
    # due to missing Qdrant, which is fine for this unit test
    resp = await api_client.post(
        "/api/v1/query",
        json={"question": "test question"},
    )
    # Should not be 401/403
    assert resp.status_code != 401
    assert resp.status_code != 403
```

**Step 2: Run test to verify current state**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_auth_wiring.py -v`
Expected: Tests should pass since we haven't added auth yet (endpoints are currently unprotected). This establishes the baseline.

**Step 3: Add auth to upload endpoint**

Edit `backend/app/api/v1/documents.py` — add `Depends(get_current_user)`:

```python
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, File, UploadFile

from app.auth import get_current_user
from app.config import settings
from app.exceptions import ValidationError
from app.models import User
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
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
) -> DocumentUploadResponse:
    """Upload a document for ingestion into the RAG pipeline."""
    if not file.filename:
        raise ValidationError("Filename is required")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise ValidationError(f"Unsupported file type: {file.content_type}. Allowed: PDF, DOCX, PPTX")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise ValidationError(f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)}MB")

    doc_id = uuid4()
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        haystack_docs = ingest_document(
            file_path=tmp_path,
            filename=file.filename,
            client_id=current_user.client_id,
        )

        logger.info(
            "document_upload_complete",
            doc_id=str(doc_id),
            filename=file.filename,
            chunk_count=len(haystack_docs),
            user_id=str(current_user.id),
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

Edit `backend/app/api/v1/query.py` — add auth:

```python
from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends

from app.auth import get_current_user
from app.models import User
from app.pipelines.query import query_documents
from app.schemas.query import Citation, QueryRequest, QueryResponseSchema

logger = structlog.get_logger()
router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponseSchema)
async def submit_query(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
) -> QueryResponseSchema:
    """Submit a question to the RAG pipeline."""
    result = query_documents(
        question=request.question,
        client_id=current_user.client_id,
    )

    citations = [
        Citation(
            source=c["source"],
            page=c.get("page"),
            content_preview=c.get("content_preview", ""),
        )
        for c in result["citations"]
    ]

    logger.info(
        "query_response_built",
        question_length=len(request.question),
        citation_count=len(citations),
        latency_ms=result["latency_ms"],
        user_id=str(current_user.id),
    )

    return QueryResponseSchema(
        answer=result["answer"],
        citations=citations,
        model_used=result["model_used"],
        latency_ms=result["latency_ms"],
    )
```

**Step 4: Run tests to verify they pass**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_auth_wiring.py tests/unit/test_auth.py -v`
Expected: All PASSED (dev bypass means auth doesn't block)

**Step 5: Commit**

```bash
git add backend/app/api/v1/documents.py backend/app/api/v1/query.py tests/unit/test_auth_wiring.py
git commit -m "feat: wire auth dependency into upload and query endpoints"
```

---

## Task 6: Upload Endpoint — DB Persistence + File Saving

**Files:**
- Modify: `backend/app/api/v1/documents.py`
- Modify: `backend/app/schemas/document.py`
- Test: `tests/unit/test_upload_persistence.py` (create)

This task rewrites the upload endpoint to:
1. Save the file to `uploads/{client_id}/{doc_id}{suffix}`
2. Create a Document row in PostgreSQL with `status=queued`
3. Run ingestion synchronously (Celery dispatch comes in Task 10)
4. Return 202 with document ID and status

**Step 1: Write the failing test**

Create `tests/unit/test_upload_persistence.py`:

```python
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def api_client() -> AsyncClient:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.mark.anyio
@patch("app.api.v1.documents.ingest_document")
@patch("app.api.v1.documents.get_db_session")
async def test_upload_creates_document_row(
    mock_get_db: MagicMock,
    mock_ingest: MagicMock,
    api_client: AsyncClient,
    tmp_path: object,
) -> None:
    """Upload should create a Document row in the database."""
    # Mock DB session
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()

    async def fake_get_db():
        yield mock_session

    mock_get_db.return_value = fake_get_db()

    # Mock ingestion to avoid actual pipeline calls
    mock_ingest.return_value = []

    resp = await api_client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.pdf", b"fake pdf content", "application/pdf")},
    )

    # Should succeed (either 200 or 202)
    assert resp.status_code in (200, 202)
    data = resp.json()
    assert "id" in data
    assert data["status"] in ("queued", "completed")
```

**Step 2: Run test to verify it fails**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_upload_persistence.py -v`
Expected: FAIL (current endpoint doesn't do DB persistence)

**Step 3: Rewrite upload endpoint with DB persistence**

Edit `backend/app/api/v1/documents.py` — full replacement:

```python
from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit import write_audit_log
from app.auth import get_current_user
from app.config import settings
from app.database import get_db_session
from app.exceptions import ValidationError
from app.models import Document, DocumentStatus, User
from app.models.base import AuditAction
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


def _ensure_upload_dir(client_id: str) -> Path:
    """Create the upload directory for a client if it doesn't exist."""
    upload_dir = Path(settings.UPLOAD_DIR) / client_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


@router.post("/upload", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> DocumentUploadResponse:
    """Upload a document for ingestion into the RAG pipeline.

    Validates the file, creates a Document row, saves to disk, and runs
    ingestion. Phase 2: synchronous with Celery fallback. Phase 2+: async.

    Returns 202 with document ID and status.
    """
    if not file.filename:
        raise ValidationError("Filename is required")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise ValidationError(f"Unsupported file type: {file.content_type}. Allowed: PDF, DOCX, PPTX")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise ValidationError(f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)}MB")

    # 1. Create Document row
    doc_id = uuid4()
    suffix = Path(file.filename).suffix
    stored_filename = f"{doc_id}{suffix}"

    doc = Document(
        id=doc_id,
        filename=stored_filename,
        original_filename=file.filename,
        file_size=len(content),
        content_type=file.content_type or "application/octet-stream",
        status=DocumentStatus.QUEUED,
        client_id=current_user.client_id,
        uploaded_by=current_user.id,
    )
    session.add(doc)
    await session.flush()  # Get the ID assigned

    # 2. Save file to disk
    upload_dir = _ensure_upload_dir(current_user.client_id)
    file_path = upload_dir / stored_filename
    file_path.write_bytes(content)

    # 3. Write audit log
    await write_audit_log(
        session=session,
        user_id=current_user.id,
        action=AuditAction.DOCUMENT_UPLOAD,
        resource_type="document",
        resource_id=doc_id,
        details={"filename": file.filename, "file_size": len(content), "content_type": file.content_type},
        client_id=current_user.client_id,
    )

    # 4. Run ingestion synchronously (Celery dispatch in Task 10)
    try:
        doc.status = DocumentStatus.PROCESSING
        await session.flush()

        haystack_docs = ingest_document(
            file_path=str(file_path),
            filename=file.filename,
            client_id=current_user.client_id,
        )

        doc.status = DocumentStatus.COMPLETED
        doc.chunk_count = len(haystack_docs)
        await session.flush()

        logger.info(
            "document_upload_complete",
            doc_id=str(doc_id),
            filename=file.filename,
            chunk_count=len(haystack_docs),
            user_id=str(current_user.id),
        )
    except Exception:
        doc.status = DocumentStatus.FAILED
        await session.flush()
        logger.exception("document_ingestion_failed", doc_id=str(doc_id))
        raise

    return DocumentUploadResponse(
        id=doc_id,
        filename=file.filename,
        status=doc.status,
        chunk_count=doc.chunk_count,
        message=f"Document processed successfully. {doc.chunk_count} chunks indexed.",
    )
```

Also update `backend/app/schemas/document.py` to accept DocumentStatus enum:

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


class DocumentListRequest(BaseModel):
    page: int = 1
    page_size: int = 20
    status: DocumentStatus | None = None
    language: str | None = None


class DocumentDetailResponse(BaseModel):
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
    uploaded_by: UUID | None = None
    created_at: datetime
    updated_at: datetime


class DocumentStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: DocumentStatus
    chunk_count: int | None = None
    page_count: int | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


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

Update `backend/app/schemas/__init__.py`:

```python
from app.schemas.document import (
    DocumentDetailResponse,
    DocumentListRequest,
    DocumentResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
from app.schemas.health import HealthResponse, ServiceStatus
from app.schemas.query import Citation, QueryRequest, QueryResponseSchema

__all__ = [
    "Citation",
    "DocumentDetailResponse",
    "DocumentListRequest",
    "DocumentResponse",
    "DocumentStatusResponse",
    "DocumentUploadResponse",
    "HealthResponse",
    "QueryRequest",
    "QueryResponseSchema",
    "ServiceStatus",
]
```

**Step 4: Run tests to verify they pass**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_upload_persistence.py -v`
Expected: PASSED

Run full suite: `cd E:/LCMGoCloudRAG && python -m pytest tests/ -v`
Expected: All PASSED

**Step 5: Commit**

```bash
git add backend/app/api/v1/documents.py backend/app/schemas/document.py backend/app/schemas/__init__.py tests/unit/test_upload_persistence.py
git commit -m "feat: upload endpoint with DB persistence and file saving"
```

---

## Task 7: Persist Query + QueryResponse to DB

**Files:**
- Modify: `backend/app/api/v1/query.py`
- Modify: `backend/app/schemas/query.py`
- Test: `tests/unit/test_query_persistence.py` (create)

**Step 1: Write the failing test**

Create `tests/unit/test_query_persistence.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def api_client() -> AsyncClient:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.mark.anyio
@patch("app.api.v1.query.query_documents")
async def test_query_returns_db_backed_response(
    mock_query: MagicMock,
    api_client: AsyncClient,
) -> None:
    """Query endpoint should return response with model and latency."""
    mock_query.return_value = {
        "answer": "Test answer [Source: test.pdf, Page 1]",
        "citations": [{"source": "test.pdf", "page": 1, "content_preview": "preview"}],
        "model_used": "qwen3:30b-a3b",
        "latency_ms": 500,
        "retrieved_docs": [],
    }

    resp = await api_client.post(
        "/api/v1/query",
        json={"question": "What is in the document?"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Test answer [Source: test.pdf, Page 1]"
    assert data["model_used"] == "qwen3:30b-a3b"
    assert data["latency_ms"] == 500
    assert len(data["citations"]) == 1
```

**Step 2: Run test to verify it fails**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_query_persistence.py -v`
Expected: May pass or fail depending on Qdrant availability. The important thing is testing DB integration.

**Step 3: Add DB persistence to query endpoint**

Edit `backend/app/api/v1/query.py`:

```python
from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit import write_audit_log
from app.auth import get_current_user
from app.database import get_db_session
from app.models import Query, QueryResponse, User
from app.models.base import AuditAction
from app.pipelines.query import query_documents
from app.schemas.query import Citation, QueryRequest, QueryResponseSchema

logger = structlog.get_logger()
router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponseSchema)
async def submit_query(
    request_body: QueryRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> QueryResponseSchema:
    """Submit a question to the RAG pipeline.

    Creates a Query row, runs the pipeline, creates a QueryResponse row,
    and writes audit log entries for both the query and response.
    """
    # 1. Create Query row
    query_row = Query(
        user_id=current_user.id,
        query_text=request_body.question,
        client_id=current_user.client_id,
    )
    session.add(query_row)
    await session.flush()

    # 2. Audit: query submitted
    client_ip = request.client.host if request.client else None
    await write_audit_log(
        session=session,
        user_id=current_user.id,
        action=AuditAction.QUERY_SUBMITTED,
        resource_type="query",
        resource_id=query_row.id,
        details={"query_text": request_body.question[:200]},
        ip_address=client_ip,
        client_id=current_user.client_id,
    )

    # 3. Run RAG pipeline
    result = query_documents(
        question=request_body.question,
        client_id=current_user.client_id,
    )

    # 4. Build citations
    citations = [
        Citation(
            source=c["source"],
            page=c.get("page"),
            content_preview=c.get("content_preview", ""),
        )
        for c in result["citations"]
    ]

    citations_json = [c.model_dump() for c in citations]

    # 5. Create QueryResponse row
    response_row = QueryResponse(
        query_id=query_row.id,
        response_text=result["answer"],
        citations=citations_json,
        latency_ms=result["latency_ms"],
        model_used=result["model_used"],
    )
    session.add(response_row)
    await session.flush()

    # 6. Audit: response generated
    await write_audit_log(
        session=session,
        user_id=current_user.id,
        action=AuditAction.RESPONSE_GENERATED,
        resource_type="query_response",
        resource_id=response_row.id,
        details={"latency_ms": result["latency_ms"], "citation_count": len(citations)},
        ip_address=client_ip,
        client_id=current_user.client_id,
    )

    logger.info(
        "query_response_built",
        query_id=str(query_row.id),
        question_length=len(request_body.question),
        citation_count=len(citations),
        latency_ms=result["latency_ms"],
        user_id=str(current_user.id),
    )

    return QueryResponseSchema(
        answer=result["answer"],
        citations=citations,
        model_used=result["model_used"],
        latency_ms=result["latency_ms"],
    )
```

**Step 4: Run tests to verify they pass**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_query_persistence.py -v`
Expected: PASSED

**Step 5: Commit**

```bash
git add backend/app/api/v1/query.py tests/unit/test_query_persistence.py
git commit -m "feat: persist queries and responses to database with audit logging"
```

---

## Task 8: Create DocumentChunk Rows During Ingestion

**Files:**
- Modify: `backend/app/pipelines/ingestion.py`
- Test: `tests/unit/test_chunk_persistence.py` (create)

This task modifies the ingestion pipeline to save DocumentChunk rows to PostgreSQL alongside the Qdrant writes. The ingestion function gains optional `doc_id` and `session` parameters; when provided, it creates DocumentChunk rows and updates their `embedding_id` with Qdrant point IDs.

**Step 1: Write the failing test**

Create `tests/unit/test_chunk_persistence.py`:

```python
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.pipelines.ingestion import create_document_chunks


@pytest.mark.anyio
async def test_create_document_chunks_returns_chunks() -> None:
    """create_document_chunks should create DocumentChunk rows."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()

    doc_id = uuid4()
    chunks = ["First chunk of text.", "Second chunk of text."]

    result = await create_document_chunks(
        session=session,
        doc_id=doc_id,
        chunks=chunks,
        client_id="default",
    )

    assert len(result) == 2
    assert result[0].chunk_seq == 0
    assert result[1].chunk_seq == 1
    assert result[0].document_id == doc_id
    assert result[0].client_id == "default"
    assert session.add.call_count == 2
```

**Step 2: Run test to verify it fails**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_chunk_persistence.py -v`
Expected: FAIL — `ImportError: cannot import name 'create_document_chunks' from 'app.pipelines.ingestion'`

**Step 3: Add create_document_chunks to ingestion.py**

Add this function to `backend/app/pipelines/ingestion.py` (before `ingest_document`):

```python
async def create_document_chunks(
    session: AsyncSession,
    doc_id: UUID,
    chunks: list[str],
    client_id: str = "default",
) -> list[DocumentChunk]:
    """Create DocumentChunk rows in the database for each text chunk.

    Args:
        session: Active async database session.
        doc_id: Parent Document UUID.
        chunks: List of chunk text strings.
        client_id: Tenant identifier.

    Returns:
        List of DocumentChunk ORM objects (not yet committed).
    """
    chunk_rows: list[DocumentChunk] = []
    for seq, chunk_content in enumerate(chunks):
        chunk_row = DocumentChunk(
            document_id=doc_id,
            chunk_seq=seq,
            content=chunk_content,
            token_count=count_tokens(chunk_content),
            client_id=client_id,
        )
        session.add(chunk_row)
        chunk_rows.append(chunk_row)

    await session.flush()
    return chunk_rows
```

Add necessary imports at the top of `backend/app/pipelines/ingestion.py`:

```python
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DocumentChunk
```

**Step 4: Run tests to verify they pass**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_chunk_persistence.py -v`
Expected: PASSED

**Step 5: Commit**

```bash
git add backend/app/pipelines/ingestion.py tests/unit/test_chunk_persistence.py
git commit -m "feat: add create_document_chunks for DB persistence during ingestion"
```

---

## Task 9: Celery Ingest Document Task

**Files:**
- Create: `backend/app/tasks/ingestion.py`
- Test: `tests/unit/test_ingest_task.py` (create)

**Step 1: Write the failing test**

Create `tests/unit/test_ingest_task.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.tasks.ingestion import ingest_document_task


def test_ingest_task_is_registered() -> None:
    """ingest_document_task should be a registered Celery task."""
    assert ingest_document_task.name == "app.tasks.ingestion.ingest_document_task"


def test_ingest_task_has_max_retries() -> None:
    """ingest_document_task should have max_retries=3."""
    assert ingest_document_task.max_retries == 3
```

**Step 2: Run test to verify it fails**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_ingest_task.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.tasks.ingestion'`

**Step 3: Create the Celery task**

Create `backend/app/tasks/ingestion.py`:

```python
from __future__ import annotations

import asyncio

import structlog
from sqlalchemy import select

from app.celery_app import celery_app
from app.config import settings
from app.database import async_session_factory
from app.models import Document, DocumentStatus
from app.pipelines.chunking import count_tokens
from app.pipelines.ingestion import create_document_chunks, ingest_document, parse_document

logger = structlog.get_logger()


def _run_async(coro):
    """Run an async coroutine from synchronous Celery task context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _do_ingest(doc_id: str, file_path: str, client_id: str) -> dict:
    """Async implementation of document ingestion with DB persistence.

    Steps:
    1. Update Document status to PROCESSING
    2. Parse document with Docling
    3. Chunk text
    4. Create DocumentChunk rows in DB
    5. Embed + write to Qdrant via Haystack pipeline
    6. Update Document status to COMPLETED with counts
    """
    from uuid import UUID

    from app.pipelines.chunking import chunk_text

    doc_uuid = UUID(doc_id)

    async with async_session_factory() as session:
        # Fetch the Document row
        result = await session.execute(select(Document).where(Document.id == doc_uuid))
        doc = result.scalar_one_or_none()
        if not doc:
            raise ValueError(f"Document {doc_id} not found in database")

        try:
            # 1. Mark as processing
            doc.status = DocumentStatus.PROCESSING
            await session.flush()

            # 2. Parse
            text, page_count = parse_document(file_path)
            logger.info("task_document_parsed", doc_id=doc_id, page_count=page_count)

            # 3. Chunk
            chunks = chunk_text(
                text,
                target_tokens=settings.CHUNK_TARGET_TOKENS,
                max_tokens=settings.CHUNK_MAX_TOKENS,
                min_tokens=settings.CHUNK_MIN_TOKENS,
                overlap_pct=settings.CHUNK_OVERLAP_PCT,
            )
            logger.info("task_document_chunked", doc_id=doc_id, num_chunks=len(chunks))

            # 4. Create DB chunk rows
            chunk_rows = await create_document_chunks(
                session=session,
                doc_id=doc_uuid,
                chunks=chunks,
                client_id=client_id,
            )

            # 5. Embed + write to Qdrant (reuse existing pipeline)
            haystack_docs = ingest_document(
                file_path=file_path,
                filename=doc.original_filename,
                client_id=client_id,
            )

            # 6. Update Document
            doc.status = DocumentStatus.COMPLETED
            doc.chunk_count = len(chunks)
            doc.page_count = page_count

            await session.commit()

            logger.info(
                "task_ingestion_complete",
                doc_id=doc_id,
                chunk_count=len(chunks),
                page_count=page_count,
            )

            return {"doc_id": doc_id, "status": "completed", "chunk_count": len(chunks)}

        except Exception as e:
            doc.status = DocumentStatus.FAILED
            await session.commit()
            logger.exception("task_ingestion_failed", doc_id=doc_id, error=str(e))
            raise


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def ingest_document_task(self, doc_id: str, file_path: str, client_id: str) -> dict:
    """Celery task: parse, chunk, embed, and store a document.

    Args:
        doc_id: UUID string of the Document row.
        file_path: Path to the uploaded file on disk.
        client_id: Tenant identifier.

    Returns:
        Dict with doc_id, status, and chunk_count.

    Retries up to 3 times with exponential backoff (10s, 60s, 300s).
    """
    try:
        return _run_async(_do_ingest(doc_id, file_path, client_id))
    except Exception as exc:
        retry_delays = [10, 60, 300]
        delay = retry_delays[min(self.request.retries, len(retry_delays) - 1)]
        logger.warning(
            "task_retrying",
            doc_id=doc_id,
            attempt=self.request.retries + 1,
            delay=delay,
        )
        raise self.retry(exc=exc, countdown=delay)
```

**Step 4: Run test to verify it passes**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_ingest_task.py -v`
Expected: 2 PASSED

**Step 5: Commit**

```bash
git add backend/app/tasks/ingestion.py tests/unit/test_ingest_task.py
git commit -m "feat: add Celery ingest_document_task with retry logic"
```

---

## Task 10: Async Upload with Celery Dispatch + Status Endpoint

**Files:**
- Modify: `backend/app/api/v1/documents.py`
- Create: `backend/app/api/v1/status.py` (or add to documents.py)
- Modify: `backend/app/api/v1/router.py`
- Test: `tests/unit/test_async_upload.py` (create)

This task adds:
1. Celery dispatch in upload (with sync fallback when `ENVIRONMENT=dev` and Celery is unavailable)
2. `GET /api/v1/documents/{id}/status` endpoint for polling

**Step 1: Write the failing test**

Create `tests/unit/test_async_upload.py`:

```python
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.schemas.document import DocumentStatusResponse


@pytest.fixture
async def api_client() -> AsyncClient:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


def test_document_status_response_schema() -> None:
    """DocumentStatusResponse should accept valid data."""
    from datetime import datetime, timezone
    from uuid import uuid4

    resp = DocumentStatusResponse(
        id=uuid4(),
        status="queued",
        chunk_count=None,
        page_count=None,
        error_message=None,
        created_at=datetime.now(tz=timezone.utc),
        updated_at=datetime.now(tz=timezone.utc),
    )
    assert resp.status == "queued"
```

**Step 2: Run test to verify it fails**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_async_upload.py -v`
Expected: May pass if schema exists from Task 6. If not, it will fail.

**Step 3: Add status endpoint to documents.py**

Add to `backend/app/api/v1/documents.py` (after the upload endpoint):

```python
from app.auth import RoleChecker
from app.exceptions import NotFoundError
from app.models.base import UserRole
from app.schemas.document import DocumentStatusResponse


@router.get("/{doc_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    doc_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> DocumentStatusResponse:
    """Get the processing status of a document.

    Lightweight endpoint for polling during async ingestion.
    Users can only see their own documents; managers can see all.
    """
    from uuid import UUID

    from sqlalchemy import select

    doc_uuid = UUID(doc_id)
    stmt = select(Document).where(Document.id == doc_uuid)

    # ACL: users see own docs only, managers+ see all
    if current_user.role in (UserRole.USER, UserRole.VIEWER):
        stmt = stmt.where(Document.uploaded_by == current_user.id)

    stmt = stmt.where(Document.client_id == current_user.client_id)

    result = await session.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise NotFoundError(f"Document {doc_id} not found")

    return DocumentStatusResponse(
        id=doc.id,
        status=doc.status,
        chunk_count=doc.chunk_count,
        page_count=doc.page_count,
        error_message=None,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )
```

**Step 4: Run test to verify it passes**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_async_upload.py -v`
Expected: PASSED

**Step 5: Commit**

```bash
git add backend/app/api/v1/documents.py tests/unit/test_async_upload.py
git commit -m "feat: add document status endpoint for polling"
```

---

## Task 11: Document List Endpoint

**Files:**
- Modify: `backend/app/api/v1/documents.py`
- Test: `tests/unit/test_document_list.py` (create)

**Step 1: Write the failing test**

Create `tests/unit/test_document_list.py`:

```python
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def api_client() -> AsyncClient:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.mark.anyio
async def test_list_documents_returns_paginated_list(api_client: AsyncClient) -> None:
    """GET /documents should return a paginated list."""
    resp = await api_client.get("/api/v1/documents")
    # Should succeed even with empty DB
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert isinstance(data["items"], list)
```

**Step 2: Run test to verify it fails**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_document_list.py -v`
Expected: FAIL — 404 or 405 (endpoint doesn't exist yet)

**Step 3: Add list endpoint**

Add to `backend/app/schemas/document.py`:

```python
class PaginatedDocumentsResponse(BaseModel):
    items: list[DocumentResponse]
    total: int
    page: int
    page_size: int
```

Add to `backend/app/api/v1/documents.py`:

```python
from app.schemas.document import DocumentResponse, PaginatedDocumentsResponse


@router.get("", response_model=PaginatedDocumentsResponse)
async def list_documents(
    page: int = 1,
    page_size: int = 20,
    status: str | None = None,
    language: str | None = None,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> PaginatedDocumentsResponse:
    """List documents with pagination and optional filters.

    Users see their own documents. Managers and admins see all client documents.
    Sorted by created_at descending (newest first).
    """
    from sqlalchemy import func, select

    # Base query with client_id filter
    stmt = select(Document).where(Document.client_id == current_user.client_id)

    # ACL: users see own docs only
    if current_user.role in (UserRole.USER, UserRole.VIEWER):
        stmt = stmt.where(Document.uploaded_by == current_user.id)

    # Optional filters
    if status:
        stmt = stmt.where(Document.status == status)
    if language:
        stmt = stmt.where(Document.language == language)

    # Count total
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await session.execute(count_stmt)
    total = total_result.scalar() or 0

    # Paginate
    stmt = stmt.order_by(Document.created_at.desc())
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)

    result = await session.execute(stmt)
    docs = result.scalars().all()

    return PaginatedDocumentsResponse(
        items=[DocumentResponse.model_validate(d) for d in docs],
        total=total,
        page=page,
        page_size=page_size,
    )
```

Update `backend/app/schemas/__init__.py` to export `PaginatedDocumentsResponse`.

**Step 4: Run test to verify it passes**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_document_list.py -v`
Expected: PASSED

**Step 5: Commit**

```bash
git add backend/app/api/v1/documents.py backend/app/schemas/document.py backend/app/schemas/__init__.py tests/unit/test_document_list.py
git commit -m "feat: add paginated document list endpoint with ACL filtering"
```

---

## Task 12: Document Detail Endpoint

**Files:**
- Modify: `backend/app/api/v1/documents.py`
- Test: `tests/unit/test_document_detail.py` (create)

**Step 1: Write the failing test**

Create `tests/unit/test_document_detail.py`:

```python
from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def api_client() -> AsyncClient:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.mark.anyio
async def test_get_document_not_found(api_client: AsyncClient) -> None:
    """GET /documents/{id} should return 404 for non-existent document."""
    fake_id = str(uuid4())
    resp = await api_client.get(f"/api/v1/documents/{fake_id}")
    assert resp.status_code == 404
```

**Step 2: Run test to verify it fails**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_document_detail.py -v`
Expected: FAIL — 404 or 405 (endpoint doesn't exist)

**Step 3: Add detail endpoint**

Add to `backend/app/api/v1/documents.py`:

```python
@router.get("/{doc_id}", response_model=DocumentDetailResponse)
async def get_document(
    doc_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> DocumentDetailResponse:
    """Get document details by ID.

    Returns document metadata and chunk count.
    Users can only see their own documents; managers see all client documents.
    """
    from uuid import UUID

    from sqlalchemy import select

    doc_uuid = UUID(doc_id)
    stmt = select(Document).where(
        Document.id == doc_uuid,
        Document.client_id == current_user.client_id,
    )

    if current_user.role in (UserRole.USER, UserRole.VIEWER):
        stmt = stmt.where(Document.uploaded_by == current_user.id)

    result = await session.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise NotFoundError(f"Document {doc_id} not found")

    # Audit: document viewed
    await write_audit_log(
        session=session,
        user_id=current_user.id,
        action=AuditAction.DOCUMENT_VIEW,
        resource_type="document",
        resource_id=doc.id,
        client_id=current_user.client_id,
    )

    return DocumentDetailResponse.model_validate(doc)
```

Import `DocumentDetailResponse` in the file's imports.

**Step 4: Run test to verify it passes**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_document_detail.py -v`
Expected: PASSED (404 as expected for non-existent doc)

**Step 5: Commit**

```bash
git add backend/app/api/v1/documents.py tests/unit/test_document_detail.py
git commit -m "feat: add document detail endpoint with ACL and audit logging"
```

---

## Task 13: Document Delete with Cascade

**Files:**
- Modify: `backend/app/api/v1/documents.py`
- Test: `tests/unit/test_document_delete.py` (create)

This is the most complex CRUD endpoint. It must:
1. Check manager+ role
2. Delete Qdrant points for this document
3. Delete DocumentChunk rows (cascade handles this via FK)
4. Delete Document row
5. Delete file from disk
6. Write audit log

**Step 1: Write the failing test**

Create `tests/unit/test_document_delete.py`:

```python
from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def api_client() -> AsyncClient:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.mark.anyio
async def test_delete_document_not_found(api_client: AsyncClient) -> None:
    """DELETE /documents/{id} should return 404 for non-existent document."""
    fake_id = str(uuid4())
    resp = await api_client.delete(f"/api/v1/documents/{fake_id}")
    assert resp.status_code == 404
```

**Step 2: Run test to verify it fails**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_document_delete.py -v`
Expected: FAIL — 405 Method Not Allowed (endpoint doesn't exist)

**Step 3: Add delete endpoint**

Add to `backend/app/api/v1/documents.py`:

```python
from fastapi import Response
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue


@router.delete("/{doc_id}", status_code=204, dependencies=[Depends(RoleChecker(UserRole.MANAGER))])
async def delete_document(
    doc_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    """Delete a document and all associated data.

    Cascade: Qdrant points -> DocumentChunk rows -> Document row -> file on disk.
    Requires manager role or higher.
    """
    from uuid import UUID

    from sqlalchemy import select

    doc_uuid = UUID(doc_id)
    stmt = select(Document).where(
        Document.id == doc_uuid,
        Document.client_id == current_user.client_id,
    )

    result = await session.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise NotFoundError(f"Document {doc_id} not found")

    # 1. Delete Qdrant points for this document
    try:
        qdrant = QdrantClient(url=settings.QDRANT_URL, timeout=10)
        qdrant.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="meta.source",
                        match=MatchValue(value=doc.original_filename),
                    ),
                ],
            ),
        )
        qdrant.close()
        logger.info("qdrant_points_deleted", doc_id=doc_id, filename=doc.original_filename)
    except Exception:
        logger.warning("qdrant_delete_failed", doc_id=doc_id, exc_info=True)

    # 2. Delete Document (DocumentChunks cascade via FK ondelete=CASCADE)
    await session.delete(doc)
    await session.flush()

    # 3. Delete file from disk
    file_path = Path(settings.UPLOAD_DIR) / doc.client_id / doc.filename
    if file_path.exists():
        file_path.unlink()
        logger.info("file_deleted", path=str(file_path))

    # 4. Audit log
    await write_audit_log(
        session=session,
        user_id=current_user.id,
        action=AuditAction.DOCUMENT_DELETE,
        resource_type="document",
        resource_id=doc_uuid,
        details={"filename": doc.original_filename, "file_size": doc.file_size},
        client_id=current_user.client_id,
    )

    logger.info(
        "document_deleted",
        doc_id=doc_id,
        filename=doc.original_filename,
        user_id=str(current_user.id),
    )

    return Response(status_code=204)
```

**Step 4: Run test to verify it passes**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_document_delete.py -v`
Expected: PASSED (404 as expected)

**Step 5: Commit**

```bash
git add backend/app/api/v1/documents.py tests/unit/test_document_delete.py
git commit -m "feat: add document delete with Qdrant + DB + file cascade"
```

---

## Task 14: Keycloak JWT Validation (Prod Mode)

**Files:**
- Modify: `backend/app/auth.py`
- Test: `tests/unit/test_jwt_validation.py` (create)

**Step 1: Write the failing test**

Create `tests/unit/test_jwt_validation.py`:

```python
from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from jose import jwt

from app.auth import _decode_jwt, _validate_jwt_claims


def _make_jwt(payload: dict, key: str = "test-secret") -> str:
    """Create a test JWT token."""
    return jwt.encode(payload, key, algorithm="HS256")


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
        "exp": int(time.time()) - 3600,  # expired
        "realm_access": {"roles": ["user"]},
    }
    with pytest.raises(Exception):
        _validate_jwt_claims(claims, expected_issuer="http://localhost:8080/realms/docintel")


def test_validate_jwt_claims_wrong_issuer() -> None:
    """Wrong issuer should fail validation."""
    claims = {
        "sub": str(uuid4()),
        "iss": "http://evil.com/realms/docintel",
        "exp": int(time.time()) + 3600,
        "realm_access": {"roles": ["user"]},
    }
    with pytest.raises(Exception):
        _validate_jwt_claims(claims, expected_issuer="http://localhost:8080/realms/docintel")
```

**Step 2: Run test to verify it fails**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_jwt_validation.py -v`
Expected: FAIL — `ImportError: cannot import name '_decode_jwt'`

**Step 3: Install python-jose and implement JWT validation**

Run: `cd E:/LCMGoCloudRAG && pip install "python-jose[cryptography]"`

Add JWT validation functions to `backend/app/auth.py`:

```python
import time
from functools import lru_cache

import httpx
from jose import JWTError, jwt as jose_jwt


def _validate_jwt_claims(claims: dict, expected_issuer: str) -> bool:
    """Validate standard JWT claims.

    Raises:
        AuthError: If claims are invalid.
    """
    # Check expiration
    exp = claims.get("exp")
    if not exp or exp < time.time():
        raise AuthError("Token has expired")

    # Check issuer
    iss = claims.get("iss")
    if iss != expected_issuer:
        raise AuthError(f"Invalid issuer: {iss}")

    # Check subject exists
    if not claims.get("sub"):
        raise AuthError("Missing subject claim")

    return True


_jwks_cache: dict[str, tuple[dict, float]] = {}
JWKS_CACHE_TTL = 3600  # 1 hour


async def _fetch_jwks(keycloak_url: str, realm: str) -> dict:
    """Fetch JWKS keys from Keycloak with 1-hour cache."""
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
        raise AuthError(f"Invalid token header: {e}")

    # Find matching key
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
        raise AuthError(f"Token validation failed: {e}")
```

Update `get_current_user_prod` in `backend/app/auth.py`:

```python
async def get_current_user_prod(request: Request) -> User:
    """Prod-mode auth: validate JWT from Keycloak and return User from DB."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise AuthError("Missing or invalid Authorization header")

    token = auth_header.split(" ", 1)[1]

    # Fetch JWKS and decode
    jwks = await _fetch_jwks(settings.KEYCLOAK_URL, settings.KEYCLOAK_REALM)
    claims = _decode_jwt(token, jwks)

    expected_issuer = f"{settings.KEYCLOAK_URL}/realms/{settings.KEYCLOAK_REALM}"
    _validate_jwt_claims(claims, expected_issuer=expected_issuer)

    keycloak_id = claims["sub"]

    # Look up user in DB
    from app.database import async_session_factory

    async with async_session_factory() as session:
        result = await session.execute(select(User).where(User.keycloak_id == keycloak_id))
        user = result.scalar_one_or_none()
        if not user:
            raise AuthError(f"User with keycloak_id '{keycloak_id}' not found")
        if not user.is_active:
            raise AuthError("User account is deactivated")
        return user
```

**Step 4: Run test to verify it passes**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/unit/test_jwt_validation.py -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add backend/app/auth.py tests/unit/test_jwt_validation.py
git commit -m "feat: add Keycloak JWT validation with JWKS caching"
```

---

## Task 15: Keycloak Docker Realm Export

**Files:**
- Create: `docker/keycloak/realm-export.json`
- Modify: `docker/docker-compose.dev.yml`

This task creates a Keycloak realm configuration that auto-imports on startup, providing:
- Realm: `docintel`
- Client: `docintel-api` (confidential)
- Roles: admin, manager, user, viewer
- Test users: admin@test.com (admin), user@test.com (user)

**Step 1: Create the realm export**

Create `docker/keycloak/realm-export.json`:

```json
{
  "realm": "docintel",
  "enabled": true,
  "roles": {
    "realm": [
      { "name": "admin", "description": "Full system access" },
      { "name": "manager", "description": "Manage documents and users" },
      { "name": "user", "description": "Upload and query documents" },
      { "name": "viewer", "description": "Read-only access" }
    ]
  },
  "clients": [
    {
      "clientId": "docintel-api",
      "enabled": true,
      "protocol": "openid-connect",
      "publicClient": false,
      "secret": "docintel-api-secret",
      "directAccessGrantsEnabled": true,
      "serviceAccountsEnabled": true,
      "standardFlowEnabled": true,
      "redirectUris": ["http://localhost:3000/*"],
      "webOrigins": ["http://localhost:3000"]
    }
  ],
  "users": [
    {
      "username": "admin@test.com",
      "email": "admin@test.com",
      "firstName": "Admin",
      "lastName": "User",
      "enabled": true,
      "credentials": [
        {
          "type": "password",
          "value": "admin123",
          "temporary": false
        }
      ],
      "realmRoles": ["admin"]
    },
    {
      "username": "user@test.com",
      "email": "user@test.com",
      "firstName": "Test",
      "lastName": "User",
      "enabled": true,
      "credentials": [
        {
          "type": "password",
          "value": "user123",
          "temporary": false
        }
      ],
      "realmRoles": ["user"]
    }
  ]
}
```

**Step 2: Update docker-compose to import realm**

Edit `docker/docker-compose.dev.yml` — update the keycloak service:

```yaml
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
    volumes:
      - ./keycloak/realm-export.json:/opt/keycloak/data/import/realm-export.json:ro
    command: start-dev --import-realm
    depends_on:
      postgres:
        condition: service_healthy
```

**Step 3: Verify**

Run: `cd E:/LCMGoCloudRAG && docker compose -f docker/docker-compose.dev.yml up keycloak -d`
Then check: `curl http://localhost:8080/realms/docintel` should return the realm config.

**Step 4: Add KEYCLOAK_CLIENT_SECRET to .env.example**

Add to `.env.example`:

```
KEYCLOAK_REALM=docintel
KEYCLOAK_CLIENT_ID=docintel-api
KEYCLOAK_CLIENT_SECRET=docintel-api-secret
```

**Step 5: Commit**

```bash
git add docker/keycloak/realm-export.json docker/docker-compose.dev.yml .env.example
git commit -m "feat: add Keycloak realm export with roles and test users"
```

---

## Task 16: Integration Tests + Full Verification

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_full_flow.py`
- Modify: `tests/conftest.py`

**Step 1: Update conftest with DB fixtures**

Edit `tests/conftest.py`:

```python
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.main import app
from app.models import Base


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def api_client() -> AsyncClient:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client
```

**Step 2: Create integration test**

Create `tests/integration/__init__.py` (empty).

Create `tests/integration/test_full_flow.py`:

```python
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def api_client() -> AsyncClient:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.mark.anyio
async def test_health_endpoint(api_client: AsyncClient) -> None:
    """Health endpoint should return status."""
    resp = await api_client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "database" in data


@pytest.mark.anyio
async def test_document_list_empty(api_client: AsyncClient) -> None:
    """Document list should return empty paginated response."""
    resp = await api_client.get("/api/v1/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0
    assert data["page"] == 1
```

**Step 3: Run full test suite**

Run: `cd E:/LCMGoCloudRAG && python -m pytest tests/ -v`
Expected: All PASSED

**Step 4: Run lint**

Run: `cd E:/LCMGoCloudRAG && ruff check backend/ tests/ && ruff format --check backend/ tests/`
Expected: No errors. Fix any that appear.

**Step 5: Commit**

```bash
git add tests/conftest.py tests/integration/__init__.py tests/integration/test_full_flow.py
git commit -m "feat: add integration tests for Phase 2 endpoints"
```

---

## Post-Implementation Checklist

After all tasks are complete:

1. **Run full test suite:** `python -m pytest tests/ -v --tb=short`
2. **Lint check:** `ruff check backend/ tests/ && ruff format --check backend/ tests/`
3. **Docker stack up:** `docker compose -f docker/docker-compose.dev.yml up -d`
4. **Run migrations:** `cd backend && alembic upgrade head`
5. **Start FastAPI:** `cd backend && uvicorn app.main:app --reload`
6. **E2E verification:**
   - `GET /api/v1/health` — all services healthy
   - `POST /api/v1/documents/upload` — upload a PDF, get 202 with doc ID
   - `GET /api/v1/documents/{id}/status` — poll until completed
   - `GET /api/v1/documents` — see the uploaded document in list
   - `GET /api/v1/documents/{id}` — get document detail
   - `POST /api/v1/query` — query with citations
   - `DELETE /api/v1/documents/{id}` — cascade delete
   - `GET /api/v1/documents` — confirm empty list
