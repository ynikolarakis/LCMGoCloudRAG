---
globs: backend/**/*.py
---

# Python Backend Rules

## Language & Version
- Python 3.12+ exclusively
- Strict type hints on ALL functions, variables, and return types
- Use `from __future__ import annotations` for forward references

## Style & Linting
- Ruff for linting AND formatting (single tool)
- Line length: 120 characters
- Target: py312
- Import order: stdlib → third-party → local (ruff handles automatically)
- Google-style docstrings on all public functions and classes
- No `print()` — use `structlog` for all logging

## FastAPI Patterns
- All endpoints: `async def` (never sync `def` for route handlers)
- Request/response: Pydantic v2 `BaseModel` subclasses (never return raw dicts)
- Auth: `Depends(get_current_user)` on all protected routes
- Versioning: `/api/v1/` prefix on all routes
- Errors: custom exception classes inheriting from `HTTPException`
- Health: `GET /api/v1/health` returns status of DB, Qdrant, LLM, Redis

## Pydantic Models
```python
# Always use Pydantic v2 patterns
class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    filename: str
    status: DocumentStatus
    created_at: datetime
```

## SQLAlchemy 2.0
- Use `create_async_engine` and `async_sessionmaker`
- All queries: `select()`, `insert()`, `update()` syntax (not legacy query API)
- Alembic for ALL schema changes (never raw SQL in application code)
- Connection pool: `pool_size=5`, `max_overflow=15` (configurable via env)

## Error Handling
- Custom exception hierarchy: `AppError → NotFoundError, ValidationError, AuthError, ...`
- Never bare `except:` or `except Exception:`
- Catch specific exceptions, log with structlog, re-raise as appropriate HTTP error
- All Celery tasks: try/except with retry logic (3 attempts, exponential backoff)

## Testing
- Framework: pytest + pytest-asyncio
- Fixtures: `db_session`, `api_client`, `mock_ollama`, `sample_document`
- Coverage: minimum 80% on new code
- Naming: `test_{function_name}_{scenario}_{expected_result}`
- Use `httpx.AsyncClient` for API integration tests
- RAGAS evaluation tests in `tests/evaluation/`

## Configuration
```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    # Never hardcode defaults for secrets
    DATABASE_URL: str
    REDIS_URL: str
    LLM_BASE_URL: str = "http://host.docker.internal:11434/v1"
    SECRET_KEY: SecretStr  # Forces SecretStr type — never logged
```

## Logging
```python
import structlog
logger = structlog.get_logger()

# Always include context
logger.info("document_ingested", doc_id=doc.id, chunks=chunk_count, duration_ms=elapsed)
logger.error("embedding_failed", doc_id=doc.id, error=str(e))
```
