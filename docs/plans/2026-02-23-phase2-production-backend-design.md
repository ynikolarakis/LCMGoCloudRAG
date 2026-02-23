# Phase 2 Design: Production-Ready Backend

**Date:** 2026-02-23
**Status:** Approved

## Scope

Make the backend production-ready by wiring up database persistence, async ingestion, authentication, and document management. Six workstreams:

1. Auth dependency with dev bypass
2. DB persistence for document uploads
3. DB persistence for queries/responses
4. Celery async ingestion
5. Keycloak JWT validation + RBAC
6. Document CRUD endpoints + audit logging

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Auth dev mode | Dev bypass with seed user | Testing ergonomics — no JWT needed in dev |
| Auth prod mode | Keycloak JWT + JWKS validation | Per CLAUDE.md spec |
| Ingestion mode | Celery async (Redis broker) | Upload returns 202, worker processes in background |
| File storage | Local uploads dir | S3 in prod later; local for Phase 2 |
| Cascade delete | DB chunks + Qdrant points + Document row | GDPR right-to-erasure compliance |
| Audit logging | Direct INSERT helper | Simple, synchronous; Celery audit queue in prod later |
| Role enforcement | RoleChecker dependency | Minimum role per endpoint |

## Workstream 1: Auth Dependency

### Dev Mode (ENVIRONMENT=dev)
- `get_current_user` returns a default "dev-admin" User
- `seed_dev_user()` runs at app startup, creates user if not exists
- Dev user: email=dev@docintel.local, role=admin, client_id=default

### Prod Mode
- Extract JWT from `Authorization: Bearer <token>` header
- Validate signature against Keycloak JWKS endpoint (cached)
- Extract `sub` (keycloak_id) and `realm_access.roles`
- Look up User by keycloak_id, return User object
- 401 if missing/invalid/expired token
- 403 if insufficient role

### RoleChecker
```python
class RoleChecker:
    def __init__(self, minimum_role: UserRole):
        self.minimum_role = minimum_role

    async def __call__(self, user: User = Depends(get_current_user)):
        if user.role.value < self.minimum_role.value:
            raise ForbiddenError()
        return user
```

### RBAC Matrix

| Endpoint | Minimum Role |
|----------|-------------|
| POST /documents/upload | user |
| GET /documents | user (own) / manager (all) |
| GET /documents/{id} | user (own) / manager (all) |
| DELETE /documents/{id} | manager |
| GET /documents/{id}/status | user (own) |
| POST /query | user |
| GET /health | (no auth) |

## Workstream 2: DB Persistence for Uploads

### Flow
1. Validate file (type whitelist, 50MB max)
2. Create Document row: `status=queued`, `uploaded_by=current_user.id`, `client_id`, `file_size`, `content_type`, `original_filename`
3. Generate filename: `{doc_id}{suffix}`, save to `uploads/{client_id}/`
4. Dispatch Celery task (or run sync in dev fallback)
5. Return 202 with `{id, filename, status: "queued"}`

### Celery Worker
1. Update Document `status=processing`
2. Docling parse -> get text + page_count
3. chunk_text() -> get chunks
4. For each chunk: INSERT DocumentChunk (chunk_seq, content, page_num, token_count, client_id)
5. Embed + write to Qdrant via Haystack pipeline
6. Update DocumentChunk.embedding_id with Qdrant point IDs
7. Update Document: `status=completed`, `chunk_count`, `page_count`
8. On failure: `status=failed`, log error, retry up to 3x with exponential backoff

### File Organization
```
uploads/
  default/           # client_id
    {doc_id}.pdf
    {doc_id}.docx
```

## Workstream 3: DB Persistence for Queries

### Flow
1. Create Query row: `user_id`, `query_text`, `language`, `client_id`
2. Run existing Haystack query pipeline
3. Create QueryResponse row:
   - `response_text`: LLM answer
   - `citations`: `[{"source": "file.pdf", "page": 3, "chunk_id": "uuid"}]` JSONB
   - `faithfulness_score`: null (Phase 3 with HHEM)
   - `latency_ms`: from pipeline timing
   - `model_used`: from settings
4. Write audit log: `query_submitted`
5. Return response with DB-generated IDs

## Workstream 4: Celery Async Ingestion

### Configuration
- Broker: Redis (same instance as cache)
- Result backend: Redis
- Serializer: JSON
- `task_acks_late = True`
- `task_reject_on_worker_lost = True`
- Max retries: 3, exponential backoff (10s, 60s, 300s)
- Concurrency: 2 workers (CPU-bound Docling parsing)

### Task
```python
@celery_app.task(bind=True, max_retries=3)
def ingest_document_task(self, doc_id: str, file_path: str, client_id: str) -> dict:
    """Parse, chunk, embed, and store a document."""
```

### Status Endpoint
```
GET /api/v1/documents/{id}/status
Response: {id, status, chunk_count, page_count, error_message, created_at, updated_at}
```

## Workstream 5: Keycloak JWT Validation

### Docker Setup
- Keycloak realm: `docintel` (auto-created via realm export JSON in docker/)
- Client: `docintel-api` (confidential, service account enabled)
- Roles: admin, manager, user, viewer
- Test users: admin@test.com (admin), user@test.com (user)

### JWT Middleware
- Library: `python-jose[cryptography]` for JWT decode
- JWKS endpoint: `{KEYCLOAK_URL}/realms/docintel/protocol/openid-connect/certs`
- Cache JWKS keys for 1 hour (httpx fetch + cache)
- Validate: issuer, audience, expiration, signature

### Settings Additions
```
KEYCLOAK_REALM=docintel
KEYCLOAK_CLIENT_ID=docintel-api
KEYCLOAK_CLIENT_SECRET=<from env>
```

## Workstream 6: Document CRUD + Audit

### Endpoints

**GET /api/v1/documents**
- Pagination: `?page=1&page_size=20`
- Filters: `?status=completed`, `?language=en`
- Sorted by `created_at` descending
- Users see own docs; managers see all client docs

**GET /api/v1/documents/{id}**
- Returns Document with chunk metadata (count, not full content)
- 404 if not found or not authorized

**DELETE /api/v1/documents/{id}**
- Cascade: delete Qdrant points by doc_id filter → delete DocumentChunk rows → delete Document row
- Write audit_log: `document_deleted`
- Manager+ role required
- Returns 204 No Content

**GET /api/v1/documents/{id}/status**
- Lightweight: returns only status fields (for polling during async ingestion)

### Audit Helper
```python
async def write_audit_log(
    session: AsyncSession,
    user_id: UUID | None,
    action: AuditAction,
    resource_type: str,
    resource_id: UUID | None,
    details: dict | None = None,
    ip_address: str | None = None,
    client_id: str = "default",
) -> None:
    """Write an immutable audit log entry."""
```

Called from:
- Document upload (document_uploaded)
- Document delete (document_deleted)
- Query submitted (query_submitted)
- Response generated (response_generated)

## Explicitly Out of Scope

- Frontend chat UI (Phase 3)
- Hybrid search / BM25 / HyDE / query expansion (Phase 3+)
- Reranking with BGE-Reranker (Phase 3+)
- Guardrails / NeMo / LLM Guard / HHEM (Phase 3+)
- RAGAS evaluation suite (Phase 3+)
- S3 file storage (local uploads dir for now)
- WebSocket streaming (Phase 3 with frontend)
- Per-client Keycloak realm management (single realm for now)
- Cross-lingual search (Phase 3+)
- OCR with Surya/EasyOCR (Phase 3+)
