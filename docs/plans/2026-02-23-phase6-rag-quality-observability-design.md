# Phase 6: RAG Quality & Observability — Design Document

**Date:** 2026-02-23
**Status:** Approved
**Scope:** Semantic chunking, real Ollama streaming, conversation history, RAGAS evaluation, Prometheus/Grafana observability, security hardening

---

## Goal

Improve RAG quality, add real-time streaming, persist conversations, measure pipeline quality with RAGAS, add production observability, and harden security.

## Architecture Overview

Phase 6 builds on the production-ready Phase 5 infrastructure by fixing the two biggest functional gaps (simulated streaming and basic chunking), adding stateful conversations, proving quality with RAGAS evaluation, and layering on observability and security controls.

**Dependency chain:** Chunking → Streaming → Conversations → RAGAS → Observability → Security

---

## Workstream 1: Semantic/Structure-Aware Chunking

### Problem

Current chunking in `backend/app/pipelines/chunking.py` splits by `\n\n` paragraph boundaries. Docling exports to flat markdown via `export_to_markdown()`, discarding structural metadata (headings, tables, page boundaries). The `section_title`, `page_num`, and `language` columns in `document_chunks` are never populated.

### Approach

Replace `export_to_markdown()` with direct iteration over Docling's document model. Docling exposes `result.document.pages`, each page containing typed items (headings, paragraphs, tables, lists). Walk the document tree to:

1. **Track current section** — when a heading is encountered, store it as `section_title` for subsequent chunks
2. **Respect page boundaries** — never merge chunks across pages, set accurate `page_num`
3. **Keep tables intact** — if a table < 512 tokens, keep as single chunk; if larger, split by rows
4. **Detect language** — use `langdetect` on each chunk to populate `language` field
5. **Add contextual prefix** — generate a 2-3 sentence document summary via LLM on first ingestion, prepend to every chunk as context

### New/Modified Files

**New files:**
- `backend/app/pipelines/summarizer.py` — generate document summary for contextual prefix (cached in Document model)

**Modified files:**
- `backend/app/pipelines/chunking.py` — new `chunk_document()` function that takes Docling's parsed document model
- `backend/app/pipelines/ingestion.py` — `parse_document()` returns structured data; `create_document_chunks()` populates all metadata fields
- `backend/app/tasks/ingestion.py` — pass structured data through pipeline
- `backend/app/models/document.py` — add `summary` field to Document model
- New Alembic migration for `summary` column
- Updated chunking and ingestion tests

---

## Workstream 2: Real Ollama Streaming

### Problem

In `backend/app/api/v1/ws.py`, the WebSocket handler calls `query_documents()` which returns a complete answer string. Lines 224-228 split it by spaces and send fake tokens. The full LLM generation happens synchronously before any tokens are sent.

### Approach

Split the pipeline into two phases: **retrieval** (synchronous, returns context) and **generation** (streaming, token-by-token from Ollama).

1. **New function:** `retrieve_context()` in `query.py` — runs the existing pipeline up to and including reranking, returns ranked documents + formatted context. Does NOT call the LLM.
2. **Direct Ollama streaming:** Use `httpx.AsyncClient` to call Ollama's `/v1/chat/completions` with `stream=True`. Parse SSE chunks, extract token deltas.
3. **WebSocket handler:** Call `retrieve_context()`, then stream tokens directly from Ollama to the WebSocket as they arrive. Citations extracted from retrieved docs (already available before generation). Faithfulness check runs after full response is assembled.
4. **Graceful fallback:** If streaming fails mid-response, send error message and persist partial response.

### New/Modified Files

**New files:**
- `backend/app/pipelines/streaming.py` — `stream_llm_response()` async generator yielding tokens from Ollama

**Modified files:**
- `backend/app/pipelines/query.py` — extract `retrieve_context()` from `query_documents()`; keep `query_documents()` as non-streaming wrapper for REST endpoint
- `backend/app/api/v1/ws.py` — replace simulated streaming with real streaming loop
- `backend/app/config.py` — add `LLM_STREAM_TIMEOUT` setting

---

## Workstream 3: Conversation History Persistence

### Problem

Each query is independent. No `Conversation` model exists. Frontend `WebSocketManager.send()` has a `conversation_id` field defined but never used. Users lose chat context on page refresh.

### Approach

**Backend:**
1. **New model:** `Conversation` — `id` (UUID), `user_id` (FK), `title` (auto-generated from first question), `client_id`, `created_at`, `updated_at`
2. **Modify model:** `Query` — add `conversation_id` (FK to Conversation, nullable for backward compat)
3. **New endpoints:**
   - `GET /api/v1/conversations` — list user's conversations (paginated, most recent first)
   - `POST /api/v1/conversations` — create new conversation
   - `GET /api/v1/conversations/{id}/messages` — get conversation history
   - `DELETE /api/v1/conversations/{id}` — delete conversation + cascade
4. **WebSocket changes:** Accept `conversation_id` in query message. Load last N messages (configurable, default 5) and prepend to LLM context. Auto-create conversation if none provided.
5. **Title generation:** After first response, generate a short title via LLM (async, fire-and-forget).

**Frontend:**
1. **New component:** `ConversationList` — sidebar panel showing conversation history
2. **Modify:** `ChatInterface` — track active `conversation_id`, pass to WebSocket, load history on conversation switch
3. **Modify:** `Sidebar` — add conversation list below nav items
4. **New i18n keys:** `chat.newConversation`, `chat.conversations`, `chat.deleteConversation`, `chat.noConversations`

### New/Modified Files

**Backend new:**
- `backend/app/models/conversation.py`
- `backend/app/api/v1/conversations.py`
- New Alembic migration

**Backend modified:**
- `backend/app/models/query.py` — add `conversation_id` FK
- `backend/app/api/v1/ws.py` — conversation context loading
- `backend/app/api/v1/router.py` — register conversations router

**Frontend new:**
- `frontend/src/components/chat/ConversationList.tsx`

**Frontend modified:**
- `frontend/src/components/chat/ChatInterface.tsx`
- `frontend/src/components/Sidebar.tsx`
- `frontend/src/lib/api.ts` — conversation API functions
- `frontend/messages/en.json`, `frontend/messages/el.json`

---

## Workstream 4: RAGAS Evaluation Suite

### Problem

`tests/evaluation/` is empty. No automated way to measure RAG quality. CLAUDE.md targets: Faithfulness >0.95, Answer Relevancy >0.90, Context Precision >0.85, Context Recall >0.90.

### Approach

1. **Golden test set:** 20-30 curated question-answer pairs with expected source documents and ground truth answers. Mix Greek and English. Stored as `tests/evaluation/golden_set.json`.
2. **RAGAS evaluation script:** Runs each question through `query_documents()`, scores results with RAGAS library.
3. **Metrics:** Faithfulness, Answer Relevancy, Context Precision, Context Recall, HHEM score, Citation Coverage.
4. **CI integration:** Optional `ragas-eval` job via `workflow_dispatch` (too slow for every PR).
5. **Reporting:** HTML report via RAGAS, saved as CI artifact.

### New/Modified Files

- `tests/evaluation/golden_set.json`
- `tests/evaluation/conftest.py`
- `tests/evaluation/test_ragas.py`
- `tests/evaluation/test_citation_coverage.py`
- `backend/requirements.txt` — add `ragas>=0.2.0`
- `.github/workflows/ci.yml` — add optional evaluation job

---

## Workstream 5: Prometheus + Grafana Observability

### Problem

No metrics collected. No dashboards. Qdrant and Redis clients created ad-hoc (not pooled). No visibility into pipeline performance.

### Approach

**Metrics Collection:**
- `prometheus-fastapi-instrumentator` for auto HTTP metrics
- Custom metrics via `prometheus_client`:
  - `rag_query_duration_seconds`, `rag_retrieval_duration_seconds`, `rag_generation_duration_seconds` (histograms)
  - `rag_faithfulness_score` (histogram)
  - `rag_guardrail_blocks_total` (counter)
  - `celery_tasks_total`, `celery_task_duration_seconds`
  - `websocket_connections_active` (gauge)
  - `documents_ingested_total` (counter)
- Expose `GET /metrics` endpoint

**Connection Pooling:**
- New `backend/app/clients.py` — singleton Qdrant + Redis clients
- Initialized lazily, closed on app shutdown via lifespan
- Health endpoint uses singletons

**Docker Services:**
- Prometheus in `docker-compose.dev.yml` — scrapes `/metrics` every 15s
- Grafana with pre-provisioned dashboards (RAG Pipeline Overview, Celery Tasks, System Health)

### New/Modified Files

**New:**
- `backend/app/metrics.py`, `backend/app/clients.py`
- `docker/prometheus/prometheus.yml`
- `docker/grafana/dashboards/rag-overview.json`
- `docker/grafana/provisioning/datasources.yml`, `docker/grafana/provisioning/dashboards.yml`

**Modified:**
- `backend/requirements.txt` — add `prometheus-client`, `prometheus-fastapi-instrumentator`
- `backend/app/main.py` — instrument app, register metrics, lifespan cleanup
- `backend/app/api/v1/health.py` — use client singletons
- `backend/app/api/v1/ws.py` — WebSocket gauge
- `backend/app/tasks/ingestion.py` — Celery metrics
- `backend/app/pipelines/query.py` — timing histograms
- `backend/app/guardrails.py` — guardrail block counter
- `docker/docker-compose.dev.yml`, `docker/docker-compose.prod.yml` — add prometheus + grafana

---

## Workstream 6: Security Hardening

### Problem

No rate limiting, no CSRF protection, no security headers, no Redis caching. CORS allows `*` methods/headers.

### Approach

**Rate Limiting:**
- `slowapi` on FastAPI — auth endpoints (10/min), query (30/min per user), upload (10/min per user)
- Redis-backed storage (reuse connection pool from WS5)

**Security Headers (Backend):**
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 0`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: camera=(), microphone=(), geolocation=()`

**Security Headers (Frontend):**
- CSP in `next.config.ts` allowing Keycloak domain, self, inline styles
- `X-Frame-Options: DENY`

**Redis Query Caching:**
- Cache RAG results with TTL (5 min default, configurable)
- Key: hash of `(question, client_id)`
- Invalidate on document upload/delete

**CORS Tightening:**
- Explicit method and header lists instead of `*`

### New/Modified Files

**New:**
- `backend/app/middleware/rate_limit.py`
- `backend/app/middleware/security_headers.py`
- `backend/app/cache.py`

**Modified:**
- `backend/requirements.txt` — add `slowapi`
- `backend/app/main.py` — register middlewares, tighten CORS
- `backend/app/pipelines/query.py` — cache check/store
- `backend/app/api/v1/documents.py` — cache invalidation
- `backend/app/config.py` — rate limit + cache settings
- `frontend/next.config.ts` — security headers

---

## Dependency Order

```
WS1 (Chunking) → WS2 (Streaming) → WS3 (Conversations) → WS4 (RAGAS) → WS5 (Observability) → WS6 (Security)
```

- WS1 fixes the data foundation before WS2 changes how it's delivered
- WS2 must be done before WS3 (conversations need real streaming)
- WS4 evaluates the complete pipeline after WS1-3
- WS5 and WS6 are infrastructure layers independent of RAG logic but ordered last to avoid disruption

## Testing Strategy

**Backend:** Existing 71 tests continue passing. New tests for: chunking (structure-aware), streaming (mock Ollama SSE), conversations (CRUD + WebSocket context), cache (hit/miss/invalidation), rate limiting, metrics exposure.

**Frontend:** Existing 28 Vitest tests continue passing. New tests for: ConversationList, conversation switching in ChatInterface.

**E2E:** Existing 5 Playwright flows continue working. New flows for: conversation persistence, conversation switching.

**Evaluation:** RAGAS suite with 20-30 golden test pairs.

## Tech Stack Additions

| Component | Library/Tool | Purpose |
|-----------|-------------|---------|
| Language detection | `langdetect` | Per-chunk language identification |
| HTTP streaming | `httpx` | Async Ollama SSE streaming |
| RAG evaluation | `ragas` | Faithfulness, relevancy, precision, recall |
| Metrics | `prometheus-client` + `prometheus-fastapi-instrumentator` | Application metrics |
| Dashboards | Grafana | Pre-provisioned monitoring dashboards |
| Rate limiting | `slowapi` | Per-endpoint request throttling |
| Metrics scraping | Prometheus | Time-series metrics collection |
