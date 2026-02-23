# Phase 4: Frontend + WebSocket Streaming — Design Document

**Date:** 2026-02-23
**Status:** Approved
**Scope:** WebSocket streaming backend endpoint, full Next.js frontend (chat, documents, admin), i18n (Greek+English), dev-mode auth bypass

---

## Goal

Make LCM DocIntel usable and demo-ready by building the full frontend: a chat interface with real-time WebSocket streaming, a document management portal, and an admin dashboard — all with Greek+English i18n support.

## Architecture Overview

Phase 4 adds a WebSocket streaming endpoint to the backend and builds three frontend pages on top of the existing REST API + new WebSocket.

**Approach:** Backend-first — build the WebSocket endpoint first, then the frontend foundation (deps, layout, API client, i18n), then all three pages.

**Auth:** Dev-mode bypass only. No Keycloak integration in Phase 4.

---

## Workstream 1: WebSocket Backend Endpoint

**New file:** `backend/app/api/v1/ws.py`

FastAPI WebSocket endpoint at `/api/v1/ws` for streaming RAG queries.

**Protocol (JSON Lines):**

Client sends:
```json
{"type": "query", "question": "What is the contract term?", "conversation_id": "uuid"}
```

Server streams back:
```json
{"type": "status", "status": "processing"}
{"type": "token", "content": "The"}
{"type": "token", "content": " contract"}
{"type": "citations", "citations": [{"source": "contract.pdf", "page": 3, "content_preview": "..."}]}
{"type": "done", "latency_ms": 1234}
```

On error:
```json
{"type": "error", "detail": "Query blocked by guardrail: prompt_injection"}
```

**Streaming from Ollama:** The retrieval pipeline (embed → hybrid retrieve → rerank) runs first synchronously. Then the LLM generation step uses the OpenAI Python client with `stream=True` on the `/v1/chat/completions` endpoint, which Ollama supports natively. Tokens are forwarded to the WebSocket as they arrive.

**Guardrails:**
- Input guardrails (LLM Guard) run before retrieval
- Output guardrails (HHEM faithfulness) run after the full response is buffered — can't check partial tokens. If HHEM fails, a correction message replaces the streamed answer.

**Heartbeat:** Server sends `{"type": "ping"}` every 30s. Client responds with `{"type": "pong"}`.

**DB persistence:** Query and QueryResponse rows are created in PostgreSQL, same as the REST endpoint. Audit log entries written for query_submitted and response_generated.

**New backend endpoint:** `GET /api/v1/admin/audit-logs` — paginated audit log listing (ADMIN role required). Returns `{items: [...], total, page, size}` with optional `action` filter.

---

## Workstream 2: Frontend Foundation

**New dependencies:**

| Package | Purpose |
|---------|---------|
| `@tanstack/react-query` | Server state management |
| `next-intl` | i18n with locale routing (`/en/chat`, `/el/chat`) |
| `react-hook-form` + `zod` + `@hookform/resolvers` | Form validation |
| `lucide-react` | Icons |
| `clsx` + `tailwind-merge` | Class merging (shadcn pattern) |
| `dompurify` + `@types/dompurify` | Sanitize LLM output |

**shadcn/ui components:** button, input, card, dialog, badge, table, toast, tabs, dropdown-menu, separator, scroll-area. Copied into `src/components/ui/`.

**Directory structure:**
```
frontend/src/
├── app/
│   └── [locale]/
│       ├── layout.tsx          ← sidebar nav + providers
│       ├── chat/page.tsx
│       ├── documents/page.tsx
│       └── admin/page.tsx
├── components/
│   ├── ui/                     ← shadcn/ui primitives
│   ├── chat/                   ← ChatInterface, MessageList, MessageBubble, CitationCard
│   ├── documents/              ← UploadForm, DocumentList, DocumentStatusBadge
│   └── admin/                  ← HealthPanel, AuditLogTable
├── lib/
│   ├── api.ts                  ← typed fetch wrappers for REST endpoints
│   ├── websocket.ts            ← WebSocket manager
│   └── utils.ts                ← cn() helper, formatDate
├── hooks/
│   ├── use-documents.ts        ← TanStack Query hooks for document CRUD
│   └── use-health.ts           ← TanStack Query hook for health endpoint
└── i18n/
    ├── request.ts              ← next-intl getRequestConfig
    └── routing.ts              ← locale routing config
```

**Layout:** Sidebar navigation (Chat / Documents / Admin) with locale switcher (EN/EL). Main content area to the right. Responsive — sidebar collapses to hamburger on mobile.

**API client (`lib/api.ts`):** Typed functions wrapping `fetch()` against `http://localhost:8000/api/v1/`. No auth tokens in Phase 4 (dev bypass). Returns typed responses matching backend Pydantic schemas.

---

## Workstream 3: Chat Page

**Route:** `/[locale]/chat`

**Components:**
- `ChatInterface` — main container, manages conversation state and WebSocket connection
- `MessageList` — scrollable list of user questions and assistant responses
- `MessageBubble` — single message. Assistant bubbles render markdown and stream tokens in real-time
- `CitationCard` — citation reference rendered as clickable chip, expands to show content preview
- `ConnectionIndicator` — WebSocket status dot (connected/reconnecting/disconnected)

**Data flow:**
1. User types question, hits Send
2. `ChatInterface` sends `{"type": "query", "question": "..."}` over WebSocket
3. As `token` messages arrive, `MessageBubble` appends text character-by-character
4. When `citations` message arrives, citation chips render below the answer
5. On `done`, message marked complete, input re-enables
6. On `error`, error toast, input re-enables

**Conversation state:** React `useState` array of `{role, content, citations, isStreaming}`. No server-side conversation persistence in Phase 4 — refresh clears history. Individual queries/responses still persisted in PostgreSQL by the backend.

**LLM output sanitization:** All assistant content passes through DOMPurify before rendering.

**WebSocket manager (`lib/websocket.ts`):**
- Auto-reconnect with exponential backoff (1s, 2s, 4s, 8s, 16s — max 5 retries)
- Heartbeat: responds to server `ping` with `pong`
- Exposes `onToken`, `onCitations`, `onDone`, `onError` callbacks
- `send(message)` queues if disconnected, sends on reconnect

---

## Workstream 4: Documents Page

**Route:** `/[locale]/documents`

**Components:**
- `UploadForm` — drag-and-drop zone + file picker. Validates file type (PDF, DOCX, TXT) and size (max 50MB) via Zod schema. POSTs to `/api/v1/documents/upload`. Shows progress toast.
- `DocumentList` — table from `GET /api/v1/documents`. Columns: filename, status, page count, uploaded date, actions. Pagination (page/size). Auto-refetches every 5s while any document is queued/processing.
- `DocumentStatusBadge` — colored badge: queued (yellow), processing (blue pulse), completed (green), failed (red)
- `DeleteConfirmDialog` — shadcn AlertDialog. Calls `DELETE /api/v1/documents/{id}`. Invalidates query cache on success.

**Error handling:** Upload failures show toast. Network errors show retry prompt. Failed documents show error detail on hover.

---

## Workstream 5: Admin Page

**Route:** `/[locale]/admin`

**Components:**
- `AdminPage` — tabbed layout: Health and Audit Log tabs
- `HealthPanel` — calls `GET /api/v1/health` via TanStack Query (refetch every 10s). Status cards for Database, Qdrant, Redis, LLM with green/red indicators.
- `AuditLogTable` — paginated audit log entries from `GET /api/v1/admin/audit-logs` (newest first). Columns: timestamp, user, action, resource type, details (truncated, expandable). Filter by action type dropdown. Read-only — audit logs are immutable.

**No user management in Phase 4** — dev bypass means single user. User CRUD deferred to Keycloak auth phase.

---

## Testing Strategy

**Backend:**
- Unit test for WebSocket endpoint (mock pipeline, verify message protocol)
- Unit test for audit log listing endpoint (pagination, role check)
- Existing 65 tests must continue passing

**Frontend:**
- Vitest + React Testing Library for component unit tests
- Key tests: ChatInterface renders messages, UploadForm validates file types, DocumentList shows status badges, HealthPanel renders service cards
- No Playwright E2E in Phase 4 — deferred to CI/CD phase
- `data-testid` attributes on all interactive elements

---

## Tech Stack Additions

| Component | Library | Purpose |
|-----------|---------|---------|
| Server state | `@tanstack/react-query` v5 | Document list, health, audit log |
| i18n | `next-intl` | Greek+English locale routing |
| Forms | `react-hook-form` + `zod` | Upload validation |
| UI primitives | `shadcn/ui` | Buttons, cards, tables, dialogs |
| Icons | `lucide-react` | Navigation and status icons |
| Sanitization | `dompurify` | XSS prevention on LLM output |
| WebSocket | Native browser API | Streaming chat |
| Streaming LLM | `openai` Python client (stream=True) | Token-by-token generation |

## Dependency Order

```
Workstream 1 (WebSocket backend) ──┐
                                    ├─→ Workstream 3 (Chat page)
Workstream 2 (Frontend foundation) ┤
                                    ├─→ Workstream 4 (Documents page)
                                    └─→ Workstream 5 (Admin page)
```

- Workstream 1 (backend) and 2 (frontend foundation) are independent of each other
- Workstreams 3, 4, 5 depend on Workstream 2 (foundation)
- Workstream 3 (chat) depends on Workstream 1 (WebSocket)
- Workstreams 4 and 5 only need REST, not WebSocket
