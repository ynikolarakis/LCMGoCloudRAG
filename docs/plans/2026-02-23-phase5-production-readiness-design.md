# Phase 5: Production Readiness — Design Document

**Date:** 2026-02-23
**Status:** Approved
**Scope:** Frontend Keycloak auth, production Docker builds, CI/CD pipeline (GitHub Actions → ECR → ECS), Playwright E2E tests

---

## Goal

Make LCM DocIntel deployable and secure by adding real authentication, production-grade containers, automated CI/CD, and end-to-end test coverage.

## Architecture Overview

Phase 5 builds on the functional MVP from Phase 4 by adding the production infrastructure layer. The approach is auth-first: Keycloak frontend integration changes the app fundamentally (protected routes, token flow, role-based rendering), then Docker and CI/CD containerize and automate the auth-integrated app, and E2E tests validate the complete system.

**Dependency chain:** Auth → Docker → CI/CD → E2E

---

## Workstream 1: Frontend Keycloak Authentication

### Approach

Use `keycloak-js` adapter (official Keycloak JS library) initialized in a React context provider. The adapter handles the full OIDC Authorization Code flow — redirect to Keycloak login page, token exchange, silent refresh.

### Token Strategy

The `keycloak-js` adapter manages tokens in memory. No JWTs in localStorage (per security rules).

- Access token (15-min TTL) held in the adapter instance, injected into API requests via header interceptor
- Refresh token (7-day TTL) managed automatically by `keycloak-js` with `onTokenExpired` callback triggering silent refresh
- Adapter's `init({ onLoad: 'login-required' })` forces login before any page renders

### Keycloak Client

Add a **public** client (`docintel-frontend`) to the realm export. The existing `docintel-api` client is confidential (has a secret) — not suitable for browser-based OIDC. Public clients use PKCE instead of client secrets.

### New/Modified Files

**New files:**
- `frontend/src/lib/auth.ts` — Keycloak instance singleton, init config
- `frontend/src/components/AuthProvider.tsx` — React context wrapping keycloak-js, exposes `user`, `roles`, `token`, `logout()`
- `frontend/src/components/AuthGuard.tsx` — wrapper that checks role before rendering children

**Modified files:**
- `frontend/src/lib/api.ts` — inject `Authorization: Bearer <token>` from auth context
- `frontend/src/components/Providers.tsx` — wrap with AuthProvider
- `frontend/src/components/Sidebar.tsx` — hide Admin link for non-admin users, add logout button
- `frontend/src/app/[locale]/admin/page.tsx` — wrap with role guard
- `docker/keycloak/realm-export.json` — add `docintel-frontend` public client
- `backend/app/main.py` — CORS origins from env var instead of hardcoded

---

## Workstream 2: Production Docker Builds

### Dockerfiles (multi-stage, non-root)

**`docker/Dockerfile.backend`** — FastAPI production image:
- Stage 1: `python:3.12-slim` builder — install deps into virtualenv
- Stage 2: `python:3.12-slim` runtime — copy virtualenv + app code only
- Runs: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4`
- Non-root user (`appuser`), health check via curl

**`docker/Dockerfile.frontend`** — Next.js production image:
- Stage 1: `node:20-alpine` deps — `npm ci`
- Stage 2: `node:20-alpine` builder — `npm run build` (standalone output)
- Stage 3: `node:20-alpine` runner — copy `.next/standalone` + `public` + `static`
- Runs: `node server.js` (Next.js standalone mode)
- Non-root user, port 3000

**`docker/Dockerfile.worker`** — Update existing to multi-stage (same pattern as backend, different CMD)

### Next.js Config Change

Add `output: "standalone"` to `next.config.ts` for Docker-optimized builds.

### Production Compose

**`docker/docker-compose.prod.yml`:**
- All services use pre-built images (`${ECR_REGISTRY}/docintel-*:${TAG}`)
- Environment from `.env.prod` file
- Keycloak in production mode (not `start-dev`)
- Health checks on all services
- Restart policies: `unless-stopped`

---

## Workstream 3: CI/CD Pipeline (GitHub Actions)

### CI Workflow (`.github/workflows/ci.yml`)

**Trigger:** push to any branch, pull requests to `master`

**Jobs (parallel where possible):**
1. `backend-lint` — `ruff check . && ruff format --check .`
2. `backend-test` — PostgreSQL + Redis + Qdrant service containers, `pytest tests/ -x -v` with coverage
3. `frontend-lint` — `npm run lint && npx tsc --noEmit`
4. `frontend-test` — `npm run test`
5. `frontend-build` — `npm run build`
6. `secrets-check` — `gitleaks detect --source .`
7. `e2e-test` — depends on backend-test + frontend-build. Full stack via docker-compose, Playwright tests

### CD Workflow (`.github/workflows/cd.yml`)

**Trigger:** push to `master` only (after CI passes)

**Steps:**
1. Build Docker images (backend, frontend, worker)
2. Tag with git SHA + `latest`
3. Push to ECR via `aws-actions/amazon-ecr-login`
4. Update ECS task definitions via `aws-actions/amazon-ecs-deploy-task-definition`
5. Wait for ECS service stability

**AWS auth:** GitHub OIDC provider with role assumption (`aws-actions/configure-aws-credentials`). No static access keys.

**Required GitHub secrets:** `AWS_ROLE_ARN`, `AWS_REGION`, `ECR_REGISTRY`

---

## Workstream 4: Playwright E2E Tests

### Config

- `frontend/playwright.config.ts` — Chromium only, `http://localhost:3000` base URL, 30s timeout
- `frontend/e2e/` — test directory

### Test Flows (5 tests)

1. **Login flow** — navigate to `/en/chat`, redirect to Keycloak, enter admin credentials, land on chat page
2. **Upload document** — upload test PDF, verify it appears in list, wait for status progression (queued → processing → completed)
3. **Chat query + streaming** — type question, verify connection indicator, verify tokens stream in, verify citations appear
4. **Admin dashboard** — verify health panel (4 cards), audit log table loads, action filter works
5. **Role-based access** — login as non-admin, verify admin tab hidden, verify direct URL access denied

### Fixtures & Auth Setup

- `e2e/fixtures/sample.pdf` — small 2-page test PDF
- `e2e/auth.setup.ts` — Playwright auth setup, logs in via Keycloak, saves storage state for test reuse

### CI Integration

E2E job spins up full stack via docker-compose, waits for health checks, runs `npx playwright test`. Screenshots and traces uploaded as artifacts on failure.

---

## Testing Strategy

**Backend:** Existing 71 tests continue passing. No new backend tests in Phase 5 (auth validation already tested).

**Frontend:** Existing 24 Vitest tests continue passing. New unit tests for AuthProvider and AuthGuard components.

**E2E:** 5 Playwright tests covering core user flows with real Keycloak login.

---

## Dependency Order

```
Workstream 1 (Keycloak auth) ──→ Workstream 2 (Docker builds) ──→ Workstream 3 (CI/CD) ──→ Workstream 4 (E2E tests)
```

- Each workstream depends on the previous one
- Workstream 2 containerizes the auth-integrated app
- Workstream 3 automates building/deploying the containers
- Workstream 4 validates the full deployed stack

## Tech Stack Additions

| Component | Library/Tool | Purpose |
|-----------|-------------|---------|
| Auth adapter | `keycloak-js` | OIDC Authorization Code flow with PKCE |
| E2E testing | `@playwright/test` | Browser-based integration tests |
| CI/CD | GitHub Actions | Automated lint, test, build, deploy |
| Container registry | AWS ECR | Docker image storage |
| Container orchestration | AWS ECS Fargate | Production deployment |
| Secrets | GitHub OIDC + IAM roles | Keyless AWS authentication in CI |
