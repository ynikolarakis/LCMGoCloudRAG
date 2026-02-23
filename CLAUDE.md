# LCM DocIntel — Enterprise Multilingual RAG Platform

## Project Overview
Self-hosted, open-source RAG platform for enterprise clients (legal firms, production companies).
Per-client AWS deployments with physical isolation (separate VPC per client).
Languages: Greek + English with cross-lingual retrieval.
Hallucination tolerance: near-zero. Every answer MUST cite sources.

## Architecture
- **Backend:** FastAPI + Haystack 2.x + Celery + Redis
- **Frontend:** Next.js 15 (App Router) + TypeScript + Tailwind + shadcn/ui
- **LLM:** Qwen3-30B-A3B (Ollama dev / vLLM prod)
- **Embeddings:** Qwen3-Embedding-0.6B (512 dims, multilingual)
- **Reranker:** BGE-Reranker-v2-m3 (cross-encoder, multilingual)
- **Vector DB:** Qdrant (hybrid search, payload filtering for ACLs)
- **Database:** PostgreSQL 16 (metadata, users, audit logs)
- **Auth:** Keycloak (SSO/LDAP/OIDC, per-client realm)
- **IaC:** Terraform (per-client VPC modules)
- **Containers:** Docker (dev) + AWS ECS (prod)
- **OCR:** Docling + Surya + EasyOCR (Greek/English)
- **Guardrails:** NeMo Guardrails + LLM Guard + HHEM hallucination scoring

## Project Structure
```
backend/          → FastAPI app, Haystack pipelines, Celery tasks
frontend/         → Next.js app (chat UI, document portal, admin dashboard)
terraform/        → Per-client infrastructure modules
docker/           → Dockerfiles and compose files
docs/             → Architecture docs, API specs, runbooks
tests/            → Pytest + RAGAS evaluation suites
scripts/          → Utility scripts (setup, provisioning, model download)
```

## Dev Environment
- **OS:** Windows 11 Enterprise + WSL2 (Ubuntu)
- **CPU:** Ryzen 9 7950X3D (16C/32T, 128MB V-Cache)
- **RAM:** 64 GB
- **GPU:** RTX 4070 (12 GB VRAM)
- **LLM serving:** Ollama on Windows host (Qwen3-30B-A3B with GPU+RAM offload, ~10-15 tok/s)
- **Ollama endpoint:** http://host.docker.internal:11434
- **Docker services:** Qdrant (6333), PostgreSQL (5432), Redis (6379), Keycloak (8080)
- **FastAPI + Next.js:** run directly in WSL2 with hot-reload

## Dev vs Prod Endpoints
```
# Dev (local Ollama)
LLM_BASE_URL=http://host.docker.internal:11434/v1
EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1

# Prod (AWS vLLM on g5.2xlarge)
LLM_BASE_URL=http://vllm:8000/v1
EMBEDDING_BASE_URL=http://vllm-embed:8000/v1
```
Same OpenAI-compatible API interface. Only URL changes.

## Key Rules
1. **Citations mandatory:** EVERY response must include `[Source: filename, page]` citations
2. **Encryption everywhere:** AES-256/KMS at rest, TLS 1.3 in transit
3. **No hardcoded secrets:** Use `.env` (dev) or AWS Secrets Manager (prod)
4. **Client isolation:** Separate VPC, Qdrant collection, PostgreSQL schema, S3 bucket, KMS key per client
5. **Audit everything:** All queries + responses logged to immutable audit table
6. **Python standards:** 3.12+, strict type hints, ruff lint+format, pytest (80% coverage minimum)
7. **TypeScript standards:** strict mode, no `any`, functional components only
8. **Terraform standards:** fmt, validate, plan before apply. Tag ALL resources.
9. **If no answer found:** Respond "Cannot find answer in the available documents" — never hallucinate

## Before Committing
```bash
ruff check . && ruff format --check .       # Python lint
pytest tests/ -x                             # Python tests
tsc --noEmit                                 # TypeScript check (frontend/)
terraform validate                           # If terraform/ changed
gitleaks detect --source .                   # No secrets in code
```

## Reference Docs
- @docs/architecture.md    → Full platform architecture
- @docs/rag-pipeline.md    → RAG pipeline design details
- @docs/security.md        → Security architecture & GDPR
- @docs/api-spec.md        → REST + WebSocket API contracts
- @docs/evaluation.md      → RAGAS evaluation methodology
