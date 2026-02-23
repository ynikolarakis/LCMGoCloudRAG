# First Prompt for Claude Code

Copy and paste this as your first message in Claude Code after setting up MCP servers and placing all files in the project root.

---

```
I'm building LCM DocIntel — a self-hosted enterprise RAG platform for Greek and English documents. Read the CLAUDE.md, all agents in .claude/agents/, and all rules in .claude/rules/ to understand the full project.

Now bootstrap Phase 1 of the project. Here's exactly what I need:

## 1. Project Scaffolding
Create the full directory structure as defined in CLAUDE.md. Initialize:
- `backend/`: FastAPI project with app/, workers/, alembic/, requirements.txt
- `frontend/`: Next.js 15 project with TypeScript, Tailwind, shadcn/ui
- `terraform/`: Module structure (vpc, compute, database, storage, security, monitoring)
- `docker/`: docker-compose.dev.yml with Qdrant, PostgreSQL, Redis, Keycloak
- `tests/`: unit/, integration/, evaluation/ directories
- `scripts/`: setup-dev.sh (one-command local dev setup)
- `docs/`: placeholder architecture.md, api-spec.md, rag-pipeline.md, security.md, evaluation.md

## 2. Backend Foundation
- FastAPI app with health check endpoint (/api/v1/health)
- Settings class (Pydantic BaseSettings) with dev defaults for Ollama, Qdrant, PostgreSQL, Redis
- SQLAlchemy 2.0 async setup with Alembic
- Initial database models: User, Document, DocumentChunk, Query, Response, AuditLog
- First Alembic migration
- structlog configuration
- Basic middleware: CORS, request logging

## 3. Docker Dev Stack
- docker-compose.dev.yml that starts: Qdrant (6333), PostgreSQL (5432), Redis (6379), Keycloak (8080)
- .env.example with all required environment variables
- Ollama connects via host.docker.internal:11434 (runs on Windows host)

## 4. Basic RAG Pipeline (Proof of Concept)
Using Haystack 2.x, build a minimal working pipeline:
- Ingestion: accept a PDF file → parse with Docling → chunk (400-512 tokens) → embed with Ollama (Qwen3-Embedding) → store in Qdrant
- Query: accept a question → embed → search Qdrant (top-5) → send context + question to Ollama (Qwen3-30B) → return answer with citations
- Wire it up to two FastAPI endpoints: POST /api/v1/documents/upload and POST /api/v1/query

## 5. Verification
After scaffolding, run:
- docker compose -f docker/docker-compose.dev.yml up -d (verify all services start)
- pytest tests/ (verify test infrastructure works)
- A curl command I can use to test the upload + query flow end-to-end

Start with the project scaffolding and docker-compose, then build the backend and RAG pipeline. Use context7 for latest Haystack and Qdrant docs.
```
