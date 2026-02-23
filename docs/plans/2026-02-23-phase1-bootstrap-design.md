# Phase 1 Design: Project Bootstrap

**Date:** 2026-02-23
**Status:** Approved

## Scope

Bootstrap the LCM DocIntel enterprise RAG platform with:
1. Full project scaffolding (backend, frontend, terraform, docker, tests, scripts, docs)
2. FastAPI backend foundation (health check, settings, database, models, middleware)
3. Docker dev stack (Qdrant, PostgreSQL, Redis, Keycloak)
4. Basic RAG pipeline PoC (PDF upload + query with citations via Haystack 2.x)

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| PDF parsing | Docling | Full pipeline from day one, handles tables/structure |
| Chunking | Token-based (tiktoken) | 400-512 tokens, 15% overlap per RAG rules |
| Embeddings | Ollama OpenAI-compat API | Qwen3-Embedding-0.6B, 512 dims |
| Vector store | Qdrant via qdrant-client | Collection per client, HNSW cosine |
| LLM | Ollama OpenAI-compat API | Qwen3-30B-A3B, temp=0.1 |
| Pipeline orchestration | Haystack 2.x | Separate ingestion + query Pipeline objects |
| Database | SQLAlchemy 2.0 async + asyncpg | As specified in rules |
| Migrations | Alembic async | First migration creates all 6 tables |

## Database Models

- **User**: id, email, full_name, role (admin/manager/user/viewer), keycloak_id, client_id, is_active, timestamps
- **Document**: id, filename, original_filename, file_size, content_type, s3_key, status (queued/processing/completed/failed), language, page_count, chunk_count, client_id, uploaded_by FK, timestamps
- **DocumentChunk**: id, document_id FK, chunk_seq, content, section_title, page_num, language, token_count, embedding_id (Qdrant point ID), client_id, created_at
- **Query**: id, user_id FK, query_text, language, client_id, created_at
- **Response**: id, query_id FK, response_text, citations JSONB, faithfulness_score, latency_ms, model_used, created_at
- **AuditLog**: id, user_id, action enum, resource_type, resource_id, details JSONB, ip_address, client_id, created_at (immutable - no DELETE)

## RAG Pipeline (PoC)

**Ingestion:** PDF file -> Docling parse -> token chunking (400-512, 15% overlap) -> Ollama embed (Qwen3-Embedding) -> Qdrant write

**Query:** Question -> Ollama embed -> Qdrant retrieve (top-5 cosine) -> prompt build (citation template) -> Ollama generate (Qwen3-30B) -> response with [Source: file, page]

## Docker Dev Stack

- qdrant/qdrant:latest on 6333/6334
- postgres:16-alpine on 5432 (db: docintel, user: ragadmin)
- redis:7-alpine on 6379
- keycloak/keycloak:latest on 8080 (dev mode)
- Ollama on host via host.docker.internal:11434

## Explicitly Out of Scope

- Frontend (scaffold only)
- Celery async processing (synchronous ingestion in Phase 1)
- Keycloak auth integration (unprotected endpoints)
- Guardrails (NeMo, LLM Guard)
- Reranking (BGE-Reranker)
- BM25 hybrid search / HyDE / query expansion
- OCR (Surya/EasyOCR)
- AWS Terraform actual deployment (module scaffolding only)
