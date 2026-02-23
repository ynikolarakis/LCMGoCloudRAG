# Phase 3: RAG Quality Stack — Design Document

**Date:** 2026-02-23
**Status:** Approved
**Scope:** Backend RAG pipeline upgrades — async ingestion, ACL filtering, hybrid search, reranking, guardrails, cross-lingual query expansion

---

## Goal

Make the RAG pipeline production-grade: fix critical gaps (synchronous ingestion, missing tenant isolation), add hybrid search with reranking for retrieval quality, add guardrails for safety, and enable cross-lingual Greek+English retrieval.

## Architecture Overview

Phase 3 upgrades the query pipeline from a simple dense retriever to a full hybrid search → rerank → guardrails pipeline, while fixing async ingestion and tenant isolation on the ingestion side.

**Before (Phase 2):**
```
Query → dense embed → Qdrant top-5 → prompt → LLM → response
```

**After (Phase 3):**
```
Query → input guardrails → query expansion (Greek+English variants)
      → sparse embed (BM25) + dense embed (Qwen3) per variant
      → QdrantHybridRetriever (RRF fusion, top-50, ACL filtered)
      → BGE-Reranker-v2-m3 (top-5, threshold 0.3)
      → prompt → LLM → HHEM faithfulness check → output guardrails → response
```

## Tech Stack Additions

| Component | Library/Model | Purpose |
|-----------|--------------|---------|
| Sparse embedding | `Qdrant/bm25` via FastEmbed | BM25 sparse vectors (language-agnostic) |
| Hybrid retrieval | `QdrantHybridRetriever` (Haystack) | RRF fusion of dense + sparse |
| Reranker | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranking (multilingual) |
| Input guardrails | LLM Guard (`llm-guard`) | Prompt injection + PII detection |
| Faithfulness scoring | HHEM 2.1 (`vectara/hallucination_evaluation_model`) | Hallucination detection |
| Query expansion | Qwen3-30B via Ollama | Cross-lingual variant generation |

## New Dependencies

```
fastembed>=0.4.0
fastembed-haystack>=1.0.0
llm-guard>=0.3.0
sentence-transformers>=3.0.0
```

---

## Workstream 1: Async Ingestion

**Problem:** Upload endpoint calls `ingest_document()` synchronously, blocking HTTP response for 30-60s.

**Solution:**
- Upload endpoint dispatches `ingest_document_task.apply_async()` and returns 202 with `status=QUEUED` immediately
- Celery worker picks up task, processes document (parse → chunk → embed → store)
- Client polls `GET /documents/{id}/status` for progress
- Add Celery worker service to `docker-compose.dev.yml`

**Files modified:**
- `backend/app/api/v1/documents.py` — replace sync `ingest_document()` call with Celery dispatch
- `docker/docker-compose.dev.yml` — add `celery-worker` service

**Data flow:**
```
Client → POST /upload → save file + create Document(QUEUED) → Celery dispatch → 202 returned
                                                                    ↓
                                              Worker picks up → PROCESSING → parse/chunk/embed → COMPLETED/FAILED
```

---

## Workstream 2: ACL Filtering

**Problem:** `client_id` stored in Qdrant payload but never used in retrieval. Multi-tenancy breach — users can see results from other clients' documents.

**Solution:**
- Add `client_id` must-filter to Qdrant retrieval at query time
- Filter applied BEFORE similarity search (efficient, not post-filtering)

**Files modified:**
- `backend/app/pipelines/query.py` — pass `filters` parameter to retriever with `client_id == current_user.client_id`

---

## Workstream 3: Hybrid Search

**Problem:** Dense-only retrieval misses keyword matches. Important for legal/technical documents where exact terms matter.

**Solution:**
- Add BM25 sparse embedding via FastEmbed during ingestion
- Switch `QdrantDocumentStore` to `use_sparse_embeddings=True`
- Switch retriever from `QdrantEmbeddingRetriever` to `QdrantHybridRetriever` with RRF fusion
- Re-indexing migration for existing documents

**Sparse model choice:** `Qdrant/bm25` — language-agnostic (works for Greek+English). SPLADE++ is English-only, BM42 has mixed benchmarks.

**Ingestion pipeline (updated):**
```
Docling parse → chunk → FastembedSparseDocumentEmbedder (BM25) → OpenAIDocumentEmbedder (Qwen3) → DocumentWriter
```

**Query pipeline (updated):**
```
Question → FastembedSparseTextEmbedder → QdrantHybridRetriever (RRF, top_k=50)
         → OpenAITextEmbedder         ↗
```

**Migration:** Script to recreate Qdrant collection with sparse vector support, re-embed all existing DocumentChunks from PostgreSQL.

**Files modified/created:**
- `backend/app/pipelines/ingestion.py` — add sparse embedder to pipeline
- `backend/app/pipelines/query.py` — switch to hybrid retriever
- `backend/requirements.txt` — add fastembed, fastembed-haystack
- `scripts/reindex_hybrid.py` — migration script

---

## Workstream 4: Reranking

**Problem:** Retrieval returns top-50 candidates ranked by vector similarity, which may not reflect true relevance to the query.

**Solution:**
- Add `TransformersSimilarityRanker` with `BAAI/bge-reranker-v2-m3` after hybrid retrieval
- Cross-encoder takes (query, passage) pairs, outputs relevance scores
- Top-50 → reranked → top-5 with minimum score threshold 0.3

**Configuration:**
- `model="BAAI/bge-reranker-v2-m3"` (~568M params, multilingual)
- `top_k=5`, `score_threshold=0.3`, `scale_score=True`
- CPU inference acceptable for dev (~1-2s for 50 candidates)
- Model auto-downloads from HuggingFace on first `warm_up()`

**Files modified:**
- `backend/app/pipelines/query.py` — add ranker component after retriever
- `backend/app/config.py` — add `RERANKER_TOP_K`, `RERANKER_THRESHOLD` settings

---

## Workstream 5: Guardrails

**Problem:** No input validation (prompt injection, PII) or output validation (hallucination detection, PII leakage).

**Solution:** LLM Guard for scanning + HHEM for faithfulness scoring. No NeMo Guardrails in Phase 3 — it requires its own LLM reasoning calls (2-5s latency) and Colang configuration. Can add in a later phase if dialog-level rails are needed.

**Input guardrails (pre-pipeline):**
- LLM Guard `PromptInjection` scanner — detects injection attacks
- LLM Guard `Anonymize` scanner — detects PII in queries
- Blocked queries return error response and log `GUARDRAIL_BLOCKED` audit entry

**Output guardrails (post-generation):**
- HHEM 2.1 faithfulness scoring — cross-encoder on (retrieved_context, llm_response)
- Below threshold: response replaced with "Cannot verify answer against available documents"
- LLM Guard output scanners for PII leakage detection

**Configuration (via Settings):**
- `GUARDRAIL_INJECTION_THRESHOLD: float = 0.5`
- `GUARDRAIL_HHEM_THRESHOLD: float = 0.5`
- `GUARDRAILS_ENABLED: bool = True` (toggle for dev/testing)

**Files modified/created:**
- `backend/app/guardrails.py` — new module with input/output scanning functions
- `backend/app/pipelines/query.py` — wrap pipeline with guardrail checks
- `backend/app/config.py` — add guardrail settings
- `backend/app/models/base.py` — add `GUARDRAIL_BLOCKED` to AuditAction enum
- `backend/requirements.txt` — add llm-guard, sentence-transformers

---

## Workstream 6: Query Expansion (Cross-lingual)

**Problem:** Greek query won't match English document content and vice versa. Single query phrasing may miss relevant results.

**Solution:**
- LLM generates 2-3 query variants (at least one Greek, one English) before retrieval
- Each variant goes through full embed → retrieve path independently
- Results merged by document ID (dedup, keep highest score) then fed to reranker

**LLM prompt:**
```
System: "Generate 2-3 search query variants for the following question.
Include at least one Greek and one English variant. Return JSON array of strings."
User: "{original_query}"
```

Uses existing Ollama endpoint (qwen3:30b-a3b), low max_tokens (200), temperature 0.3.

**Configuration:**
- `QUERY_EXPANSION_ENABLED: bool = True` — toggle for single-language deployments
- Adds ~3-5s latency for the variant generation LLM call

**Files modified:**
- `backend/app/pipelines/query.py` — add expansion step before retrieval
- `backend/app/config.py` — add `QUERY_EXPANSION_ENABLED` setting

---

## Dependency Order

```
Workstream 1 (async ingestion) ─┐
Workstream 2 (ACL filtering)   ─┼─→ Workstream 3 (hybrid search) → Workstream 4 (reranking)
                                 │
                                 ├─→ Workstream 5 (guardrails)     [independent]
                                 └─→ Workstream 6 (query expansion) [independent]
```

- Workstreams 1 and 2 are independent of each other and of 3-6
- Workstream 3 must come before 4 (reranker operates on hybrid results)
- Workstreams 5 and 6 are independent of each other and of 3-4

## Testing Strategy

- Unit tests for each component (sparse embedder, reranker, guardrails, query expansion)
- Integration tests for full pipeline (hybrid search → rerank → guardrails)
- Re-indexing migration tested against existing document set
- RAGAS evaluation before/after to measure retrieval quality improvement
- E2E verification against running Docker stack (same as Phase 2)
