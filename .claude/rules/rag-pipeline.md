---
globs: backend/app/pipeline/**,backend/app/services/rag/**,backend/workers/tasks/**
---

# RAG Pipeline Rules

## Chunking
- Method: semantic + structure-aware (respect headings, tables, page boundaries)
- Target size: 400-512 tokens per chunk
- Minimum size: 200 tokens (merge smaller chunks with neighbors)
- Maximum size: 512 tokens (hard ceiling)
- Overlap: 10-20% between consecutive chunks
- Contextual prefix: prepend 2-3 sentence document summary to each chunk
- Metadata per chunk: `doc_id`, `page_num`, `section_title`, `chunk_seq`, `language`, `client_id`
- Tables: keep entire table in one chunk if <512 tokens, split rows if larger

## Embedding
- Model: Qwen3-Embedding-0.6B (512 dimensions, instruction-aware)
- Instruction prefix for queries: "Retrieve relevant passages to answer: {query}"
- Instruction prefix for documents: "Passage from {doc_type} document: {chunk_text}"
- Batch size: 32 chunks per embedding call
- Normalize vectors before indexing (cosine distance)

## Vector Database (Qdrant)
- Collection per client: `{client_id}_documents`
- Distance: Cosine
- HNSW config: m=16, ef_construct=128 (balance quality vs index speed)
- Payload indexes: `doc_id` (keyword), `client_id` (keyword), `language` (keyword), `page_num` (integer)
- ACL filtering: apply `must` filter on `allowed_users` payload BEFORE similarity search

## Hybrid Retrieval
- Dense search: cosine similarity, top-50 candidates
- Sparse search: BM25 keyword search, top-50 candidates
- Fusion: Reciprocal Rank Fusion (RRF) with k=60
- Metadata filter: apply BEFORE similarity search (not after)
- Query expansion: generate 2-3 alternative phrasings
- Cross-lingual: always search with BOTH Greek AND English query variants
- HyDE: generate hypothetical answer, embed it, search with both original + HyDE vectors

## Reranking
- Model: BGE-Reranker-v2-m3 (cross-encoder, multilingual)
- Input: top-50 from hybrid retrieval
- Output: top-5 (configurable via `RERANKER_TOP_K`)
- Minimum score threshold: 0.3 (drop chunks below this)
- CPU inference is acceptable for dev (reranker is fast)

## Generation
- System prompt MUST include citation requirement (see rag-specialist agent for template)
- Context format: numbered chunks with metadata
  ```
  [1] Source: contract.pdf, Page 3
  The agreement shall remain in force for a period of twenty-four (24) months...
  
  [2] Source: appendix_a.pdf, Page 1
  Payment terms are net 30 days from invoice date...
  ```
- Response format: answer text with inline `[Source: filename, page X]`
- If no relevant context: "Cannot find answer in the available documents"
- Max context tokens: 4096 (leave room for system prompt + response)
- Temperature: 0.1 (low creativity, high faithfulness)
- Max response tokens: 1024

## Guardrails
- Input: NeMo Guardrails + LLM Guard check EVERY query before processing
  - Prompt injection detection
  - Off-topic detection (configurable per client)
  - PII detection in queries
- Output: check EVERY response before returning to user
  - Faithfulness: verify claims against retrieved chunks
  - PII detection: mask any PII that leaked through
  - Topic boundary: ensure response stays within document scope
  - HHEM score: block responses with hallucination score < threshold (configurable)

## Evaluation Targets
- Faithfulness: >0.95 (RAGAS)
- Answer Relevancy: >0.90 (RAGAS)
- Context Precision: >0.85 (RAGAS)
- Context Recall: >0.90 (RAGAS golden test set)
- Hallucination Rate: <2% (HHEM v2)
- Citation Coverage: 100% (every claim has [Source])
- Run FULL evaluation on every pipeline configuration change
