from __future__ import annotations

import time

import structlog
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from app.config import settings

logger = structlog.get_logger()

SYSTEM_PROMPT = """\
You are a document research assistant.
You ONLY answer based on the provided context. You MUST cite every claim.

Rules:
1. Every statement must end with [Source: filename, page X]
2. If the context does not contain the answer, say "Cannot find answer in the available documents"
3. Never make claims beyond what the context explicitly states
4. Respond in the same language as the user's question
5. If context is in a different language than the question, translate the answer but keep citations in original language\
"""

USER_TEMPLATE = """\
Context:
{% for doc in documents %}
[{{ loop.index }}] Source: {{ doc.meta.source }}, Page {{ doc.meta.page_num }}
{{ doc.content }}

{% endfor %}
Question: {{ query }}\
"""


def _get_document_store() -> QdrantDocumentStore:
    """Create a QdrantDocumentStore instance with sparse embedding support."""
    return QdrantDocumentStore(
        url=settings.QDRANT_URL,
        index=settings.QDRANT_COLLECTION,
        embedding_dim=settings.EMBEDDING_DIMENSION,
        recreate_index=False,
        use_sparse_embeddings=True,
    )


def _build_query_pipeline(document_store: QdrantDocumentStore) -> Pipeline:
    """Build the Haystack hybrid query pipeline: sparse+dense embed -> hybrid retrieve -> prompt -> generate."""
    pipeline = Pipeline()

    # Sparse text embedder (BM25)
    sparse_embedder = FastembedSparseTextEmbedder(
        model=settings.SPARSE_EMBEDDING_MODEL,
    )

    # Dense text embedder (Qwen3 via Ollama)
    dense_embedder = OpenAITextEmbedder(
        api_key=Secret.from_token("ollama"),
        model=settings.EMBEDDING_MODEL,
        api_base_url=settings.EMBEDDING_BASE_URL,
    )

    # Hybrid retriever (RRF fusion of dense + sparse)
    retriever = QdrantHybridRetriever(
        document_store=document_store,
        top_k=settings.RETRIEVER_TOP_K,
    )

    # Reranker (cross-encoder)
    ranker = TransformersSimilarityRanker(
        model=settings.RERANKER_MODEL,
        top_k=settings.RERANKER_TOP_K,
        score_threshold=settings.RERANKER_THRESHOLD,
        scale_score=True,
    )

    # Chat prompt builder with system + user template
    messages_template = [
        ChatMessage.from_system(SYSTEM_PROMPT),
        ChatMessage.from_user(USER_TEMPLATE),
    ]
    prompt_builder = ChatPromptBuilder(template=messages_template)

    # LLM generator (OpenAI-compatible — works with Ollama and vLLM)
    llm = OpenAIChatGenerator(
        api_key=Secret.from_token("ollama"),
        model=settings.LLM_MODEL,
        api_base_url=settings.LLM_BASE_URL,
        generation_kwargs={
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
        },
    )

    pipeline.add_component("sparse_embedder", sparse_embedder)
    pipeline.add_component("dense_embedder", dense_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("ranker", ranker)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm)

    pipeline.connect("sparse_embedder.sparse_embedding", "retriever.query_sparse_embedding")
    pipeline.connect("dense_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "ranker.documents")
    pipeline.connect("ranker.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.messages")

    return pipeline


def query_documents(question: str, client_id: str = "default") -> dict:
    """Run the hybrid RAG query pipeline.

    Args:
        question: The user's question.
        client_id: Client identifier for multi-tenancy filtering.

    Returns:
        Dict with keys: answer, citations, model_used, latency_ms, retrieved_docs.
    """
    logger.info("query_started", question=question[:100], client_id=client_id)
    start_time = time.perf_counter()

    store = _get_document_store()
    pipeline = _build_query_pipeline(store)

    result = pipeline.run(
        {
            "sparse_embedder": {"text": question},
            "dense_embedder": {"text": question},
            "retriever": {
                "filters": {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.client_id", "operator": "==", "value": client_id},
                    ],
                },
            },
            "ranker": {"query": question},
            "prompt_builder": {"query": question},
        }
    )

    latency_ms = round((time.perf_counter() - start_time) * 1000)

    # Extract answer from LLM response
    replies = result.get("llm", {}).get("replies", [])
    answer_text = replies[0].text if replies else "Cannot find answer in the available documents"

    # Extract retrieved documents for citation metadata
    retrieved_docs = result.get("retriever", {}).get("documents", [])
    citations = [
        {
            "source": doc.meta.get("source", "unknown"),
            "page": doc.meta.get("page_num"),
            "content_preview": doc.content[:200] if doc.content else "",
        }
        for doc in retrieved_docs
    ]

    logger.info(
        "query_complete",
        latency_ms=latency_ms,
        num_retrieved=len(retrieved_docs),
        answer_length=len(answer_text),
    )

    return {
        "answer": answer_text,
        "citations": citations,
        "model_used": settings.LLM_MODEL,
        "latency_ms": latency_ms,
        "retrieved_docs": retrieved_docs,
    }
