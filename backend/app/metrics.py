from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# RAG pipeline metrics
rag_query_duration = Histogram(
    "rag_query_duration_seconds",
    "Total RAG query pipeline duration",
    buckets=(0.5, 1, 2, 5, 10, 30, 60),
)

rag_retrieval_duration = Histogram(
    "rag_retrieval_duration_seconds",
    "Retrieval phase duration",
    buckets=(0.1, 0.5, 1, 2, 5, 10),
)

rag_generation_duration = Histogram(
    "rag_generation_duration_seconds",
    "LLM generation phase duration",
    buckets=(0.5, 1, 2, 5, 10, 30, 60),
)

rag_faithfulness_score = Histogram(
    "rag_faithfulness_score",
    "HHEM faithfulness score distribution",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

guardrail_blocks_total = Counter(
    "rag_guardrail_blocks_total",
    "Total guardrail blocks",
    ["type"],
)

# Celery metrics
celery_tasks_total = Counter(
    "celery_tasks_total",
    "Total Celery tasks",
    ["task_name", "status"],
)

celery_task_duration = Histogram(
    "celery_task_duration_seconds",
    "Celery task duration",
    ["task_name"],
    buckets=(1, 5, 10, 30, 60, 120, 300),
)

# WebSocket metrics
websocket_connections = Gauge(
    "websocket_connections_active",
    "Active WebSocket connections",
)

# Document metrics
documents_ingested_total = Counter(
    "documents_ingested_total",
    "Total documents ingested",
    ["status"],
)
