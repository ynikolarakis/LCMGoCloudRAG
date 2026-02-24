from __future__ import annotations


def test_metrics_module_imports() -> None:
    """All metric objects should be importable."""
    from app.metrics import (
        rag_query_duration,
        rag_retrieval_duration,
        rag_generation_duration,
        rag_faithfulness_score,
        guardrail_blocks_total,
        celery_tasks_total,
        celery_task_duration,
        websocket_connections,
        documents_ingested_total,
    )
    assert rag_query_duration is not None
    assert websocket_connections is not None


def test_clients_module_imports() -> None:
    """Client singleton functions should be importable."""
    from app.clients import get_qdrant_client, get_redis_client, close_clients
    assert callable(get_qdrant_client)
    assert callable(get_redis_client)
    assert callable(close_clients)
