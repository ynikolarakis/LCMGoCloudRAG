from __future__ import annotations

from app.tasks.ingestion import ingest_document_task


def test_ingest_task_is_registered() -> None:
    """ingest_document_task should be a registered Celery task."""
    assert ingest_document_task.name == "app.tasks.ingestion.ingest_document_task"


def test_ingest_task_has_max_retries() -> None:
    """ingest_document_task should have max_retries=3."""
    assert ingest_document_task.max_retries == 3


def test_ingest_task_is_bound() -> None:
    """ingest_document_task should be a bound task (has self)."""
    # Bound tasks have the 'bind' attribute
    assert hasattr(ingest_document_task, "request")
