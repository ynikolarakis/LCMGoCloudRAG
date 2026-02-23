from __future__ import annotations

from app.celery_app import celery_app


def test_celery_app_name() -> None:
    """Celery app should be named 'docintel'."""
    assert celery_app.main == "docintel"


def test_celery_task_serializer_is_json() -> None:
    """Celery should use JSON serializer."""
    assert celery_app.conf.task_serializer == "json"


def test_celery_acks_late() -> None:
    """Celery tasks should ack late for reliability."""
    assert celery_app.conf.task_acks_late is True


def test_celery_reject_on_worker_lost() -> None:
    """Tasks should be rejected (requeued) if worker dies."""
    assert celery_app.conf.task_reject_on_worker_lost is True
