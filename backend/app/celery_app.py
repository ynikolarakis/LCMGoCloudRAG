from __future__ import annotations

from celery import Celery

from app.config import settings

celery_app = Celery(
    "docintel",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    worker_concurrency=2,
    broker_connection_retry_on_startup=True,
)

# Explicitly include task modules so Celery registers them at startup
celery_app.conf.include = ["app.tasks.ingestion"]
