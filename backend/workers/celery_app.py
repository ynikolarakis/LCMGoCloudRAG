from __future__ import annotations

from celery import Celery

from app.config import settings

celery_app = Celery(
    "docintel",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(name="tasks.health_check")
def health_check_task() -> dict[str, str]:
    """Simple health check task to verify Celery is running.

    Returns:
        Dict with a single ``status`` key set to ``"ok"``.
    """
    return {"status": "ok"}
