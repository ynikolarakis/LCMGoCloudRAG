from __future__ import annotations

import httpx
import structlog
from fastapi import APIRouter
from sqlalchemy import text

from app.clients import get_qdrant_client, get_redis_client
from app.config import settings
from app.database import async_session_factory
from app.schemas.health import HealthResponse, ServiceStatus

logger = structlog.get_logger()
router = APIRouter()


async def _check_database() -> ServiceStatus:
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        return ServiceStatus(status="healthy")
    except Exception as e:
        logger.error("health_check_database_failed", error=str(e))
        return ServiceStatus(status="unhealthy", detail=str(e))


async def _check_qdrant() -> ServiceStatus:
    try:
        client = get_qdrant_client()
        client.get_collections()
        return ServiceStatus(status="healthy")
    except Exception as e:
        logger.error("health_check_qdrant_failed", error=str(e))
        return ServiceStatus(status="unhealthy", detail=str(e))


async def _check_redis() -> ServiceStatus:
    try:
        client = get_redis_client()
        await client.ping()
        return ServiceStatus(status="healthy")
    except Exception as e:
        logger.error("health_check_redis_failed", error=str(e))
        return ServiceStatus(status="unhealthy", detail=str(e))


async def _check_llm() -> ServiceStatus:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            base_url = settings.LLM_BASE_URL.rstrip("/v1").rstrip("/")
            resp = await client.get(f"{base_url}/api/tags")
            if resp.status_code == 200:
                return ServiceStatus(status="healthy")
            return ServiceStatus(status="unhealthy", detail=f"HTTP {resp.status_code}")
    except Exception as e:
        logger.error("health_check_llm_failed", error=str(e))
        return ServiceStatus(status="unhealthy", detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check health of all backend services."""
    db = await _check_database()
    qdrant = await _check_qdrant()
    redis_status = await _check_redis()
    llm = await _check_llm()

    all_healthy = all(s.status == "healthy" for s in [db, qdrant, redis_status, llm])

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        database=db,
        qdrant=qdrant,
        redis=redis_status,
        llm=llm,
    )
