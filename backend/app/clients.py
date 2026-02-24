from __future__ import annotations

import structlog
from qdrant_client import QdrantClient
from redis.asyncio import Redis

from app.config import settings

logger = structlog.get_logger()

_qdrant_client: QdrantClient | None = None
_redis_client: Redis | None = None


def get_qdrant_client() -> QdrantClient:
    """Get or create the singleton Qdrant client.

    Returns:
        The shared QdrantClient instance. Created on first call and reused
        on subsequent calls (singleton pattern).
    """
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=settings.QDRANT_URL, timeout=10)
        logger.info("qdrant_client_created", url=settings.QDRANT_URL)
    return _qdrant_client


def get_redis_client() -> Redis:
    """Get or create the singleton async Redis client.

    Returns:
        The shared async Redis instance. Created on first call and reused
        on subsequent calls (singleton pattern).
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = Redis.from_url(settings.REDIS_URL, decode_responses=True)
        logger.info("redis_client_created", url=settings.REDIS_URL)
    return _redis_client


async def close_clients() -> None:
    """Close all singleton clients. Called on app shutdown."""
    global _qdrant_client, _redis_client
    if _qdrant_client:
        _qdrant_client.close()
        _qdrant_client = None
        logger.info("qdrant_client_closed")
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("redis_client_closed")
