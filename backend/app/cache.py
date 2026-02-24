from __future__ import annotations

import hashlib
import json

import structlog

from app.clients import get_redis_client
from app.config import settings

logger = structlog.get_logger()

CACHE_PREFIX = "docintel:query:"


def make_cache_key(question: str, client_id: str) -> str:
    """Generate a deterministic cache key from question and client_id.

    Args:
        question: The user's query string. Whitespace is stripped and the text
            is lowercased before hashing so that trivially equivalent queries
            share the same cache entry.
        client_id: The tenant/client identifier used to namespace the key and
            prevent cross-client cache leakage.

    Returns:
        A Redis key string of the form ``docintel:query:<sha256hex>``.
    """
    normalized = question.strip().lower()
    raw = f"{client_id}:{normalized}"
    return CACHE_PREFIX + hashlib.sha256(raw.encode()).hexdigest()


async def get_cached_response(question: str, client_id: str) -> dict | None:
    """Get a cached RAG response if available.

    Returns ``None`` when the cache is disabled, the key is not present, or
    a Redis error occurs (fail-open so that a cache outage never blocks queries).

    Args:
        question: The user's query string.
        client_id: The tenant/client identifier.

    Returns:
        The cached response dict, or ``None`` on cache miss or error.
    """
    if not settings.CACHE_ENABLED:
        return None
    try:
        client = get_redis_client()
        key = make_cache_key(question, client_id)
        data = await client.get(key)
        if data:
            logger.info("cache_hit", key=key[:30])
            return json.loads(data)
        return None
    except Exception as exc:
        logger.warning("cache_get_error", error=str(exc))
        return None


async def set_cached_response(question: str, client_id: str, response: dict) -> None:
    """Cache a RAG response with a TTL.

    Silently swallows Redis errors so that a cache outage never prevents a
    response from reaching the user.

    Args:
        question: The user's query string (used to derive the cache key).
        client_id: The tenant/client identifier.
        response: The RAG response dict to serialise and store.
    """
    if not settings.CACHE_ENABLED:
        return
    try:
        client = get_redis_client()
        key = make_cache_key(question, client_id)
        await client.setex(key, settings.CACHE_TTL_SECONDS, json.dumps(response))
        logger.info("cache_set", key=key[:30], ttl=settings.CACHE_TTL_SECONDS)
    except Exception as exc:
        logger.warning("cache_set_error", error=str(exc))


async def invalidate_client_cache(client_id: str) -> None:
    """Invalidate all cached responses matching the global cache prefix.

    Uses ``SCAN`` (non-blocking, cursor-based) rather than ``KEYS`` to avoid
    stalling Redis on large keyspaces.

    Note: The current implementation scans *all* ``docintel:query:*`` keys,
    not just those belonging to ``client_id``, because the SHA-256 key does
    not encode the client identifier in a scannable way.  A future improvement
    would store per-client key sets in a Redis Set to enable targeted
    invalidation.

    Args:
        client_id: The tenant/client identifier (used only for audit logging).
    """
    if not settings.CACHE_ENABLED:
        return
    try:
        client = get_redis_client()
        pattern = f"{CACHE_PREFIX}*"
        cursor: int = 0
        deleted: int = 0
        while True:
            cursor, keys = await client.scan(cursor, match=pattern, count=100)
            if keys:
                await client.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        logger.info("cache_invalidated", client_id=client_id, keys_deleted=deleted)
    except Exception as exc:
        logger.warning("cache_invalidate_error", error=str(exc))
