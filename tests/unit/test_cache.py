from __future__ import annotations

from app.cache import make_cache_key


def test_make_cache_key_deterministic() -> None:
    """Same inputs should produce the same cache key."""
    key1 = make_cache_key("What are the terms?", "client-1")
    key2 = make_cache_key("What are the terms?", "client-1")
    assert key1 == key2


def test_make_cache_key_different_for_different_inputs() -> None:
    """Different inputs should produce different keys."""
    key1 = make_cache_key("Question A", "client-1")
    key2 = make_cache_key("Question B", "client-1")
    assert key1 != key2
