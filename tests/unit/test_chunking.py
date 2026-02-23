from __future__ import annotations

from app.pipelines.chunking import chunk_text, count_tokens


def test_count_tokens_returns_positive_int() -> None:
    """count_tokens should return a positive integer for non-empty input."""
    result = count_tokens("Hello, world!")
    assert isinstance(result, int)
    assert result > 0


def test_chunk_text_empty_string_returns_empty_list() -> None:
    """chunk_text should return an empty list when given an empty string."""
    assert chunk_text("") == []


def test_chunk_text_short_text_returns_single_chunk() -> None:
    """chunk_text should keep short text in a single chunk without splitting."""
    text = "This is a short paragraph."
    chunks = chunk_text(text, target_tokens=450, max_tokens=512)
    assert len(chunks) == 1
    assert text in chunks[0]


def test_chunk_text_respects_max_tokens() -> None:
    """No chunk produced by chunk_text should vastly exceed max_tokens."""
    # Create text with many paragraphs
    text = "\n\n".join([f"Paragraph number {i}. " * 20 for i in range(30)])
    chunks = chunk_text(text, target_tokens=100, max_tokens=150)
    for chunk in chunks:
        tokens = count_tokens(chunk)
        # Allow some tolerance due to overlap
        assert tokens < 300, f"Chunk has {tokens} tokens, expected < 300"


def test_chunk_text_creates_multiple_chunks_for_long_text() -> None:
    """chunk_text should split long text into more than one chunk."""
    text = "\n\n".join([f"This is paragraph {i} with enough text to matter." for i in range(50)])
    chunks = chunk_text(text, target_tokens=50, max_tokens=100)
    assert len(chunks) > 1
