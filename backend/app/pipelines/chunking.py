from __future__ import annotations

import structlog
import tiktoken

logger = structlog.get_logger()

# Use cl100k_base encoding (GPT-4 / general-purpose)
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(_ENCODING.encode(text))


def chunk_text(
    text: str,
    target_tokens: int = 450,
    max_tokens: int = 512,
    min_tokens: int = 200,
    overlap_pct: float = 0.15,
) -> list[str]:
    """Split text into token-sized chunks with overlap.

    Args:
        text: Input text to chunk.
        target_tokens: Target chunk size in tokens.
        max_tokens: Hard ceiling for chunk size.
        min_tokens: Minimum chunk size; smaller chunks merged with neighbors.
        overlap_pct: Fraction of overlap between consecutive chunks.

    Returns:
        List of text chunks, each within [min_tokens, max_tokens] range.
    """
    if not text.strip():
        return []

    # Split into paragraphs first (respects natural boundaries)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Build chunks by accumulating paragraphs up to target token size
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # If single paragraph exceeds max, split by sentences
        if para_tokens > max_tokens:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            # Split long paragraph by sentences
            sentences = para.replace(". ", ".\n").split("\n")
            for sentence in sentences:
                sent_tokens = count_tokens(sentence)
                if current_tokens + sent_tokens > target_tokens and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                current_chunk.append(sentence)
                current_tokens += sent_tokens
            continue

        if current_tokens + para_tokens > target_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_tokens = 0

        current_chunk.append(para)
        current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    # Merge small trailing chunks with previous (but never exceed max_tokens)
    merged: list[str] = []
    for chunk in chunks:
        if merged and count_tokens(chunk) < min_tokens:
            candidate = merged[-1] + "\n\n" + chunk
            if count_tokens(candidate) <= max_tokens:
                merged[-1] = candidate
            else:
                merged.append(chunk)
        else:
            merged.append(chunk)

    # Add overlap between consecutive chunks
    if len(merged) <= 1 or overlap_pct <= 0:
        return merged

    overlap_tokens = int(target_tokens * overlap_pct)
    overlapped: list[str] = [merged[0]]

    for i in range(1, len(merged)):
        prev_tokens = _ENCODING.encode(merged[i - 1])
        overlap_text = _ENCODING.decode(prev_tokens[-overlap_tokens:]) if len(prev_tokens) > overlap_tokens else ""
        if overlap_text.strip():
            overlapped.append(overlap_text.strip() + "\n\n" + merged[i])
        else:
            overlapped.append(merged[i])

    logger.info("chunking_complete", num_chunks=len(overlapped))
    return overlapped
