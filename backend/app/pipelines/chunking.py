from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class ChunkResult:
    """A chunk with extracted metadata from document structure."""

    content: str
    page_num: int
    section_title: str | None = None
    language: str | None = None
    token_count: int = 0


def _detect_language(text: str) -> str | None:
    """Detect language of text using langdetect. Returns ISO 639-1 code or None.

    Args:
        text: Text to detect language for.

    Returns:
        ISO 639-1 language code (e.g. "en", "el") or None if detection fails
        or text is too short.
    """
    try:
        from langdetect import detect  # noqa: PLC0415

        if len(text.strip()) < 20:
            return None
        return detect(text)  # type: ignore[no-any-return]
    except Exception:  # noqa: BLE001
        return None


def chunk_document(
    document: object,
    target_tokens: int = 450,
    max_tokens: int = 512,
    min_tokens: int = 200,
    overlap_pct: float = 0.15,
) -> list[ChunkResult]:
    """Split a Docling document into structure-aware chunks.

    Walks the document page-by-page, tracking headings as section titles,
    respecting page boundaries, keeping tables intact when possible, and
    detecting language per chunk.

    Args:
        document: A Docling document object with a ``pages`` attribute. Each
            page must expose ``page_no`` (int) and ``items`` (list of items
            with ``label`` and ``text`` attributes).
        target_tokens: Target chunk size in tokens.
        max_tokens: Hard ceiling for chunk size.
        min_tokens: Minimum chunk size; smaller chunks are merged with the
            preceding chunk on the same page when the merge stays within
            ``max_tokens``.
        overlap_pct: Fraction of overlap between consecutive chunks on the
            same page (reserved for future use — overlap is tracked per page
            but not yet prepended in this implementation).

    Returns:
        List of ChunkResult objects with populated metadata.
    """
    pages = getattr(document, "pages", None)
    if not pages:
        return []

    all_chunks: list[ChunkResult] = []
    current_section: str | None = None

    for page in pages:
        page_num: int = getattr(page, "page_no", 1)
        items = getattr(page, "items", [])

        # Collect text segments for this page with their metadata.
        # Each entry is (text, section_title_at_time_of_item).
        page_segments: list[tuple[str, str | None]] = []

        for item in items:
            label: str = getattr(item, "label", "paragraph")
            text: str = getattr(item, "text", "").strip()
            if not text:
                continue

            if label == "section_header":
                current_section = text
                continue

            if label == "table":
                # Tables: keep intact if small enough, otherwise split by rows.
                table_tokens = count_tokens(text)
                if table_tokens <= max_tokens:
                    page_segments.append((text, current_section))
                else:
                    rows = text.split("\n")
                    row_group: list[str] = []
                    row_tokens = 0
                    for row in rows:
                        rt = count_tokens(row)
                        if row_tokens + rt > target_tokens and row_group:
                            page_segments.append(("\n".join(row_group), current_section))
                            row_group = []
                            row_tokens = 0
                        row_group.append(row)
                        row_tokens += rt
                    if row_group:
                        page_segments.append(("\n".join(row_group), current_section))
                continue

            # Regular text (paragraph, list_item, caption, etc.)
            page_segments.append((text, current_section))

        if not page_segments:
            continue

        # Build chunks from page segments — never cross page boundaries.
        current_texts: list[str] = []
        current_tokens_count: int = 0
        current_section_for_chunk: str | None = page_segments[0][1]

        page_chunks: list[ChunkResult] = []

        for text, section in page_segments:
            seg_tokens = count_tokens(text)

            # If a single segment exceeds max, flush the buffer then split by sentences.
            if seg_tokens > max_tokens:
                if current_texts:
                    chunk_content = "\n\n".join(current_texts)
                    page_chunks.append(
                        ChunkResult(
                            content=chunk_content,
                            page_num=page_num,
                            section_title=current_section_for_chunk,
                            token_count=count_tokens(chunk_content),
                        )
                    )
                    current_texts = []
                    current_tokens_count = 0

                sentences = text.replace(". ", ".\n").split("\n")
                sent_group: list[str] = []
                sent_tokens: int = 0
                for sent in sentences:
                    st = count_tokens(sent)
                    if sent_tokens + st > target_tokens and sent_group:
                        chunk_content = " ".join(sent_group)
                        page_chunks.append(
                            ChunkResult(
                                content=chunk_content,
                                page_num=page_num,
                                section_title=section,
                                token_count=count_tokens(chunk_content),
                            )
                        )
                        sent_group = []
                        sent_tokens = 0
                    sent_group.append(sent)
                    sent_tokens += st
                if sent_group:
                    chunk_content = " ".join(sent_group)
                    page_chunks.append(
                        ChunkResult(
                            content=chunk_content,
                            page_num=page_num,
                            section_title=section,
                            token_count=count_tokens(chunk_content),
                        )
                    )
                current_section_for_chunk = section
                continue

            # Flush buffer when adding this segment would exceed the target size.
            if current_tokens_count + seg_tokens > target_tokens and current_texts:
                chunk_content = "\n\n".join(current_texts)
                page_chunks.append(
                    ChunkResult(
                        content=chunk_content,
                        page_num=page_num,
                        section_title=current_section_for_chunk,
                        token_count=count_tokens(chunk_content),
                    )
                )
                current_texts = []
                current_tokens_count = 0

            if not current_texts:
                # First segment in a new buffer sets the section heading.
                current_section_for_chunk = section

            current_texts.append(text)
            current_tokens_count += seg_tokens

        # Flush remaining buffer for this page.
        if current_texts:
            chunk_content = "\n\n".join(current_texts)
            page_chunks.append(
                ChunkResult(
                    content=chunk_content,
                    page_num=page_num,
                    section_title=current_section_for_chunk,
                    token_count=count_tokens(chunk_content),
                )
            )

        # Merge small trailing chunks into the preceding chunk on the same page.
        merged_page_chunks: list[ChunkResult] = []
        for chunk in page_chunks:
            if merged_page_chunks and chunk.token_count < min_tokens:
                candidate_content = merged_page_chunks[-1].content + "\n\n" + chunk.content
                candidate_tokens = count_tokens(candidate_content)
                if candidate_tokens <= max_tokens:
                    merged_page_chunks[-1] = ChunkResult(
                        content=candidate_content,
                        page_num=page_num,
                        section_title=merged_page_chunks[-1].section_title,
                        token_count=candidate_tokens,
                    )
                    continue
            merged_page_chunks.append(chunk)

        # Detect language for each finalised chunk on this page.
        for chunk in merged_page_chunks:
            chunk.language = _detect_language(chunk.content)

        all_chunks.extend(merged_page_chunks)

    logger.info("semantic_chunking_complete", num_chunks=len(all_chunks))
    return all_chunks
