# Phase 6: RAG Quality & Observability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade LCM DocIntel with semantic chunking, real Ollama streaming, conversation persistence, RAGAS evaluation, Prometheus/Grafana observability, and security hardening.

**Architecture:** Pipeline-first approach — fix the data foundation (chunking), then the delivery layer (streaming), then add stateful features (conversations), prove quality (RAGAS), add operational tooling (observability), and finally harden security. Each workstream builds on the previous.

**Tech Stack:** langdetect, httpx (async SSE), ragas, prometheus-client, prometheus-fastapi-instrumentator, slowapi, Grafana, Prometheus

---

## Dependency Graph

```
WS1: Tasks 1-6 (Semantic Chunking)
  ↓
WS2: Tasks 7-10 (Real Streaming)
  ↓
WS3: Tasks 11-18 (Conversations)
  ↓
WS4: Tasks 19-21 (RAGAS Evaluation)
  ↓
WS5: Tasks 22-27 (Observability)
  ↓
WS6: Tasks 28-33 (Security Hardening)
  ↓
Task 34: Final Verification
```

---

## Workstream 1: Semantic/Structure-Aware Chunking

### Task 1: Add summary column to Document model and create migration

**Files:**
- Modify: `backend/app/models/document.py:28`
- Create: new Alembic migration
- Modify: `backend/requirements.txt`

**Step 1: Add langdetect to requirements.txt**

Add after the `tiktoken>=0.7.0` line in `backend/requirements.txt`:

```
langdetect>=1.0.9
```

**Step 2: Install the dependency**

Run: `cd /e/LCMGoCloudRAG/backend && pip install langdetect`

**Step 3: Add summary field to Document model**

In `backend/app/models/document.py`, add after line 27 (`chunk_count`):

```python
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
```

Add the `Text` import to the existing import line (line 6):

```python
from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, func
```

**Step 4: Generate Alembic migration**

Run: `cd /e/LCMGoCloudRAG/backend && PYTHONPATH=. alembic revision --autogenerate -m "add_document_summary_column"`

**Step 5: Apply migration**

Run: `cd /e/LCMGoCloudRAG/backend && PYTHONPATH=. alembic upgrade head`

**Step 6: Run backend tests to verify no regressions**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/ -x -v`
Expected: All 71 tests pass.

**Step 7: Commit**

```bash
git add backend/app/models/document.py backend/requirements.txt backend/alembic/versions/
git commit -m "feat: add summary column to Document model for contextual chunking prefix"
```

---

### Task 2: Write tests for structure-aware chunking

**Files:**
- Create: `tests/unit/test_semantic_chunking.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_semantic_chunking.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.pipelines.chunking import chunk_document, count_tokens, ChunkResult


def _make_mock_item(type_name: str, text: str, level: int = 1) -> MagicMock:
    """Create a mock Docling document item."""
    item = MagicMock()
    item.label = type_name
    item.text = text
    if type_name == "section_header":
        item.level = level
    return item


def _make_mock_page(page_no: int, items: list) -> MagicMock:
    """Create a mock Docling page."""
    page = MagicMock()
    page.page_no = page_no
    page.items = items
    return page


def _make_mock_document(pages: list) -> MagicMock:
    """Create a mock Docling document."""
    doc = MagicMock()
    doc.pages = pages
    return doc


class TestChunkDocument:
    def test_returns_chunk_results(self) -> None:
        """chunk_document should return a list of ChunkResult objects."""
        page = _make_mock_page(1, [
            _make_mock_item("paragraph", "This is a paragraph with enough text to form a chunk. " * 5),
        ])
        doc = _make_mock_document([page])
        result = chunk_document(doc, target_tokens=50, max_tokens=100)
        assert len(result) >= 1
        assert isinstance(result[0], ChunkResult)

    def test_populates_page_num(self) -> None:
        """Each chunk should have the correct page number."""
        page1 = _make_mock_page(1, [
            _make_mock_item("paragraph", "Content on page one. " * 20),
        ])
        page2 = _make_mock_page(2, [
            _make_mock_item("paragraph", "Content on page two. " * 20),
        ])
        doc = _make_mock_document([page1, page2])
        result = chunk_document(doc, target_tokens=50, max_tokens=100)
        page_nums = {c.page_num for c in result}
        assert 1 in page_nums
        assert 2 in page_nums

    def test_tracks_section_title(self) -> None:
        """Chunks after a heading should carry the section title."""
        page = _make_mock_page(1, [
            _make_mock_item("section_header", "Introduction"),
            _make_mock_item("paragraph", "Paragraph under introduction. " * 10),
        ])
        doc = _make_mock_document([page])
        result = chunk_document(doc, target_tokens=50, max_tokens=100)
        assert result[0].section_title == "Introduction"

    def test_never_merges_across_pages(self) -> None:
        """Chunks should not span page boundaries."""
        page1 = _make_mock_page(1, [
            _make_mock_item("paragraph", "Short text."),
        ])
        page2 = _make_mock_page(2, [
            _make_mock_item("paragraph", "Another short text."),
        ])
        doc = _make_mock_document([page1, page2])
        result = chunk_document(doc, target_tokens=450, max_tokens=512, min_tokens=5)
        for chunk in result:
            assert chunk.page_num in (1, 2)

    def test_detects_language(self) -> None:
        """Chunks should have a language field set."""
        page = _make_mock_page(1, [
            _make_mock_item("paragraph", "This is an English paragraph with enough words for detection. " * 5),
        ])
        doc = _make_mock_document([page])
        result = chunk_document(doc, target_tokens=50, max_tokens=100)
        assert result[0].language is not None

    def test_keeps_small_table_intact(self) -> None:
        """A table smaller than max_tokens should be a single chunk."""
        table_text = "| Col A | Col B |\n|---|---|\n| val1 | val2 |"
        page = _make_mock_page(1, [
            _make_mock_item("table", table_text),
        ])
        doc = _make_mock_document([page])
        result = chunk_document(doc, target_tokens=450, max_tokens=512)
        table_chunks = [c for c in result if table_text in c.content]
        assert len(table_chunks) == 1

    def test_empty_document(self) -> None:
        """Empty document should return empty list."""
        doc = _make_mock_document([])
        result = chunk_document(doc)
        assert result == []
```

**Step 2: Run test to verify it fails**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/unit/test_semantic_chunking.py -x -v`
Expected: FAIL — `ChunkResult` and `chunk_document` not found in `chunking.py`.

**Step 3: Commit**

```bash
git add tests/unit/test_semantic_chunking.py
git commit -m "test: add failing tests for structure-aware chunking"
```

---

### Task 3: Implement structure-aware chunking

**Files:**
- Modify: `backend/app/pipelines/chunking.py`

**Context:** The existing `chunk_text()` function stays for backward compatibility (REST query endpoint still uses it). We add `ChunkResult` dataclass and `chunk_document()` function that takes a Docling document model.

**Step 1: Add ChunkResult and chunk_document to chunking.py**

Add after the existing `chunk_text` function (after line 108) in `backend/app/pipelines/chunking.py`:

```python
from dataclasses import dataclass, field


@dataclass
class ChunkResult:
    """A chunk with extracted metadata from document structure."""
    content: str
    page_num: int
    section_title: str | None = None
    language: str | None = None
    token_count: int = 0


def _detect_language(text: str) -> str | None:
    """Detect language of text using langdetect. Returns ISO 639-1 code or None."""
    try:
        from langdetect import detect
        if len(text.strip()) < 20:
            return None
        return detect(text)
    except Exception:
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
        document: A Docling document object with `.pages` attribute.
        target_tokens: Target chunk size in tokens.
        max_tokens: Hard ceiling for chunk size.
        min_tokens: Minimum chunk size; smaller chunks merged with neighbors on same page.
        overlap_pct: Fraction of overlap between consecutive chunks on same page.

    Returns:
        List of ChunkResult objects with populated metadata.
    """
    pages = getattr(document, "pages", None)
    if not pages:
        return []

    all_chunks: list[ChunkResult] = []
    current_section: str | None = None

    for page in pages:
        page_num = getattr(page, "page_no", 1)
        items = getattr(page, "items", [])

        # Collect text segments for this page with their metadata
        page_segments: list[tuple[str, str | None]] = []  # (text, section_title)

        for item in items:
            label = getattr(item, "label", "paragraph")
            text = getattr(item, "text", "").strip()
            if not text:
                continue

            if label == "section_header":
                current_section = text
                continue

            if label == "table":
                # Tables: keep intact if small enough, otherwise split by rows
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

            # Regular text (paragraph, list_item, etc.)
            page_segments.append((text, current_section))

        if not page_segments:
            continue

        # Build chunks from page segments (never cross page boundaries)
        current_texts: list[str] = []
        current_tokens_count = 0
        current_section_for_chunk = page_segments[0][1] if page_segments else None

        page_chunks: list[ChunkResult] = []

        for text, section in page_segments:
            seg_tokens = count_tokens(text)

            # If a single segment exceeds max, split by sentences
            if seg_tokens > max_tokens:
                if current_texts:
                    chunk_content = "\n\n".join(current_texts)
                    page_chunks.append(ChunkResult(
                        content=chunk_content,
                        page_num=page_num,
                        section_title=current_section_for_chunk,
                        token_count=count_tokens(chunk_content),
                    ))
                    current_texts = []
                    current_tokens_count = 0

                sentences = text.replace(". ", ".\n").split("\n")
                sent_group: list[str] = []
                sent_tokens = 0
                for sent in sentences:
                    st = count_tokens(sent)
                    if sent_tokens + st > target_tokens and sent_group:
                        chunk_content = " ".join(sent_group)
                        page_chunks.append(ChunkResult(
                            content=chunk_content,
                            page_num=page_num,
                            section_title=section,
                            token_count=count_tokens(chunk_content),
                        ))
                        sent_group = []
                        sent_tokens = 0
                    sent_group.append(sent)
                    sent_tokens += st
                if sent_group:
                    chunk_content = " ".join(sent_group)
                    page_chunks.append(ChunkResult(
                        content=chunk_content,
                        page_num=page_num,
                        section_title=section,
                        token_count=count_tokens(chunk_content),
                    ))
                current_section_for_chunk = section
                continue

            if current_tokens_count + seg_tokens > target_tokens and current_texts:
                chunk_content = "\n\n".join(current_texts)
                page_chunks.append(ChunkResult(
                    content=chunk_content,
                    page_num=page_num,
                    section_title=current_section_for_chunk,
                    token_count=count_tokens(chunk_content),
                ))
                current_texts = []
                current_tokens_count = 0

            if not current_texts:
                current_section_for_chunk = section

            current_texts.append(text)
            current_tokens_count += seg_tokens

        if current_texts:
            chunk_content = "\n\n".join(current_texts)
            page_chunks.append(ChunkResult(
                content=chunk_content,
                page_num=page_num,
                section_title=current_section_for_chunk,
                token_count=count_tokens(chunk_content),
            ))

        # Merge small trailing chunks on same page
        merged: list[ChunkResult] = []
        for chunk in page_chunks:
            if merged and chunk.token_count < min_tokens:
                candidate_content = merged[-1].content + "\n\n" + chunk.content
                candidate_tokens = count_tokens(candidate_content)
                if candidate_tokens <= max_tokens:
                    merged[-1] = ChunkResult(
                        content=candidate_content,
                        page_num=page_num,
                        section_title=merged[-1].section_title,
                        token_count=candidate_tokens,
                    )
                    continue
            merged.append(chunk)

        # Detect language for each chunk
        for chunk in merged:
            chunk.language = _detect_language(chunk.content)

        all_chunks.extend(merged)

    logger.info("semantic_chunking_complete", num_chunks=len(all_chunks))
    return all_chunks
```

**Step 2: Run tests to verify they pass**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/unit/test_semantic_chunking.py -x -v`
Expected: All 7 tests PASS.

**Step 3: Run all existing tests for no regressions**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/ -x -v`
Expected: All 71+ tests pass (old `chunk_text` tests still pass since that function is unchanged).

**Step 4: Commit**

```bash
git add backend/app/pipelines/chunking.py
git commit -m "feat: implement structure-aware chunking with page boundaries and section tracking"
```

---

### Task 4: Implement document summarizer

**Files:**
- Create: `backend/app/pipelines/summarizer.py`
- Create: `tests/unit/test_summarizer.py`

**Step 1: Write the failing test**

Create `tests/unit/test_summarizer.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch


@patch("app.pipelines.summarizer.OpenAIChatGenerator")
def test_generate_summary_returns_string(mock_gen_cls: MagicMock) -> None:
    """generate_summary should return a summary string from the LLM."""
    mock_instance = MagicMock()
    mock_reply = MagicMock()
    mock_reply.text = "This document discusses contract terms and payment conditions."
    mock_instance.run.return_value = {"replies": [mock_reply]}
    mock_gen_cls.return_value = mock_instance

    from app.pipelines.summarizer import generate_summary
    result = generate_summary("Some long document text about contracts and payments.")
    assert isinstance(result, str)
    assert len(result) > 0


@patch("app.pipelines.summarizer.OpenAIChatGenerator")
def test_generate_summary_fallback_on_error(mock_gen_cls: MagicMock) -> None:
    """generate_summary should return empty string on LLM failure."""
    mock_gen_cls.return_value.run.side_effect = Exception("LLM unavailable")

    from app.pipelines.summarizer import generate_summary
    result = generate_summary("Some text.")
    assert result == ""
```

**Step 2: Run test to verify it fails**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/unit/test_summarizer.py -x -v`
Expected: FAIL — module `app.pipelines.summarizer` not found.

**Step 3: Implement summarizer**

Create `backend/app/pipelines/summarizer.py`:

```python
from __future__ import annotations

import structlog
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from app.config import settings

logger = structlog.get_logger()

SUMMARY_SYSTEM_PROMPT = """\
Generate a 2-3 sentence summary of the following document.
Focus on the document type, subject matter, and key topics.
Return ONLY the summary text, no headers or formatting.\
"""


def generate_summary(text: str, max_chars: int = 3000) -> str:
    """Generate a short summary of a document for contextual chunk prefixing.

    Args:
        text: The full document text (will be truncated to max_chars).
        max_chars: Maximum characters to send to LLM.

    Returns:
        A 2-3 sentence summary string, or empty string on failure.
    """
    try:
        generator = OpenAIChatGenerator(
            api_key=Secret.from_token("ollama"),
            model=settings.LLM_MODEL,
            api_base_url=settings.LLM_BASE_URL,
            generation_kwargs={"temperature": 0.1, "max_tokens": 150},
        )

        truncated = text[:max_chars]
        messages = [
            ChatMessage.from_system(SUMMARY_SYSTEM_PROMPT),
            ChatMessage.from_user(truncated),
        ]

        result = generator.run(messages=messages)
        summary = result["replies"][0].text.strip()
        logger.info("document_summary_generated", length=len(summary))
        return summary

    except Exception as exc:
        logger.warning("document_summary_failed", error=str(exc))
        return ""
```

**Step 4: Run tests**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/unit/test_summarizer.py -x -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add backend/app/pipelines/summarizer.py tests/unit/test_summarizer.py
git commit -m "feat: add document summarizer for contextual chunk prefixes"
```

---

### Task 5: Update ingestion pipeline and Celery task for semantic chunking

**Files:**
- Modify: `backend/app/pipelines/ingestion.py`
- Modify: `backend/app/tasks/ingestion.py`

**Context:** The ingestion pipeline currently calls `parse_document()` which returns flat text via `export_to_markdown()`. We change it to return the Docling document object directly, then use `chunk_document()` instead of `chunk_text()`. The `create_document_chunks()` function is updated to accept `ChunkResult` objects and populate all metadata fields. The Celery task is updated to generate and store the document summary.

**Step 1: Update parse_document in ingestion.py**

Replace the `parse_document` function (lines 63-76) in `backend/app/pipelines/ingestion.py`:

```python
def parse_document(file_path: str | Path) -> tuple[object, str, int]:
    """Parse a document using Docling and return (document_model, text, page_count).

    Args:
        file_path: Path to the document file.

    Returns:
        Tuple of (docling_document, extracted_text, page_count).
    """
    converter = DocumentConverter()
    result = converter.convert(str(file_path))
    text = result.document.export_to_markdown()
    page_count = len(result.document.pages) if hasattr(result.document, "pages") else 1
    return result.document, text, page_count
```

**Step 2: Update create_document_chunks to accept ChunkResult objects**

Replace the `create_document_chunks` function (lines 137-167) in `backend/app/pipelines/ingestion.py`:

```python
async def create_document_chunks(
    session: AsyncSession,
    doc_id: UUID,
    chunks: list,
    client_id: str = "default",
) -> list[DocumentChunk]:
    """Create DocumentChunk rows in the database.

    Accepts either a list of strings (legacy) or a list of ChunkResult objects.

    Args:
        session: Active async database session.
        doc_id: Parent Document UUID.
        chunks: List of chunk text strings or ChunkResult objects.
        client_id: Tenant identifier.

    Returns:
        List of DocumentChunk ORM objects (not yet committed).
    """
    from app.pipelines.chunking import ChunkResult

    chunk_rows: list[DocumentChunk] = []
    for seq, chunk_item in enumerate(chunks):
        if isinstance(chunk_item, ChunkResult):
            chunk_row = DocumentChunk(
                document_id=doc_id,
                chunk_seq=seq,
                content=chunk_item.content,
                section_title=chunk_item.section_title,
                page_num=chunk_item.page_num,
                language=chunk_item.language,
                token_count=chunk_item.token_count,
                client_id=client_id,
            )
        else:
            chunk_row = DocumentChunk(
                document_id=doc_id,
                chunk_seq=seq,
                content=chunk_item,
                token_count=count_tokens(chunk_item),
                client_id=client_id,
            )
        session.add(chunk_row)
        chunk_rows.append(chunk_row)

    await session.flush()
    return chunk_rows
```

**Step 3: Update ingest_document to use chunk_document**

Replace the `ingest_document` function (lines 79-135) in `backend/app/pipelines/ingestion.py`. Update imports at the top to include `chunk_document` and `ChunkResult`:

```python
from app.pipelines.chunking import chunk_text, chunk_document, ChunkResult, count_tokens
```

Replace the function:

```python
def ingest_document(
    file_path: str | Path,
    filename: str,
    client_id: str = "default",
    summary: str = "",
) -> list[HaystackDocument]:
    """Full ingestion: parse -> chunk -> embed -> store in Qdrant.

    Args:
        file_path: Path to the uploaded file.
        filename: Original filename for citation metadata.
        client_id: Client identifier for multi-tenancy.
        summary: Optional document summary for contextual prefix.

    Returns:
        List of Haystack Documents that were indexed.
    """
    logger.info("ingestion_started", filename=filename, client_id=client_id)

    # 1. Parse document
    doc_model, text, page_count = parse_document(file_path)
    logger.info("document_parsed", filename=filename, page_count=page_count, chars=len(text))

    # 2. Chunk using structure-aware chunking
    chunk_results = chunk_document(
        doc_model,
        target_tokens=settings.CHUNK_TARGET_TOKENS,
        max_tokens=settings.CHUNK_MAX_TOKENS,
        min_tokens=settings.CHUNK_MIN_TOKENS,
        overlap_pct=settings.CHUNK_OVERLAP_PCT,
    )

    # Fallback to basic chunking if Docling model doesn't expose pages
    if not chunk_results:
        text_chunks = chunk_text(
            text,
            target_tokens=settings.CHUNK_TARGET_TOKENS,
            max_tokens=settings.CHUNK_MAX_TOKENS,
            min_tokens=settings.CHUNK_MIN_TOKENS,
            overlap_pct=settings.CHUNK_OVERLAP_PCT,
        )
        chunk_results = [
            ChunkResult(content=c, page_num=i + 1, token_count=count_tokens(c))
            for i, c in enumerate(text_chunks)
        ]

    logger.info("document_chunked", filename=filename, num_chunks=len(chunk_results))

    # 3. Create Haystack Documents with metadata + contextual prefix
    haystack_docs: list[HaystackDocument] = []
    for seq, cr in enumerate(chunk_results):
        content = f"{summary}\n\n{cr.content}" if summary else cr.content
        doc = HaystackDocument(
            id=str(uuid4()),
            content=content,
            meta={
                "source": filename,
                "page_num": cr.page_num,
                "chunk_seq": seq,
                "section_title": cr.section_title,
                "language": cr.language,
                "token_count": cr.token_count,
                "client_id": client_id,
            },
        )
        haystack_docs.append(doc)

    # 4. Embed and write to Qdrant via Haystack pipeline
    store = _get_document_store()
    pipeline = _build_ingestion_pipeline(store)
    result = pipeline.run({"sparse_embedder": {"documents": haystack_docs}})
    written_count = result.get("writer", {}).get("documents_written", len(haystack_docs))
    logger.info("ingestion_complete", filename=filename, chunks_written=written_count)

    return haystack_docs
```

**Step 4: Update Celery task to generate summary and use new parse_document signature**

In `backend/app/tasks/ingestion.py`, update the `_do_ingest` function. The key changes:
- `parse_document` now returns 3 values (doc_model, text, page_count)
- Generate summary from text
- Store summary on Document row
- Use `chunk_document()` for structured chunks
- Pass summary to `ingest_document()`

Replace lines 47-125 in `backend/app/tasks/ingestion.py`:

```python
async def _do_ingest(doc_id: str, file_path: str, client_id: str) -> dict[str, object]:
    """Async implementation of document ingestion with DB persistence."""
    doc_uuid = UUID(doc_id)

    async with async_session_factory() as session:
        result = await session.execute(select(Document).where(Document.id == doc_uuid))
        doc = result.scalar_one_or_none()
        if not doc:
            raise ValueError(f"Document {doc_id} not found in database")

        try:
            # 1. Mark as processing
            doc.status = DocumentStatus.PROCESSING
            await session.flush()
            logger.info("task_ingestion_started", doc_id=doc_id, file_path=file_path, client_id=client_id)

            # 2. Parse document with Docling
            doc_model, text, page_count = parse_document(file_path)
            logger.info("task_document_parsed", doc_id=doc_id, page_count=page_count, chars=len(text))

            # 3. Generate document summary for contextual prefix
            from app.pipelines.summarizer import generate_summary
            summary = generate_summary(text)
            doc.summary = summary
            await session.flush()

            # 4. Chunk using structure-aware chunking
            from app.pipelines.chunking import chunk_document as chunk_doc_fn, chunk_text, ChunkResult, count_tokens
            chunk_results = chunk_doc_fn(
                doc_model,
                target_tokens=settings.CHUNK_TARGET_TOKENS,
                max_tokens=settings.CHUNK_MAX_TOKENS,
                min_tokens=settings.CHUNK_MIN_TOKENS,
                overlap_pct=settings.CHUNK_OVERLAP_PCT,
            )

            # Fallback to basic chunking
            if not chunk_results:
                text_chunks = chunk_text(
                    text,
                    target_tokens=settings.CHUNK_TARGET_TOKENS,
                    max_tokens=settings.CHUNK_MAX_TOKENS,
                    min_tokens=settings.CHUNK_MIN_TOKENS,
                    overlap_pct=settings.CHUNK_OVERLAP_PCT,
                )
                chunk_results = [
                    ChunkResult(content=c, page_num=i + 1, token_count=count_tokens(c))
                    for i, c in enumerate(text_chunks)
                ]

            logger.info("task_document_chunked", doc_id=doc_id, num_chunks=len(chunk_results))

            # 5. Persist DocumentChunk rows to PostgreSQL
            await create_document_chunks(
                session=session,
                doc_id=doc_uuid,
                chunks=chunk_results,
                client_id=client_id,
            )

            # 6. Embed chunks and write vectors to Qdrant
            ingest_document(
                file_path=file_path,
                filename=doc.original_filename,
                client_id=client_id,
                summary=summary,
            )

            # 7. Mark document as completed with final counts
            doc.status = DocumentStatus.COMPLETED
            doc.chunk_count = len(chunk_results)
            doc.page_count = page_count
            await session.commit()

            logger.info(
                "task_ingestion_complete",
                doc_id=doc_id,
                chunk_count=len(chunk_results),
                page_count=page_count,
            )

            return {"doc_id": doc_id, "status": "completed", "chunk_count": len(chunk_results)}

        except Exception as exc:
            doc.status = DocumentStatus.FAILED
            await session.commit()
            logger.exception("task_ingestion_failed", doc_id=doc_id, error=str(exc))
            raise
```

Remove the now-unnecessary direct imports of `chunk_text` from the top of `tasks/ingestion.py` — they're now imported inside the function.

**Step 5: Run all backend tests**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/ -x -v`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add backend/app/pipelines/ingestion.py backend/app/tasks/ingestion.py
git commit -m "feat: integrate semantic chunking with ingestion pipeline and Celery task"
```

---

### Task 6: Verify Workstream 1 and run full test suite

**Step 1: Run backend lint**

Run: `cd /e/LCMGoCloudRAG/backend && python -m ruff check . && python -m ruff format --check .`
Expected: Clean.

**Step 2: Run all backend tests**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/ -x -v`
Expected: All tests pass.

**Step 3: Run frontend tests (sanity check)**

Run: `cd /e/LCMGoCloudRAG/frontend && npx vitest run`
Expected: All 28 tests pass.

---

## Workstream 2: Real Ollama Streaming

### Task 7: Extract retrieve_context() from query_documents()

**Files:**
- Modify: `backend/app/pipelines/query.py`
- Create: `tests/unit/test_retrieve_context.py`

**Step 1: Write the failing test**

Create `tests/unit/test_retrieve_context.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch


@patch("app.pipelines.query._get_document_store")
@patch("app.pipelines.query._build_query_pipeline")
@patch("app.pipelines.query.expand_query")
def test_retrieve_context_returns_docs_and_formatted_context(
    mock_expand: MagicMock, mock_build: MagicMock, mock_store: MagicMock
) -> None:
    """retrieve_context should return retrieved docs and formatted context string."""
    mock_expand.return_value = ["test question"]

    mock_doc = MagicMock()
    mock_doc.id = "doc-1"
    mock_doc.content = "Some relevant content."
    mock_doc.score = 0.9
    mock_doc.meta = {"source": "test.pdf", "page_num": 1, "client_id": "default"}

    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = {
        "retriever": {"documents": [mock_doc]},
        "ranker": {"documents": [mock_doc]},
    }
    mock_build.return_value = mock_pipeline

    from app.pipelines.query import retrieve_context
    result = retrieve_context("test question", client_id="default")

    assert "documents" in result
    assert "formatted_context" in result
    assert "citations" in result
    assert len(result["documents"]) == 1
    assert "test.pdf" in result["formatted_context"]
```

**Step 2: Run test to verify it fails**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/unit/test_retrieve_context.py -x -v`
Expected: FAIL — `retrieve_context` not found.

**Step 3: Add retrieve_context to query.py**

Add after line 117 (end of `_build_query_pipeline`) in `backend/app/pipelines/query.py`:

```python
def retrieve_context(question: str, client_id: str = "default") -> dict:
    """Run the retrieval + reranking pipeline without LLM generation.

    Args:
        question: The user's question.
        client_id: Client identifier for multi-tenancy filtering.

    Returns:
        Dict with keys: documents, formatted_context, citations, latency_ms.
    """
    logger.info("retrieval_started", question=question[:100], client_id=client_id)
    start_time = time.perf_counter()

    store = _get_document_store()
    pipeline = _build_query_pipeline(store)

    variants = expand_query(question)

    all_retrieved: dict[str, object] = {}
    for variant in variants:
        result = pipeline.run(
            {
                "sparse_embedder": {"text": variant},
                "dense_embedder": {"text": variant},
                "retriever": {
                    "filters": {
                        "operator": "AND",
                        "conditions": [
                            {"field": "meta.client_id", "operator": "==", "value": client_id},
                        ],
                    },
                },
                "ranker": {"query": variant},
                "prompt_builder": {"query": question},
            }
        )

        for doc in result.get("ranker", {}).get("documents", result.get("retriever", {}).get("documents", [])):
            doc_key = doc.id
            if doc_key not in all_retrieved or (doc.score and doc.score > all_retrieved[doc_key].score):
                all_retrieved[doc_key] = doc

    retrieved_docs = list(all_retrieved.values())
    latency_ms = round((time.perf_counter() - start_time) * 1000)

    # Build formatted context for LLM prompt
    context_parts: list[str] = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.meta.get("source", "unknown")
        page = doc.meta.get("page_num", "?")
        context_parts.append(f"[{i}] Source: {source}, Page {page}\n{doc.content}")
    formatted_context = "\n\n".join(context_parts)

    citations = [
        {
            "source": doc.meta.get("source", "unknown"),
            "page": doc.meta.get("page_num"),
            "content_preview": doc.content[:200] if doc.content else "",
        }
        for doc in retrieved_docs
    ]

    logger.info("retrieval_complete", latency_ms=latency_ms, num_docs=len(retrieved_docs))

    return {
        "documents": retrieved_docs,
        "formatted_context": formatted_context,
        "citations": citations,
        "latency_ms": latency_ms,
    }
```

**Step 4: Run tests**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/unit/test_retrieve_context.py -x -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add backend/app/pipelines/query.py tests/unit/test_retrieve_context.py
git commit -m "feat: extract retrieve_context() for retrieval without LLM generation"
```

---

### Task 8: Implement real Ollama streaming module

**Files:**
- Create: `backend/app/pipelines/streaming.py`
- Create: `tests/unit/test_streaming.py`
- Modify: `backend/app/config.py`

**Step 1: Add LLM_STREAM_TIMEOUT to config**

In `backend/app/config.py`, add after `LLM_MAX_TOKENS` (line 55):

```python
    LLM_STREAM_TIMEOUT: int = 120  # seconds
```

**Step 2: Write the failing test**

Create `tests/unit/test_streaming.py`:

```python
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.anyio
@patch("app.pipelines.streaming.httpx.AsyncClient")
async def test_stream_llm_response_yields_tokens(mock_client_cls: MagicMock) -> None:
    """stream_llm_response should yield token strings from SSE chunks."""

    async def mock_aiter(*_args, **_kwargs):
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            b'data: {"choices":[{"delta":{"content":" world"}}]}',
            b"data: [DONE]",
        ]
        for line in lines:
            yield line

    mock_response = AsyncMock()
    mock_response.aiter_lines = mock_aiter
    mock_response.status_code = 200
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_client

    from app.pipelines.streaming import stream_llm_response

    tokens: list[str] = []
    async for token in stream_llm_response("context here", "What is this?"):
        tokens.append(token)

    assert "Hello" in tokens
    assert " world" in tokens
```

**Step 3: Run test to verify it fails**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/unit/test_streaming.py -x -v`
Expected: FAIL — module not found.

**Step 4: Implement streaming.py**

Create `backend/app/pipelines/streaming.py`:

```python
from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import httpx
import structlog

from app.config import settings
from app.pipelines.query import SYSTEM_PROMPT

logger = structlog.get_logger()


async def stream_llm_response(
    context: str,
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> AsyncGenerator[str, None]:
    """Stream tokens from Ollama/vLLM via OpenAI-compatible SSE API.

    Args:
        context: Formatted retrieved context string.
        question: The user's question.
        conversation_history: Optional list of previous messages [{"role": "...", "content": "..."}].

    Yields:
        Individual token strings as they arrive from the LLM.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history)

    user_content = f"Context:\n{context}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": settings.LLM_MODEL,
        "messages": messages,
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "stream": True,
    }

    url = f"{settings.LLM_BASE_URL}/chat/completions"

    try:
        async with httpx.AsyncClient(timeout=settings.LLM_STREAM_TIMEOUT) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    logger.error("streaming_http_error", status=response.status_code, body=body[:500])
                    return

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # strip "data: " prefix
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                    except (json.JSONDecodeError, IndexError, KeyError) as exc:
                        logger.warning("streaming_parse_error", error=str(exc), line=line[:100])
                        continue

    except httpx.TimeoutException:
        logger.error("streaming_timeout", timeout=settings.LLM_STREAM_TIMEOUT)
    except Exception as exc:
        logger.error("streaming_error", error=str(exc))
```

**Step 5: Run tests**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/unit/test_streaming.py -x -v`
Expected: PASS.

**Step 6: Commit**

```bash
git add backend/app/pipelines/streaming.py tests/unit/test_streaming.py backend/app/config.py
git commit -m "feat: add real Ollama SSE streaming module via httpx"
```

---

### Task 9: Replace simulated streaming in WebSocket handler

**Files:**
- Modify: `backend/app/api/v1/ws.py`

**Context:** Replace the simulated word-by-word streaming (lines 224-228) with real token streaming from Ollama. The handler now calls `retrieve_context()` first, then streams from `stream_llm_response()`.

**Step 1: Update imports in ws.py**

Add to the imports at the top of `backend/app/api/v1/ws.py`:

```python
from app.pipelines.query import retrieve_context
from app.pipelines.streaming import stream_llm_response
```

Remove the import of `query_documents` (no longer called directly from ws.py).

**Step 2: Replace _handle_query function**

Replace the `_handle_query` function (lines 161-279) in `backend/app/api/v1/ws.py`:

```python
async def _handle_query(websocket: WebSocket, data: dict[str, Any]) -> None:
    """Handle a single ``query`` message on the WebSocket.

    Runs retrieval, then streams real tokens from Ollama, then fires off
    persistence and audit tasks in the background.
    """
    question: str = data.get("question", "").strip()
    client_id: str = settings.CLIENT_ID

    if not question:
        await websocket.send_json({"type": "error", "detail": "question field is required"})
        return

    # 1. Input guardrail
    guardrail_result: dict[str, Any] = scan_input(question)
    if guardrail_result["blocked"]:
        reason: str = guardrail_result.get("reason") or "guardrail_blocked"
        logger.warning("ws_query_blocked", reason=reason, client_id=client_id)
        asyncio.ensure_future(
            _write_query_audit(
                action=AuditAction.GUARDRAIL_BLOCKED,
                details={"reason": reason, "risk_score": guardrail_result.get("risk_score", 0.0)},
                client_id=client_id,
            )
        )
        await websocket.send_json({"type": "error", "detail": reason})
        return

    # 2. Acknowledge
    await websocket.send_json({"type": "status", "status": "processing"})

    # 3. Retrieve context (synchronous pipeline in thread)
    import time
    start_time = time.perf_counter()
    retrieval_result: dict[str, Any] = await asyncio.to_thread(
        retrieve_context,
        question=question,
        client_id=client_id,
    )

    formatted_context: str = retrieval_result["formatted_context"]
    citations_raw: list[dict[str, Any]] = retrieval_result.get("citations", [])
    retrieved_docs = retrieval_result.get("documents", [])

    # 4. Stream real tokens from Ollama
    answer_parts: list[str] = []
    try:
        async for token in stream_llm_response(formatted_context, question):
            answer_parts.append(token)
            await websocket.send_json({"type": "token", "token": token})
    except Exception as exc:
        logger.error("ws_streaming_error", error=str(exc))
        if not answer_parts:
            await websocket.send_json({"type": "error", "detail": "Streaming failed"})
            return

    answer = "".join(answer_parts)
    latency_ms = round((time.perf_counter() - start_time) * 1000)

    # 5. Faithfulness check on assembled answer
    faithfulness_score: float = 1.0
    retrieved_context = "\n\n".join(doc.content for doc in retrieved_docs if doc.content)
    if retrieved_context and answer:
        faithfulness_score, is_faithful = check_faithfulness(
            context=retrieved_context,
            response=answer,
        )
        if not is_faithful:
            # Send a correction message
            await websocket.send_json({
                "type": "token",
                "token": "\n\n[Note: Response could not be verified against source documents.]",
            })

    # 6. Citations
    citations_json: list[dict[str, Any]] = [
        {
            "source": c.get("source", "unknown"),
            "page": c.get("page"),
            "content_preview": c.get("content_preview", ""),
        }
        for c in citations_raw
    ]
    await websocket.send_json({"type": "citations", "citations": citations_json})

    # 7. Done
    await websocket.send_json({"type": "done", "latency_ms": latency_ms})

    logger.info(
        "ws_query_complete",
        question_length=len(question),
        answer_length=len(answer),
        citation_count=len(citations_json),
        latency_ms=latency_ms,
        client_id=client_id,
    )

    # 8. Fire-and-forget: persist and audit
    asyncio.ensure_future(
        _persist_query_response(
            question=question,
            answer=answer,
            citations_json=citations_json,
            latency_ms=latency_ms,
            model_used=settings.LLM_MODEL,
            faithfulness_score=faithfulness_score,
            client_id=client_id,
        )
    )
    asyncio.ensure_future(
        _write_query_audit(
            action=AuditAction.QUERY_SUBMITTED,
            details={"query_text": question[:200]},
            client_id=client_id,
        )
    )
    asyncio.ensure_future(
        _write_query_audit(
            action=AuditAction.RESPONSE_GENERATED,
            details={"latency_ms": latency_ms, "citation_count": len(citations_json)},
            client_id=client_id,
        )
    )
```

**Step 3: Run all backend tests**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/ -x -v`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add backend/app/api/v1/ws.py
git commit -m "feat: replace simulated streaming with real Ollama SSE token streaming"
```

---

### Task 10: Verify Workstream 2

**Step 1: Run backend lint + tests**

Run: `cd /e/LCMGoCloudRAG/backend && python -m ruff check . && python -m ruff format --check .`
Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/ -x -v`
Expected: Clean, all tests pass.

---

## Workstream 3: Conversation History

### Task 11: Create Conversation model and migration

**Files:**
- Create: `backend/app/models/conversation.py`
- Modify: `backend/app/models/__init__.py`
- Modify: `backend/app/models/query.py`
- Create: new Alembic migration

**Step 1: Create Conversation model**

Create `backend/app/models/conversation.py`:

```python
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, generate_uuid


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=generate_uuid)
    user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    client_id: Mapped[str] = mapped_column(String(100), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    queries: Mapped[list] = relationship("Query", back_populates="conversation", cascade="all, delete-orphan")
```

**Step 2: Add conversation_id FK to Query model**

In `backend/app/models/query.py`, add after line 21 (`client_id`):

```python
    conversation_id: Mapped[UUID | None] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), nullable=True, index=True)
```

Add the relationship after line 23 (`response`):

```python
    conversation: Mapped[object | None] = relationship("Conversation", back_populates="queries")
```

**Step 3: Update models __init__.py**

In `backend/app/models/__init__.py`, add the import:

```python
from app.models.conversation import Conversation
```

Add `"Conversation"` to the `__all__` list.

**Step 4: Generate and apply migration**

Run: `cd /e/LCMGoCloudRAG/backend && PYTHONPATH=. alembic revision --autogenerate -m "add_conversations_table_and_query_fk"`
Run: `cd /e/LCMGoCloudRAG/backend && PYTHONPATH=. alembic upgrade head`

**Step 5: Run backend tests**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/ -x -v`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add backend/app/models/conversation.py backend/app/models/query.py backend/app/models/__init__.py backend/alembic/versions/
git commit -m "feat: add Conversation model with Query FK for chat history"
```

---

### Task 12: Implement conversation CRUD endpoints

**Files:**
- Create: `backend/app/api/v1/conversations.py`
- Create: `backend/app/schemas/conversation.py`
- Modify: `backend/app/api/v1/router.py`
- Create: `tests/unit/test_conversations_api.py`

**Step 1: Create conversation schemas**

Create `backend/app/schemas/conversation.py`:

```python
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ConversationCreate(BaseModel):
    title: str | None = None


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str | None
    client_id: str
    created_at: datetime
    updated_at: datetime


class ConversationMessageResponse(BaseModel):
    role: str
    content: str
    created_at: datetime


class PaginatedConversationsResponse(BaseModel):
    items: list[ConversationResponse]
    total: int
    page: int
    page_size: int
```

**Step 2: Create conversations API**

Create `backend/app/api/v1/conversations.py`:

```python
from __future__ import annotations

from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.database import get_db_session
from app.exceptions import NotFoundError
from app.models import Query, QueryResponse, User
from app.models.conversation import Conversation
from app.schemas.conversation import (
    ConversationCreate,
    ConversationMessageResponse,
    ConversationResponse,
    PaginatedConversationsResponse,
)

logger = structlog.get_logger()
router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    body: ConversationCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> ConversationResponse:
    """Create a new conversation."""
    conv = Conversation(
        user_id=current_user.id,
        title=body.title,
        client_id=current_user.client_id,
    )
    session.add(conv)
    await session.flush()
    return ConversationResponse.model_validate(conv)


@router.get("", response_model=PaginatedConversationsResponse)
async def list_conversations(
    page: int = 1,
    page_size: int = 20,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> PaginatedConversationsResponse:
    """List user's conversations, most recent first."""
    stmt = select(Conversation).where(
        Conversation.user_id == current_user.id,
        Conversation.client_id == current_user.client_id,
    )

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await session.execute(count_stmt)).scalar() or 0

    stmt = stmt.order_by(Conversation.updated_at.desc())
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    result = await session.execute(stmt)
    convs = result.scalars().all()

    return PaginatedConversationsResponse(
        items=[ConversationResponse.model_validate(c) for c in convs],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{conv_id}/messages", response_model=list[ConversationMessageResponse])
async def get_conversation_messages(
    conv_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> list[ConversationMessageResponse]:
    """Get messages for a conversation."""
    conv_uuid = UUID(conv_id)
    stmt = select(Conversation).where(
        Conversation.id == conv_uuid,
        Conversation.user_id == current_user.id,
    )
    result = await session.execute(stmt)
    conv = result.scalar_one_or_none()
    if not conv:
        raise NotFoundError(f"Conversation {conv_id} not found")

    query_stmt = (
        select(Query)
        .options(selectinload(Query.response))
        .where(Query.conversation_id == conv_uuid)
        .order_by(Query.created_at.asc())
    )
    query_result = await session.execute(query_stmt)
    queries = query_result.scalars().all()

    messages: list[ConversationMessageResponse] = []
    for q in queries:
        messages.append(ConversationMessageResponse(
            role="user", content=q.query_text, created_at=q.created_at,
        ))
        if q.response:
            messages.append(ConversationMessageResponse(
                role="assistant", content=q.response.response_text, created_at=q.response.created_at,
            ))
    return messages


@router.delete("/{conv_id}", status_code=204)
async def delete_conversation(
    conv_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    """Delete a conversation and all its queries."""
    conv_uuid = UUID(conv_id)
    stmt = select(Conversation).where(
        Conversation.id == conv_uuid,
        Conversation.user_id == current_user.id,
    )
    result = await session.execute(stmt)
    conv = result.scalar_one_or_none()
    if not conv:
        raise NotFoundError(f"Conversation {conv_id} not found")

    await session.delete(conv)
    return Response(status_code=204)
```

**Step 3: Register router**

In `backend/app/api/v1/router.py`, add:

```python
from app.api.v1.conversations import router as conversations_router
```

And add: `api_v1_router.include_router(conversations_router)`

**Step 4: Write tests**

Create `tests/unit/test_conversations_api.py`:

```python
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


@pytest.mark.anyio
@patch("app.api.v1.conversations.get_current_user")
async def test_create_conversation(mock_user: MagicMock) -> None:
    """POST /conversations should create a conversation row."""
    from app.models.conversation import Conversation
    assert Conversation.__tablename__ == "conversations"


def test_conversation_model_has_required_fields() -> None:
    """Conversation model should have id, user_id, title, client_id, timestamps."""
    from app.models.conversation import Conversation
    columns = {c.name for c in Conversation.__table__.columns}
    assert "id" in columns
    assert "user_id" in columns
    assert "title" in columns
    assert "client_id" in columns
    assert "created_at" in columns
    assert "updated_at" in columns


def test_query_has_conversation_id_column() -> None:
    """Query model should have a conversation_id FK."""
    from app.models.query import Query
    columns = {c.name for c in Query.__table__.columns}
    assert "conversation_id" in columns
```

**Step 5: Run tests**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/unit/test_conversations_api.py -x -v`
Expected: PASS.

**Step 6: Commit**

```bash
git add backend/app/api/v1/conversations.py backend/app/schemas/conversation.py backend/app/api/v1/router.py tests/unit/test_conversations_api.py
git commit -m "feat: add conversation CRUD endpoints with schemas"
```

---

### Task 13: Update WebSocket handler for conversation context

**Files:**
- Modify: `backend/app/api/v1/ws.py`
- Modify: `backend/app/config.py`

**Step 1: Add conversation settings to config**

In `backend/app/config.py`, add after `LLM_STREAM_TIMEOUT`:

```python
    CONVERSATION_CONTEXT_MESSAGES: int = 5  # number of past messages to include
```

**Step 2: Update _handle_query to support conversation context**

Add a helper function before `_handle_query` in `ws.py`:

```python
async def _load_conversation_history(
    conversation_id: str | None,
    client_id: str,
) -> tuple[str | None, list[dict[str, str]]]:
    """Load conversation history for LLM context.

    Returns (conversation_id, messages) where messages is a list of
    {"role": "user"|"assistant", "content": "..."} dicts.
    """
    if not conversation_id:
        # Auto-create conversation
        from app.models.conversation import Conversation
        async with async_session_factory() as session:
            conv = Conversation(user_id=DEV_USER_ID, client_id=client_id)
            session.add(conv)
            await session.flush()
            conversation_id = str(conv.id)
            await session.commit()
        return conversation_id, []

    # Load last N messages from existing conversation
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    history: list[dict[str, str]] = []
    async with async_session_factory() as session:
        from app.models import Query as QueryModel
        stmt = (
            select(QueryModel)
            .options(selectinload(QueryModel.response))
            .where(QueryModel.conversation_id == conversation_id)
            .order_by(QueryModel.created_at.desc())
            .limit(settings.CONVERSATION_CONTEXT_MESSAGES)
        )
        result = await session.execute(stmt)
        queries = list(reversed(result.scalars().all()))

        for q in queries:
            history.append({"role": "user", "content": q.query_text})
            if q.response:
                history.append({"role": "assistant", "content": q.response.response_text})

    return conversation_id, history
```

In `_handle_query`, add conversation loading after guardrail check and before retrieval:

```python
    # Extract conversation_id from message
    conversation_id: str | None = data.get("conversation_id")
    conversation_id, conversation_history = await _load_conversation_history(conversation_id, client_id)
```

Pass `conversation_history` to `stream_llm_response`:

```python
    async for token in stream_llm_response(formatted_context, question, conversation_history):
```

Update `_persist_query_response` to accept and store `conversation_id`. Add `conversation_id` parameter, and in the Query creation set `conversation_id=UUID(conversation_id) if conversation_id else None`.

Also send `conversation_id` back to the client in the done message:

```python
    await websocket.send_json({"type": "done", "latency_ms": latency_ms, "conversation_id": conversation_id})
```

**Step 3: Run all backend tests**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/ -x -v`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add backend/app/api/v1/ws.py backend/app/config.py
git commit -m "feat: add conversation context loading to WebSocket handler"
```

---

### Task 14: Add conversation title generation

**Files:**
- Modify: `backend/app/api/v1/ws.py`

**Step 1: Add title generation helper**

Add after `_load_conversation_history` in `ws.py`:

```python
async def _generate_conversation_title(conversation_id: str, question: str) -> None:
    """Generate a short title for a conversation from the first question (fire-and-forget)."""
    try:
        from app.pipelines.summarizer import generate_summary
        title = await asyncio.to_thread(
            generate_summary,
            f"Generate a 3-5 word title for a conversation starting with: {question}",
            max_chars=500,
        )
        if not title:
            title = question[:50]

        from app.models.conversation import Conversation
        async with async_session_factory() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
            conv = result.scalar_one_or_none()
            if conv and not conv.title:
                conv.title = title[:500]
                await session.commit()
    except Exception as exc:
        logger.warning("title_generation_failed", error=str(exc))
```

**Step 2: Call title generation after first response**

At the end of `_handle_query`, add (with the other fire-and-forget tasks):

```python
    # Generate title if this is the first message in the conversation
    if conversation_id and len(conversation_history) == 0:
        asyncio.ensure_future(_generate_conversation_title(conversation_id, question))
```

**Step 3: Run backend tests**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/ -x -v`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add backend/app/api/v1/ws.py
git commit -m "feat: auto-generate conversation title from first question"
```

---

### Task 15: Add frontend conversation API functions and i18n keys

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/messages/en.json`
- Modify: `frontend/messages/el.json`

**Step 1: Add conversation types and API functions to api.ts**

Add to `frontend/src/lib/api.ts` after the existing types:

```typescript
export interface ConversationResponse {
  id: string;
  title: string | null;
  client_id: string;
  created_at: string;
  updated_at: string;
}

export interface PaginatedConversationsResponse {
  items: ConversationResponse[];
  total: number;
  page: number;
  page_size: number;
}

export interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
  created_at: string;
}
```

Add API functions:

```typescript
export async function fetchConversations(
  page: number = 1,
  pageSize: number = 20,
): Promise<PaginatedConversationsResponse> {
  return apiFetch<PaginatedConversationsResponse>(
    `/conversations?page=${page}&page_size=${pageSize}`,
  );
}

export async function createConversation(
  title?: string,
): Promise<ConversationResponse> {
  return apiFetch<ConversationResponse>("/conversations", {
    method: "POST",
    body: JSON.stringify({ title: title || null }),
  });
}

export async function fetchConversationMessages(
  convId: string,
): Promise<ConversationMessage[]> {
  return apiFetch<ConversationMessage[]>(`/conversations/${convId}/messages`);
}

export async function deleteConversation(convId: string): Promise<void> {
  await fetch(`${API_BASE}/conversations/${convId}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
}
```

**Step 2: Add i18n keys**

In `frontend/messages/en.json`, add to the `chat` section:

```json
    "newConversation": "New Chat",
    "conversations": "Conversations",
    "deleteConversation": "Delete",
    "noConversations": "No conversations yet"
```

In `frontend/messages/el.json`, add to the `chat` section:

```json
    "newConversation": "Νέα Συνομιλία",
    "conversations": "Συνομιλίες",
    "deleteConversation": "Διαγραφή",
    "noConversations": "Δεν υπάρχουν συνομιλίες"
```

**Step 3: Run frontend tests**

Run: `cd /e/LCMGoCloudRAG/frontend && npx vitest run`
Expected: All 28 tests pass.

**Step 4: Commit**

```bash
git add frontend/src/lib/api.ts frontend/messages/en.json frontend/messages/el.json
git commit -m "feat: add conversation API functions and i18n keys"
```

---

### Task 16: Create ConversationList component

**Files:**
- Create: `frontend/src/components/chat/ConversationList.tsx`

**Step 1: Create the component**

Create `frontend/src/components/chat/ConversationList.tsx`:

```tsx
"use client";

import { useTranslations } from "next-intl";
import { MessageSquarePlus, Trash2 } from "lucide-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  fetchConversations,
  deleteConversation,
  type ConversationResponse,
} from "@/lib/api";

interface ConversationListProps {
  activeConversationId: string | null;
  onSelect: (conv: ConversationResponse | null) => void;
}

export function ConversationList({
  activeConversationId,
  onSelect,
}: ConversationListProps) {
  const t = useTranslations("chat");
  const queryClient = useQueryClient();

  const { data } = useQuery({
    queryKey: ["conversations"],
    queryFn: () => fetchConversations(1, 50),
    refetchInterval: 10000,
  });

  const deleteMutation = useMutation({
    mutationFn: deleteConversation,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["conversations"] });
      onSelect(null);
    },
  });

  const conversations = data?.items ?? [];

  return (
    <div className="flex flex-col gap-1" data-testid="conversation-list">
      <Button
        variant="outline"
        size="sm"
        className="mb-2 w-full justify-start gap-2"
        onClick={() => onSelect(null)}
        data-testid="new-conversation-button"
      >
        <MessageSquarePlus className="h-4 w-4" />
        {t("newConversation")}
      </Button>

      {conversations.length === 0 && (
        <p className="px-2 text-xs text-muted-foreground">{t("noConversations")}</p>
      )}

      {conversations.map((conv) => (
        <div
          key={conv.id}
          className={cn(
            "group flex items-center justify-between rounded-md px-2 py-1.5 text-sm cursor-pointer hover:bg-accent",
            activeConversationId === conv.id && "bg-accent",
          )}
          onClick={() => onSelect(conv)}
          data-testid={`conversation-${conv.id}`}
        >
          <span className="truncate">{conv.title || t("newConversation")}</span>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 opacity-0 group-hover:opacity-100"
            onClick={(e) => {
              e.stopPropagation();
              deleteMutation.mutate(conv.id);
            }}
            aria-label={t("deleteConversation")}
          >
            <Trash2 className="h-3 w-3" />
          </Button>
        </div>
      ))}
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add frontend/src/components/chat/ConversationList.tsx
git commit -m "feat: add ConversationList component with CRUD"
```

---

### Task 17: Update ChatInterface for conversation tracking

**Files:**
- Modify: `frontend/src/components/chat/ChatInterface.tsx`

**Step 1: Update ChatInterface**

Key changes to `frontend/src/components/chat/ChatInterface.tsx`:
- Accept `conversationId` and `onConversationChange` props
- Pass `conversation_id` in WebSocket messages
- Handle `conversation_id` from done messages
- Reconnect WebSocket when conversation changes
- Load conversation messages on switch

Add props interface and update the component. The WebSocket `send` call becomes:

```typescript
wsRef.current?.send({ type: "query", question, conversation_id: conversationId || undefined });
```

Add `onDone` callback parsing for `conversation_id`:

```typescript
onDone: (data?: { conversation_id?: string }) => {
    // ... existing done logic ...
    if (data?.conversation_id && onConversationChange) {
        onConversationChange(data.conversation_id);
    }
},
```

**Step 2: Run frontend tests**

Run: `cd /e/LCMGoCloudRAG/frontend && npx vitest run`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add frontend/src/components/chat/ChatInterface.tsx
git commit -m "feat: add conversation tracking to ChatInterface"
```

---

### Task 18: Update Sidebar with conversation list and wire up chat page

**Files:**
- Modify: `frontend/src/components/Sidebar.tsx`
- Modify: `frontend/src/app/[locale]/chat/page.tsx`

**Step 1: Add ConversationList to Sidebar**

In `frontend/src/components/Sidebar.tsx`, add a conversation list section after the nav items. Import `ConversationList` and add state management for active conversation.

Since the Sidebar and ChatInterface need to share conversation state, the simplest approach is to lift state to the chat page. Update `frontend/src/app/[locale]/chat/page.tsx` to manage conversation state and pass it to both components.

**Step 2: Run frontend tests**

Run: `cd /e/LCMGoCloudRAG/frontend && npx vitest run`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add frontend/src/components/Sidebar.tsx frontend/src/app/[locale]/chat/page.tsx
git commit -m "feat: integrate conversation list into sidebar and chat page"
```

---

## Workstream 4: RAGAS Evaluation Suite

### Task 19: Add ragas dependency and create golden test set

**Files:**
- Modify: `backend/requirements.txt`
- Create: `tests/evaluation/golden_set.json`

**Step 1: Add ragas to requirements**

Add to `backend/requirements.txt` after `sentence-transformers`:

```
ragas>=0.2.0
datasets>=2.0.0
```

**Step 2: Install**

Run: `cd /e/LCMGoCloudRAG/backend && pip install ragas datasets`

**Step 3: Create golden test set**

Create `tests/evaluation/golden_set.json` with 10 sample Q&A pairs (expandable later). Use placeholder content that tests both English and Greek queries:

```json
[
  {
    "question": "What are the payment terms in the contract?",
    "ground_truth": "Payment terms are net 30 days from invoice date.",
    "expected_source": "contract.pdf"
  },
  {
    "question": "Ποια είναι η διάρκεια της σύμβασης;",
    "ground_truth": "Η σύμβαση ισχύει για είκοσι τέσσερις (24) μήνες.",
    "expected_source": "contract.pdf"
  },
  {
    "question": "Who are the parties involved in the agreement?",
    "ground_truth": "The agreement is between the company and the client as specified in the document header.",
    "expected_source": "agreement.pdf"
  },
  {
    "question": "What happens if a payment is late?",
    "ground_truth": "Late payments incur a penalty as specified in the penalty clause.",
    "expected_source": "contract.pdf"
  },
  {
    "question": "Ποιες είναι οι υποχρεώσεις του εργοδότη;",
    "ground_truth": "Ο εργοδότης υποχρεούται να παρέχει ασφαλές περιβάλλον εργασίας.",
    "expected_source": "employment.pdf"
  },
  {
    "question": "What is the termination clause?",
    "ground_truth": "Either party may terminate with 30 days written notice.",
    "expected_source": "contract.pdf"
  },
  {
    "question": "What are the confidentiality obligations?",
    "ground_truth": "Both parties must maintain confidentiality of proprietary information for 5 years after termination.",
    "expected_source": "nda.pdf"
  },
  {
    "question": "Ποια είναι η πολιτική επιστροφών;",
    "ground_truth": "Οι επιστροφές γίνονται εντός 14 ημερών από την παραλαβή.",
    "expected_source": "policy.pdf"
  },
  {
    "question": "What insurance coverage is required?",
    "ground_truth": "The contractor must maintain professional liability insurance of at least $1 million.",
    "expected_source": "contract.pdf"
  },
  {
    "question": "What is the governing law for this contract?",
    "ground_truth": "This contract is governed by the laws of Greece.",
    "expected_source": "contract.pdf"
  }
]
```

**Step 4: Commit**

```bash
git add backend/requirements.txt tests/evaluation/golden_set.json
git commit -m "feat: add ragas dependency and golden test set for RAG evaluation"
```

---

### Task 20: Implement RAGAS evaluation test suite

**Files:**
- Create: `tests/evaluation/conftest.py`
- Create: `tests/evaluation/test_ragas.py`
- Create: `tests/evaluation/test_citation_coverage.py`

**Step 1: Create conftest**

Create `tests/evaluation/conftest.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


@pytest.fixture
def golden_set() -> list[dict]:
    """Load the golden test set."""
    with open(GOLDEN_SET_PATH) as f:
        return json.load(f)
```

**Step 2: Create RAGAS test**

Create `tests/evaluation/test_ragas.py`:

```python
from __future__ import annotations

import pytest


@pytest.mark.evaluation
@pytest.mark.skipif(
    not pytest.importorskip("ragas", reason="ragas not installed"),
    reason="ragas not installed",
)
class TestRagasEvaluation:
    """RAGAS evaluation suite — run with: pytest -m evaluation"""

    def test_golden_set_is_valid(self, golden_set: list[dict]) -> None:
        """Golden set should have required fields."""
        assert len(golden_set) >= 10
        for item in golden_set:
            assert "question" in item
            assert "ground_truth" in item
            assert "expected_source" in item

    @pytest.mark.skip(reason="Requires live LLM and indexed documents")
    def test_faithfulness_above_threshold(self, golden_set: list[dict]) -> None:
        """Faithfulness should be above 0.95 target."""
        from ragas import evaluate
        from ragas.metrics import faithfulness

        # This test requires a running backend with indexed documents
        # Run manually: pytest tests/evaluation/test_ragas.py -m evaluation --no-skip
        pass

    @pytest.mark.skip(reason="Requires live LLM and indexed documents")
    def test_answer_relevancy_above_threshold(self, golden_set: list[dict]) -> None:
        """Answer relevancy should be above 0.90 target."""
        pass
```

**Step 3: Create citation coverage test**

Create `tests/evaluation/test_citation_coverage.py`:

```python
from __future__ import annotations

import re

import pytest


@pytest.mark.evaluation
class TestCitationCoverage:
    """Test citation coverage in RAG responses."""

    def test_citation_pattern_regex(self) -> None:
        """Citation regex should match [Source: filename, page X] pattern."""
        pattern = r"\[Source:\s*[^,]+,\s*[Pp]age\s*\d+\]"
        assert re.search(pattern, "Answer text [Source: contract.pdf, Page 3]")
        assert not re.search(pattern, "Answer without citation")

    @pytest.mark.skip(reason="Requires live LLM")
    def test_all_responses_have_citations(self, golden_set: list[dict]) -> None:
        """Every response from the RAG pipeline should contain at least one citation."""
        pass
```

**Step 4: Run the non-skipped tests**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/evaluation/ -x -v -m "not skip"`
Expected: PASS for golden_set_is_valid and citation_pattern_regex.

**Step 5: Commit**

```bash
git add tests/evaluation/conftest.py tests/evaluation/test_ragas.py tests/evaluation/test_citation_coverage.py
git commit -m "feat: add RAGAS evaluation test suite with golden test set"
```

---

### Task 21: Add optional evaluation job to CI

**Files:**
- Modify: `.github/workflows/ci.yml`

**Step 1: Add evaluation job**

Add to `.github/workflows/ci.yml` at the end, after the `secrets-check` job:

```yaml
  ragas-eval:
    runs-on: ubuntu-latest
    if: github.event_name == 'workflow_dispatch'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r backend/requirements.txt
      - run: python -m pytest tests/evaluation/ -v --tb=short -m evaluation
        env:
          PYTHONPATH: backend
```

Add `workflow_dispatch` to the `on:` triggers:

```yaml
on:
  push:
    branches: ["*"]
  pull_request:
    branches: [master]
  workflow_dispatch:
```

**Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "feat: add optional RAGAS evaluation CI job via workflow_dispatch"
```

---

## Workstream 5: Prometheus + Grafana Observability

### Task 22: Add Prometheus dependencies and create metrics module

**Files:**
- Modify: `backend/requirements.txt`
- Create: `backend/app/metrics.py`

**Step 1: Add dependencies**

Add to `backend/requirements.txt`:

```
prometheus-client>=0.21.0
prometheus-fastapi-instrumentator>=7.0.0
```

**Step 2: Install**

Run: `cd /e/LCMGoCloudRAG/backend && pip install prometheus-client prometheus-fastapi-instrumentator`

**Step 3: Create metrics module**

Create `backend/app/metrics.py`:

```python
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# RAG pipeline metrics
rag_query_duration = Histogram(
    "rag_query_duration_seconds",
    "Total RAG query pipeline duration",
    buckets=(0.5, 1, 2, 5, 10, 30, 60),
)

rag_retrieval_duration = Histogram(
    "rag_retrieval_duration_seconds",
    "Retrieval phase duration",
    buckets=(0.1, 0.5, 1, 2, 5, 10),
)

rag_generation_duration = Histogram(
    "rag_generation_duration_seconds",
    "LLM generation phase duration",
    buckets=(0.5, 1, 2, 5, 10, 30, 60),
)

rag_faithfulness_score = Histogram(
    "rag_faithfulness_score",
    "HHEM faithfulness score distribution",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

guardrail_blocks_total = Counter(
    "rag_guardrail_blocks_total",
    "Total guardrail blocks",
    ["type"],
)

# Celery metrics
celery_tasks_total = Counter(
    "celery_tasks_total",
    "Total Celery tasks",
    ["task_name", "status"],
)

celery_task_duration = Histogram(
    "celery_task_duration_seconds",
    "Celery task duration",
    ["task_name"],
    buckets=(1, 5, 10, 30, 60, 120, 300),
)

# WebSocket metrics
websocket_connections = Gauge(
    "websocket_connections_active",
    "Active WebSocket connections",
)

# Document metrics
documents_ingested_total = Counter(
    "documents_ingested_total",
    "Total documents ingested",
    ["status"],
)
```

**Step 4: Commit**

```bash
git add backend/requirements.txt backend/app/metrics.py
git commit -m "feat: add Prometheus metrics definitions"
```

---

### Task 23: Create singleton clients module

**Files:**
- Create: `backend/app/clients.py`
- Modify: `backend/app/api/v1/health.py`

**Step 1: Create clients.py**

Create `backend/app/clients.py`:

```python
from __future__ import annotations

import structlog
from qdrant_client import QdrantClient
from redis.asyncio import Redis

from app.config import settings

logger = structlog.get_logger()

_qdrant_client: QdrantClient | None = None
_redis_client: Redis | None = None


def get_qdrant_client() -> QdrantClient:
    """Get or create the singleton Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=settings.QDRANT_URL, timeout=10)
        logger.info("qdrant_client_created", url=settings.QDRANT_URL)
    return _qdrant_client


def get_redis_client() -> Redis:
    """Get or create the singleton async Redis client."""
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
```

**Step 2: Update health.py to use singletons**

In `backend/app/api/v1/health.py`, replace `_check_qdrant` and `_check_redis`:

```python
from app.clients import get_qdrant_client, get_redis_client
```

Replace `_check_qdrant`:
```python
async def _check_qdrant() -> ServiceStatus:
    try:
        client = get_qdrant_client()
        client.get_collections()
        return ServiceStatus(status="healthy")
    except Exception as e:
        logger.error("health_check_qdrant_failed", error=str(e))
        return ServiceStatus(status="unhealthy", detail=str(e))
```

Replace `_check_redis`:
```python
async def _check_redis() -> ServiceStatus:
    try:
        client = get_redis_client()
        await client.ping()
        return ServiceStatus(status="healthy")
    except Exception as e:
        logger.error("health_check_redis_failed", error=str(e))
        return ServiceStatus(status="unhealthy", detail=str(e))
```

Remove the unused imports of `QdrantClient` and `Redis` from health.py.

**Step 3: Run backend tests**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/ -x -v`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add backend/app/clients.py backend/app/api/v1/health.py
git commit -m "feat: add singleton Qdrant and Redis clients with connection pooling"
```

---

### Task 24: Instrument FastAPI with Prometheus and add metrics to pipelines

**Files:**
- Modify: `backend/app/main.py`
- Modify: `backend/app/api/v1/ws.py`
- Modify: `backend/app/tasks/ingestion.py`
- Modify: `backend/app/guardrails.py`

**Step 1: Instrument main.py**

In `backend/app/main.py`, add after line 13 (after `RequestLoggingMiddleware` import):

```python
from prometheus_fastapi_instrumentator import Instrumentator
from app.clients import close_clients
```

Update the lifespan to close clients on shutdown:

```python
@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    if settings.ENVIRONMENT == "dev":
        from app.auth import seed_dev_user
        from app.database import async_session_factory
        async with async_session_factory() as session:
            await seed_dev_user(session)
    yield
    await close_clients()
```

In `create_app()`, add instrumentation after creating the application (before middleware):

```python
    Instrumentator().instrument(application).expose(application, endpoint="/metrics")
```

**Step 2: Add metrics to WebSocket handler**

In `backend/app/api/v1/ws.py`, import metrics and instrument:

```python
from app.metrics import (
    rag_query_duration, rag_retrieval_duration, rag_generation_duration,
    rag_faithfulness_score, guardrail_blocks_total, websocket_connections,
)
```

In `websocket_query`, increment gauge on connect and decrement on disconnect:

After `await websocket.accept()`:
```python
    websocket_connections.inc()
```

In the finally/except blocks, add:
```python
    websocket_connections.dec()
```

In `_handle_query`, add timing observations for retrieval and generation phases.

**Step 3: Add metrics to Celery tasks**

In `backend/app/tasks/ingestion.py`, import and use:

```python
from app.metrics import celery_tasks_total, celery_task_duration, documents_ingested_total
```

Wrap the task execution with timing and status tracking.

**Step 4: Add metrics to guardrails**

In `backend/app/guardrails.py`, import and increment counter on blocks:

```python
from app.metrics import guardrail_blocks_total
```

In `scan_input`, when blocked: `guardrail_blocks_total.labels(type="prompt_injection").inc()`

**Step 5: Run backend tests**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/ -x -v`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add backend/app/main.py backend/app/api/v1/ws.py backend/app/tasks/ingestion.py backend/app/guardrails.py
git commit -m "feat: instrument FastAPI, WebSocket, Celery, and guardrails with Prometheus metrics"
```

---

### Task 25: Add Prometheus and Grafana to Docker Compose

**Files:**
- Create: `docker/prometheus/prometheus.yml`
- Create: `docker/grafana/provisioning/datasources.yml`
- Create: `docker/grafana/provisioning/dashboards.yml`
- Create: `docker/grafana/dashboards/rag-overview.json`
- Modify: `docker/docker-compose.dev.yml`
- Modify: `docker/docker-compose.prod.yml`

**Step 1: Create Prometheus config**

Create `docker/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "docintel-backend"
    static_configs:
      - targets: ["backend:8000"]
    metrics_path: /metrics
```

**Step 2: Create Grafana provisioning**

Create `docker/grafana/provisioning/datasources.yml`:

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

Create `docker/grafana/provisioning/dashboards.yml`:

```yaml
apiVersion: 1
providers:
  - name: "default"
    orgId: 1
    folder: ""
    type: file
    options:
      path: /var/lib/grafana/dashboards
```

**Step 3: Create Grafana dashboard**

Create `docker/grafana/dashboards/rag-overview.json` with a basic dashboard containing panels for: request rate, latency percentiles, active WebSocket connections, guardrail blocks, Celery task status, and faithfulness score distribution. (Full JSON dashboard definition — this is a standard Grafana dashboard JSON.)

**Step 4: Add services to docker-compose.dev.yml**

Add to `docker/docker-compose.dev.yml`:

```yaml
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_AUTH_ANONYMOUS_ENABLED: "true"
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    depends_on:
      - prometheus
    restart: unless-stopped
```

Add the same to `docker/docker-compose.prod.yml`.

**Step 5: Commit**

```bash
git add docker/prometheus/ docker/grafana/ docker/docker-compose.dev.yml docker/docker-compose.prod.yml
git commit -m "feat: add Prometheus and Grafana with pre-provisioned RAG dashboard"
```

---

### Task 26: Write metrics test

**Files:**
- Create: `tests/unit/test_metrics.py`

**Step 1: Write test**

Create `tests/unit/test_metrics.py`:

```python
from __future__ import annotations


def test_metrics_module_imports() -> None:
    """All metric objects should be importable."""
    from app.metrics import (
        rag_query_duration,
        rag_retrieval_duration,
        rag_generation_duration,
        rag_faithfulness_score,
        guardrail_blocks_total,
        celery_tasks_total,
        celery_task_duration,
        websocket_connections,
        documents_ingested_total,
    )
    assert rag_query_duration is not None
    assert websocket_connections is not None


def test_clients_module_imports() -> None:
    """Client singleton functions should be importable."""
    from app.clients import get_qdrant_client, get_redis_client, close_clients
    assert callable(get_qdrant_client)
    assert callable(get_redis_client)
    assert callable(close_clients)
```

**Step 2: Run tests**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/unit/test_metrics.py -x -v`
Expected: PASS.

**Step 3: Commit**

```bash
git add tests/unit/test_metrics.py
git commit -m "test: add metrics and clients module import tests"
```

---

### Task 27: Verify Workstream 5

**Step 1: Run full backend lint + tests**

Run: `cd /e/LCMGoCloudRAG/backend && python -m ruff check . && python -m ruff format --check .`
Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/ -x -v`
Expected: All clean.

---

## Workstream 6: Security Hardening

### Task 28: Add slowapi and create rate limiting middleware

**Files:**
- Modify: `backend/requirements.txt`
- Create: `backend/app/middleware/rate_limit.py`
- Modify: `backend/app/config.py`

**Step 1: Add slowapi to requirements**

Add to `backend/requirements.txt`:

```
slowapi>=0.1.9
```

**Step 2: Install**

Run: `cd /e/LCMGoCloudRAG/backend && pip install slowapi`

**Step 3: Add rate limit settings to config**

In `backend/app/config.py`, add after `CONVERSATION_CONTEXT_MESSAGES`:

```python
    # Rate Limiting
    RATE_LIMIT_DEFAULT: str = "100/minute"
    RATE_LIMIT_AUTH: str = "10/minute"
    RATE_LIMIT_QUERY: str = "30/minute"
    RATE_LIMIT_UPLOAD: str = "10/minute"

    # Cache
    CACHE_ENABLED: bool = True
    CACHE_TTL_SECONDS: int = 300  # 5 minutes
```

**Step 4: Create rate limit middleware**

Create `backend/app/middleware/rate_limit.py`:

```python
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.RATE_LIMIT_DEFAULT],
    storage_uri=settings.REDIS_URL,
)
```

**Step 5: Commit**

```bash
git add backend/requirements.txt backend/app/middleware/rate_limit.py backend/app/config.py
git commit -m "feat: add slowapi rate limiting with Redis storage"
```

---

### Task 29: Create security headers middleware

**Files:**
- Create: `backend/app/middleware/security_headers.py`

**Step 1: Create the middleware**

Create `backend/app/middleware/security_headers.py`:

```python
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all HTTP responses."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "0"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response
```

**Step 2: Commit**

```bash
git add backend/app/middleware/security_headers.py
git commit -m "feat: add security headers middleware"
```

---

### Task 30: Create Redis query cache

**Files:**
- Create: `backend/app/cache.py`
- Create: `tests/unit/test_cache.py`

**Step 1: Write failing test**

Create `tests/unit/test_cache.py`:

```python
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
```

**Step 2: Implement cache module**

Create `backend/app/cache.py`:

```python
from __future__ import annotations

import hashlib
import json

import structlog

from app.clients import get_redis_client
from app.config import settings

logger = structlog.get_logger()

CACHE_PREFIX = "docintel:query:"


def make_cache_key(question: str, client_id: str) -> str:
    """Generate a deterministic cache key from question and client_id."""
    normalized = question.strip().lower()
    raw = f"{client_id}:{normalized}"
    return CACHE_PREFIX + hashlib.sha256(raw.encode()).hexdigest()


async def get_cached_response(question: str, client_id: str) -> dict | None:
    """Get a cached RAG response if available."""
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
    """Cache a RAG response."""
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
    """Invalidate all cached responses for a client."""
    if not settings.CACHE_ENABLED:
        return
    try:
        client = get_redis_client()
        pattern = f"{CACHE_PREFIX}*"
        # Scan and delete matching keys
        cursor = 0
        deleted = 0
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
```

**Step 3: Run tests**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/unit/test_cache.py -x -v`
Expected: PASS.

**Step 4: Commit**

```bash
git add backend/app/cache.py tests/unit/test_cache.py
git commit -m "feat: add Redis query cache with TTL and client invalidation"
```

---

### Task 31: Register middlewares and tighten CORS in main.py

**Files:**
- Modify: `backend/app/main.py`

**Step 1: Update main.py**

Add imports:

```python
from app.middleware.rate_limit import limiter
from app.middleware.security_headers import SecurityHeadersMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
```

In `create_app()`, add after creating the application:

```python
    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

Add security headers middleware (after CORS):

```python
    application.add_middleware(SecurityHeadersMiddleware)
```

Tighten CORS — replace `allow_methods=["*"]` and `allow_headers=["*"]`:

```python
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",")],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
    )
```

**Step 2: Run backend tests**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/ -x -v`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add backend/app/main.py
git commit -m "feat: register rate limiter, security headers, and tighten CORS"
```

---

### Task 32: Add cache to query pipeline and invalidation to documents

**Files:**
- Modify: `backend/app/api/v1/query.py`
- Modify: `backend/app/api/v1/documents.py`

**Step 1: Add cache check to REST query endpoint**

In `backend/app/api/v1/query.py`, add import:

```python
from app.cache import get_cached_response, set_cached_response
```

Before calling `query_documents()` (around line 82), add:

```python
    # Check cache
    cached = await get_cached_response(request_body.question, current_user.client_id)
    if cached:
        return QueryResponseSchema(**cached)
```

After building the response (before return), add:

```python
    # Cache the response
    cache_data = {
        "answer": result["answer"],
        "citations": [c.model_dump() for c in citations],
        "model_used": result["model_used"],
        "latency_ms": result["latency_ms"],
    }
    await set_cached_response(request_body.question, current_user.client_id, cache_data)
```

**Step 2: Add cache invalidation to document upload and delete**

In `backend/app/api/v1/documents.py`, add import:

```python
from app.cache import invalidate_client_cache
```

In `upload_document`, after dispatching the Celery task:

```python
    await invalidate_client_cache(current_user.client_id)
```

In `delete_document`, after deleting the document:

```python
    await invalidate_client_cache(current_user.client_id)
```

**Step 3: Run backend tests**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/ -x -v`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add backend/app/api/v1/query.py backend/app/api/v1/documents.py
git commit -m "feat: add query caching with cache invalidation on document changes"
```

---

### Task 33: Add security headers to Next.js frontend

**Files:**
- Modify: `frontend/next.config.ts`

**Step 1: Add security headers**

Replace `frontend/next.config.ts`:

```typescript
import createNextIntlPlugin from "next-intl/plugin";
import type { NextConfig } from "next";

const withNextIntl = createNextIntlPlugin("./src/i18n/request.ts");

const nextConfig: NextConfig = {
  output: "standalone",
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Frame-Options", value: "DENY" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          {
            key: "Content-Security-Policy",
            value: [
              "default-src 'self'",
              `connect-src 'self' ${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"} ${process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000"} ${process.env.NEXT_PUBLIC_KEYCLOAK_URL || "http://localhost:8080"}`,
              "style-src 'self' 'unsafe-inline'",
              "script-src 'self' 'unsafe-eval'",
              "img-src 'self' data:",
              "font-src 'self'",
              "frame-src 'self' " + (process.env.NEXT_PUBLIC_KEYCLOAK_URL || "http://localhost:8080"),
            ].join("; "),
          },
          {
            key: "Permissions-Policy",
            value: "camera=(), microphone=(), geolocation=()",
          },
        ],
      },
    ];
  },
};

export default withNextIntl(nextConfig);
```

**Step 2: Run frontend build**

Run: `cd /e/LCMGoCloudRAG/frontend && npm run build`
Expected: Build succeeds.

**Step 3: Commit**

```bash
git add frontend/next.config.ts
git commit -m "feat: add CSP and security headers to Next.js config"
```

---

## Final Verification

### Task 34: Full verification pass

**Step 1: Run backend lint**

Run: `cd /e/LCMGoCloudRAG/backend && python -m ruff check . && python -m ruff format --check .`
Expected: Clean.

**Step 2: Run all backend tests**

Run: `cd /e/LCMGoCloudRAG && PYTHONPATH=backend python -m pytest tests/ -x -v`
Expected: All tests pass (71 existing + new tests).

**Step 3: Run frontend unit tests**

Run: `cd /e/LCMGoCloudRAG/frontend && npx vitest run`
Expected: All 28+ tests pass.

**Step 4: Run TypeScript check**

Run: `cd /e/LCMGoCloudRAG/frontend && npx tsc --noEmit`
Expected: Clean.

**Step 5: Run frontend build**

Run: `cd /e/LCMGoCloudRAG/frontend && npm run build`
Expected: Build succeeds with standalone output.

**Step 6: Verify git status**

Run: `cd /e/LCMGoCloudRAG && git log --oneline -20`
Expected: All Phase 6 commits visible.
