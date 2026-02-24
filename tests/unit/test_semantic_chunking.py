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
