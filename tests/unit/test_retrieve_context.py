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
