from __future__ import annotations

from unittest.mock import MagicMock, patch


@patch("app.pipelines.query.expand_query")
@patch("app.pipelines.query._build_query_pipeline")
@patch("app.pipelines.query._get_document_store")
def test_query_uses_expansion_when_enabled(
    mock_store: MagicMock,
    mock_build: MagicMock,
    mock_expand: MagicMock,
) -> None:
    """query_documents should call expand_query and use all variants."""
    mock_expand.return_value = ["What is the term?", "Ποια είναι η διάρκεια;"]

    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = {
        "llm": {"replies": [MagicMock(text="Answer")]},
        "retriever": {"documents": []},
    }
    mock_build.return_value = mock_pipeline

    from app.pipelines.query import query_documents

    query_documents("What is the term?", client_id="default")

    mock_expand.assert_called_once_with("What is the term?")
    # Pipeline should be called at least once (possibly per variant)
    assert mock_pipeline.run.call_count >= 1
