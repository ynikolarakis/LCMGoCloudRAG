from __future__ import annotations

from unittest.mock import MagicMock, patch


@patch("app.pipelines.query.QdrantDocumentStore")
def test_query_document_store_has_sparse_embeddings(mock_store_cls: MagicMock) -> None:
    """Query document store should enable sparse embeddings for hybrid search."""
    from app.pipelines.query import _get_document_store

    _get_document_store()

    call_kwargs = mock_store_cls.call_args[1]
    assert call_kwargs["use_sparse_embeddings"] is True


@patch("app.pipelines.query.QdrantHybridRetriever")
@patch("app.pipelines.query.QdrantDocumentStore")
def test_query_pipeline_has_hybrid_retriever(mock_store_cls: MagicMock, mock_retriever_cls: MagicMock) -> None:
    """Query pipeline should use QdrantHybridRetriever, not QdrantEmbeddingRetriever."""
    from app.pipelines.query import _build_query_pipeline

    # Use a real mock pipeline to avoid Haystack's @component validation checks
    mock_pipeline = MagicMock()
    added_components: list[str] = []

    def _capture_add(name: str, _instance: object) -> None:
        added_components.append(name)

    mock_pipeline.add_component.side_effect = _capture_add
    mock_store = MagicMock()

    with patch("app.pipelines.query.Pipeline", return_value=mock_pipeline):
        _build_query_pipeline(mock_store)

    assert "sparse_embedder" in added_components
    assert "dense_embedder" in added_components
    assert "retriever" in added_components


@patch("app.pipelines.query._build_query_pipeline")
@patch("app.pipelines.query._get_document_store")
def test_query_passes_question_to_both_embedders(
    mock_store: MagicMock,
    mock_build: MagicMock,
) -> None:
    """query_documents should pass the question to both sparse and dense embedders."""
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = {
        "llm": {"replies": [MagicMock(text="Answer")]},
        "retriever": {"documents": []},
    }
    mock_build.return_value = mock_pipeline

    from app.pipelines.query import query_documents

    query_documents("What is the term?", client_id="default")

    call_args = mock_pipeline.run.call_args[0][0]
    assert "sparse_embedder" in call_args
    assert "dense_embedder" in call_args
    assert call_args["sparse_embedder"]["text"] == "What is the term?"
    assert call_args["dense_embedder"]["text"] == "What is the term?"
