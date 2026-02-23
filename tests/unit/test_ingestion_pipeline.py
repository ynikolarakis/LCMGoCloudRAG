from __future__ import annotations

from unittest.mock import MagicMock, patch


@patch("app.pipelines.ingestion.QdrantDocumentStore")
def test_document_store_has_sparse_embeddings_enabled(mock_store_cls: MagicMock) -> None:
    """QdrantDocumentStore should be created with use_sparse_embeddings=True."""
    from app.pipelines.ingestion import _get_document_store

    _get_document_store()

    mock_store_cls.assert_called_once()
    call_kwargs = mock_store_cls.call_args[1]
    assert call_kwargs["use_sparse_embeddings"] is True


@patch("app.pipelines.ingestion.QdrantDocumentStore")
def test_ingestion_pipeline_has_sparse_embedder(mock_store_cls: MagicMock) -> None:
    """Ingestion pipeline should contain a sparse_embedder component."""
    from app.pipelines.ingestion import _build_ingestion_pipeline

    mock_store = MagicMock()
    pipeline = _build_ingestion_pipeline(mock_store)

    component_names = list(pipeline.graph.nodes.keys())
    assert "sparse_embedder" in component_names
    assert "embedder" in component_names
    assert "writer" in component_names
