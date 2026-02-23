from __future__ import annotations

from unittest.mock import MagicMock, patch


@patch("app.pipelines.query.TransformersSimilarityRanker")
@patch("app.pipelines.query.QdrantHybridRetriever")
@patch("app.pipelines.query.QdrantDocumentStore")
def test_query_pipeline_has_ranker_component(
    mock_store_cls: MagicMock,
    mock_retriever_cls: MagicMock,
    mock_ranker_cls: MagicMock,
) -> None:
    """Query pipeline should include a ranker component after retriever."""
    from app.pipelines.query import _build_query_pipeline

    # Build a mock pipeline that captures add_component calls in order,
    # and exposes them via graph.nodes.keys() to satisfy the assertion.
    added_components: list[str] = []

    mock_pipeline = MagicMock()

    def _capture_add(name: str, _instance: object) -> None:
        added_components.append(name)

    mock_pipeline.add_component.side_effect = _capture_add
    mock_pipeline.graph.nodes.keys.side_effect = lambda: iter(added_components)

    mock_store = MagicMock()

    with patch("app.pipelines.query.Pipeline", return_value=mock_pipeline):
        _build_query_pipeline(mock_store)

    component_names = list(mock_pipeline.graph.nodes.keys())
    assert "ranker" in component_names
    # Ranker should be between retriever and prompt_builder
    assert component_names.index("retriever") < component_names.index("ranker")
    assert component_names.index("ranker") < component_names.index("prompt_builder")
