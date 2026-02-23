from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.pipelines.query import query_documents


@patch("app.pipelines.query._build_query_pipeline")
@patch("app.pipelines.query._get_document_store")
def test_query_passes_client_id_filter_to_pipeline(
    mock_store: MagicMock,
    mock_build: MagicMock,
) -> None:
    """query_documents should pass client_id as a filter to the pipeline run."""
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = {
        "llm": {"replies": [MagicMock(text="Test answer")]},
        "retriever": {"documents": []},
    }
    mock_build.return_value = mock_pipeline

    query_documents("What is the contract term?", client_id="acme-corp")

    # Verify the pipeline.run was called with filters containing client_id
    call_kwargs = mock_pipeline.run.call_args
    run_data = call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1]
    # The retriever input should include filters
    retriever_input = run_data.get("retriever", {})
    assert "filters" in retriever_input
    filters = retriever_input["filters"]
    assert filters["conditions"][0]["field"] == "meta.client_id"
    assert filters["conditions"][0]["value"] == "acme-corp"


@patch("app.pipelines.query._build_query_pipeline")
@patch("app.pipelines.query._get_document_store")
def test_query_default_client_id_is_default(
    mock_store: MagicMock,
    mock_build: MagicMock,
) -> None:
    """Default client_id 'default' should still be passed as filter."""
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = {
        "llm": {"replies": [MagicMock(text="Answer")]},
        "retriever": {"documents": []},
    }
    mock_build.return_value = mock_pipeline

    query_documents("test question")

    call_kwargs = mock_pipeline.run.call_args
    run_data = call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1]
    retriever_input = run_data.get("retriever", {})
    assert retriever_input["filters"]["conditions"][0]["value"] == "default"
