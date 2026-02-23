from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


@patch("app.pipelines.expansion.OpenAIChatGenerator")
def test_expand_query_returns_variants(mock_gen_cls: MagicMock) -> None:
    """expand_query should return a list of query variants including the original."""
    mock_gen = MagicMock()
    mock_gen.run.return_value = {
        "replies": [
            MagicMock(text=json.dumps(["What is the contract duration?", "Ποια είναι η διάρκεια της σύμβασης;"]))
        ]
    }
    mock_gen_cls.return_value = mock_gen

    from app.pipelines.expansion import expand_query

    variants = expand_query("What is the contract term?")

    assert len(variants) >= 2
    assert "What is the contract term?" in variants


@patch("app.pipelines.expansion.settings")
def test_expand_query_disabled_returns_original(mock_settings: MagicMock) -> None:
    """When expansion is disabled, should return only the original query."""
    mock_settings.QUERY_EXPANSION_ENABLED = False

    from app.pipelines.expansion import expand_query

    variants = expand_query("What is the contract term?")

    assert variants == ["What is the contract term?"]


@patch("app.pipelines.expansion.OpenAIChatGenerator")
def test_expand_query_handles_malformed_llm_response(mock_gen_cls: MagicMock) -> None:
    """If LLM returns non-JSON, should fall back to original query only."""
    mock_gen = MagicMock()
    mock_gen.run.return_value = {"replies": [MagicMock(text="not valid json")]}
    mock_gen_cls.return_value = mock_gen

    from app.pipelines.expansion import expand_query

    variants = expand_query("test query")

    assert variants == ["test query"]
