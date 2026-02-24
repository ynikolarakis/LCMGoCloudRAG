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
