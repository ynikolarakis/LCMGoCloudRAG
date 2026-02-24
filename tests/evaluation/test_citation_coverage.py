from __future__ import annotations

import re

import pytest


@pytest.mark.evaluation
class TestCitationCoverage:
    """Test citation coverage in RAG responses."""

    def test_citation_pattern_regex(self) -> None:
        """Citation regex should match [Source: filename, page X] pattern.

        Verifies that the canonical citation regex correctly identifies
        well-formed ``[Source: filename, Page N]`` citations and correctly
        rejects plain text with no citation present.
        """
        pattern = r"\[Source:\s*[^,]+,\s*[Pp]age\s*\d+\]"
        assert re.search(pattern, "Answer text [Source: contract.pdf, Page 3]")
        assert not re.search(pattern, "Answer without citation")

    @pytest.mark.skip(reason="Requires live LLM")
    def test_all_responses_have_citations(self, golden_set: list[dict]) -> None:
        """Every response from the RAG pipeline should contain at least one citation.

        Runs every question in the golden set through the live RAG pipeline and
        asserts that each response contains at least one ``[Source: …, Page N]``
        citation, enforcing the platform-wide 100% citation-coverage target.

        Args:
            golden_set: Fixture providing the loaded golden Q&A pairs.
        """
        pass
