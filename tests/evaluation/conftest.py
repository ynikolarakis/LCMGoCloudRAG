from __future__ import annotations

import json
from pathlib import Path

import pytest

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


@pytest.fixture
def golden_set() -> list[dict]:
    """Load the golden test set.

    Returns:
        A list of dicts, each containing ``question``, ``ground_truth``,
        and ``expected_source`` keys loaded from the JSON golden set file.
    """
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        return json.load(f)
