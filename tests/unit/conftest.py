from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub out the `fastembed` package so that fastembed-haystack components can
# be imported without requiring the full Rust-compiled fastembed wheel.
# This only affects the test process; production code uses the real package.
# ---------------------------------------------------------------------------

_fastembed_stub = ModuleType("fastembed")
_fastembed_sparse_stub = ModuleType("fastembed.sparse")
_fastembed_sparse_text_stub = ModuleType("fastembed.sparse.sparse_text_embedding")

# Minimal class stubs that satisfy fastembed_backend.py imports
_fastembed_stub.TextEmbedding = MagicMock  # type: ignore[attr-defined]
_fastembed_sparse_text_stub.SparseTextEmbedding = MagicMock  # type: ignore[attr-defined]

sys.modules.setdefault("fastembed", _fastembed_stub)
sys.modules.setdefault("fastembed.sparse", _fastembed_sparse_stub)
sys.modules.setdefault("fastembed.sparse.sparse_text_embedding", _fastembed_sparse_text_stub)
