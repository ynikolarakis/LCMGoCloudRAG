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

# ---------------------------------------------------------------------------
# Stub out `llm_guard` and `sentence_transformers` so that guardrail modules
# can be imported without the heavy optional wheels installed in the dev env.
# Lazy-loaded singletons in app.guardrails are patched per-test where needed.
# ---------------------------------------------------------------------------

_llm_guard_stub = ModuleType("llm_guard")
_llm_guard_input_stub = ModuleType("llm_guard.input_scanners")

# PromptInjection stub: instances have .scan() that returns (text, is_valid, score)
_prompt_injection_instance = MagicMock()
_prompt_injection_instance.scan.return_value = ("query", True, 0.0)
_PromptInjectionClass = MagicMock(return_value=_prompt_injection_instance)
_llm_guard_input_stub.PromptInjection = _PromptInjectionClass  # type: ignore[attr-defined]

sys.modules.setdefault("llm_guard", _llm_guard_stub)
sys.modules.setdefault("llm_guard.input_scanners", _llm_guard_input_stub)

_sentence_transformers_stub = ModuleType("sentence_transformers")

# CrossEncoder stub: instances have .predict() that returns [1.0] (faithful by default)
_cross_encoder_instance = MagicMock()
_cross_encoder_instance.predict.return_value = [1.0]
_CrossEncoderClass = MagicMock(return_value=_cross_encoder_instance)
_sentence_transformers_stub.CrossEncoder = _CrossEncoderClass  # type: ignore[attr-defined]

sys.modules.setdefault("sentence_transformers", _sentence_transformers_stub)
