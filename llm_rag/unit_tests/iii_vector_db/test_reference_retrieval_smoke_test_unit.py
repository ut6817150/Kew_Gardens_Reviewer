"""Unit tests for the reference retrieval smoke-test script.

Purpose:
    This module verifies argument parsing and console-output behavior for
    `llm_rag/iii_vector_db/reference_retrieval_smoke_test.py` using mocked
    retrieval responses.
"""

from __future__ import annotations

import importlib
import io
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SOURCE_DIR = Path(__file__).resolve().parents[2] / "iii_vector_db"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

fake_retrieval_engine = types.ModuleType("retrieval_engine")
fake_retrieval_engine.answer_query = lambda *args, **kwargs: {}
sys.modules["retrieval_engine"] = fake_retrieval_engine
sys.modules.pop("reference_retrieval_smoke_test", None)
smoke = importlib.import_module("reference_retrieval_smoke_test")


def test_parse_args_uses_defaults():
    with patch.object(sys, "argv", ["reference_retrieval_smoke_test.py"]):
        args = smoke.parse_args()

    assert args.query == "What supporting information is required for a threatened species assessment?"
    assert args.dense_k == 24
    assert args.sparse_k == 24
    assert args.k == 8


def test_parse_args_accepts_custom_values():
    with patch.object(
        sys,
        "argv",
        [
            "reference_retrieval_smoke_test.py",
            "What are the Criterion B thresholds for EOO and AOO?",
            "--dense-k",
            "12",
            "--sparse-k",
            "7",
            "--k",
            "4",
        ],
    ):
        args = smoke.parse_args()

    assert args.query == "What are the Criterion B thresholds for EOO and AOO?"
    assert args.dense_k == 12
    assert args.sparse_k == 7
    assert args.k == 4


def test_main_prints_deterministic_threshold_answer_and_returns_early():
    payload = {
        "route": "deterministic_threshold_lookup",
        "query": "What is the AOO threshold?",
        "threshold_answer": "Criterion B2 AOO thresholds: CR < 10 km2.",
    }

    with patch.object(smoke, "parse_args", return_value=SimpleNamespace(query=payload["query"], dense_k=24, sparse_k=24, k=8)):
        with patch.object(smoke, "answer_query", return_value=payload) as mock_answer_query:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                smoke.main()

    output = buffer.getvalue()
    mock_answer_query.assert_called_once_with(payload["query"], dense_k=24, sparse_k=24, final_k=8)
    assert "Route" in output
    assert "deterministic_threshold_lookup" in output
    assert "Answer" in output
    assert "Criterion B2 AOO thresholds" in output
    assert "Results" not in output


def test_main_prints_hybrid_results_and_parent_context():
    candidate = SimpleNamespace(
        metadata={
            "source_file": "support.pdf",
            "page": 4,
            "block_type": "table_row",
            "table_id": "table_002",
            "row_id": 3,
            "section_title": "Required supporting information",
            "table_title": "Table 2 required supporting information",
            "fallback": False,
            "chunk_id": "chunk-1",
        },
        forced_for_coverage=True,
        dense_score=0.71,
        bm25_score=5.25,
        rerank_score=0.88,
        text="Row-level supporting-information evidence",
        parent_text="Parent table context for the hit",
    )
    payload = {
        "route": "hybrid_rag",
        "query": "What extra supporting information is needed beyond the basics?",
        "subqueries": [
            "required supporting information under all conditions for IUCN Red List assessments",
            "additional required supporting information under specific conditions for threatened taxa beyond the basics",
        ],
        "answer_scaffold": "Answer scaffold\n---------------\nRequired for all assessments:",
        "results": [candidate],
        "threshold_fallback_miss": True,
    }

    with patch.object(smoke, "parse_args", return_value=SimpleNamespace(query=payload["query"], dense_k=10, sparse_k=11, k=2)):
        with patch.object(smoke, "answer_query", return_value=payload):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                smoke.main()

    output = buffer.getvalue()
    assert "hybrid_rag" in output
    assert "Threshold lookup did not match exactly" in output
    assert "Internal retrieval queries" in output
    assert payload["subqueries"][0] in output
    assert "Results" in output
    assert "support.pdf" in output
    assert "Parent context" in output
    assert "Parent table context for the hit" in output
