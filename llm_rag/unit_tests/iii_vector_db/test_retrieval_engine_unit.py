"""Unit tests for the reference retrieval engine.

Purpose:
    This module verifies query classification, sparse and dense hit merging,
    table coverage, parent-context attachment, answer scaffolds, and routing
    behavior in `llm_rag/iii_vector_db/retrieval_engine.py`.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

SOURCE_DIR = Path(__file__).resolve().parents[2] / "iii_vector_db"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

fake_langchain_chroma = types.ModuleType("langchain_chroma")


class FakeChroma:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


fake_langchain_chroma.Chroma = FakeChroma
sys.modules.setdefault("langchain_chroma", fake_langchain_chroma)

fake_embedding_loader = types.ModuleType("embedding_loader")
fake_embedding_loader.build_huggingface_embeddings = lambda *args, **kwargs: None
sys.modules.setdefault("embedding_loader", fake_embedding_loader)

fake_threshold_lookup = types.ModuleType("threshold_lookup")
fake_threshold_lookup.answer_threshold_query = lambda query: None
fake_threshold_lookup.is_threshold_query = lambda query: False
sys.modules.setdefault("threshold_lookup", fake_threshold_lookup)

sys.modules.pop("retrieval_engine", None)
re_engine = importlib.import_module("retrieval_engine")


def make_candidate(
    chunk_id: str,
    *,
    text: str = "Content: example",
    search_text: str = "example search text",
    rerank_score: float = 0.0,
    dense_score: float = 0.0,
    bm25_score: float = 0.0,
    source_file: str = "source.pdf",
    page: int = 1,
    block_type: str = "text",
    table_id: str | None = None,
    table_title: str | None = None,
    row_id: int | None = None,
    section_title: str = "Section",
    priority: float = 1.0,
):
    metadata = {
        "source_file": source_file,
        "page": page,
        "block_type": block_type,
        "table_id": table_id,
        "table_title": table_title,
        "row_id": row_id,
        "section_title": section_title,
        "priority": priority,
    }
    return re_engine.Candidate(
        chunk_id=chunk_id,
        text=text,
        search_text=search_text,
        metadata=metadata,
        rerank_score=rerank_score,
        dense_score=dense_score,
        bm25_score=bm25_score,
    )


class FakeDenseDoc:
    def __init__(self, chunk_id: str, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = {"chunk_id": chunk_id, **metadata}


def test_tokenize_and_has_phrase():
    tokens = re_engine.tokenize("Criterion B2, AOO < 10 km2.")
    assert "criterion" in tokens
    assert "b2," in tokens
    assert "aoo" in tokens

    assert re_engine.has_phrase("required supporting information for all assessments", "supporting information") is True
    assert re_engine.has_phrase("threshold-like text", "threshold-like") is True
    assert re_engine.has_phrase("unrelated text", "supporting information") is False


def test_detect_query_mode_and_supporting_info_queries():
    mode = re_engine.detect_query_mode("What supporting information is required for threatened species?")
    assert mode["supporting_info"] is True
    assert mode["threatened"] is True

    subqueries = re_engine.build_supporting_info_queries("What extra supporting information is needed beyond the basics?")
    assert len(subqueries) == 3
    assert "all conditions" in subqueries[0]
    assert "beyond the basics" in subqueries[1]


def test_candidate_from_doc_and_table_label():
    doc = {
        "chunk_id": "chunk-1",
        "text": "Displayed text",
        "search_text": "Search text",
        "table_title": "Table 2 required info",
        "source_file": "sample.pdf",
        "page": 3,
        "block_type": "table_row",
    }

    candidate = re_engine.candidate_from_doc(doc)

    assert candidate.chunk_id == "chunk-1"
    assert candidate.text == "Displayed text"
    assert candidate.metadata["source_file"] == "sample.pdf"
    assert re_engine.table_label(candidate) == "table2"


def test_pick_best_table_candidate_prefers_block_type():
    table_parent = make_candidate("parent", block_type="table", table_title="Table 1 supporting information", rerank_score=0.5)
    table_row = make_candidate("row", block_type="table_row", table_title="Table 1 supporting information", rerank_score=0.9)

    chosen = re_engine.pick_best_table_candidate([table_row, table_parent], "table1", preferred_block_type="table")

    assert chosen.chunk_id == "parent"


def test_pick_corpus_backfill_returns_preferred_table_candidate():
    corpus = [
        {
            "chunk_id": "row-1",
            "text": "row",
            "search_text": "row",
            "table_title": "Table 2 required supporting information",
            "block_type": "table_row",
            "page": 8,
            "row_id": 2,
            "source_file": "support.pdf",
        },
        {
            "chunk_id": "parent-1",
            "text": "parent",
            "search_text": "parent",
            "table_title": "Table 2 required supporting information",
            "block_type": "table",
            "page": 7,
            "row_id": None,
            "source_file": "support.pdf",
        },
    ]

    chosen = re_engine.pick_corpus_backfill(corpus, "table2", preferred_block_type="table_row")

    assert chosen is not None
    assert chosen.chunk_id == "row-1"


def test_upsert_dense_and_sparse_hits_merge_scores():
    by_id = {}
    dense_doc = FakeDenseDoc(
        "chunk-1",
        "dense page content",
        {"display_text": "dense text", "source_file": "source.pdf", "page": 1, "block_type": "text"},
    )
    corpus = [
        {
            "chunk_id": "chunk-1",
            "text": "sparse text",
            "search_text": "sparse search text",
            "source_file": "source.pdf",
            "page": 1,
            "block_type": "text",
        }
    ]

    re_engine._upsert_dense_hits(by_id, [(dense_doc, 0.25)])
    re_engine._upsert_sparse_hits(by_id, [(0, 4.0)], corpus)
    candidate = by_id["chunk-1"]

    assert candidate.text == "dense text"
    assert candidate.dense_rank == 1
    assert candidate.sparse_rank == 1
    assert candidate.dense_score > 0
    assert candidate.bm25_score == 4.0
    assert candidate.fused_score > 0


def test_merge_and_select_forces_supporting_info_coverage():
    existing = make_candidate(
        "other-1",
        rerank_score=0.9,
        source_file="general.pdf",
        table_title="Other table",
        block_type="text",
    )
    corpus = [
        {
            "chunk_id": "table1-parent",
            "text": "Content: table 1 parent",
            "search_text": "table 1 parent",
            "table_title": "Table 1 required supporting information",
            "block_type": "table",
            "page": 1,
            "row_id": None,
            "source_file": "support.pdf",
        },
        {
            "chunk_id": "table2-row",
            "text": "Content: table 2 row",
            "search_text": "table 2 row",
            "table_title": "Table 2 additional information",
            "block_type": "table_row",
            "page": 2,
            "row_id": 1,
            "source_file": "support.pdf",
        },
    ]

    selected = re_engine.merge_and_select(
        [("query", [existing])],
        corpus,
        final_k=3,
        support_info_mode=True,
        per_source_limit=8,
    )

    assert len(selected) == 3
    assert selected[0].forced_for_coverage is True
    assert selected[1].forced_for_coverage is True
    assert {re_engine.table_label(item) for item in selected[:2]} == {"table1", "table2"}


def test_attach_parent_context_and_extract_parent_rows():
    candidate = make_candidate("row-1", table_id="table_001", source_file="support.pdf", block_type="table_row")
    parent_map = {
        ("support.pdf", "table_001"): {
            "text": "Table 1\nSchema: cols\nPage 1 row a\nPage 2 row b\nOther",
            "source_file": "support.pdf",
            "table_id": "table_001",
            "page": 4,
        }
    }

    re_engine.attach_parent_context([candidate], parent_map)
    rows = re_engine.extract_parent_rows(candidate.parent_text, max_rows=2)

    assert candidate.parent_metadata["page"] == 4
    assert rows == ["Page 1 row a", "Page 2 row b"]


def test_build_answer_summary_for_general_and_supporting_info():
    general_candidate = make_candidate(
        "general-1",
        text="Content: threshold explanation",
        source_file="guide.pdf",
        page=9,
        section_title="Criterion B",
    )
    general = re_engine.build_answer_summary([general_candidate], support_info_mode=False, query="Criterion B thresholds")
    assert "Most relevant retrieved reference evidence" in general
    assert "Criterion B" in general

    table1_candidate = make_candidate(
        "table1-parent",
        text="Content: parent text",
        block_type="table",
        table_title="Table 1 required supporting information",
        source_file="support.pdf",
    )
    table1_candidate.parent_text = "Table 1\nSchema: x\nPage 1 basic row\nPage 2 another row"

    table2_candidate = make_candidate(
        "table2-row",
        text="Content: extra row detail",
        block_type="table_row",
        table_title="Table 2 additional information",
        source_file="support.pdf",
    )

    supporting = re_engine.build_answer_summary(
        [table1_candidate, table2_candidate],
        support_info_mode=True,
        query="What extra supporting information is needed beyond the basics?",
    )
    assert "Additional required under specific conditions" in supporting
    assert "Required for all assessments" in supporting
    assert "Page 1 basic row" in supporting
    assert "extra row detail" in supporting


def test_answer_query_uses_deterministic_threshold_route():
    with patch.object(re_engine, "is_threshold_query", return_value=True):
        with patch.object(re_engine, "answer_threshold_query", return_value="Exact threshold answer"):
            payload = re_engine.answer_query("What are the AOO thresholds?")

    assert payload["route"] == "deterministic_threshold_lookup"
    assert payload["threshold_answer"] == "Exact threshold answer"
    assert payload["query"] == "What are the AOO thresholds?"


def test_answer_query_uses_hybrid_route_when_threshold_lookup_misses():
    final_candidate = make_candidate(
        "chunk-1",
        text="Content: retrieved evidence",
        search_text="retrieved evidence",
        table_title="Table 1 required supporting information",
        block_type="table",
        source_file="support.pdf",
    )

    with patch.object(re_engine, "is_threshold_query", return_value=True):
        with patch.object(re_engine, "answer_threshold_query", return_value=None):
            with patch.object(re_engine, "load_corpus", return_value=[{"chunk_id": "chunk-1"}]):
                with patch.object(re_engine, "get_bm25_index", return_value=MagicMock()):
                    with patch.object(re_engine, "load_parent_contexts", return_value={}):
                        with patch.object(
                            re_engine,
                            "hybrid_search_single",
                            side_effect=[[final_candidate], [final_candidate], [final_candidate]],
                        ) as mock_hybrid:
                            with patch.object(re_engine, "merge_and_select", return_value=[final_candidate]) as mock_merge:
                                with patch.object(re_engine, "attach_parent_context") as mock_attach:
                                    payload = re_engine.answer_query(
                                        "What supporting information is required?",
                                        final_k=4,
                                    )

    assert payload["route"] == "hybrid_rag"
    assert payload["threshold_fallback_miss"] is True
    assert len(payload["subqueries"]) == 3
    assert payload["results"] == [final_candidate]
    assert "Answer scaffold" in payload["answer_scaffold"]
    assert mock_hybrid.call_count == 3
    mock_merge.assert_called_once()
    mock_attach.assert_called_once_with([final_candidate], {})


def run_all_tests():
    tests = [
        obj
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
    print(f"Passed {len(tests)} retrieval_engine unit tests.")


if __name__ == "__main__":
    run_all_tests()
