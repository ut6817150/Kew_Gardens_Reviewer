"""Unit tests for reference asset building.

Purpose:
    This module verifies the helper functions in
    `llm_rag/iii_vector_db/build_reference_db.py`, including JSONL loading,
    chunk construction, metadata preservation, parent-context output, and
    Chroma build orchestration with test doubles.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import types
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

SOURCE_DIR = Path(__file__).resolve().parents[2] / "iii_vector_db"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

fake_embedding_loader = types.ModuleType("embedding_loader")
fake_embedding_loader.build_huggingface_embeddings = lambda *args, **kwargs: None
sys.modules.setdefault("embedding_loader", fake_embedding_loader)

fake_langchain_chroma = types.ModuleType("langchain_chroma")


class FakeChroma:
    @classmethod
    def from_documents(cls, *args, **kwargs):
        return None


fake_langchain_chroma.Chroma = FakeChroma
sys.modules.setdefault("langchain_chroma", fake_langchain_chroma)

fake_langchain_core = types.ModuleType("langchain_core")
fake_langchain_core_documents = types.ModuleType("langchain_core.documents")


class FakeDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


fake_langchain_core_documents.Document = FakeDocument
sys.modules.setdefault("langchain_core", fake_langchain_core)
sys.modules.setdefault("langchain_core.documents", fake_langchain_core_documents)

fake_langchain_splitters = types.ModuleType("langchain_text_splitters")


class FakeRecursiveCharacterTextSplitter:
    def __init__(self, chunk_size, chunk_overlap, separators):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def split_text(self, text):
        if len(text) <= self.chunk_size:
            return [text]
        pieces = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            pieces.append(text[start:end])
            if end >= len(text):
                break
            start = max(end - self.chunk_overlap, start + 1)
        return pieces


fake_langchain_splitters.RecursiveCharacterTextSplitter = FakeRecursiveCharacterTextSplitter
sys.modules.setdefault("langchain_text_splitters", fake_langchain_splitters)

import build_reference_db as brd


def make_record(**overrides):
    record = {
        "doc_id": "doc-1",
        "source_file": "sample.pdf",
        "pdf_stem": "sample",
        "page": 2,
        "block_type": "text",
        "text": "Source: sample.pdf\nContent: Example body text",
        "table_id": None,
        "table_title": None,
        "row_id": None,
        "parent_id": None,
        "section_title": "Overview",
        "section_path": ["Root", "Overview"],
        "section_path_str": "Root > Overview",
        "priority": 1.0,
        "metadata": {},
    }
    record.update(overrides)
    return record


def test_extract_display_text():
    assert brd.extract_display_text("Source: x\nContent: Main text") == "Main text"
    assert brd.extract_display_text("  Plain text  ") == "Plain text"
    assert brd.extract_display_text("") == ""


def test_build_search_text_omits_none_fields():
    metadata = {
        "source_file": "sample.pdf",
        "section_title": "Overview",
        "section_path": "Root > Overview",
        "table_title": None,
        "block_type": "text",
        "page": 2,
    }

    text = brd.build_search_text("Chunk body", metadata)

    assert "Source file: sample.pdf" in text
    assert "Section title: Overview" in text
    assert "Section path: Root > Overview" in text
    assert "Table title:" not in text
    assert text.endswith("Content: Chunk body")


def test_parse_record_line_sets_defaults():
    line = json.dumps(
        {
            "text": "Body text",
            "section_path": ["A", "B"],
        }
    )

    record = brd._parse_record_line(line, "example_pdf", 7)

    assert record is not None
    assert record["section_path_str"] == "A > B"
    assert record["doc_id"] == "example_pdf::line::00007"
    assert record["pdf_stem"] == "example_pdf"
    assert record["priority"] == 1.0


def test_parse_record_line_skips_blank_or_empty_text():
    assert brd._parse_record_line("   ", "example_pdf", 1) is None
    assert brd._parse_record_line(json.dumps({"text": "   "}), "example_pdf", 2) is None


def test_load_retrieval_records_reads_all_document_folders():
    temp_dir = Path.cwd() / "llm_rag" / "iii_vector_db" / f"_test_tmp_{uuid.uuid4().hex}"
    try:
        temp_dir.mkdir(parents=True, exist_ok=False)
        processed_dir = temp_dir
        first_dir = processed_dir / "doc_one"
        second_dir = processed_dir / "doc_two"
        first_dir.mkdir()
        second_dir.mkdir()

        (first_dir / "retrieval_blocks.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"text": "First", "section_path": ["One"]}),
                    json.dumps({"text": "Second", "section_path": ["Two"]}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (second_dir / "retrieval_blocks.jsonl").write_text(
            json.dumps({"text": "Third", "section_path": ["Three"]}) + "\n",
            encoding="utf-8",
        )

        records = brd.load_retrieval_records(processed_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    assert len(records) == 3
    assert records[0]["pdf_stem"] == "doc_one"
    assert records[-1]["pdf_stem"] == "doc_two"


def test_build_record_metadata_preserves_expected_fields():
    record = make_record(
        block_type="table_row",
        table_id="table_001",
        table_title="Example Table",
        row_id=3,
        parent_id="table_001",
        priority=1.3,
        metadata={"fallback": True, "table_name": "criteria_table"},
    )

    metadata = brd.build_record_metadata(record)

    assert metadata["block_type"] == "table_row"
    assert metadata["table_id"] == "table_001"
    assert metadata["row_id"] == 3
    assert metadata["priority"] == 1.3
    assert metadata["fallback"] is True
    assert metadata["table_name"] == "criteria_table"


def test_split_record_keeps_table_rows_atomic():
    record = make_record(
        block_type="table_row",
        text="Source: sample.pdf\nContent: Atomic row text",
        table_id="table_001",
        row_id=1,
    )

    documents = brd.split_record(record, chunk_size=10, chunk_overlap=2)

    assert len(documents) == 1
    assert documents[0].metadata["chunk_id"] == "doc-1::0"
    assert documents[0].metadata["display_text"] == "Atomic row text"
    assert "Content: Atomic row text" in documents[0].page_content


def test_split_record_uses_splitter_for_text_blocks():
    record = make_record(text="Source: sample.pdf\nContent: Chunk me")
    fake_splitter = MagicMock()
    fake_splitter.split_text.return_value = ["Part one", "Part two"]

    with patch.object(brd, "_splitter_for_block", return_value=fake_splitter) as mock_splitter:
        documents = brd.split_record(record, chunk_size=10, chunk_overlap=2)

    assert mock_splitter.called
    assert [doc.metadata["chunk_id"] for doc in documents] == ["doc-1::0", "doc-1::1"]
    assert [doc.metadata["display_text"] for doc in documents] == ["Part one", "Part two"]


def test_build_documents_flattens_all_split_results():
    records = [make_record(doc_id="doc-1"), make_record(doc_id="doc-2")]

    with patch.object(brd, "split_record", side_effect=[["a"], ["b", "c"]]) as mock_split:
        documents = brd.build_documents(records, chunk_size=100, chunk_overlap=10)

    assert documents == ["a", "b", "c"]
    assert mock_split.call_count == 2


def test_write_reference_corpus_serializes_documents():
    document = brd.Document(
        page_content="search text",
        metadata={"chunk_id": "doc-1::0", "display_text": "display text", "source_file": "sample.pdf"},
    )
    mocked_file = mock_open()

    with patch.object(brd, "CORPUS_PATH", Path("virtual_corpus.jsonl")):
        with patch.object(Path, "open", mocked_file):
            brd.write_reference_corpus([document])

    written = "".join(call.args[0] for call in mocked_file().write.call_args_list)
    payload = json.loads(written.strip())
    assert payload["text"] == "display text"
    assert payload["search_text"] == "search text"
    assert payload["chunk_id"] == "doc-1::0"


def test_write_parent_contexts_keeps_unique_tables_only():
    records = [
        make_record(
            block_type="table",
            text="Source: sample.pdf\nContent: Parent table text",
            source_file="sample.pdf",
            table_id="table_001",
            table_title="Table 1",
            metadata={"fallback": True},
        ),
        make_record(
            block_type="table",
            text="Source: sample.pdf\nContent: Duplicate parent table text",
            source_file="sample.pdf",
            table_id="table_001",
            table_title="Table 1",
        ),
        make_record(
            block_type="table_row",
            text="Source: sample.pdf\nContent: Row text",
            source_file="sample.pdf",
            table_id="table_001",
        ),
    ]
    mocked_file = mock_open()

    with patch.object(brd, "PARENT_CONTEXTS_PATH", Path("virtual_parent_contexts.jsonl")):
        with patch.object(Path, "open", mocked_file):
            brd.write_parent_contexts(records)

    written = "".join(call.args[0] for call in mocked_file().write.call_args_list).strip().splitlines()
    assert len(written) == 1
    payload = json.loads(written[0])
    assert payload["table_id"] == "table_001"
    assert payload["text"] == "Parent table text"


def test_build_embeddings_requires_non_none_result():
    with patch.object(brd, "build_huggingface_embeddings", return_value="embeddings") as mock_builder:
        embeddings = brd.build_embeddings()

    assert embeddings == "embeddings"
    mock_builder.assert_called_once_with(brd.EMBED_MODEL, device="cpu", normalize_embeddings=True, strict=True)

    with patch.object(brd, "build_huggingface_embeddings", return_value=None):
        with pytest.raises(RuntimeError) as exc_info:
            brd.build_embeddings()

    assert brd.EMBED_MODEL in str(exc_info.value)


def test_build_chroma_resets_directory_and_persists_documents():
    documents = [brd.Document(page_content="text", metadata={"chunk_id": "doc-1::0"})]
    fake_persist_dir = MagicMock()
    fake_persist_dir.exists.return_value = True
    fake_embeddings = object()

    with patch.object(brd, "PERSIST_DIR", fake_persist_dir):
        with patch.object(brd, "build_embeddings", return_value=fake_embeddings):
            with patch.object(brd.shutil, "rmtree") as mock_rmtree:
                with patch.object(brd.Chroma, "from_documents") as mock_from_documents:
                    brd.build_chroma(documents, reset=True)

    mock_rmtree.assert_called_once_with(fake_persist_dir)
    fake_persist_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_from_documents.assert_called_once()
    _, kwargs = mock_from_documents.call_args
    assert kwargs["documents"] == documents
    assert kwargs["embedding"] is fake_embeddings
    assert kwargs["ids"] == ["doc-1::0"]


def test_write_summary_counts_records_correctly():
    records = [
        make_record(block_type="table_row", metadata={"fallback": True}),
        make_record(block_type="table_row", metadata={}),
        make_record(block_type="table", metadata={"fallback": True}),
    ]
    documents = [
        brd.Document(page_content="x", metadata={"chunk_id": "a"}),
        brd.Document(page_content="y", metadata={"chunk_id": "b"}),
    ]
    mocked_file = mock_open()

    with patch.object(brd, "SUMMARY_PATH", Path("virtual_summary.json")):
        with patch.object(Path, "open", mocked_file):
            brd.write_summary(records, documents, chunk_size=800, chunk_overlap=100, chroma_built=True)

    written = "".join(call.args[0] for call in mocked_file().write.call_args_list)
    payload = json.loads(written)
    assert payload["raw_record_count"] == 3
    assert payload["chunk_count"] == 2
    assert payload["table_row_count"] == 2
    assert payload["fallback_table_row_count"] == 1
    assert payload["synthetic_parent_count"] == 1
    assert payload["chroma_built"] is True


def test_main_runs_build_pipeline_with_skip_chroma():
    fake_args = argparse_namespace = MagicMock()
    argparse_namespace.chunk_size = 512
    argparse_namespace.chunk_overlap = 64
    argparse_namespace.skip_chroma = True
    argparse_namespace.reset = False
    records = [make_record()]
    documents = [brd.Document(page_content="search", metadata={"chunk_id": "doc-1::0"})]
    fake_parent = MagicMock()
    fake_corpus_path = MagicMock()
    fake_corpus_path.parent = fake_parent

    with patch.object(brd, "parse_args", return_value=fake_args):
        with patch.object(brd, "load_retrieval_records", return_value=records) as mock_load:
            with patch.object(brd, "build_documents", return_value=documents) as mock_build_docs:
                with patch.object(brd, "write_reference_corpus") as mock_write_corpus:
                    with patch.object(brd, "write_parent_contexts") as mock_write_parents:
                        with patch.object(brd, "build_chroma") as mock_build_chroma:
                            with patch.object(brd, "write_summary") as mock_write_summary:
                                with patch.object(brd, "CORPUS_PATH", fake_corpus_path):
                                    with redirect_stdout(io.StringIO()):
                                        brd.main()

    mock_load.assert_called_once_with(brd.PROCESSED_DIR)
    mock_build_docs.assert_called_once_with(records, 512, 64)
    fake_parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_write_corpus.assert_called_once_with(documents)
    mock_write_parents.assert_called_once_with(records)
    mock_build_chroma.assert_not_called()
    mock_write_summary.assert_called_once_with(records, documents, 512, 64, chroma_built=False)


def run_all_tests():
    tests = [
        obj
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
    print(f"Passed {len(tests)} build_reference_db unit tests.")


if __name__ == "__main__":
    run_all_tests()
