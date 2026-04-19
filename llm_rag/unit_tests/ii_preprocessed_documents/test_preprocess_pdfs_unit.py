"""Unit tests for PDF preprocessing helpers.

Purpose:
    This module verifies the small parsing, cleanup, table-recovery, and output
    helpers used by `llm_rag/ii_preprocessed_documents/preprocess_pdfs.py`
    without requiring the full raw PDF corpus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

SOURCE_DIR = Path(__file__).resolve().parents[2] / "ii_preprocessed_documents"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import preprocess_pdfs as pp


class FakeFitzDoc:
    def __init__(self, pages):
        self._pages = pages
        self.closed = False

    def __iter__(self):
        return iter(self._pages)

    def close(self):
        self.closed = True


class FakeFitzPage:
    def __init__(self, page_dict):
        self._page_dict = page_dict

    def get_text(self, mode, sort=True):
        assert mode == "dict"
        assert sort is True
        return self._page_dict


class FakePdfPlumberDoc:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakePdfPlumberPage:
    def __init__(self, text="", tables=None):
        self._text = text
        self._tables = tables or []

    def extract_text(self):
        return self._text

    def extract_tables(self):
        return self._tables


def test_make_retrieval_record():
    record = pp.make_retrieval_record(
        Path("sample.pdf"),
        doc_id="doc-1",
        page=2,
        block_type="text",
        raw_text="Hello world",
        section_path=["A", "B"],
        priority=1.2,
        table_id="table_001",
        table_title="Example Table",
        row_id=3,
        parent_id="table_001",
        metadata={"x": 1},
    )

    assert record.source_file == "sample.pdf"
    assert record.section_title == "B"
    assert record.priority == 1.2
    assert "Source: sample.pdf" in record.text
    assert "Table title: Example Table" in record.text


def test_clean_text():
    value = "A\x00  B\u00ad \u2013 \u2014  C\nD"
    assert pp.clean_text(value) == "A B - - C D"


def test_normalize_repeat_text():
    value = pp.normalize_repeat_text("Page 12 Header 99")
    assert value == "page header #"


def test_looks_like_heading():
    assert pp.looks_like_heading("2.1 Title", 10, False, 10) is True
    assert pp.looks_like_heading("IMPORTANT NOTICE", 10, False, 10) is True
    assert pp.looks_like_heading("Short Bold Heading", 10, True, 10) is True
    assert pp.looks_like_heading("This is a long prose block that should not be a heading because it reads like body text.", 12, True, 10) is False


def test_heading_level():
    assert pp.heading_level("1 Title") == 1
    assert pp.heading_level("2.3 Subtitle") == 2
    assert pp.heading_level("Unnumbered") == 1


def test_update_section_path():
    current = ["1 Root", "1.1 Child"]
    updated = pp.update_section_path(current, "1.2 Sibling")
    assert updated == ["1 Root", "1.2 Sibling"]


def test_contextualize_text():
    result = pp.contextualize_text("Body", "x.pdf", 4, ["A", "B"], "table_row", "Table X")
    assert "Source: x.pdf" in result
    assert "Page: 4" in result
    assert "Section path: A > B" in result
    assert result.endswith("Body")


def test_extract_layout_blocks():
    fake_doc = FakeFitzDoc(
        [
            FakeFitzPage(
                {
                    "blocks": [
                        {
                            "type": 0,
                            "bbox": [1, 2, 3, 4],
                            "lines": [
                                {
                                    "spans": [
                                        {"text": "Main", "size": 12, "font": "Times-Bold", "flags": 0},
                                        {"text": "text", "size": 12, "font": "Times-Roman", "flags": 0},
                                    ]
                                }
                            ],
                        }
                    ]
                }
            )
        ]
    )

    with patch.object(pp.fitz, "open", return_value=fake_doc):
        blocks, body_font = pp.extract_layout_blocks(Path("sample.pdf"))

    assert len(blocks) == 1
    assert blocks[0].text == "Main text"
    assert blocks[0].is_bold is True
    assert body_font == 12
    assert fake_doc.closed is True


def test_detect_repeated_header_footer_text():
    blocks = [
        pp.LayoutBlock(1, "Header", [0, 0, 1, 1], 10, False),
        pp.LayoutBlock(1, "Body one", [0, 10, 1, 11], 10, False),
        pp.LayoutBlock(1, "Footer", [0, 20, 1, 21], 10, False),
        pp.LayoutBlock(2, "Header", [0, 0, 1, 1], 10, False),
        pp.LayoutBlock(2, "Body two", [0, 10, 1, 11], 10, False),
        pp.LayoutBlock(2, "Footer", [0, 20, 1, 21], 10, False),
    ]
    repeated = pp.detect_repeated_header_footer_text(blocks)
    assert "header" in repeated
    assert "footer" in repeated


def test_merge_text_blocks():
    raw_blocks = [
        pp.LayoutBlock(1, "1 Heading", [0, 0, 1, 1], 14, True),
        pp.LayoutBlock(1, "Body text first sentence.", [0, 10, 1, 11], 10, False),
        pp.LayoutBlock(1, "Body text second sentence.", [0, 20, 1, 21], 10, False),
    ]

    records, page_section_map = pp.merge_text_blocks(Path("sample.pdf"), raw_blocks, set(), 10)

    assert len(records) == 1
    assert records[0].block_type == "text"
    assert records[0].section_title == "1 Heading"
    assert "Body text first sentence." in records[0].raw_text
    assert page_section_map[1] == ["1 Heading"]


def test_normalize_table_matrix():
    rows = pp.normalize_table_matrix([[None, " A "], ["", ""], ["B", " C "]])
    assert rows == [["", "A"], ["B", "C"]]


def test_make_headers():
    assert pp.make_headers([["", "B"]]) == ["column_1", "B"]
    assert pp.make_headers([]) == []


def test_row_to_pairs():
    assert pp.row_to_pairs(["A", "B"], ["1", "2"]) == "A: 1 | B: 2"


def test_nearest_section_path():
    page_map = {1: ["A"], 3: ["B"]}
    assert pp.nearest_section_path(3, page_map) == ["B"]
    assert pp.nearest_section_path(2, page_map) == ["A"]
    assert pp.nearest_section_path(0, page_map) is None


def test_infer_table_title():
    assert pp.infer_table_title(["Section", "Subsection"], ["A", "B"]) == "Subsection"
    assert pp.infer_table_title(None, ["A", "B"]) == "Table with headers: A | B"
    assert pp.infer_table_title(None, []) == "Untitled table"


def test_get_pdfplumber_page_text():
    fake_doc = FakePdfPlumberDoc([FakePdfPlumberPage(" A \n B "), FakePdfPlumberPage("")])
    with patch.object(pp.pdfplumber, "open", return_value=fake_doc):
        texts = pp.get_pdfplumber_page_text(Path("sample.pdf"))

    assert texts == {1: "A B", 2: ""}


def test_detect_supporting_info_table():
    assert pp.detect_supporting_info_table("Table 1 required supporting information") == "Table 1"
    assert pp.detect_supporting_info_table("Table 3 recommended supporting information") == "Table 3"
    assert pp.detect_supporting_info_table("Other text") is None


def test_detect_flattened_table_schema():
    assert pp.detect_flattened_table_schema("Required information specific condition purpose guidance notes") is True
    assert pp.detect_flattened_table_schema("No table schema here") is False


def test_split_enumerated_entries():
    entries = pp.split_enumerated_entries(
        "1. First entry contains enough words to be counted. 2. Second entry also contains enough words to be counted."
    )
    assert len(entries) == 2
    assert entries[0][0] == 1
    assert entries[1][0] == 2


def test_add_required_info_fallback_rows():
    pdf_path = Path("Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf")
    page_section_map = {1: ["Support"]}
    out_records: list[pp.RetrievalRecord] = []
    fake_texts = {
        1: (
            "Table 1 required supporting information 1. First fallback entry has enough detail to count. "
            "2. Second fallback entry also has enough detail to count."
        )
    }

    with patch.object(pp, "get_pdfplumber_page_text", return_value=fake_texts):
        new_index = pp.add_required_info_fallback_rows(pdf_path, page_section_map, out_records, 0)

    assert new_index == 3
    assert len(out_records) == 3
    assert out_records[0].block_type == "table"
    assert out_records[1].block_type == "table_row"
    assert out_records[1].metadata["fallback"] is True


def test_extract_tables():
    pdf_path = Path("sample.pdf")
    fake_doc = FakePdfPlumberDoc(
        [
            FakePdfPlumberPage(
                tables=[
                    [
                        ["Col A", "Col B"],
                        ["1", "2"],
                        ["3", "4"],
                    ]
                ]
            )
        ]
    )

    fake_df = MagicMock()
    fake_tables_dir = MagicMock()
    fake_tables_dir.mkdir = MagicMock()
    fake_out_dir = MagicMock()
    fake_out_dir.__truediv__.side_effect = lambda name: fake_tables_dir if name == "tables" else MagicMock()
    fake_tables_dir.__truediv__.side_effect = lambda name: Path(f"/virtual/{name}")

    with patch.object(pp.pdfplumber, "open", return_value=fake_doc):
        with patch.object(pp, "add_required_info_fallback_rows", side_effect=lambda *args: args[-1]):
            with patch.object(pp.pd, "DataFrame", return_value=fake_df) as mock_df:
                records, next_index = pp.extract_tables(pdf_path, fake_out_dir, {1: ["Section"]}, 0)

    assert next_index == 3
    assert len(records) == 3
    assert records[0].block_type == "table"
    assert records[1].block_type == "table_row"
    assert mock_df.called
    assert fake_df.to_csv.called


def test_drop_overlapping_text_records():
    text_record = pp.make_retrieval_record(Path("x.pdf"), "1", 1, "text", "Table 1 details", ["Sec"])
    table_record = pp.make_retrieval_record(Path("x.pdf"), "2", 1, "table", "Table 1 details", ["Sec"], table_id="t1", parent_id="t1")
    filtered = pp.drop_overlapping_text_records([text_record, table_record])
    assert len(filtered) == 1
    assert filtered[0].block_type == "table"


def test_write_jsonl():
    mocked_file = mock_open()
    with patch.object(Path, "open", mocked_file):
        pp.write_jsonl(Path("virtual.jsonl"), [{"a": 1}, {"b": 2}])

    written = "".join(call.args[0] for call in mocked_file().write.call_args_list)
    lines = written.strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1}


def test_process_pdf():
    raw_blocks = [pp.LayoutBlock(1, "Body", [0, 0, 1, 1], 10, False)]
    text_records = [pp.make_retrieval_record(Path("sample.pdf"), "doc1", 1, "text", "Narrative content long enough", ["Sec"])]
    table_records = [pp.make_retrieval_record(Path("sample.pdf"), "doc2", 1, "table", "Table title: X", ["Sec"], table_id="t1", parent_id="t1")]
    fake_out_dir = MagicMock()
    fake_doc_dir = MagicMock()
    fake_manifest_path = MagicMock()
    fake_raw_path = MagicMock()
    fake_retrieval_path = MagicMock()

    def output_div(name):
        return fake_doc_dir if name == "sample" else MagicMock()

    def doc_div(name):
        mapping = {
            "manifest.json": fake_manifest_path,
            "raw_page_blocks.jsonl": fake_raw_path,
            "retrieval_blocks.jsonl": fake_retrieval_path,
        }
        return mapping.get(name, MagicMock())

    fake_out_dir.__truediv__.side_effect = output_div
    fake_doc_dir.__truediv__.side_effect = doc_div
    fake_doc_dir.mkdir = MagicMock()
    fake_manifest_path.open = mock_open()

    with patch.object(pp, "extract_layout_blocks", return_value=(raw_blocks, 10)):
        with patch.object(pp, "detect_repeated_header_footer_text", return_value={"header"}):
            with patch.object(pp, "merge_text_blocks", return_value=(text_records, {1: ["Sec"]})):
                with patch.object(pp, "extract_tables", return_value=(table_records, 2)):
                    with patch.object(pp, "drop_overlapping_text_records", return_value=text_records + table_records):
                        with patch.object(pp, "write_jsonl") as mock_write_jsonl:
                            manifest = pp.process_pdf(Path("sample.pdf"), fake_out_dir)

    assert manifest["source_file"] == "sample.pdf"
    assert manifest["retrieval_block_count"] == 2
    assert manifest["repeated_edge_text_removed"] == ["header"]
    assert fake_doc_dir.mkdir.called
    assert mock_write_jsonl.call_count == 2
    assert fake_manifest_path.open.called


def test_main():
    fake_summary = [{"source_file": "a.pdf"}]
    fake_output_dir = MagicMock()
    fake_input_dir = MagicMock()
    fake_pdf = Path("a.pdf")
    fake_summary_path = MagicMock()
    fake_summary_path.open = mock_open()

    fake_output_dir.__truediv__.side_effect = lambda name: fake_summary_path if name == "summary.json" else MagicMock()
    fake_output_dir.mkdir = MagicMock()
    fake_input_dir.exists.return_value = True
    fake_input_dir.glob.return_value = [fake_pdf]

    with patch.object(pp, "OUTPUT_DIR", fake_output_dir):
        with patch.object(pp, "INPUT_DIR", fake_input_dir):
            with patch.object(pp, "process_pdf", return_value=fake_summary[0]) as mock_process:
                pp.main()

    written = "".join(call.args[0] for call in fake_summary_path.open().write.call_args_list)
    assert mock_process.call_count == 1
    assert json.loads(written) == fake_summary
