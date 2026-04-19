"""Unit tests for the inference assessment parser.

Purpose:
    This module verifies that `InferenceAssessmentParser` converts nested
    assessment dictionaries into section-level HTML reports while handling
    paragraphs, tables, rich cells, unsupported blocks, and malformed rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SOURCE_DIR = Path(__file__).resolve().parents[2] / "iv_inference"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import inference_assessment_parser as parser_module


def test_parse_rejects_non_dict_input():
    parser = parser_module.InferenceAssessmentParser()

    with pytest.raises(TypeError):
        parser.parse(None)


def test_parse_returns_empty_report_for_empty_document():
    parser = parser_module.InferenceAssessmentParser()

    assert parser.parse({}) == {}


def test_parse_builds_nested_section_paths():
    parser = parser_module.InferenceAssessmentParser()
    assessment = {
        "title": "Assessment Information",
        "blocks": [{"type": "paragraph", "text_rich": "Summary text"}],
        "children": [
            {
                "title": "Population",
                "blocks": [{"type": "paragraph", "text_rich": "Declining"}],
                "children": [],
            }
        ],
    }

    report = parser.parse(assessment)

    assert report["Assessment Information"] == "<p>Summary text</p>"
    assert report["Assessment Information > Population"] == "<p>Declining</p>"


def test_parse_uses_assessment_fallback_when_titles_missing():
    parser = parser_module.InferenceAssessmentParser()
    assessment = {
        "blocks": [{"type": "paragraph", "text_rich": "Fallback section text"}],
        "children": [],
    }

    report = parser.parse(assessment)

    assert report == {"Assessment": "<p>Fallback section text</p>"}


def test_render_paragraph_wraps_plain_text_rich():
    parser = parser_module.InferenceAssessmentParser()

    html = parser.render_paragraph_html({"type": "paragraph", "text_rich": "Plain paragraph"})

    assert html == "<p>Plain paragraph</p>"


def test_render_paragraph_does_not_double_wrap_existing_p_tag():
    parser = parser_module.InferenceAssessmentParser()

    html = parser.render_paragraph_html({"type": "paragraph", "text_rich": "<p>Ready wrapped</p>"})

    assert html == "<p>Ready wrapped</p>"


def test_render_table_builds_html_rows_and_cells():
    parser = parser_module.InferenceAssessmentParser()
    block = {
        "type": "table",
        "rows_rich": [
            ["Status", "EN"],
            ["Trend", "Decreasing"],
        ],
    }

    html = parser.render_table_html(block)

    assert html == (
        "<table>\n"
        "  <tr><td>Status</td><td>EN</td></tr>\n"
        "  <tr><td>Trend</td><td>Decreasing</td></tr>\n"
        "</table>"
    )


def test_render_table_skips_empty_or_malformed_rows_rich():
    parser = parser_module.InferenceAssessmentParser()

    assert parser.render_table_html({"type": "table", "rows_rich": []}) == ""
    assert parser.render_table_html({"type": "table", "rows_rich": None}) == ""


def test_render_table_row_flattens_nested_cells_with_breaks():
    parser = parser_module.InferenceAssessmentParser()

    row_html = parser.render_table_row_html([["A", "B"], "C"])

    assert row_html == "<td>A<br/>B</td><td>C</td>"


def test_render_table_row_accepts_scalar_rows():
    parser = parser_module.InferenceAssessmentParser()

    row_html = parser.render_table_row_html("Single value")

    assert row_html == "<td>Single value</td>"


def test_parse_ignores_unsupported_block_types():
    parser = parser_module.InferenceAssessmentParser()
    assessment = {
        "title": "Assessment Information",
        "blocks": [{"type": "image", "url": "x"}],
        "children": [],
    }

    report = parser.parse(assessment)

    assert report == {}


def test_parse_merges_multiple_supported_blocks_under_same_section():
    parser = parser_module.InferenceAssessmentParser()
    assessment = {
        "title": "Assessment Information",
        "blocks": [
            {"type": "paragraph", "text_rich": "Summary text"},
            {"type": "table", "rows_rich": [["Status", "EN"]]},
        ],
        "children": [],
    }

    report = parser.parse(assessment)

    assert report["Assessment Information"] == (
        "<p>Summary text</p>\n"
        "<table>\n"
        "  <tr><td>Status</td><td>EN</td></tr>\n"
        "</table>"
    )


def test_stringify_rich_cell_coerces_none_and_bytes():
    parser = parser_module.InferenceAssessmentParser()

    assert parser.stringify_rich_cell(None) == ""
    assert parser.stringify_rich_cell(b"Byte text") == "Byte text"

