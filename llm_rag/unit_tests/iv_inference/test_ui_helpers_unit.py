"""Unit tests for RAG UI helper functions.

Purpose:
    This module verifies display-only cleanup behavior for draft section names
    produced by parser and retrieval helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[2] / "iv_inference"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import ui_helpers


def test_normalize_display_section_name_removes_paragraph_marker():
    assert ui_helpers.normalize_display_section_name("Assessment Information [paragraph 3]") == (
        "Assessment Information"
    )


def test_normalize_display_section_name_removes_table_marker():
    assert ui_helpers.normalize_display_section_name("Threats Classification Scheme [table 1]") == (
        "Threats Classification Scheme"
    )


def test_normalize_display_section_name_removes_row_marker():
    assert ui_helpers.normalize_display_section_name("Population Details [row 4]") == "Population Details"


def test_normalize_display_section_name_is_case_insensitive_for_markers():
    assert ui_helpers.normalize_display_section_name("Locations Information [PaRaGrApH 2]") == (
        "Locations Information"
    )
    assert ui_helpers.normalize_display_section_name("Countries of Occurrence [TABLE 5]") == (
        "Countries of Occurrence"
    )
    assert ui_helpers.normalize_display_section_name("Range Notes [RoW 7]") == "Range Notes"


def test_normalize_display_section_name_leaves_plain_section_name_unchanged():
    assert ui_helpers.normalize_display_section_name("Assessment Rationale") == "Assessment Rationale"


def test_normalize_display_section_name_only_removes_supported_suffix_patterns():
    assert ui_helpers.normalize_display_section_name("Assessment [figure 1]") == "Assessment [figure 1]"
    assert ui_helpers.normalize_display_section_name("Assessment [paragraph]") == "Assessment [paragraph]"

