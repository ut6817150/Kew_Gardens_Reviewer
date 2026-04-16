"""
Tests for select_sections() in llm_checker_v2.py.

Covers all three scope modes:
  - "full_document"
  - "relevant_sections: X, Y"
  - "section_type:*"

And edge cases: case-insensitivity, whitespace, missing children key, unknown scope.
"""
import pytest

from llm_checker_v2 import select_sections


# ── full_document ─────────────────────────────────────────────────────────────

class TestFullDocument:
    def test_returns_tree_as_single_element(self, simple_document_tree):
        result = select_sections(simple_document_tree, "full_document")
        assert result == [simple_document_tree]

    def test_leading_trailing_whitespace_stripped(self, simple_document_tree):
        result = select_sections(simple_document_tree, "  full_document  ")
        assert result == [simple_document_tree]

    def test_returns_list_of_length_one(self, simple_document_tree):
        result = select_sections(simple_document_tree, "full_document")
        assert len(result) == 1

    def test_works_on_empty_document_tree(self, empty_document_tree):
        result = select_sections(empty_document_tree, "full_document")
        assert result == [empty_document_tree]


# ── relevant_sections ─────────────────────────────────────────────────────────

class TestRelevantSections:
    def test_single_match(self, simple_document_tree):
        result = select_sections(simple_document_tree, "relevant_sections: Threats")
        assert len(result) == 1
        assert result[0]["title"] == "Threats"

    def test_two_matches(self, simple_document_tree):
        result = select_sections(simple_document_tree, "relevant_sections: Threats, Conservation")
        titles = {s["title"] for s in result}
        assert titles == {"Threats", "Conservation"}

    def test_three_matches(self, simple_document_tree):
        result = select_sections(simple_document_tree, "relevant_sections: Threats, Conservation, Bibliography")
        assert len(result) == 3

    def test_case_insensitive(self, simple_document_tree):
        result = select_sections(simple_document_tree, "relevant_sections: THREATS")
        assert len(result) == 1
        assert result[0]["title"] == "Threats"

    def test_extra_whitespace_in_names(self, simple_document_tree):
        result = select_sections(simple_document_tree, "relevant_sections:  Threats , Bibliography ")
        titles = {s["title"] for s in result}
        assert titles == {"Threats", "Bibliography"}

    def test_no_match_returns_empty_list(self, simple_document_tree):
        result = select_sections(simple_document_tree, "relevant_sections: Nonexistent")
        assert result == []

    def test_all_keyword_returns_full_tree(self, simple_document_tree):
        result = select_sections(simple_document_tree, "relevant_sections: all")
        assert result == [simple_document_tree]

    def test_all_keyword_case_insensitive(self, simple_document_tree):
        result = select_sections(simple_document_tree, "relevant_sections: ALL")
        assert result == [simple_document_tree]

    def test_no_children_key_returns_empty(self):
        tree_without_children = {"title": "Root", "level": 0, "blocks": []}
        result = select_sections(tree_without_children, "relevant_sections: Threats")
        assert result == []

    def test_empty_children_returns_empty(self, empty_document_tree):
        result = select_sections(empty_document_tree, "relevant_sections: Threats")
        assert result == []

    def test_returns_actual_child_dicts_not_copies(self, simple_document_tree):
        result = select_sections(simple_document_tree, "relevant_sections: Threats")
        assert result[0] is simple_document_tree["children"][0]


# ── section_type:* ────────────────────────────────────────────────────────────

class TestSectionTypeStar:
    def test_returns_all_children(self, simple_document_tree):
        result = select_sections(simple_document_tree, "section_type:*")
        assert result == simple_document_tree["children"]

    def test_returns_correct_count(self, simple_document_tree):
        result = select_sections(simple_document_tree, "section_type:*")
        assert len(result) == 3

    def test_whitespace_stripped(self, simple_document_tree):
        result = select_sections(simple_document_tree, "  section_type:*  ")
        assert len(result) == 3

    def test_empty_children_returns_empty(self, empty_document_tree):
        result = select_sections(empty_document_tree, "section_type:*")
        assert result == []

    def test_no_children_key_returns_empty(self):
        tree = {"title": "Root", "level": 0}
        result = select_sections(tree, "section_type:*")
        assert result == []


# ── Unknown scope ─────────────────────────────────────────────────────────────

class TestUnknownScope:
    def test_raises_value_error(self, simple_document_tree):
        with pytest.raises(ValueError):
            select_sections(simple_document_tree, "some_random_scope")

    def test_error_message_contains_scope(self, simple_document_tree):
        scope = "weird: value"
        with pytest.raises(ValueError, match="weird"):
            select_sections(simple_document_tree, scope)

    def test_partial_prefix_match_raises(self, simple_document_tree):
        with pytest.raises(ValueError):
            select_sections(simple_document_tree, "full_doc")
