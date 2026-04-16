"""
Tests for pure/stateless helper functions in llm_checker_v2.py:
  - _coerce_token_count
  - _extract_text_content
  - _estimate_token_count
  - _normalize
  - _serialize_prompt_payload
  - _build_user_message
"""
import json
from types import SimpleNamespace

import pytest

from llm_checker_v2 import (
    _build_user_message,
    _coerce_token_count,
    _estimate_token_count,
    _extract_text_content,
    _normalize,
    _serialize_prompt_payload,
)


# ── _coerce_token_count ───────────────────────────────────────────────────────

class TestCoerceTokenCount:
    def test_none_returns_none(self):
        assert _coerce_token_count(None) is None

    def test_bool_true_returns_one(self):
        assert _coerce_token_count(True) == 1

    def test_bool_false_returns_zero(self):
        assert _coerce_token_count(False) == 0

    def test_int_passthrough(self):
        assert _coerce_token_count(42) == 42

    def test_zero(self):
        assert _coerce_token_count(0) == 0

    def test_negative_int_preserved(self):
        assert _coerce_token_count(-5) == -5

    def test_float_rounds_up(self):
        assert _coerce_token_count(3.7) == 4

    def test_float_rounds_down(self):
        assert _coerce_token_count(3.2) == 3

    def test_string_integer(self):
        assert _coerce_token_count("100") == 100

    def test_string_float(self):
        assert _coerce_token_count("99.9") == 100

    def test_string_nonnumeric_returns_none(self):
        assert _coerce_token_count("abc") is None

    def test_empty_string_returns_none(self):
        assert _coerce_token_count("") is None

    def test_list_returns_none(self):
        assert _coerce_token_count([1, 2]) is None

    def test_returns_int_type(self):
        result = _coerce_token_count(42)
        assert isinstance(result, int)


# ── _extract_text_content ─────────────────────────────────────────────────────

class TestExtractTextContent:
    def test_plain_string_passthrough(self):
        assert _extract_text_content("hello") == "hello"

    def test_empty_string(self):
        assert _extract_text_content("") == ""

    def test_list_of_text_dicts_concatenated(self):
        content = [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]
        assert _extract_text_content(content) == "AB"

    def test_list_skips_non_text_type_dicts(self):
        content = [{"type": "image", "url": "x"}, {"type": "text", "text": "C"}]
        assert _extract_text_content(content) == "C"

    def test_list_of_objects_with_text_attr(self):
        content = [SimpleNamespace(text="X"), SimpleNamespace(text="Y")]
        assert _extract_text_content(content) == "XY"

    def test_list_mixed_dict_and_object(self):
        content = [{"type": "text", "text": "A"}, SimpleNamespace(text="B")]
        assert _extract_text_content(content) == "AB"

    def test_single_object_with_text_attr(self):
        assert _extract_text_content(SimpleNamespace(text="Z")) == "Z"

    def test_non_string_object_falls_back_to_str(self):
        assert _extract_text_content(42) == "42"

    def test_list_item_text_attr_not_string_skipped(self):
        content = [SimpleNamespace(text=999)]
        assert _extract_text_content(content) == ""

    def test_list_dict_missing_text_key_skipped(self):
        content = [{"other": "val"}]
        assert _extract_text_content(content) == ""

    def test_empty_list_returns_empty_string(self):
        assert _extract_text_content([]) == ""


# ── _estimate_token_count ─────────────────────────────────────────────────────

class TestEstimateTokenCount:
    def test_empty_string_returns_zero(self):
        assert _estimate_token_count("") == 0

    def test_nonempty_returns_at_least_one(self):
        assert _estimate_token_count("Hello world") >= 1

    def test_returns_int_type(self):
        assert isinstance(_estimate_token_count("anything"), int)

    def test_longer_text_produces_more_tokens(self):
        short = _estimate_token_count("Hi")
        long = _estimate_token_count("This is a much longer piece of text with many words.")
        assert long > short


# ── _normalize ────────────────────────────────────────────────────────────────

class TestNormalize:
    def test_removes_spaces(self):
        assert _normalize("Red List") == "redlist"

    def test_lowercases(self):
        assert _normalize("THREATS") == "threats"

    def test_strips_leading_trailing(self):
        assert _normalize("  Threats  ") == "threats"

    def test_internal_spaces_removed(self):
        assert _normalize("Habitats and Ecology") == "habitatsandecology"

    def test_empty_string(self):
        assert _normalize("") == ""

    def test_mixed(self):
        assert _normalize("  Red List Assessment ") == "redlistassessment"


# ── _serialize_prompt_payload ─────────────────────────────────────────────────

class TestSerializePromptPayload:
    def test_anthropic_has_top_level_system_key(self):
        result = json.loads(_serialize_prompt_payload("anthropic", "claude-3", "sys", "user msg"))
        assert "system" in result
        assert result["system"] == "sys"

    def test_anthropic_messages_has_single_user_entry(self):
        result = json.loads(_serialize_prompt_payload("anthropic", "claude-3", "sys", "user msg"))
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_anthropic_messages_no_system_role(self):
        result = json.loads(_serialize_prompt_payload("anthropic", "claude-3", "sys", "user msg"))
        roles = [m["role"] for m in result["messages"]]
        assert "system" not in roles

    def test_non_anthropic_messages_first_is_system(self):
        result = json.loads(_serialize_prompt_payload("openrouter", "gpt-4", "sys", "user msg"))
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "sys"

    def test_non_anthropic_messages_second_is_user(self):
        result = json.loads(_serialize_prompt_payload("openrouter", "gpt-4", "sys", "user msg"))
        assert result["messages"][1]["role"] == "user"

    def test_returns_valid_json(self):
        raw = _serialize_prompt_payload("huggingface", "model-x", "sys", "msg")
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_model_name_in_output(self):
        raw = _serialize_prompt_payload("anthropic", "my-special-model", "s", "m")
        assert "my-special-model" in raw

    def test_non_ascii_preserved(self):
        raw = _serialize_prompt_payload("anthropic", "m", "Système: résumé", "msg")
        assert "résumé" in raw


# ── _build_user_message ───────────────────────────────────────────────────────

class TestBuildUserMessage:
    def test_contains_rule_opening_tag(self):
        result = _build_user_message("Do this check.", [])
        assert "<rule>" in result

    def test_contains_rule_closing_tag(self):
        result = _build_user_message("Do this check.", [])
        assert "</rule>" in result

    def test_rule_body_between_tags(self):
        result = _build_user_message("My rule body.", [])
        assert "<rule>\nMy rule body.\n</rule>" in result

    def test_contains_document_opening_tag(self):
        result = _build_user_message("body", [])
        assert "<document>" in result

    def test_contains_document_closing_tag(self):
        result = _build_user_message("body", [])
        assert "</document>" in result

    def test_document_section_is_valid_json(self):
        sections = [{"title": "Threats", "blocks": []}]
        result = _build_user_message("body", sections)
        start = result.index("<document>\n") + len("<document>\n")
        end = result.index("\n</document>")
        doc_content = result[start:end]
        parsed = json.loads(doc_content)
        assert isinstance(parsed, list)

    def test_document_json_round_trips(self):
        sections = [{"title": "Threats", "blocks": []}, {"title": "Ecology", "blocks": []}]
        result = _build_user_message("body", sections)
        start = result.index("<document>\n") + len("<document>\n")
        end = result.index("\n</document>")
        doc_content = result[start:end]
        assert json.loads(doc_content) == sections

    def test_empty_sections_produces_empty_array(self):
        result = _build_user_message("body", [])
        start = result.index("<document>\n") + len("<document>\n")
        end = result.index("\n</document>")
        doc_content = result[start:end]
        assert json.loads(doc_content) == []
