"""Unit tests for RAG runtime orchestration.

Purpose:
    This module verifies prompt assembly, session-state synchronization,
    external LLM response normalization, debug-payload formatting, and the
    high-level `answer_rag_question(...)` flow with mocked dependencies.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

SOURCE_DIR = Path(__file__).resolve().parents[2] / "iv_inference"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

fake_huggingface_hub = types.ModuleType("huggingface_hub")
fake_huggingface_hub.snapshot_download = lambda *args, **kwargs: "fake-model-path"
sys.modules.setdefault("huggingface_hub", fake_huggingface_hub)

fake_langchain_huggingface = types.ModuleType("langchain_huggingface")


class FakeHuggingFaceEmbeddings:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


fake_langchain_huggingface.HuggingFaceEmbeddings = FakeHuggingFaceEmbeddings
sys.modules.setdefault("langchain_huggingface", fake_langchain_huggingface)

fake_langchain_chroma = types.ModuleType("langchain_chroma")


class FakeChroma:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


fake_langchain_chroma.Chroma = FakeChroma
sys.modules.setdefault("langchain_chroma", fake_langchain_chroma)

import rag_runtime as rr


def test_build_openai_base_url_trims_chat_completions_suffix():
    assert rr._build_openai_base_url("https://example.com/v1/chat/completions") == "https://example.com/v1"
    assert rr._build_openai_base_url("https://example.com/v1/") == "https://example.com/v1"


def test_build_llm_request_kwargs_includes_messages_and_optional_reasoning():
    basic = rr._build_llm_request_kwargs("Prompt text", "demo-model", reasoning_enabled=False)
    reasoning = rr._build_llm_request_kwargs("Prompt text", "demo-model", reasoning_enabled=True)

    assert basic["model"] == "demo-model"
    assert basic["messages"][0]["content"] == rr.SYSTEM_MESSAGE
    assert basic["messages"][1] == {"role": "user", "content": "Prompt text"}
    assert basic["temperature"] == 0.1
    assert "extra_body" not in basic

    assert reasoning["extra_body"] == {"reasoning": {"enabled": True}}


def test_completion_to_dict_supports_common_response_shapes():
    class WithModelDump:
        def model_dump(self):
            return {"choices": []}

    class WithToDict:
        def to_dict(self):
            return {"choices": [{"message": {}}]}

    class WithJsonDump:
        def model_dump_json(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}}]})

    assert rr._completion_to_dict({"choices": []}) == {"choices": []}
    assert rr._completion_to_dict(WithModelDump()) == {"choices": []}
    assert rr._completion_to_dict(WithToDict()) == {"choices": [{"message": {}}]}
    assert rr._completion_to_dict(WithJsonDump()) == {"choices": [{"message": {"content": "ok"}}]}


def test_completion_to_dict_rejects_unsupported_shape():
    with pytest.raises(TypeError):
        rr._completion_to_dict(object())


def test_stringify_response_text_handles_nested_structures():
    value = [
        {"text": "First"},
        {"content": ["Second", {"summary": "Third"}]},
    ]

    assert rr._stringify_response_text(value) == "First\nSecond\nThird"
    assert rr._stringify_response_text("  hello  ") == "hello"
    assert rr._stringify_response_text(None) is None


def test_extract_reasoning_text_uses_reasoning_details_when_direct_reasoning_missing():
    message = {
        "reasoning_details": [
            {"type": "reasoning.summary", "summary": "Short summary"},
            {"type": "reasoning.text", "text": "Detailed chain"},
            {"type": "reasoning.encrypted"},
        ]
    }

    thinking = rr._extract_reasoning_text(message)

    assert "Summary: Short summary" in thinking
    assert "Detailed chain" in thinking
    assert "[Encrypted reasoning block]" in thinking


def test_extract_message_parts_returns_normalized_output_and_reasoning():
    response = {
        "choices": [
            {
                "message": {
                    "content": [{"text": "Final answer"}],
                    "reasoning_details": [{"type": "reasoning.text", "text": "Why"}],
                }
            }
        ]
    }

    parts = rr._extract_message_parts(response)

    assert parts == {
        "output": "Final answer",
        "thinking": "Why",
        "reasoning_details": [{"type": "reasoning.text", "text": "Why"}],
    }


def test_llm_error_result_returns_normalized_error_payload():
    assert rr._llm_error_result("boom") == {
        "output": None,
        "thinking": None,
        "reasoning_details": None,
        "error": "boom",
    }


def test_init_rag_session_state_sets_missing_defaults_only():
    session_state = {"rag_messages": ["existing"]}

    rr.init_rag_session_state(session_state)

    assert session_state["rag_messages"] == ["existing"]
    assert session_state["rag_draft_store"] is None
    assert session_state["rag_report_dict"] is None


def test_sync_rag_state_with_upload_resets_cached_values_on_signature_change():
    session_state = {
        "rag_upload_signature": "old",
        "rag_messages": ["message"],
        "rag_draft_store": [{"chunk": 1}],
        "rag_draft_signature": "sig",
        "rag_assessment_input_dict": {"a": 1},
        "rag_assessment_input_signature": "assessment-sig",
        "rag_report_dict": {"section": "text"},
    }

    rr.sync_rag_state_with_upload(session_state, "new")

    assert session_state["rag_upload_signature"] == "new"
    assert session_state["rag_messages"] == []
    assert session_state["rag_draft_store"] is None
    assert session_state["rag_report_dict"] is None


def test_sync_rag_state_with_upload_preserves_state_when_signature_matches():
    session_state = {
        "rag_upload_signature": "same",
        "rag_messages": ["message"],
        "rag_draft_store": [{"chunk": 1}],
        "rag_draft_signature": "sig",
        "rag_assessment_input_dict": {"a": 1},
        "rag_assessment_input_signature": "assessment-sig",
        "rag_report_dict": {"section": "text"},
    }

    rr.sync_rag_state_with_upload(session_state, "same")

    assert session_state["rag_messages"] == ["message"]
    assert session_state["rag_draft_store"] == [{"chunk": 1}]


def test_build_report_from_assessment_returns_section_level_mapping():
    assessment = {
        "title": "Assessment Information",
        "blocks": [{"type": "paragraph", "text_rich": "Summary"}],
        "children": [],
    }

    report = rr.build_report_from_assessment(assessment)

    assert report == {"Assessment Information": "<p>Summary</p>"}


def test_ensure_draft_store_from_report_builds_once_and_then_reuses_cache():
    session_state = {}
    report = {"Section": "Text"}

    with patch.object(rr, "build_draft_store_from_report", return_value=[{"section_path": "Section"}]) as builder:
        first = rr.ensure_draft_store_from_report(session_state, report, "sig-1")
        second = rr.ensure_draft_store_from_report(session_state, report, "sig-1")

    assert first == [{"section_path": "Section"}]
    assert second == [{"section_path": "Section"}]
    assert builder.call_count == 1


def test_build_reference_result_lines_formats_results_and_parent_context():
    candidate = types.SimpleNamespace(
        text="Reference text",
        parent_text="Parent details",
        metadata={
            "source_file": "doc.pdf",
            "page": 3,
            "block_type": "table_row",
            "section_title": "Section A",
        },
    )

    lines = rr.build_reference_result_lines({"results": [candidate]})

    assert "Source: doc.pdf" in lines[0]
    assert "Page: 3" in lines[0]
    assert "Text: Reference text" in lines[0]
    assert lines[1] == "  Parent context: Parent details"


def test_build_generation_prompt_includes_deterministic_reference_and_draft_sections():
    prompt = rr.build_generation_prompt(
        query="What is the status?",
        deterministic_answer="Criterion facts",
        reference_payload={"answer_scaffold": "Reference scaffold", "results": []},
        draft_hits=[
            {
                "section_path": "Assessment Information",
                "source_key": "Assessment Information",
                "score": 4.2,
                "text": "Draft evidence text",
            }
        ],
        has_draft=True,
    )

    assert "User question: What is the status?" in prompt
    assert "Criterion facts" in prompt
    assert "Reference scaffold" in prompt
    assert "Draft assessment evidence (top 6 chunks):" in prompt
    assert "Section: Assessment Information" in prompt


def test_external_llm_is_configured_requires_base_url_and_model():
    assert rr.external_llm_is_configured({"base_url": "https://x", "model": "m"}) is True
    assert rr.external_llm_is_configured({"base_url": "https://x", "model": ""}) is False
    assert rr.external_llm_is_configured(None) is False


def test_maybe_call_external_llm_returns_none_when_config_is_incomplete():
    assert rr.maybe_call_external_llm("Prompt", {"base_url": "", "model": "m"}) is None
    assert rr.maybe_call_external_llm("Prompt", {"base_url": "https://x", "model": ""}) is None


def test_maybe_call_external_llm_returns_error_when_openai_client_missing():
    with patch.object(rr, "OpenAI", None), patch.object(rr, "OPENAI_IMPORT_ERROR", Exception("not installed")):
        result = rr.maybe_call_external_llm("Prompt", {"base_url": "https://x", "model": "m"})

    assert "OpenAI client is not installed" in result["error"]


def test_maybe_call_external_llm_calls_client_and_normalizes_response():
    class FakeCompletions:
        def create(self, **kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Model answer",
                            "reasoning_details": [{"type": "reasoning.text", "text": "Reasoning"}],
                        }
                    }
                ]
            }

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = FakeChat()

    with patch.object(rr, "OpenAI", FakeClient):
        result = rr.maybe_call_external_llm(
            "Prompt body",
            {
                "base_url": "https://example.com/v1/chat/completions",
                "model": "demo-model",
                "api_key": "secret",
                "reasoning_enabled": True,
            },
        )

    assert result["output"] == "Model answer"
    assert result["thinking"] == "Reasoning"


def test_answer_rag_question_combines_threshold_reference_draft_and_model_output():
    with (
        patch.object(rr, "is_threshold_query", return_value=True),
        patch.object(rr, "answer_threshold_query", return_value="Threshold answer"),
        patch.object(
            rr,
            "answer_query",
            return_value={"route": "hybrid_rag", "answer_scaffold": "Reference scaffold", "results": []},
        ),
        patch.object(
            rr,
            "retrieve_from_draft",
            return_value=[{"section_path": "Population Information", "source_key": "Population", "score": 3.0, "text": "Declining"}],
        ),
        patch.object(rr, "external_llm_is_configured", return_value=True),
        patch.object(rr, "maybe_call_external_llm", return_value={"output": "Final grounded answer", "thinking": None}),
    ):
        response = rr.answer_rag_question(
            "What is the population trend?",
            draft_store=[{"dummy": True}],
            llm_config={"base_url": "https://x", "model": "m"},
        )

    assert response["answer"] == "Final grounded answer"
    assert response["route"] == "deterministic_plus_hybrid_rag"
    assert response["deterministic_answer"] == "Threshold answer"
    assert response["draft_hits"][0]["section_path"] == "Population Information"
    assert response["llm_configured"] is True
    assert response["used_external_llm"] is True


def test_format_rag_debug_payload_serializes_debug_view():
    payload = rr.format_rag_debug_payload(
        {
            "answer": "Answer",
            "route": "hybrid_rag",
            "llm_configured": False,
            "used_external_llm": False,
            "deterministic_answer": None,
            "reference_payload": {"subqueries": [], "answer_scaffold": "Scaffold", "results": []},
            "draft_hits": [],
            "prompt": "Prompt body",
            "model_response": {"output": None, "thinking": None, "reasoning_details": None},
        }
    )

    parsed = json.loads(payload)
    assert parsed["route"] == "hybrid_rag"
    assert parsed["full_final_prompt"] == "Prompt body"

