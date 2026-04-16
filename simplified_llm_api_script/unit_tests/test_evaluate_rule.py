"""
Tests for the async evaluate_rule() function in llm_checker_v2.py.

Uses mock providers — no real API calls are made.
Covers:
  - Happy path (bare JSON, single finding)
  - Markdown code fence stripping (```json and ```)
  - Provider exception → error status, estimated tokens
  - Invalid JSON response → error status, provider tokens preserved
  - Pydantic validation error in finding (bad severity)
  - Finding section_path coercion from list to string
"""
import json

import pytest

from llm_checker_v2 import evaluate_rule
from helpers import make_failing_provider, make_mock_provider

SYSTEM_PROMPT = "You are a helpful assistant."

VALID_JSON_NO_FINDINGS = json.dumps({
    "rule_name": "rule_test",
    "findings": [],
})

VALID_JSON_ONE_FINDING = json.dumps({
    "rule_name": "rule_test",
    "findings": [
        {
            "section_path": "Threats",
            "issue": "Data missing",
            "severity": "high",
            "suggestion": "Add data",
        }
    ],
})

FENCED_JSON_WITH_LANG = f"```json\n{VALID_JSON_NO_FINDINGS}\n```"
FENCED_JSON_NO_LANG = f"```\n{VALID_JSON_NO_FINDINGS}\n```"


# ── Happy path ────────────────────────────────────────────────────────────────

class TestEvaluateRuleSuccess:
    async def test_bare_json_no_findings_success(self, minimal_rule, simple_document_tree):
        provider = make_mock_provider(VALID_JSON_NO_FINDINGS)
        sections = [simple_document_tree]
        result = await evaluate_rule(provider, SYSTEM_PROMPT, minimal_rule, sections)
        assert result.metrics.status == "success"
        assert result.rule_name == "rule_test"
        assert result.findings == []

    async def test_bare_json_one_finding(self, minimal_rule, simple_document_tree):
        provider = make_mock_provider(VALID_JSON_ONE_FINDING)
        result = await evaluate_rule(provider, SYSTEM_PROMPT, minimal_rule, [simple_document_tree])
        assert result.metrics.status == "success"
        assert len(result.findings) == 1

    async def test_finding_fields_populated(self, minimal_rule, simple_document_tree):
        provider = make_mock_provider(VALID_JSON_ONE_FINDING)
        result = await evaluate_rule(provider, SYSTEM_PROMPT, minimal_rule, [simple_document_tree])
        f = result.findings[0]
        assert f.section_path == "Threats"
        assert f.issue == "Data missing"
        assert f.severity == "high"
        assert f.suggestion == "Add data"

    async def test_token_source_from_provider(self, minimal_rule, simple_document_tree):
        provider = make_mock_provider(VALID_JSON_NO_FINDINGS, input_tokens=15, output_tokens=25)
        result = await evaluate_rule(provider, SYSTEM_PROMPT, minimal_rule, [simple_document_tree])
        assert result.metrics.token_source == "provider"

    async def test_token_counts_match_completion(self, minimal_rule, simple_document_tree):
        provider = make_mock_provider(VALID_JSON_NO_FINDINGS, input_tokens=15, output_tokens=25)
        result = await evaluate_rule(provider, SYSTEM_PROMPT, minimal_rule, [simple_document_tree])
        assert result.metrics.input_tokens == 15
        assert result.metrics.output_tokens == 25
        assert result.metrics.total_tokens == 40

    async def test_duration_seconds_is_positive(self, minimal_rule, simple_document_tree):
        provider = make_mock_provider(VALID_JSON_NO_FINDINGS)
        result = await evaluate_rule(provider, SYSTEM_PROMPT, minimal_rule, [simple_document_tree])
        assert result.metrics.duration_seconds > 0


# ── Code fence stripping ──────────────────────────────────────────────────────

class TestCodeFenceStripping:
    async def test_json_fenced_with_language_tag(self, minimal_rule, simple_document_tree):
        provider = make_mock_provider(FENCED_JSON_WITH_LANG)
        result = await evaluate_rule(provider, SYSTEM_PROMPT, minimal_rule, [simple_document_tree])
        assert result.metrics.status == "success"
        assert result.rule_name == "rule_test"

    async def test_json_fenced_without_language_tag(self, minimal_rule, simple_document_tree):
        provider = make_mock_provider(FENCED_JSON_NO_LANG)
        result = await evaluate_rule(provider, SYSTEM_PROMPT, minimal_rule, [simple_document_tree])
        assert result.metrics.status == "success"

    async def test_plain_json_no_fence(self, minimal_rule, simple_document_tree):
        provider = make_mock_provider(VALID_JSON_NO_FINDINGS)
        result = await evaluate_rule(provider, SYSTEM_PROMPT, minimal_rule, [simple_document_tree])
        assert result.metrics.status == "success"

    async def test_fence_with_whitespace_padding(self, minimal_rule, simple_document_tree):
        padded = f"  \n```json\n{VALID_JSON_NO_FINDINGS}\n```\n  "
        provider = make_mock_provider(padded)
        result = await evaluate_rule(provider, SYSTEM_PROMPT, minimal_rule, [simple_document_tree])
        assert result.metrics.status == "success"


# ── Error paths ───────────────────────────────────────────────────────────────

class TestEvaluateRuleErrors:
    async def test_provider_exception_returns_error_status(self, minimal_rule, simple_document_tree):
        provider = make_failing_provider(RuntimeError("API down"))
        result = await evaluate_rule(provider, SYSTEM_PROMPT, minimal_rule, [simple_document_tree])
        assert result.metrics.status == "error"

    async def test_provider_exception_findings_empty(self, minimal_rule, simple_document_tree):
        provider = make_failing_provider(RuntimeError("API down"))
        result = await evaluate_rule(provider, SYSTEM_PROMPT, minimal_rule, [simple_document_tree])
        assert result.findings == []

    async def test_provider_exception_stores_error_message(self, minimal_rule, simple_document_tree):
        provider = make_failing_provider(RuntimeError("API down"))
        result = await evaluate_rule(provider, SYSTEM_PROMPT, minimal_rule, [simple_document_tree])
        assert result.metrics.error == "API down"

    async def test_provider_exception_uses_estimated_tokens(self, minimal_rule, simple_document_tree):
        # When complete() raises, completion is None → tokens are estimated
        provider = make_failing_provider(RuntimeError("timeout"))
        result = await evaluate_rule(provider, SYSTEM_PROMPT, minimal_rule, [simple_document_tree])
        assert result.metrics.token_source == "estimated"

    async def test_provider_exception_output_tokens_zero(self, minimal_rule, simple_document_tree):
        provider = make_failing_provider(RuntimeError("timeout"))
        result = await evaluate_rule(provider, SYSTEM_PROMPT, minimal_rule, [simple_document_tree])
        assert result.metrics.output_tokens == 0

    async def test_invalid_json_response_returns_error_status(self, minimal_rule, simple_document_tree):
        provider = make_mock_provider("not json at all")
        result = await evaluate_rule(provider, SYSTEM_PROMPT, minimal_rule, [simple_document_tree])
        assert result.metrics.status == "error"

    async def test_invalid_json_preserves_provider_token_source(self, minimal_rule, simple_document_tree):
        # complete() succeeded and returned a CompletionResult, then JSON parsing failed.
        # Tokens should come from the completion, not estimation.
        provider = make_mock_provider("not json at all", input_tokens=5, output_tokens=3)
        result = await evaluate_rule(provider, SYSTEM_PROMPT, minimal_rule, [simple_document_tree])
        assert result.metrics.token_source == "provider"
        assert result.metrics.input_tokens == 5

    async def test_finding_invalid_severity_returns_error(self, minimal_rule, simple_document_tree):
        bad_json = json.dumps({
            "rule_name": "rule_test",
            "findings": [
                {
                    "section_path": "Threats",
                    "issue": "Test",
                    "severity": "critical",  # invalid
                    "suggestion": "Fix it",
                }
            ],
        })
        provider = make_mock_provider(bad_json)
        result = await evaluate_rule(provider, SYSTEM_PROMPT, minimal_rule, [simple_document_tree])
        assert result.metrics.status == "error"

    async def test_finding_section_path_list_coerced(self, minimal_rule, simple_document_tree):
        json_with_list_path = json.dumps({
            "rule_name": "rule_test",
            "findings": [
                {
                    "section_path": ["Root", "Child"],
                    "issue": "Test",
                    "severity": "low",
                    "suggestion": "Fix it",
                }
            ],
        })
        provider = make_mock_provider(json_with_list_path)
        result = await evaluate_rule(provider, SYSTEM_PROMPT, minimal_rule, [simple_document_tree])
        assert result.metrics.status == "success"
        assert result.findings[0].section_path == "Root > Child"

    async def test_rule_name_from_result_not_rule_dict_on_success(self, minimal_rule, simple_document_tree):
        # rule_name in the JSON response is the authoritative name on success
        json_resp = json.dumps({"rule_name": "rule_from_llm", "findings": []})
        provider = make_mock_provider(json_resp)
        result = await evaluate_rule(provider, SYSTEM_PROMPT, minimal_rule, [simple_document_tree])
        assert result.rule_name == "rule_from_llm"

    async def test_rule_name_from_rule_dict_on_error(self, minimal_rule, simple_document_tree):
        # On error, rule_name falls back to the rule dict's name
        provider = make_failing_provider(RuntimeError("oops"))
        result = await evaluate_rule(provider, SYSTEM_PROMPT, minimal_rule, [simple_document_tree])
        assert result.rule_name == minimal_rule["name"]
