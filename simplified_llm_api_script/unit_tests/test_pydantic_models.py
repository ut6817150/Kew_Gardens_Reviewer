"""
Tests for Pydantic models and the ReviewDocumentError exception in llm_checker_v2.py:
  - Finding
  - CompletionResult
  - RuleMetrics
  - ReviewDocumentError
"""
import pytest
from pydantic import ValidationError

from llm_checker_v2 import (
    CompletionResult,
    Finding,
    ReviewDocumentError,
    RuleEvaluationResult,
    RuleMetrics,
)
from helpers import make_rule_result


# ── Finding ───────────────────────────────────────────────────────────────────

class TestFinding:
    def _valid_kwargs(self, **overrides):
        base = {
            "section_path": "Threats",
            "issue": "Missing data",
            "severity": "high",
            "suggestion": "Add data",
        }
        base.update(overrides)
        return base

    def test_severity_high_accepted(self):
        f = Finding(**self._valid_kwargs(severity="high"))
        assert f.severity == "high"

    def test_severity_medium_accepted(self):
        f = Finding(**self._valid_kwargs(severity="medium"))
        assert f.severity == "medium"

    def test_severity_low_accepted(self):
        f = Finding(**self._valid_kwargs(severity="low"))
        assert f.severity == "low"

    def test_invalid_severity_raises(self):
        with pytest.raises(ValidationError):
            Finding(**self._valid_kwargs(severity="critical"))

    def test_section_path_string_unchanged(self):
        f = Finding(**self._valid_kwargs(section_path="Threats > Classification"))
        assert f.section_path == "Threats > Classification"

    def test_section_path_list_coerced_to_string(self):
        f = Finding(**self._valid_kwargs(section_path=["Root", "Child", "Sub"]))
        assert f.section_path == "Root > Child > Sub"

    def test_section_path_empty_list(self):
        f = Finding(**self._valid_kwargs(section_path=[]))
        assert f.section_path == ""

    def test_section_path_single_element_list(self):
        f = Finding(**self._valid_kwargs(section_path=["Threats"]))
        assert f.section_path == "Threats"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            Finding(section_path="Threats", severity="high", suggestion="Fix it")


# ── CompletionResult ──────────────────────────────────────────────────────────

class TestCompletionResult:
    def _valid_kwargs(self, **overrides):
        base = {
            "text": "response text",
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "token_source": "provider",
        }
        base.update(overrides)
        return base

    def test_token_source_provider_accepted(self):
        r = CompletionResult(**self._valid_kwargs(token_source="provider"))
        assert r.token_source == "provider"

    def test_token_source_estimated_accepted(self):
        r = CompletionResult(**self._valid_kwargs(token_source="estimated"))
        assert r.token_source == "estimated"

    def test_invalid_token_source_raises(self):
        with pytest.raises(ValidationError):
            CompletionResult(**self._valid_kwargs(token_source="guessed"))

    def test_raw_usage_none_accepted(self):
        r = CompletionResult(**self._valid_kwargs(raw_usage=None))
        assert r.raw_usage is None

    def test_provider_details_none_accepted(self):
        r = CompletionResult(**self._valid_kwargs(provider_details=None))
        assert r.provider_details is None

    def test_raw_usage_dict_accepted(self):
        r = CompletionResult(**self._valid_kwargs(raw_usage={"input_tokens": 10}))
        assert r.raw_usage == {"input_tokens": 10}


# ── RuleMetrics ───────────────────────────────────────────────────────────────

class TestRuleMetrics:
    def _valid_kwargs(self, **overrides):
        base = {
            "duration_seconds": 1.5,
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "token_source": "provider",
            "status": "success",
        }
        base.update(overrides)
        return base

    def test_status_success_accepted(self):
        m = RuleMetrics(**self._valid_kwargs(status="success"))
        assert m.status == "success"

    def test_status_error_accepted(self):
        m = RuleMetrics(**self._valid_kwargs(status="error"))
        assert m.status == "error"

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            RuleMetrics(**self._valid_kwargs(status="pending"))

    def test_error_field_none_on_success(self):
        m = RuleMetrics(**self._valid_kwargs(status="success", error=None))
        assert m.error is None

    def test_error_field_string_on_failure(self):
        m = RuleMetrics(**self._valid_kwargs(status="error", error="timeout"))
        assert m.error == "timeout"

    def test_invalid_token_source_raises(self):
        with pytest.raises(ValidationError):
            RuleMetrics(**self._valid_kwargs(token_source="unknown"))

    def test_token_source_estimated_accepted(self):
        m = RuleMetrics(**self._valid_kwargs(token_source="estimated"))
        assert m.token_source == "estimated"


# ── ReviewDocumentError ───────────────────────────────────────────────────────

class TestReviewDocumentError:
    def test_is_runtime_error(self):
        exc = ReviewDocumentError("something failed", [])
        assert isinstance(exc, RuntimeError)

    def test_stores_results_list(self):
        results = [make_rule_result(), make_rule_result(rule_name="rule_02")]
        exc = ReviewDocumentError("partial failure", results)
        assert exc.results == results

    def test_message_accessible_as_str(self):
        exc = ReviewDocumentError("my error message", [])
        assert str(exc) == "my error message"

    def test_empty_results_stored(self):
        exc = ReviewDocumentError("failed early", [])
        assert exc.results == []

    def test_results_length_preserved(self):
        results = [make_rule_result(rule_name=f"rule_{i}") for i in range(5)]
        exc = ReviewDocumentError("failed", results)
        assert len(exc.results) == 5
