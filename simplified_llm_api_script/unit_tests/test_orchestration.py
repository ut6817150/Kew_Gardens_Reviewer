"""
Tests for orchestration/aggregation functions in llm_checker_v2.py:
  - summarize_rule_results
  - results_to_json
"""
import pytest

from llm_checker_v2 import Finding, summarize_rule_results, results_to_json
from helpers import make_finding, make_rule_result

EXPECTED_SUMMARY_KEYS = {
    "rules_attempted",
    "rules_succeeded",
    "rules_failed",
    "total_input_tokens",
    "total_output_tokens",
    "total_tokens",
    "token_estimation_used",
    "total_findings",
}


# ── summarize_rule_results ────────────────────────────────────────────────────

class TestSummarizeRuleResults:
    def test_empty_results_all_counts_zero(self):
        summary = summarize_rule_results([])
        assert summary["rules_attempted"] == 0
        assert summary["rules_succeeded"] == 0
        assert summary["rules_failed"] == 0
        assert summary["total_input_tokens"] == 0
        assert summary["total_output_tokens"] == 0
        assert summary["total_tokens"] == 0
        assert summary["total_findings"] == 0

    def test_empty_results_estimation_flag_false(self):
        summary = summarize_rule_results([])
        assert summary["token_estimation_used"] is False

    def test_all_success_counts(self):
        results = [make_rule_result("r1"), make_rule_result("r2")]
        summary = summarize_rule_results(results)
        assert summary["rules_attempted"] == 2
        assert summary["rules_succeeded"] == 2
        assert summary["rules_failed"] == 0

    def test_mixed_success_and_failure_counts(self):
        results = [
            make_rule_result("r1", status="success"),
            make_rule_result("r2", status="error", error="timeout"),
        ]
        summary = summarize_rule_results(results)
        assert summary["rules_succeeded"] == 1
        assert summary["rules_failed"] == 1

    def test_token_counts_sum_correctly(self):
        results = [
            make_rule_result("r1", input_tokens=100, output_tokens=50),
            make_rule_result("r2", input_tokens=200, output_tokens=80),
        ]
        summary = summarize_rule_results(results)
        assert summary["total_input_tokens"] == 300
        assert summary["total_output_tokens"] == 130
        assert summary["total_tokens"] == 430

    def test_estimation_flag_true_if_any_estimated(self):
        results = [
            make_rule_result("r1", token_source="provider"),
            make_rule_result("r2", token_source="estimated"),
        ]
        summary = summarize_rule_results(results)
        assert summary["token_estimation_used"] is True

    def test_estimation_flag_false_if_all_provider(self):
        results = [
            make_rule_result("r1", token_source="provider"),
            make_rule_result("r2", token_source="provider"),
        ]
        summary = summarize_rule_results(results)
        assert summary["token_estimation_used"] is False

    def test_total_findings_counted_across_results(self):
        findings_a = [make_finding(), make_finding(issue="Another issue")]
        findings_b = [make_finding(issue="Third issue")]
        results = [
            make_rule_result("r1", findings=findings_a),
            make_rule_result("r2", findings=findings_b),
        ]
        summary = summarize_rule_results(results)
        assert summary["total_findings"] == 3

    def test_total_findings_zero_when_no_findings(self):
        results = [make_rule_result("r1"), make_rule_result("r2")]
        summary = summarize_rule_results(results)
        assert summary["total_findings"] == 0

    def test_all_expected_keys_present(self):
        summary = summarize_rule_results([make_rule_result()])
        assert EXPECTED_SUMMARY_KEYS == set(summary.keys())

    def test_single_result_attempted_is_one(self):
        summary = summarize_rule_results([make_rule_result()])
        assert summary["rules_attempted"] == 1


# ── results_to_json ───────────────────────────────────────────────────────────

class TestResultsToJson:
    def test_empty_list_returns_empty_list(self):
        assert results_to_json([]) == []

    def test_each_element_has_rule_name_and_findings(self):
        output = results_to_json([make_rule_result("r1")])
        assert set(output[0].keys()) == {"rule_name", "findings"}

    def test_metrics_not_present(self):
        output = results_to_json([make_rule_result("r1")])
        assert "metrics" not in output[0]

    def test_rule_name_preserved(self):
        output = results_to_json([make_rule_result("my_rule")])
        assert output[0]["rule_name"] == "my_rule"

    def test_no_findings_gives_empty_list(self):
        output = results_to_json([make_rule_result("r1", findings=[])])
        assert output[0]["findings"] == []

    def test_finding_serialized_as_dict(self):
        f = make_finding(section_path="Threats", issue="Missing", severity="high", suggestion="Add it")
        output = results_to_json([make_rule_result("r1", findings=[f])])
        fd = output[0]["findings"][0]
        assert fd["section_path"] == "Threats"
        assert fd["issue"] == "Missing"
        assert fd["severity"] == "high"
        assert fd["suggestion"] == "Add it"

    def test_multiple_results_length_preserved(self):
        results = [make_rule_result(f"r{i}") for i in range(4)]
        output = results_to_json(results)
        assert len(output) == 4

    def test_multiple_findings_in_single_result(self):
        findings = [make_finding(issue=f"issue {i}") for i in range(3)]
        output = results_to_json([make_rule_result("r1", findings=findings)])
        assert len(output[0]["findings"]) == 3

    def test_order_preserved(self):
        results = [make_rule_result(f"rule_{i}") for i in range(3)]
        output = results_to_json(results)
        assert [o["rule_name"] for o in output] == ["rule_0", "rule_1", "rule_2"]
