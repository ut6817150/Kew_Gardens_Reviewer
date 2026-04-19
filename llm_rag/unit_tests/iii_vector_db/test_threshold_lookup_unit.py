"""Unit tests for deterministic threshold lookup.

Purpose:
    This module verifies threshold-query detection, criterion inference,
    field-specific answer construction, and non-threshold fallback behavior for
    `llm_rag/iii_vector_db/threshold_lookup.py`.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

SOURCE_DIR = Path(__file__).resolve().parents[2] / "iii_vector_db"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

sys.modules.pop("threshold_lookup", None)
tl = importlib.import_module("threshold_lookup")


def fake_threshold_data():
    return {
        "criteria": {
            "A": {
                "title": "Population reduction",
                "summary": ["Criterion A summary line 1", "Criterion A summary line 2"],
                "fields": {},
            },
            "B": {
                "title": "Geographic range",
                "summary": ["Criterion B summary line 1", "Criterion B summary line 2"],
                "fields": {
                    "eoo": "Criterion B1 EOO thresholds: CR < 100 km2, EN < 5,000 km2, VU < 20,000 km2.",
                    "aoo": "Criterion B2 AOO thresholds: CR < 10 km2, EN < 500 km2, VU < 2,000 km2.",
                    "locations": "Typical number of locations thresholds under Criterion B: CR = 1, EN <= 5, VU <= 10.",
                },
            },
            "C": {
                "title": "Small population size and decline",
                "summary": ["Criterion C summary"],
                "fields": {},
            },
            "D": {
                "title": "Very small or restricted population",
                "summary": ["Criterion D summary line 1", "Criterion D summary line 2"],
                "fields": {
                    "mature_individuals": "Criterion D thresholds: CR < 50 mature individuals, EN < 250, VU D1 < 1,000.",
                    "d2": "Criterion D2 thresholds: AOO < 20 km2 or number of locations <= 5.",
                },
            },
            "E": {
                "title": "Quantitative analysis",
                "summary": ["Criterion E summary"],
                "fields": {
                    "extinction_probability": "Criterion E extinction probability thresholds: CR >= 50%, EN >= 20%, VU >= 10%.",
                },
            },
        }
    }


def test_contains_any():
    assert tl._contains_any("criterion b threshold", ("threshold", "cutoff")) is True
    assert tl._contains_any("plain text", ("threshold", "cutoff")) is False


def test_is_threshold_query_detects_direct_and_contextual_queries():
    assert tl.is_threshold_query("What are the Criterion B thresholds for EOO and AOO?") is True
    assert tl.is_threshold_query("What is the extinction probability threshold under Criterion E?") is True
    assert tl.is_threshold_query("How many locations are allowed for D2?") is True
    assert tl.is_threshold_query("Tell me about habitat preferences.") is False


def test_infer_criterion_prefers_explicit_and_then_field_heuristics():
    assert tl.infer_criterion("What does Criterion B require?") == "B"
    assert tl.infer_criterion("How does D2 work?") == "D"
    assert tl.infer_criterion("What are the AOO and EOO cutoffs?") == "B"
    assert tl.infer_criterion("How many mature individuals are allowed?") == "D"
    assert tl.infer_criterion("What is the extinction probability?") == "E"
    assert tl.infer_criterion("General background information") is None


def test_requested_fields_marks_present_fields():
    fields = tl._requested_fields("aoo eoo number of locations mature individuals extinction probability")

    assert fields["aoo"] is True
    assert fields["eoo"] is True
    assert fields["locations"] is True
    assert fields["mature_individuals"] is True
    assert fields["extinction_probability"] is True


def test_join_non_empty_filters_blank_lines():
    assert tl._join_non_empty(["a", "", "b"]) == "a\nb"
    assert tl._join_non_empty(["", ""]) is None


def test_answer_threshold_query_returns_field_specific_b_answers():
    with patch.object(tl, "load_thresholds", return_value=fake_threshold_data()):
        answer = tl.answer_threshold_query("What are the Criterion B thresholds for EOO and AOO?")

    assert "Criterion B1 EOO thresholds" in answer
    assert "Criterion B2 AOO thresholds" in answer


def test_answer_threshold_query_returns_b_summary_when_no_specific_field_requested():
    with patch.object(tl, "load_thresholds", return_value=fake_threshold_data()):
        answer = tl.answer_threshold_query("Explain Criterion B.")

    assert answer.startswith("Criterion B: Geographic range")
    assert "Criterion B summary line 1" in answer


def test_answer_threshold_query_returns_d_specific_fields():
    with patch.object(tl, "load_thresholds", return_value=fake_threshold_data()):
        mature_answer = tl.answer_threshold_query("How many mature individuals are allowed under Criterion D?")
        d2_answer = tl.answer_threshold_query("What are the D2 thresholds?")

    assert "mature individuals" in mature_answer
    assert "D2 thresholds" in d2_answer


def test_answer_threshold_query_returns_e_specific_field():
    with patch.object(tl, "load_thresholds", return_value=fake_threshold_data()):
        answer = tl.answer_threshold_query("What is the extinction probability threshold under Criterion E?")

    assert answer == "Criterion E extinction probability thresholds: CR >= 50%, EN >= 20%, VU >= 10%."


def test_answer_threshold_query_handles_field_only_queries_without_explicit_criterion():
    with patch.object(tl, "load_thresholds", return_value=fake_threshold_data()):
        locations_answer = tl.answer_threshold_query("What is the threshold for number of locations?")
        aoo_answer = tl.answer_threshold_query("What is the AOO threshold?")
        eoo_answer = tl.answer_threshold_query("What is the EOO threshold?")

    assert "locations thresholds" in locations_answer
    assert "AOO thresholds" in aoo_answer
    assert "EOO thresholds" in eoo_answer


def test_answer_threshold_query_returns_none_when_query_does_not_map_cleanly():
    with patch.object(tl, "load_thresholds", return_value=fake_threshold_data()):
        answer = tl.answer_threshold_query("Tell me about general species documentation.")

    assert answer is None


def run_all_tests():
    tests = [
        obj
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
    print(f"Passed {len(tests)} threshold_lookup unit tests.")


if __name__ == "__main__":
    run_all_tests()
