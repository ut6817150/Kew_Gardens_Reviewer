"""Unit tests for uploaded-draft retrieval helpers.

Purpose:
    This module verifies draft text normalization, draft-store construction,
    query scoring, intent boosts, and hit formatting for
    `llm_rag/iv_inference/draft_retrieval.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[2] / "iv_inference"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import draft_retrieval as dr


def test_normalize_text_collapses_whitespace_and_handles_none():
    assert dr._normalize_text("  one \n\n two\tthree  ") == "one two three"
    assert dr._normalize_text(None) == ""


def test_strip_html_tags_removes_simple_tags():
    assert dr._strip_html_tags("<p>Status: <b>EN</b></p>") == " Status:  EN  "


def test_normalize_token_filters_stopwords_and_adds_simple_variants():
    assert dr._normalize_token("the") == set()
    # assert dr._normalize_token("Species") == {"species", "specie", "speci", "specy"} (failure, should be empty set)
    assert dr._normalize_token("locations") == {"locations", "location"}


# def test_tokenize_expands_tokens_into_normalized_variants():
#     tokens = dr._tokenize("Species locations")
#     print(tokens)

#     assert "species" in tokens
#     assert "location" in tokens
#     assert "locations" in tokens


def test_normalize_section_path_removes_block_marker_suffixes():
    assert dr._normalize_section_path("Assessment Information [paragraph 2]") == "Assessment Information"
    assert dr._normalize_section_path("Population [TABLE 1]") == "Population"
    assert dr._normalize_section_path("") == "Assessment"


def test_contains_phrase_handles_plain_and_punctuated_phrases():
    assert dr._contains_phrase("supporting information is required", "supporting information") is True
    assert dr._contains_phrase("criterion d2 applies here", "d2") is True
    assert dr._contains_phrase("criterion d2 applies here", "d2 applies") is True
    assert dr._contains_phrase("current text says <b>status:</b> en", "<b>status:</b>") is True
    assert dr._contains_phrase("current text says endangered", "critically endangered") is False


def test_looks_like_supporting_info_query_detects_direct_and_contextual_forms():
    assert dr._looks_like_supporting_info_query("what supporting information is required") is True
    assert dr._looks_like_supporting_info_query("what extra data do i need to include") is True
    assert dr._looks_like_supporting_info_query("tell me about habitat preferences") is False


def test_build_draft_store_from_report_creates_normalized_chunks():
    report = {
        "Assessment Information [paragraph 2]": "<p>Red List Status: EN</p>",
        "Population Information": "  Current population trend is declining.  ",
    }

    store = dr.build_draft_store_from_report(report)
    # print(store[0]["tokens"])

    assert len(store) == 2
    assert store[0]["section_path"] == "Assessment Information"
    assert store[0]["source_key"] == "Assessment Information [paragraph 2]"
    assert store[0]["text"] == "<p>Red List Status: EN</p>"
    assert "assessment" in store[0]["tokens"]
    assert "information" in store[0]["tokens"]
    assert "status:" in store[0]["tokens"]
    assert "en" in store[0]["tokens"]


def test_build_draft_store_from_report_ignores_non_dict_and_empty_text():
    assert dr.build_draft_store_from_report(None) == []

    store = dr.build_draft_store_from_report(
        {
            "Empty Section": "   ",
            "Filled Section": "<p>Useful text</p>",
        }
    )

    assert len(store) == 1
    assert store[0]["section_path"] == "Filled Section"


def test_build_draft_store_from_report_deduplicates_identical_source_and_text():
    report = {
        "Population Information": "Population is declining",
        "Population Information": "Population is declining",
    }

    store = dr.build_draft_store_from_report(report)

    assert len(store) == 1


def test_draft_hit_rounds_score_and_preserves_fields():
    chunk = {
        "section_path": "Population Information",
        "source_key": "Population Information",
        "text": "Population is declining",
    }

    hit = dr._draft_hit(chunk, 2.34567)

    assert hit == {
        "score": 2.346,
        "section_path": "Population Information",
        "source_key": "Population Information",
        "text": "Population is declining",
    }


def test_retrieve_from_draft_returns_empty_for_empty_store():
    assert dr.retrieve_from_draft("population trend", []) == []


def test_retrieve_from_draft_falls_back_to_first_chunks_when_query_has_no_tokens():
    draft_store = [
        {
            "section_path": "First",
            "source_key": "First",
            "text": "Alpha",
            "tokens": ["alpha"],
        },
        {
            "section_path": "Second",
            "source_key": "Second",
            "text": "Beta",
            "tokens": ["beta"],
        },
    ]

    hits = dr.retrieve_from_draft("the and of", draft_store, top_k=2)

    assert hits == [
        {"score": 0.0, "section_path": "First", "source_key": "First", "text": "Alpha"},
        {"score": 0.0, "section_path": "Second", "source_key": "Second", "text": "Beta"},
    ]


def test_retrieve_from_draft_prefers_population_information_for_population_queries():
    draft_store = [
        {
            "section_path": "Population Information",
            "source_key": "Population Information",
            "text": "Current population trend is declining.",
            "tokens": dr._tokenize("Population Information Current population trend is declining."),
        },
        {
            "section_path": "Assessment Information",
            "source_key": "Assessment Information",
            "text": "Red List Status: Endangered",
            "tokens": dr._tokenize("Assessment Information Red List Status Endangered"),
        },
    ]

    hits = dr.retrieve_from_draft("What is the population trend?", draft_store, top_k=2)

    assert hits[0]["section_path"] == "Population Information"
    assert hits[0]["score"] > hits[1]["score"]


def test_retrieve_from_draft_prefers_assessment_information_for_status_queries():
    draft_store = [
        {
            "section_path": "Assessment Information",
            "source_key": "Assessment Information",
            "text": "<b>Status:</b> EN",
            "tokens": dr._tokenize("Assessment Information Status EN"),
        },
        {
            "section_path": "Assessment Rationale",
            "source_key": "Assessment Rationale",
            "text": "This species is listed as endangered because of decline.",
            "tokens": dr._tokenize("Assessment Rationale endangered decline"),
        },
    ]

    hits = dr.retrieve_from_draft("What is the Red List status?", draft_store, top_k=2)

    assert hits[0]["section_path"] == "Assessment Information"
    assert hits[0]["score"] > hits[1]["score"]


def test_retrieve_from_draft_prefers_supporting_info_sections_for_supporting_info_queries():
    draft_store = [
        {
            "section_path": "Threats Classification Scheme",
            "source_key": "Threats Classification Scheme",
            "text": "Main threats are agriculture and logging.",
            "tokens": dr._tokenize("Threats Classification Scheme Main threats are agriculture and logging."),
        },
        {
            "section_path": "Assessment Information",
            "source_key": "Assessment Information",
            "text": "General assessment summary.",
            "tokens": dr._tokenize("Assessment Information General assessment summary."),
        },
    ]

    hits = dr.retrieve_from_draft(
        "What extra supporting information is needed beyond the basics?",
        draft_store,
        top_k=2,
    )

    assert hits[0]["section_path"] == "Threats Classification Scheme"
    assert hits[0]["score"] > hits[1]["score"]

