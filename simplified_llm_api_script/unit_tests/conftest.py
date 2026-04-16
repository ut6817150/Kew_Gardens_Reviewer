import os
import sys

import pytest

# Make the project root and this directory importable.
_here = os.path.dirname(__file__)
sys.path.insert(0, _here)                        # unit_tests/ (for helpers.py)
sys.path.insert(0, os.path.join(_here, ".."))    # project root (for llm_checker_v2)


# ── Document tree fixtures ────────────────────────────────────────────────────

@pytest.fixture
def simple_document_tree():
    """Minimal document tree with three named top-level children."""
    return {
        "title": "Test Assessment",
        "level": 0,
        "blocks": [],
        "children": [
            {"title": "Threats", "level": 1, "blocks": [{"text": "Habitat loss"}], "children": []},
            {"title": "Conservation", "level": 1, "blocks": [{"text": "Protected areas"}], "children": []},
            {"title": "Bibliography", "level": 1, "blocks": [{"text": "Smith 2020."}], "children": []},
        ],
    }


@pytest.fixture
def empty_document_tree():
    """Document tree with no children."""
    return {"title": "Empty", "level": 0, "blocks": [], "children": []}


# ── Rule fixture ──────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_rule():
    return {
        "name": "rule_test",
        "scope": "full_document",
        "severity": "high",
        "category": "test",
        "body": "Check the document.",
    }
