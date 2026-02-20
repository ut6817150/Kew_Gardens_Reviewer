import pytest
from checkers.punctuation import PunctuationChecker

def test_en_dash_for_ranges():
    """Ranges should use en dash not hyphen"""
    checker = PunctuationChecker()
    text = "Pages 50-90"
    violations = checker.check(text)
    assert any("50–90" in v.suggested_fix for v in violations)

def test_hyphen_in_compound_words():
    """Hyphens in compound words should be allowed"""
    checker = PunctuationChecker()
    text = "A well-known species"
    violations = checker.check(text)
    # should not flag this as needing en dash
    assert not any("well" in v.matched_text for v in violations)