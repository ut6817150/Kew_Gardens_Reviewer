import pytest
from checkers.numbers import NumberChecker

def test_numbers_adds_commas():
    """Numbers >= 1000 should have commas"""
    checker = NumberChecker()
    text = "Population is 5000 individuals"
    violations = checker.check(text)
    assert any("5,000" in v.suggested_fix for v in violations)

def test_numbers_ignores_small_numbers():
    """Numbers < 1000 should not be flagged"""
    checker = NumberChecker()
    text = "There are 999 species"
    violations = checker.check(text)
    assert not any("999" in v.matched_text for v in violations)

def test_numbers_written_out_1_to_9():
    """Numbers 1-9 at start of sentence should be written out"""
    checker = NumberChecker()
    text = "5 species were observed"
    violations = checker.check(text)
    assert any("five" in v.suggested_fix.lower() for v in violations)

def test_year_numbers_allowed():
    """Years in citations should not be flagged"""
    # Years in citations shouldn't trigger "don't start with number"
    # this test will help us decide if this rule needs context awareness
    checker = NumberChecker()
    text = "Smith et al. 2024"
    violations = checker.check(text)