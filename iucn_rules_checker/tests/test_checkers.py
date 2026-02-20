import pytest
from checkers.spelling import SpellingChecker
from checkers.formatting import FormattingChecker
from models import Violation, Severity


def test_formatting_finds_scientific_name():
    """Test that FormattingChecker detects scientific names needing italics"""
    checker = FormattingChecker()
    text = "The species Homo sapiens is widespread."
    
    violations = checker.check(text)
    
    print(f"\nViolations found: {len(violations)}")
    for v in violations:
        print(f"  - '{v.matched_text}': {v.message}")
    
    # Should find "Homo sapiens"
    assert len(violations) > 0
    assert any(v.matched_text == "Homo sapiens" for v in violations)


def test_formatting_no_false_positive_on_regular_text():
    """Test that regular English phrases aren't flagged as scientific names"""
    checker = FormattingChecker()
    text = "Very restricted areas need protection."
    
    violations = checker.check(text)
    
    print(f"\nText: {text}")
    print(f"Violations found: {len(violations)}")
    for v in violations:
        print(f"  - '{v.matched_text}': {v.message}")
    
    # Should not flag "Very restricted" as a scientific name
    assert len(violations) == 0, f"Should not flag regular text, but found: {[v.matched_text for v in violations]}"


def test_formatting_multiple_scientific_names():
    """Test detection of multiple scientific names"""
    checker = FormattingChecker()
    text = "Both Homo sapiens and Canis lupus are mammals."
    
    violations = checker.check(text)
    
    print(f"\nViolations found: {len(violations)}")
    for v in violations:
        print(f"  - '{v.matched_text}': {v.message}")
    
    # Should find both scientific names
    assert len(violations) >= 2
    matched_texts = [v.matched_text for v in violations]
    assert "Homo sapiens" in matched_texts
    assert "Canis lupus" in matched_texts


def test_spelling_checker_returns_list():
    """Basic test that spelling checker returns a list"""
    checker = SpellingChecker()
    text = "This is a test."
    
    violations = checker.check(text)
    
    # At minimum, should return a list (even if empty)
    assert isinstance(violations, list)