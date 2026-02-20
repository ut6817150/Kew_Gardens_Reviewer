import pytest
from checkers.formatting import FormattingChecker

def test_very_long_text():
    """Test performance with large documents"""
    checker = FormattingChecker()
    text = "Test sentence. " * 10000  # 10,000 sentences
    violations = checker.check(text)
    # Should complete without error

def test_special_characters():
    """Test handling of unicode and special characters"""
    checker = FormattingChecker()
    text = "Species: Café™ straße 中文"
    violations = checker.check(text)
    # Should not crash

def test_html_tags_in_text():
    """Test that existing HTML tags are handled"""
    checker = FormattingChecker()
    text = "The species <i>Homo sapiens</i> is common"
    violations = checker.check(text)
    # Should not flag already-italicized text
    assert not any("Homo sapiens" in v.matched_text for v in violations)