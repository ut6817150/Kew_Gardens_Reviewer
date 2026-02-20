import pytest
from engine import IUCNRuleChecker

def test_full_assessment_check():
    """Test checking a complete assessment"""
    checker = IUCNRuleChecker()
    text = """
    The species Homo sapiens has a population of 8000000000.
    Range: 50-90 km2.
    """
    
    report = checker.check(text)
    
    # Should find multiple types of violations
    assert len(report.violations) > 0
    categories = {v.category for v in report.violations}
    assert "Formatting" in categories  # Homo sapiens needs italics
    assert "Numbers" in categories     # 8000000000 needs commas

def test_empty_text():
    """Test handling of empty input"""
    checker = IUCNRuleChecker()
    report = checker.check("")
    assert len(report.violations) == 0

def test_text_with_no_violations():
    """Test perfect text with no issues"""
    checker = IUCNRuleChecker()
    text = "This is a perfectly formatted sentence."
    report = checker.check(text)
