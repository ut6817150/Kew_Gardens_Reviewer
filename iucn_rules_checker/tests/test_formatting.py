import pytest
from checkers.formatting import FormattingChecker

def test_formatting_no_false_positives():
    """Test that common false positives are fixed"""
    checker = FormattingChecker()
    
    false_positive_phrases = [
            "Very restricted areas",
            "Blue mountains region", 
            "Use map data",
            "Date restriction imposed",
            "Map created from data",
            "Continuing decline noted",
        ]
    
    for phrase in false_positive_phrases:
            violations = checker.check(phrase)
            print(f"\n'{phrase}': {len(violations)} violations")
            if violations:
                for v in violations:
                    print(f"  - Flagged: '{v.matched_text}'")
        
        # None of these should be flagged
    assert len(violations) == 0, f"'{phrase}' should not be flagged, but got: {[v.matched_text for v in violations]}"
 
def test_formatting_real_scientific_names():
    """Test that actual scientific names ARE detected"""
    checker = FormattingChecker()
    
    test_cases = [
        ("Species Homo sapiens is widespread", "Homo sapiens"),
        ("The orchid Acianthera odontotepala grows here", "Acianthera odontotepala"),
        ("Both Pinus caribaea and Pittosporum undulatum are invasive", "Pinus caribaea"),
        ]
    
    for text, expected_name in test_cases:
            violations = checker.check(text)
            print(f"\n'{text}'")
            print(f"  Expected: '{expected_name}'")
            print(f"  Found: {[v.matched_text for v in violations]}")
        
        # Should find the scientific name
    assert any(expected_name in v.matched_text for v in violations), \
            f"Should detect '{expected_name}' in '{text}'"