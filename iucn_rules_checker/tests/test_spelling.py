import pytest
from checkers.spelling import SpellingChecker

def test_ize_spelling_preferred():
    """IUCN prefers -ize spelling (not -ise)"""
    checker = SpellingChecker()
    
    # "utilise" should be flagged (UK -ise form)
    text = "This resource is utilised by researchers"
    violations = checker.check(text)
    
    print(f"\nViolations: {len(violations)}")
    for v in violations:
        print(f"  - '{v.matched_text}' → '{v.suggested_fix}'")
    
    # Should flag "utilised" and suggest "utilized"
    assert len(violations) > 0
    assert any("utilized" in v.suggested_fix.lower() for v in violations)


def test_ise_spelling_flagged():
    """UK -ise spelling should be flagged (IUCN prefers -ize)"""
    checker = SpellingChecker()
    
    test_cases = [
        ("This is organised well", "organized"),
        ("We will utilise this method", "utilize"),
        ("They specialise in conservation", "specialize"),
    ]
    
    for text, expected in test_cases:
        violations = checker.check(text)
        print(f"\n'{text}'")
        print(f"  Should suggest: {expected}")
        print(f"  Found: {[v.suggested_fix for v in violations]}")
        
        assert any(expected in v.suggested_fix for v in violations)


def test_our_spelling_preferred():
    """IUCN prefers -our spelling (UK style)"""
    checker = SpellingChecker()
    
    # "color" should be flagged (US spelling)
    text = "The color of the bird"
    violations = checker.check(text)
    
    # Should flag "color" and suggest "colour"
    assert len(violations) > 0
    assert any("colour" in v.suggested_fix.lower() for v in violations)


def test_re_spelling_preferred():
    """IUCN prefers -re spelling (UK style)"""
    checker = SpellingChecker()
    
    # "center" should be flagged (US spelling)
    text = "The center of the range"
    violations = checker.check(text)
    
    # Should flag "center" and suggest "centre"
    assert len(violations) > 0
    assert any("centre" in v.suggested_fix.lower() for v in violations)


def test_correct_spelling_not_flagged():
    """Correct IUCN spellings should not be flagged"""
    checker = SpellingChecker()
    
    # These follow IUCN rules and should NOT be flagged
    correct_texts = [
        "This resource is utilized",      # -ize form (correct for IUCN)
        "The colour is bright",            # -our form (correct for IUCN)
        "The centre of the region",        # -re form (correct for IUCN)
        "They organized the conference",   # -ize form (correct for IUCN)
    ]
    
    for text in correct_texts:
        violations = checker.check(text)
        print(f"\n'{text}': {len(violations)} violations")
        assert len(violations) == 0, f"'{text}' should not be flagged"
