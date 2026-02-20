"""IUCN Rule Checker - Check text against IUCN formatting rules.

This package provides tools for checking text blocks against IUCN
(International Union for Conservation of Nature) formatting and
style rules for Red List assessments.

Usage:
    from iucn_checker import IUCNRuleChecker, check_text

    # Using the convenience function
    report = check_text("Your text here...")
    print(report.to_json())

    # Using the checker class directly
    checker = IUCNRuleChecker()
    report = checker.check("Your text here...")

    # Accessing violations
    for violation in report.violations:
        print(f"Line {violation.position.line}: {violation.message}")
"""

from .models import (
    Violation,
    ViolationReport,
    TextPosition,
    Severity,
    RuleDefinition,
    JudgmentType,
)

from .engine import IUCNRuleChecker, check_text

from .checkers import (
    BaseChecker,
    PatternChecker,
    SpellingChecker,
    NumberChecker,
    DateChecker,
    AbbreviationChecker,
    SymbolChecker,
    PunctuationChecker,
    IUCNTermsChecker,
    GeographyChecker,
    ScientificNameChecker,
    ReferenceChecker,
    FormattingChecker, 
    LanguageChecker,
)

__version__ = "1.0.0"

__all__ = [
    # Main interface
    'IUCNRuleChecker',
    'check_text',

    # Data models
    'Violation',
    'ViolationReport',
    'TextPosition',
    'Severity',
    'RuleDefinition',
    'JudgmentType',

    # Checkers
    'BaseChecker',
    'PatternChecker',
    'SpellingChecker',
    'NumberChecker',
    'DateChecker',
    'AbbreviationChecker',
    'SymbolChecker',
    'PunctuationChecker',
    'IUCNTermsChecker',
    'GeographyChecker',
    'ScientificNameChecker',
    'ReferenceChecker',
    'FormattingChecker',
    'LanguageChecker',
]
