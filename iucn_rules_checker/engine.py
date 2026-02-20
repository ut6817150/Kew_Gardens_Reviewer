"""Main IUCN rule checker engine."""

from typing import List, Optional, Set, Dict
from .streamlit_json_validator import StreamlitAssessmentValidator

from .models import Violation, ViolationReport, Severity
from .checkers import (
    BaseChecker,
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
    LanguageChecker
)


class IUCNRuleChecker:
    """Main checker engine that orchestrates all rule checks.

    This class manages all the individual rule checkers and provides
    a unified interface for checking text against IUCN formatting rules.

    Usage:
        checker = IUCNRuleChecker()
        report = checker.check("Your text here...")
        print(report.to_json())
    """

    def __init__(
        self,
        enabled_categories: Optional[Set[str]] = None,
        disabled_rules: Optional[Set[str]] = None,
        min_severity: Severity = Severity.INFO
    ):
        """Initialize the rule checker.

        Args:
            enabled_categories: If provided, only run checkers in these categories.
                              If None, all categories are enabled.
            disabled_rules: Set of rule IDs to skip.
            min_severity: Minimum severity level to include in results.
        """
        self.enabled_categories = enabled_categories
        self.disabled_rules = disabled_rules or set()
        self.min_severity = min_severity

        # Initialize all checkers
        self._checkers: List[BaseChecker] = self._create_checkers()

        self.json_validator = StreamlitAssessmentValidator()

    def _create_checkers(self) -> List[BaseChecker]:
        """Create and return all available checkers."""
        return [
            SpellingChecker(),
            NumberChecker(),
            DateChecker(),
            AbbreviationChecker(),
            SymbolChecker(),
            PunctuationChecker(),
            IUCNTermsChecker(),
            GeographyChecker(),
            ScientificNameChecker(),
            ReferenceChecker(),
            FormattingChecker(),
            LanguageChecker(),
        ]

    def get_available_categories(self) -> Set[str]:
        """Get all available checker categories."""
        return {checker.category for checker in self._checkers}
    
    
    def check_json(self, json_data: Dict) -> ViolationReport:
        """Check assessment from Streamlit JSON.
        
        Args:
            json_data: Parsed JSON from Streamlit (hierarchical structure)
            
        Returns:
            ViolationReport with all violations found
        """
        all_violations = []
        checked_rules = []
        skipped_rules = []
        
        # 1. Run JSON structure validation
        json_report, full_text = self.json_validator.validate(json_data)
        all_violations.extend(json_report.violations)
        checked_rules.extend(json_report.checked_rules)
        
        # 2. Run text-based checkers on extracted text
        for checker in self._checkers:
            # Check if this category is enabled
            if self.enabled_categories and checker.category not in self.enabled_categories:
                skipped_rules.append(checker.rule_id)
                continue
            
            # Check if this rule is disabled
            if checker.rule_id in self.disabled_rules:
                skipped_rules.append(checker.rule_id)
                continue
            
            # Run the checker
            checked_rules.append(checker.rule_id)
            text_violations = checker.check(full_text)
            
            # Filter by severity
            for v in text_violations:
                if self._severity_value(v.severity) >= self._severity_value(self.min_severity):
                    all_violations.append(v)
        
        return self._build_report(
            text=full_text,
            violations=all_violations,
            checked_rules=checked_rules,
            skipped_rules=skipped_rules
        )


    def check(self, text: str) -> ViolationReport:
        """Check plain text for IUCN rule violations.
        
        Args:
            text: Plain text to check
            
        Returns:
            ViolationReport with all violations found
        """
        violations = []
        checked_rules = []
        skipped_rules = []
        
        for checker in self._checkers:
            # Check if this category is enabled
            if self.enabled_categories and checker.category not in self.enabled_categories:
                skipped_rules.append(checker.rule_id)
                continue
            
            # Check if this rule is disabled
            if checker.rule_id in self.disabled_rules:
                skipped_rules.append(checker.rule_id)
                continue
            
            # Run the checker
            checked_rules.append(checker.rule_id)
            checker_violations = checker.check(text)
            
            # Filter by severity
            for v in checker_violations:
                if self._severity_value(v.severity) >= self._severity_value(self.min_severity):
                    violations.append(v)
        
        return self._build_report(
            text=text,
            violations=violations,
            checked_rules=checked_rules,
            skipped_rules=skipped_rules
        )
    

    def _severity_value(self, severity: Severity) -> int:
        """Convert severity to numeric value for comparison."""
        return {Severity.INFO: 1, Severity.WARNING: 2, Severity.ERROR: 3}[severity]

    def _build_report(
        self,
        text: str,
        violations: List[Violation],
        checked_rules: List[str],
        skipped_rules: List[str]
    ) -> ViolationReport:
        """Build the final violation report."""

        # Count by severity (must use Severity enum as keys)
        by_severity: Dict[Severity, int] = {}
        for v in violations:
            if v.severity not in by_severity:
                by_severity[v.severity] = 0
            by_severity[v.severity] += 1

    # Count by category
        by_category: Dict[str, int] = {}
        for v in violations:
            if v.category not in by_category:
                by_category[v.category] = 0
            by_category[v.category] += 1

        # Sort violations by position
        violations.sort(key=lambda v: v.position.start)

        return ViolationReport(
            text_length=len(text),
            total_violations=len(violations),
            violations_by_severity=by_severity,
            violations_by_category=by_category,
            violations=violations,
            checked_rules=checked_rules,
            skipped_rules=skipped_rules
        )


def check_text(
    text: str,
    categories: Optional[Set[str]] = None,
    min_severity: Severity = Severity.INFO
) -> ViolationReport:
    """Convenience function to check text with default settings.

    Args:
        text: The text to check.
        categories: Optional set of categories to check (None = all).
        min_severity: Minimum severity to include.

    Returns:
        ViolationReport with all findings.
    """
    checker = IUCNRuleChecker(
        enabled_categories=categories,
        min_severity=min_severity
    )
    return checker.check(text)
