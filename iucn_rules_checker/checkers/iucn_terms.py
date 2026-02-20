"""IUCN-specific terminology checker."""

import re
from typing import List

from .base import BaseChecker
from ..models import Violation, Severity


class IUCNTermsChecker(BaseChecker):
    """Checker for IUCN-specific terminology and formatting."""

    # Red List categories (correct capitalization)
    CATEGORIES = [
        ('Critically Endangered', 'CR'),
        ('Endangered', 'EN'),
        ('Vulnerable', 'VU'),
        ('Near Threatened', 'NT'),
        ('Least Concern', 'LC'),
        ('Data Deficient', 'DD'),
        ('Not Evaluated', 'NE'),
        ('Extinct', 'EX'),
        ('Extinct in the Wild', 'EW'),
    ]

    def __init__(self):
        super().__init__(
            rule_id="iucn_terms",
            rule_name="IUCN terminology rules",
            category="IUCN Terms",
            severity=Severity.WARNING,
            assessment_section="Whole Document"
        )

    def check(self, text: str) -> List[Violation]:
        """Check for IUCN terminology violations."""
        violations = []

        # "the IUCN" should be "IUCN"
        violations.extend(self._check_the_iucn(text))

        # CE vs CR for Critically Endangered
        violations.extend(self._check_category_abbreviations(text))

        # Category capitalization
        violations.extend(self._check_category_capitalization(text))

        # "threatened" should be lowercase when referring to CR/EN/VU
        violations.extend(self._check_threatened_case(text))

        return violations

    def _check_the_iucn(self, text: str) -> List[Violation]:
        """Check for 'the IUCN' which should just be 'IUCN'."""
        violations = []

        pattern = re.compile(r'\bthe\s+IUCN\b', re.IGNORECASE)
        for match in pattern.finditer(text):
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message="Use 'IUCN' not 'the IUCN'",
                suggested_fix="IUCN"
            ))

        return violations

    def _check_category_abbreviations(self, text: str) -> List[Violation]:
        """Check for incorrect category abbreviations."""
        violations = []

        # CE is wrong, should be CR for Critically Endangered
        ce_pattern = re.compile(r'\b(CE)\b(?!\w)')
        for match in ce_pattern.finditer(text):
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message="Use 'CR' not 'CE' for Critically Endangered",
                suggested_fix="CR"
            ))

        return violations

    def _check_category_capitalization(self, text: str) -> List[Violation]:
        """Check that Red List categories are properly capitalized."""
        violations = []

        for category, abbrev in self.CATEGORIES:
            # Build pattern for lowercase versions
            pattern = re.compile(rf'\b{category.lower()}\b', re.IGNORECASE)

            for match in pattern.finditer(text):
                matched = match.group(0)
                # Check if it needs capitalization
                if matched != category:
                    # It's not properly capitalized
                    violations.append(self._create_violation(
                        text=text,
                        matched_text=matched,
                        start=match.start(),
                        end=match.end(),
                        message=f"Red List category should be capitalized: '{category}'",
                        suggested_fix=category
                    ))

        return violations

    def _check_threatened_case(self, text: str) -> List[Violation]:
        """Check that 'threatened' is lowercase when used as collective term."""
        violations = []

        # "Threatened" at start of sentence is OK, but mid-sentence should be lowercase
        pattern = re.compile(r'(?<=[a-z]\s)Threatened\b')
        for match in pattern.finditer(text):
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message="Use lowercase 'threatened' when referring to CR/EN/VU species collectively",
                suggested_fix="threatened"
            ))

        return violations
