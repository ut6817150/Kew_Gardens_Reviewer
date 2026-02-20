"""Reference and citation formatting checker for IUCN assessments."""

import re
from typing import List

from .base import BaseChecker
from ..models import Violation, Severity


class ReferenceChecker(BaseChecker):
    """Checker for citation and reference formatting rules."""

    def __init__(self):
        super().__init__(
            rule_id="references_format",
            rule_name="Reference formatting rules",
            category="References",
            severity=Severity.WARNING,
            assessment_section="Bibliography"
        )

    def check(self, text: str) -> List[Violation]:
        """Check for reference formatting violations."""
        violations = []

        # & vs "and" in citations
        violations.extend(self._check_ampersand_usage(text))

        # et al. format
        violations.extend(self._check_et_al_format(text))

        # Author format (Smith 2020) vs (Smith, 2020)
        violations.extend(self._check_citation_comma(text))

        return violations

    def _check_ampersand_usage(self, text: str) -> List[Violation]:
        """Check that 'and' is used instead of '&' for author names."""
        violations = []

        # Pattern: "Author1 & Author2 (year)" or "(Author1 & Author2 year)"
        ampersand_pattern = re.compile(
            r'([A-Z][a-z]+)\s*&\s*([A-Z][a-z]+)\s*[\(\[]?\d{4}'
        )

        for match in ampersand_pattern.finditer(text):
            author1, author2 = match.group(1), match.group(2)
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message=f"Use 'and' not '&' for author names: '{author1} and {author2}'",
                suggested_fix=None  # Complex replacement
            ))

        return violations

    def _check_et_al_format(self, text: str) -> List[Violation]:
        """Check et al. formatting."""
        violations = []

        # et al without period
        etal_no_period = re.compile(r'\bet al\b(?!\.)')
        for match in etal_no_period.finditer(text):
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message="Use 'et al.' with period",
                suggested_fix="et al."
            ))

        # "et. al." or "et.al" - wrong format
        etal_wrong = re.compile(r'\bet\.\s*al\.?\b')
        for match in etal_wrong.finditer(text):
            if match.group(0) != 'et al.':
                violations.append(self._create_violation(
                    text=text,
                    matched_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    message="Use 'et al.' format (period after 'al' only)",
                    suggested_fix="et al."
                ))

        return violations

    def _check_citation_comma(self, text: str) -> List[Violation]:
        """Check that there's no comma between author and date in citations."""
        violations = []

        # Pattern: "(Author, 2020)" - comma should not be there
        pattern = re.compile(r'\(([A-Z][a-z]+(?:\s+et al\.)?),\s*(\d{4})\)')

        for match in pattern.finditer(text):
            author = match.group(1)
            year = match.group(2)
            fix = f"({author} {year})"
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message=f"No comma between author and date: '{fix}'",
                suggested_fix=fix
            ))

        return violations
