"""Apply configured checkers to an already parsed assessment report."""

import re
from typing import Dict, List

try:
    from .checkers.abbreviations import AbbreviationChecker
    from .checkers.base import BaseChecker
    from .checkers.bibliography import BibliographyChecker
    from .checkers.dates import DateChecker
    from .checkers.formatting import FormattingChecker
    from .checkers.geography import GeographyChecker
    from .checkers.iucn_terms import IUCNTermsChecker
    from .checkers.numbers import NumberChecker
    from .checkers.punctuation import PunctuationChecker
    from .checkers.references import ReferenceChecker
    from .checkers.scientific import ScientificNameChecker
    from .checkers.spelling import SpellingChecker
    from .violation import Violation
except ImportError:  # pragma: no cover - direct script execution fallback
    from checkers.abbreviations import AbbreviationChecker
    from checkers.base import BaseChecker
    from checkers.bibliography import BibliographyChecker
    from checkers.dates import DateChecker
    from checkers.formatting import FormattingChecker
    from checkers.geography import GeographyChecker
    from checkers.iucn_terms import IUCNTermsChecker
    from checkers.numbers import NumberChecker
    from checkers.punctuation import PunctuationChecker
    from checkers.references import ReferenceChecker
    from checkers.scientific import ScientificNameChecker
    from checkers.spelling import SpellingChecker
    from violation import Violation


class IUCNAssessmentReviewer:
    """Review parsed assessment sections and return rule violations."""

    def __init__(self):
        self.bibliography_checker = BibliographyChecker()
        self.checkers: List[BaseChecker] = [
            AbbreviationChecker(),
            DateChecker(),
            FormattingChecker(),
            GeographyChecker(),
            IUCNTermsChecker(),
            NumberChecker(),
            PunctuationChecker(),
            ReferenceChecker(),
            ScientificNameChecker(),
            SpellingChecker(),
        ]

    def is_table_section(self, section_name: str) -> bool:
        """Return True when a parsed section key represents table-derived content."""
        return re.search(r"\[table\s+\d+\]", section_name, re.IGNORECASE) is not None

    def is_bibliography_section(self, section_name: str) -> bool:
        """Return True when a parsed section key represents bibliography content."""
        return "bibliography" in section_name.lower()

    def review_full_report(self, full_report: Dict[str, str]) -> List[Violation]:
        """Apply each rule to each ``section -> text`` pair in the parsed report."""
        if not isinstance(full_report, dict):
            raise TypeError("review_full_report() expects a dict of section paths to text.")

        violations: List[Violation] = []
        self.bibliography_checker.begin_sweep()
        for checker in self.checkers:
            checker.begin_sweep()

        try:
            for section_name, section_text in full_report.items():
                if not section_text.strip():
                    continue
                if self.is_table_section(section_name):
                    continue

                section_item = (section_name, section_text)

                if self.is_bibliography_section(section_name):
                    violations.extend(self.bibliography_checker.check(section_item))
                    continue

                for checker in self.checkers:
                    violations.extend(checker.check(section_item))
        finally:
            self.bibliography_checker.end_sweep()
            for checker in self.checkers:
                checker.end_sweep()

        return violations
