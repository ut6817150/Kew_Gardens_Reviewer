"""Reference and citation formatting checker for IUCN assessments."""

import re
from typing import List

from ..violation import Violation
from .base import BaseChecker


class ReferenceChecker(BaseChecker):
    """Checker for citation and reference formatting rules."""

    def __init__(self):
        super().__init__()

    def check_text(self, section_name: str, text: str) -> List[Violation]:
        """Check for reference formatting violations."""
        violations = []
        violations.extend(self.check_citation_comma(section_name, text))
        return violations

    def check_citation_comma(self, section_name: str, text: str) -> List[Violation]:
        """Check for a comma immediately before the year in bracketed citations.

        This rule now uses a broader citation shell rather than trying to infer
        surname shape. It looks for:
        - an opening `(`
        - any non-nested citation body
        - a comma before a 4-digit year
        - a closing `)`

        Examples flagged:
        `(Smith, 2020)` -> `(Smith 2020)`
        `(Mishra et al., 2015)` -> `(Mishra et al. 2015)`

        Examples not flagged:
        `(Smith 2020)`
        `[GBIF.org, 2021]`
        `Smith, 2020` because it is not bracketed
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )
        pattern = re.compile(r'\((?P<body>[^\(\)]*?),\s*(?P<year>\d{4})\)')

        for match in pattern.finditer(cleaned_text):
            body = match.group("body").strip()
            year = match.group("year")
            fix = f"({body} {year})"
            original_start = index_map[match.start()]
            original_end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(original_start, original_end),
                message=f"No comma between author and date: '{fix}'",
                suggested_fix=fix,
            ))

        return violations
