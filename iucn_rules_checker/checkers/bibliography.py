"""Bibliography-specific formatting checker for IUCN assessments."""

import re
from typing import List

from ..violation import Violation
from .abbreviations import AbbreviationChecker
from .base import BaseChecker
from .numbers import NumberChecker
from .punctuation import PunctuationChecker


class BibliographyChecker(BaseChecker):
    """Checker for bibliography-specific formatting rules."""

    def __init__(self):
        super().__init__()
        self.abbreviation_checker = AbbreviationChecker()
        self.number_checker = NumberChecker()
        self.punctuation_checker = PunctuationChecker()

    def begin_sweep(self) -> None:
        """Prepare helper checkers before reviewing a full report."""
        self.abbreviation_checker.begin_sweep()
        self.number_checker.begin_sweep()
        self.punctuation_checker.begin_sweep()

    def end_sweep(self) -> None:
        """Clear helper checker state after reviewing a full report."""
        self.abbreviation_checker.end_sweep()
        self.number_checker.end_sweep()
        self.punctuation_checker.end_sweep()

    def check_text(self, section_name: str, text: str) -> List[Violation]:
        """Check for bibliography formatting violations."""
        if "bibliography" not in section_name.lower():
            return []

        violations = []
        violations.extend(self.check_ampersand_usage(section_name, text))
        violations.extend(self.abbreviation_checker.check_et_al(section_name, text))
        violations.extend(self.punctuation_checker.check_range_dashes(section_name, text))
        violations.extend(self.number_checker.check_large_numbers(section_name, text))
        return violations

    def check_ampersand_usage(self, section_name: str, text: str) -> List[Violation]:
        """Flag every `&` in bibliography sections and suggest `and`.

        This rule only runs when `section_name` contains `Bibliography`
        (case-insensitive). In those sections, every literal ampersand
        character is flagged, regardless of surrounding words or reference
        structure.

        Examples flagged in bibliography sections:
        `Smith & Jones 2020`
        `Smith&Jones 2020`
        `<i>Smith</i> <b>&</b> <i>Jones</i> 2020`

        Examples not flagged:
        the same text in non-bibliography sections
        `Smith and Jones 2020`

        What it catches:
        all literal `&` characters in bibliography text, including ones inside
        simple bold or italic markup.

        What it misses:
        non-bibliography sections.
        It does not infer that some other word or symbol should become `and`
        unless an actual `&` character is present.
        """
        violations = []
        if "bibliography" not in section_name.lower():
            return violations

        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )

        for match in re.finditer(r"&", cleaned_text):
            original_start = index_map[match.start()]
            original_end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(original_start, original_end),
                message="Use 'and' not '&' in bibliography author entries",
                suggested_fix="and",
            ))

        return violations
