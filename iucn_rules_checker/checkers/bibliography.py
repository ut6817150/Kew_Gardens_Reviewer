"""Bibliography-specific formatting checker for IUCN assessments."""

import re
from typing import List

from ..violation import Violation
from .abbreviations import AbbreviationChecker
from .base import BaseChecker
from .punctuation import PunctuationChecker


class BibliographyChecker(BaseChecker):
    """
    Checker for bibliography-specific formatting rules.

    Purpose:
        This class groups related rules within the rules-based assessment workflow.
    """

    def __init__(self):
        """
        Initialise bibliography-specific helper checkers.

        Args:
            None.

        Returns:
            None (mutates helper checker attributes in place).

        Notes:
            ``BibliographyChecker`` reuses focused abbreviation and punctuation
            rules so bibliography sections can be checked without running the
            entire non-bibliography checker pipeline.
        """
        super().__init__()
        self.abbreviation_checker = AbbreviationChecker()
        self.punctuation_checker = PunctuationChecker()

    def begin_sweep(self) -> None:
        """
        Prepare helper checkers before reviewing a full report.

        Args:
            None.

        Returns:
            None: Value produced by this method.
        """
        self.abbreviation_checker.begin_sweep()
        self.punctuation_checker.begin_sweep()

    def end_sweep(self) -> None:
        """
        Clear helper checker state after reviewing a full report.

        Args:
            None.

        Returns:
            None: Value produced by this method.
        """
        self.abbreviation_checker.end_sweep()
        self.punctuation_checker.end_sweep()

    def check_text(self, section_name: str, text: str) -> List[Violation]:
        """
        Check for bibliography formatting violations.

        Args:
            section_name (str): Parsed section key supplied by the caller.
            text (str): Parsed section text supplied by the caller.

        Returns:
            List[Violation]: Violations produced by this method.
        """
        if "bibliography" not in section_name.lower():
            return []

        violations = []
        violations.extend(self.check_ampersand_usage(section_name, text))
        violations.extend(self.abbreviation_checker.check_et_al(section_name, text))
        violations.extend(self.punctuation_checker.check_range_dashes(section_name, text))
        return violations

    def check_ampersand_usage(self, section_name: str, text: str) -> List[Violation]:
        """
        Flag `&` in bibliography author text and suggest `and`.

        This rule only runs when `section_name` contains `Bibliography`
        (case-insensitive). In those sections, it first clips the cleaned text
        to the portion before the first standalone four-digit year such as
        `2020`, then flags every literal ampersand character in that clipped
        author portion only.

        Examples flagged in bibliography sections:
        `Smith & Jones 2020`
        `Smith&Jones 2020`
        `<i>Smith</i> <b>&</b> <i>Jones</i> 2020`

        Examples not flagged:
        the same text in non-bibliography sections
        `Smith and Jones 2020`
        `Smith 2020 & Brown 2021`

        What it catches:
        literal `&` characters that appear in the bibliography text before the
        first standalone four-digit year, including ones inside simple bold or
        italic markup.

        What it misses:
        non-bibliography sections.
        Ampersands that appear after the first standalone four-digit year are
        intentionally ignored.
        It does not infer that some other word or symbol should become `and`
        unless an actual `&` character is present.

        Args:
            section_name (str): Parsed section key supplied by the caller.
            text (str): Parsed section text supplied by the caller.

        Returns:
            List[Violation]: Violations produced by this method.
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
        year_match = re.search(r"\b\d{4}\b", cleaned_text)
        if year_match is not None:
            cleaned_text = cleaned_text[:year_match.start()]

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
