"""Table-specific checker for IUCN assessments."""

import re
from typing import List

from ..violation import Violation
from .abbreviations import AbbreviationChecker
from .base import BaseChecker


class TableChecker(BaseChecker):
    """
    Checker for table-specific rules.

    Purpose:
        This class groups related rules within the rules-based assessment workflow.
    """

    def __init__(self):
        """
        Initialise the table-specific helper checker.

        Args:
            None.

        Returns:
            None (mutates helper checker attributes in place).

        Notes:
            ``TableChecker`` currently reuses only
            ``AbbreviationChecker.check_et_al(...)`` so parsed table rows can
            run a narrowly scoped table-specific rule set.
        """
        super().__init__()
        self.abbreviation_checker = AbbreviationChecker()

    def begin_sweep(self) -> None:
        """
        Prepare helper checkers before reviewing a full report.

        Args:
            None.

        Returns:
            None: Value produced by this method.
        """
        self.abbreviation_checker.begin_sweep()

    def end_sweep(self) -> None:
        """
        Clear helper checker state after reviewing a full report.

        Args:
            None.

        Returns:
            None: Value produced by this method.
        """
        self.abbreviation_checker.end_sweep()

    def check_text(self, section_name: str, text: str) -> List[Violation]:
        """
        Check table sections with the current table-specific rule set.

        Args:
            section_name (str): Parsed section key supplied by the caller.
            text (str): Parsed section text supplied by the caller.

        Returns:
            List[Violation]: Violations produced by this method.
        """
        if not self.is_table_section(section_name):
            return []

        violations = []
        violations.extend(self.abbreviation_checker.check_et_al(section_name, text))
        return violations

    @staticmethod
    def is_table_section(section_name: str) -> bool:
        """
        Return True when a parsed section key represents table-derived content.

        Args:
            section_name (str): Parsed section key supplied by the caller.

        Returns:
            bool: Boolean result described by the summary line above.
        """
        return re.search(r"\[table\s+\d+\]", section_name, re.IGNORECASE) is not None
