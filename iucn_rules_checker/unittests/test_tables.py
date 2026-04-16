"""Regression tests for the table-specific checker."""

import unittest

from iucn_rules_checker.checkers.tables import TableChecker


class TableCheckerTests(unittest.TestCase):
    """
    Check the behavior of the table-specific checker.

    Purpose:
        This test case groups regression checks for the current behavior covered by the enclosed tests.
    """

    def test_check_runs_only_et_al_on_table_sections(self) -> None:
        """
        Test that check runs only et al on table sections.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = TableChecker()

        violations = checker.check_text(
            "Assessment > Notes [table 1] [row 1]",
            "Examples include Smith et al. 2020 and e.g. references.",
        )

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].rule_class, "AbbreviationChecker")
        self.assertEqual(violations[0].rule_method, "AbbreviationChecker.check_et_al")
        self.assertEqual(violations[0].message, "Use italicized 'et al.'")
        self.assertEqual(violations[0].section_name, "Assessment > Notes [table 1] [row 1]")

    def test_check_returns_no_violations_for_non_table_sections(self) -> None:
        """
        Test that check returns no violations for non table sections.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = TableChecker()

        violations = checker.check_text(
            "Assessment > Notes [paragraph 1]",
            "Examples include Smith et al. 2020.",
        )

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
