"""Regression tests for bibliography-formatting behavior."""

import unittest

from iucn_rules_checker.checkers.bibliography import BibliographyChecker


class BibliographyCheckerTests(unittest.TestCase):
    """
    Check the current bibliography rules.

    Purpose:
        This test case groups regression checks for the current behavior covered by the enclosed tests.
    """

    def test_ampersand_usage_only_runs_in_bibliography_sections(self) -> None:
        """
        Test that ampersand usage only runs in bibliography sections.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = BibliographyChecker()

        bibliography_violations = checker.check_text(
            "Assessment > Bibliography [paragraph 1]",
            "Smith & Jones 2020. Example reference."
        )
        body_violations = checker.check_text(
            "Assessment > Rationale [paragraph 1]",
            "Smith & Jones 2020 discussed the species."
        )

        ampersand_messages = [
            violation.message for violation in bibliography_violations
            if "Use 'and' not '&'" in violation.message
        ]
        body_ampersand_messages = [
            violation.message for violation in body_violations
            if "Use 'and' not '&'" in violation.message
        ]

        self.assertEqual(len(ampersand_messages), 1)
        self.assertEqual(body_ampersand_messages, [])

    def test_ampersand_usage_only_flags_ampersands_before_first_year(self) -> None:
        """
        Test that ampersand usage only flags ampersands before first year.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = BibliographyChecker()

        violations = checker.check_text(
            "Assessment > Bibliography [paragraph 1]",
            "<i>Smith</i> <b>&</b> <i>Jones</i> 2020 & Brown 2021."
        )

        ampersand_violations = [
            violation for violation in violations
            if "Use 'and' not '&'" in violation.message
        ]

        self.assertEqual(len(ampersand_violations), 1)
        self.assertTrue(all(v.suggested_fix == "and" for v in ampersand_violations))
        self.assertEqual(ampersand_violations[0].matched_text, "&")

    def test_ampersand_usage_ignores_all_ampersands_after_first_year(self) -> None:
        """
        Test that ampersand usage ignores all ampersands after first year.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = BibliographyChecker()

        violations = checker.check_text(
            "Assessment > Bibliography [paragraph 1]",
            "Smith 2020 & Brown 2021."
        )

        ampersand_violations = [
            violation for violation in violations
            if "Use 'and' not '&'" in violation.message
        ]

        self.assertEqual(ampersand_violations, [])

    def test_bibliography_checker_also_runs_et_al_rule(self) -> None:
        """
        Test that bibliography checker also runs et al rule.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = BibliographyChecker()

        violations = checker.check_text(
            "Assessment > Bibliography [paragraph 1]",
            "Mishra et al. 2015. Example reference."
        )

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].rule_class, "AbbreviationChecker")
        self.assertEqual(violations[0].rule_method, "AbbreviationChecker.check_et_al")
        self.assertEqual(violations[0].message, "Use italicized 'et al.'")

    def test_bibliography_checker_also_runs_range_dash_rule(self) -> None:
        """
        Test that bibliography checker also runs range dash rule.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = BibliographyChecker()

        violations = checker.check_text(
            "Assessment > Bibliography [paragraph 1]",
            "Smith 2020. Journal 10-20."
        )

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].rule_class, "PunctuationChecker")
        self.assertEqual(violations[0].rule_method, "PunctuationChecker.check_range_dashes")
        self.assertIn("Use an unspaced en dash", violations[0].message)

    def test_bibliography_checker_does_not_run_large_number_rule(self) -> None:
        """
        Test that bibliography checker does not run large number rule.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = BibliographyChecker()

        violations = checker.check_text(
            "Assessment > Bibliography [paragraph 1]",
            "Smith 2020. Flora 5000 species."
        )

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
