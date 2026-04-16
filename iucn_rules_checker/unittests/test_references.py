"""Regression tests for reference citation behavior."""

import unittest

from iucn_rules_checker.checkers.references import ReferenceChecker


class ReferenceCheckerTests(unittest.TestCase):
    """
    Check the current reference rules.

    Purpose:
        This test case groups regression checks for the current behavior covered by the enclosed tests.
    """

    def test_citation_comma_flags_bracketed_citations_with_final_comma_before_year(self) -> None:
        """
        Test that citation comma flags bracketed citations with final comma before year.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = ReferenceChecker()
        violations = checker.check_text(
            "Assessment > Rationale [paragraph 1]",
            "Examples include (Smith, 2020) and (Mishra et al., 2015)."
        )

        citation_violations = [
            violation for violation in violations
            if "No comma between author and date" in violation.message
        ]

        self.assertEqual(len(citation_violations), 2)
        self.assertEqual(
            [violation.suggested_fix for violation in citation_violations],
            ["(Smith 2020)", "(Mishra et al. 2015)"],
        )

    def test_citation_comma_strips_style_markers_before_matching(self) -> None:
        """
        Test that citation comma strips style markers before matching.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = ReferenceChecker()
        violations = checker.check_text(
            "Assessment > Rationale [paragraph 1]",
            "Examples include (<i>Smith</i>, <b>2020</b>) and "
            "(<sup>Mishra et al.</sup>, <sub>2015</sub>)."
        )

        citation_violations = [
            violation for violation in violations
            if "No comma between author and date" in violation.message
        ]

        self.assertEqual(len(citation_violations), 2)
        self.assertEqual(
            [violation.suggested_fix for violation in citation_violations],
            ["(Smith 2020)", "(Mishra et al. 2015)"],
        )

    def test_citation_comma_skips_square_brackets_unbracketed_and_already_correct_forms(self) -> None:
        """
        Test that citation comma skips square brackets unbracketed and already correct forms.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = ReferenceChecker()
        violations = checker.check_text(
            "Assessment > Rationale [paragraph 1]",
            "Examples include (Smith 2020), [GBIF.org, 2021], and Smith, 2020."
        )

        citation_violations = [
            violation for violation in violations
            if "No comma between author and date" in violation.message
        ]

        self.assertEqual(citation_violations, [])


if __name__ == "__main__":
    unittest.main()
