"""Regression tests for scientific-name formatting behavior."""

import unittest

from iucn_rules_checker.checkers.scientific import ScientificNameChecker


class ScientificNameCheckerTests(unittest.TestCase):
    """
    Check the current scientific-name rules.

    Purpose:
        This test case groups regression checks for the current behavior covered by the enclosed tests.
    """

    def test_species_abbreviations_ignore_simple_style_tags(self) -> None:
        """
        Test that species abbreviations ignore simple style tags.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = ScientificNameChecker()
        violations = checker.check_text(
            "Assessment > Rationale [paragraph 1]",
            "Observed <i>s</i><i>p</i> and <b>spp</b> in cultivation."
        )

        abbreviation_messages = [
            violation.message for violation in violations
            if "Use 'sp." in violation.message or "Use 'spp." in violation.message
        ]
        suggested_fixes = [violation.suggested_fix for violation in violations]

        self.assertEqual(len(abbreviation_messages), 2)
        self.assertIn("sp.", suggested_fixes)
        self.assertIn("spp.", suggested_fixes)

    def test_species_abbreviations_respect_existing_periods_inside_style_tags(self) -> None:
        """
        Test that species abbreviations respect existing periods inside style tags.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = ScientificNameChecker()
        violations = checker.check_text(
            "Assessment > Rationale [paragraph 1]",
            "Observed <i>sp.</i> and <b>spp.</b> in cultivation."
        )

        abbreviation_messages = [
            violation.message for violation in violations
            if "Use 'sp." in violation.message or "Use 'spp." in violation.message
        ]

        self.assertEqual(abbreviation_messages, [])

    def test_species_abbreviations_strip_superscript_and_subscript_tags(self) -> None:
        """
        Test that species abbreviations strip superscript and subscript tags.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = ScientificNameChecker()
        violations = checker.check_text(
            "Assessment > Rationale [paragraph 1]",
            "Observed <sup>s</sup><sub>p</sub> and <sup>spp</sup> in cultivation, "
            "but not <sub>sp.</sub>."
        )

        abbreviation_messages = [
            violation.message for violation in violations
            if "Use 'sp." in violation.message or "Use 'spp." in violation.message
        ]
        suggested_fixes = [violation.suggested_fix for violation in violations]

        self.assertEqual(len(abbreviation_messages), 2)
        self.assertIn("sp.", suggested_fixes)
        self.assertIn("spp.", suggested_fixes)


if __name__ == "__main__":
    unittest.main()
