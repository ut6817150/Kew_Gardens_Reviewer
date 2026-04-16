"""Regression tests for spelling behavior."""

import unittest

from iucn_rules_checker.checkers.spelling import SpellingChecker


class SpellingCheckerTests(unittest.TestCase):
    """
    Check the current spelling rules.

    Purpose:
        This test case groups regression checks for the current behavior covered by the enclosed tests.
    """

    def test_ize_words_use_dedicated_iucn_prefers_ize_message(self) -> None:
        """
        Test that ize words use dedicated IUCN prefers ize message.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = SpellingChecker()
        violations = checker.check_text(
            "Assessment > Rationale [paragraph 1]",
            "The programme was organised and then realised."
        )

        ize_violations = [
            violation for violation in violations
            if "IUCN prefers ize spelling" in violation.message
        ]

        self.assertEqual(len(ize_violations), 2)
        self.assertEqual(
            [violation.suggested_fix for violation in ize_violations],
            ["organized", "realized"],
        )

    def test_general_spelling_map_keeps_uk_spelling_message(self) -> None:
        """
        Test that general spelling map keeps UK spelling message.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = SpellingChecker()
        violations = checker.check_text(
            "Assessment > Rationale [paragraph 1]",
            "The color of the flower changed."
        )

        spelling_messages = [
            violation.message for violation in violations
            if "Use UK spelling" in violation.message
        ]

        self.assertEqual(len(spelling_messages), 1)
        self.assertEqual(violations[0].suggested_fix, "colour")

    def test_spelling_checks_ignore_simple_style_tags(self) -> None:
        """
        Test that spelling checks ignore simple style tags.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = SpellingChecker()
        violations = checker.check_text(
            "Assessment > Rationale [paragraph 1]",
            "The <i>col</i><i>or</i> changed and the plant was <b>organised</b> carefully."
        )

        fixes = [violation.suggested_fix for violation in violations]
        messages = [violation.message for violation in violations]

        self.assertIn("colour", fixes)
        self.assertIn("organized", fixes)
        self.assertTrue(any("Use UK spelling" in message for message in messages))
        self.assertTrue(any("IUCN prefers ize spelling" in message for message in messages))

    def test_spelling_checks_ignore_superscript_and_subscript_tags(self) -> None:
        """
        Test that spelling checks ignore superscript and subscript tags.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = SpellingChecker()
        violations = checker.check_text(
            "Assessment > Rationale [paragraph 1]",
            "The <sup>col</sup><sub>or</sub> changed and the plant was "
            "<sup>org</sup><sub>anised</sub> carefully."
        )

        fixes = [violation.suggested_fix for violation in violations]
        messages = [violation.message for violation in violations]

        self.assertIn("colour", fixes)
        self.assertIn("organized", fixes)
        self.assertTrue(any("Use UK spelling" in message for message in messages))
        self.assertTrue(any("IUCN prefers ize spelling" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
