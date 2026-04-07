"""Regression tests for punctuation behavior."""

import unittest

from iucn_rules_checker.checkers.punctuation import PunctuationChecker


class PunctuationCheckerTests(unittest.TestCase):
    """Check the current punctuation rules."""

    def test_range_dashes_use_real_en_dash_in_message_and_fix(self) -> None:
        text = "The assessment covered elevations from 10-20."

        violations = PunctuationChecker().check(("Test Section", text))
        range_violations = [
            violation for violation in violations
            if "numeric ranges" in violation.message
        ]

        self.assertEqual(len(range_violations), 1)
        self.assertIn("\u2013", range_violations[0].message)
        self.assertEqual(range_violations[0].suggested_fix, "10\u201320")

    def test_range_dashes_flag_plain_numeric_ranges_only(self) -> None:
        checker = PunctuationChecker()

        cases = {
            "The transect covered plots 10 \u2013 20.": "10\u201320",
            "The range spans 1990 - 1995.": "1990\u20131995",
            "Average temperatures were 14-26.": "14\u201326",
        }

        for text, expected_fix in cases.items():
            with self.subTest(text=text):
                violations = checker.check(("Test Section", text))
                range_violations = [
                    violation for violation in violations
                    if "numeric ranges" in violation.message
                ]

                self.assertEqual(len(range_violations), 1)
                self.assertEqual(range_violations[0].suggested_fix, expected_fix)

    def test_range_dashes_skip_already_correct_unspaced_en_dash(self) -> None:
        checker = PunctuationChecker()
        texts = [
            "The assessment covered plots 10\u201320.",
            "The range spans 1990\u20131995.",
            "Average temperatures were 14\u201326.",
        ]

        for text in texts:
            with self.subTest(text=text):
                violations = checker.check(("Test Section", text))
                range_violations = [
                    violation for violation in violations
                    if "numeric ranges" in violation.message
                ]
                self.assertEqual(range_violations, [])

    def test_range_dashes_strip_bold_and_italic_markers_before_matching(self) -> None:
        text = "The transect covered plots <b>10</b> - <i>20</i>."

        violations = PunctuationChecker().check(("Test Section", text))
        range_violations = [
            violation for violation in violations
            if "numeric ranges" in violation.message
        ]

        self.assertEqual(len(range_violations), 1)
        self.assertEqual(range_violations[0].suggested_fix, "10\u201320")

    def test_range_dashes_flag_shared_unit_ranges(self) -> None:
        checker = PunctuationChecker()
        cases = {
            "The transect was 10 - 20 km long.": "10\u201320",
            "Elevation ranged from 600-1200 m.": "600\u20131200",
            "Rainfall ranged from 500-3000 mm.": "500\u20133000",
            "Rainfall ranged from 1900-5000 mm.": "1900\u20135000",
            "Average temperatures were 14-26 \u00b0C.": "14\u201326",
        }

        for text, expected_fix in cases.items():
            with self.subTest(text=text):
                violations = checker.check(("Test Section", text))
                range_violations = [
                    violation for violation in violations
                    if "numeric ranges" in violation.message
                ]
                self.assertEqual(len(range_violations), 1)
                self.assertEqual(range_violations[0].suggested_fix, expected_fix)

    def test_range_dashes_ignore_repeated_unit_ranges(self) -> None:
        checker = PunctuationChecker()
        texts = [
            "The transect was 10km - 20 km long.",
            "The transect was 10 km - 20 km long.",
            "Cover declined from 5% - 7%.",
            "The occupied area was 10 km<sup>2</sup> - 20 km<sup>2</sup>.",
            "The occupied area was 10 cm<sup>2</sup> - 20 cm<sup>2</sup>.",
            "The occupied area was 10 ha<sup>2</sup> - 20 ha<sup>2</sup>.",
            "The volume was 10 cm<sup>3</sup> - 20 cm<sup>3</sup>.",
            "Elevation ranged from 10 m asl - 20 m asl.",
            "The occupied area was 10 sq km - 20 sq km.",
            "The occupied area was 10 sq cm - 20 sq cm.",
            "The occupied area was 10sqha - 20 sqha.",
            "The occupied area was 10sqkm - 20 sqkm.",
        ]

        for text in texts:
            with self.subTest(text=text):
                violations = checker.check(("Test Section", text))
                range_violations = [
                    violation for violation in violations
                    if "numeric ranges" in violation.message
                ]
                self.assertEqual(range_violations, [])

    def test_range_dashes_do_not_flag_hyphenated_dates(self) -> None:
        text = (
            "Assessment dates included 2022-08-01, 08-01-2022, and 08-2022-01. "
            "None of these should be treated as numeric ranges."
        )

        violations = PunctuationChecker().check(("Test Section", text))
        range_violations = [
            violation for violation in violations
            if "numeric ranges" in violation.message
        ]

        self.assertEqual(range_violations, [])

    def test_other_punctuation_rules_strip_style_markers_before_matching(self) -> None:
        text = (
            "The species <i>for</i> <b>example</b> occurs in cloud forest. "
            "Altitude <b>:</b> 200 m. "
            "Peru <i>;</i> Ecuador."
        )

        violations = PunctuationChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        self.assertIn("'for example' should be preceded by a comma", messages)
        self.assertIn("'for example' should be followed by a comma", messages)
        self.assertIn("Do not put a space before a colon", messages)
        self.assertIn("Do not put a space before a semicolon", messages)

    def test_for_example_ignores_sentence_start_after_sentence_ending_punctuation(self) -> None:
        text = (
            "This changed. For example, the species occurs in cloud forest. "
            "Was it revised? For example, another site was added. "
            "This surprised us! For example, a third site appeared."
        )

        violations = PunctuationChecker().check_for_example_commas("Test Section", text)

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
