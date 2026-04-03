"""Regression tests for number-formatting behavior."""

import unittest

from iucn_rules_checker.checkers.numbers import NumberChecker


class NumberCheckerTests(unittest.TestCase):
    """Check the current small-number matching rules."""

    def test_small_numbers_flag_plain_prose_numerals(self) -> None:
        text = "There were 3 locations and 2 subpopulations across 4 valleys"

        violations = NumberChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        self.assertIn("Numbers 1-9 should be written out: '3' should be 'three'", messages)
        self.assertIn("Numbers 1-9 should be written out: '2' should be 'two'", messages)
        self.assertIn("Numbers 1-9 should be written out: '4' should be 'four'", messages)

    def test_small_numbers_skip_dates_units_ranges_and_percentages(self) -> None:
        text = (
            "The survey took place on 3 May. "
            "Plots were 5 km apart and covered 7 ha. "
            "Humidity reached 6% and the slope was 8\u00B0. "
            "Counts ranged from 4-5 individuals or 2 \u2013 3 pairs. "
            "Decimal values such as 1.5 were also recorded."
        )

        violations = NumberChecker().check(("Test Section", text))
        small_number_messages = [
            violation.message for violation in violations
            if violation.message.startswith("Numbers 1-9 should be written out:")
        ]

        self.assertEqual(small_number_messages, [])

    def test_small_numbers_skip_extended_unit_forms(self) -> None:
        text = (
            "The range was 3 sq km and another estimate used 4 sqkm. "
            "Elevation changed by 5 m asl and 6 m bsl. "
            "Containers held 7 l and 8 ml. "
            "Biomass was 9 g and another value was 2 t. "
            "Volume reached 3 m3 and area reached 4 km2."
        )

        violations = NumberChecker().check(("Test Section", text))
        small_number_messages = [
            violation.message for violation in violations
            if violation.message.startswith("Numbers 1-9 should be written out:")
        ]

        self.assertEqual(small_number_messages, [])

    def test_small_numbers_strip_style_markers_before_matching(self) -> None:
        text = (
            "There were <b>3</b> locations. "
            "The area was 4 km<sup>2</sup> and another patch was 5 m<sub>2</sub>."
        )

        violations = NumberChecker().check(("Test Section", text))
        messages = [
            violation.message for violation in violations
            if violation.message.startswith("Numbers 1-9 should be written out:")
        ]

        self.assertEqual(
            messages,
            ["Numbers 1-9 should be written out: '3' should be 'three'"],
        )

    def test_small_numbers_still_work_when_called_directly_on_bibliography_sections(self) -> None:
        text = "There were 3 records and 2 notes in the bibliography."

        violations = NumberChecker().check(("Assessment > Bibliography [paragraph 1]", text))
        messages = [
            violation.message for violation in violations
            if violation.message.startswith("Numbers 1-9 should be written out:")
        ]

        self.assertEqual(
            messages,
            [
                "Numbers 1-9 should be written out: '3' should be 'three'",
                "Numbers 1-9 should be written out: '2' should be 'two'",
            ],
        )

    def test_very_large_numbers_flag_clean_millions_and_billions_only(self) -> None:
        text = (
            "Rounded values include 1000000, 1500000, 1570000, 25000000, "
            "1500000000, 1570000000, and 3000000000. "
            "Precise values include 1234567 and 9876543210."
        )

        violations = NumberChecker().check(("Test Section", text))
        messages = [
            violation.message for violation in violations
            if violation.message.startswith("For rounded numbers >= 1 million")
        ]

        self.assertTrue(any("1000000" in message and "1 million" in message for message in messages))
        self.assertTrue(any("1500000" in message and "1.5 million" in message for message in messages))
        self.assertTrue(any("1570000" in message and "1.57 million" in message for message in messages))
        self.assertTrue(any("25000000" in message and "25 million" in message for message in messages))
        self.assertTrue(any("1500000000" in message and "1.5 billion" in message for message in messages))
        self.assertTrue(any("1570000000" in message and "1.57 billion" in message for message in messages))
        self.assertTrue(any("3000000000" in message and "3 billion" in message for message in messages))
        self.assertFalse(any("1234567" in message for message in messages))
        self.assertFalse(any("9876543210" in message for message in messages))

    def test_very_large_numbers_strip_style_markers_before_matching(self) -> None:
        text = (
            "Rounded values include <b>1500000</b> and <i>3000000000</i>. "
            "A precise value of <sup>1234567</sup> should still be ignored."
        )

        violations = NumberChecker().check(("Test Section", text))
        messages = [
            violation.message for violation in violations
            if violation.message.startswith("For rounded numbers >= 1 million")
        ]

        self.assertTrue(any("1500000" in message and "1.5 million" in message for message in messages))
        self.assertTrue(any("3000000000" in message and "3 billion" in message for message in messages))
        self.assertFalse(any("1234567" in message for message in messages))

    def test_large_numbers_strip_style_markers_and_skip_very_large_candidates(self) -> None:
        text = (
            "There were <b>1234</b> records. "
            "A rounded total of <i>1500000</i> individuals was also reported."
        )

        violations = NumberChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]
        large_number_messages = [
            message for message in messages
            if message.startswith("Use standard comma placement for numbers:")
        ]

        self.assertIn(
            "Use standard comma placement for numbers: '1234' should be '1,234'",
            large_number_messages,
        )
        self.assertFalse(any("1500000" in message for message in large_number_messages))
        self.assertTrue(any(
            message.startswith("For rounded numbers >= 1 million") and "1500000" in message
            for message in messages
        ))

    def test_large_numbers_flag_bad_comma_grouping_and_decimal_grouping(self) -> None:
        text = (
            "Misgrouped values included 12,34, 123,4, 1234.56, 12,34.56, and 1,23. "
            "Correct values included 1,234 and 1,234.56."
        )

        violations = NumberChecker().check(("Test Section", text))
        large_number_messages = [
            violation.message for violation in violations
            if violation.message.startswith("Use standard comma placement for numbers:")
        ]

        self.assertIn(
            "Use standard comma placement for numbers: '12,34' should be '1,234'",
            large_number_messages,
        )
        self.assertIn(
            "Use standard comma placement for numbers: '123,4' should be '1,234'",
            large_number_messages,
        )
        self.assertIn(
            "Use standard comma placement for numbers: '1234.56' should be '1,234.56'",
            large_number_messages,
        )
        self.assertIn(
            "Use standard comma placement for numbers: '12,34.56' should be '1,234.56'",
            large_number_messages,
        )
        self.assertIn(
            "Use standard comma placement for numbers: '1,23' should be '123'",
            large_number_messages,
        )
        self.assertFalse(any("'1,234' should be" in message for message in large_number_messages))
        self.assertFalse(any("'1,234.56' should be" in message for message in large_number_messages))

    def test_sentence_start_strips_simple_style_markers_before_matching(self) -> None:
        text = (
            "<b>3</b> sites were surveyed. "
            "Mid-sentence values like <i>4</i> should not count here. "
            "Was it revised? <sup>12</sup> records were added."
        )

        violations = NumberChecker().check(("Test Section", text))
        messages = [
            violation.message for violation in violations
            if violation.message == "Do not start sentences with numerals; write the number out or rephrase"
        ]

        self.assertEqual(len(messages), 2)

    def test_sentence_start_ignores_years_preceded_by_et_al(self) -> None:
        text = (
            "Martinez et al. 2006 described the site. "
            "3 surveys were completed later."
        )

        violations = NumberChecker().check(("Test Section", text))
        matched_texts = [
            violation.matched_text for violation in violations
            if violation.message == "Do not start sentences with numerals; write the number out or rephrase"
        ]

        self.assertEqual(matched_texts, ["3"])

    def test_sentence_start_still_works_when_called_directly_on_bibliography_sections(self) -> None:
        text = "3 entries were reviewed. Martinez et al. 2006 described the site."

        violations = NumberChecker().check(("Assessment > Bibliography [paragraph 1]", text))
        messages = [
            violation.message for violation in violations
            if violation.message == "Do not start sentences with numerals; write the number out or rephrase"
        ]

        self.assertEqual(
            messages,
            ["Do not start sentences with numerals; write the number out or rephrase"],
        )


if __name__ == "__main__":
    unittest.main()
