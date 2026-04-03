"""Regression tests for date-formatting behavior."""

import unittest

from iucn_rules_checker.checkers.dates import DateChecker


class DateCheckerTests(unittest.TestCase):
    """Check the current ordinal-date matching rules."""

    def test_ordinal_dates_match_full_and_abbreviated_months(self) -> None:
        text = (
            "The review was completed on 11th January. "
            "Another draft was dated 3rd of May. "
            "Notes also mention 21st Jan and 22nd Feb. "
            "A final update cites 4th Sept. and 5th Dec."
        )

        violations = DateChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        self.assertEqual(len(violations), 6)
        self.assertTrue(any("11 January" in message for message in messages))
        self.assertTrue(any("3 May" in message for message in messages))
        self.assertTrue(any("21 Jan" in message for message in messages))
        self.assertTrue(any("22 Feb." in message for message in messages))
        self.assertTrue(any("4 Sept." in message for message in messages))
        self.assertTrue(any("5 Dec." in message for message in messages))

    def test_invalid_ordinal_dates_are_flagged_as_invalid_dates(self) -> None:
        text = (
            "An impossible draft date was 31st April. "
            "Another record mentioned 30th Feb. "
            "A leap-day note said 29th February. "
            "A malformed note even said 0th January."
        )

        violations = DateChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        self.assertEqual(len(violations), 4)
        self.assertTrue(any("31st April" in message and "maximum 30 days" in message for message in messages))
        self.assertTrue(any("30th Feb." in message and "maximum 28 days" in message for message in messages))
        self.assertTrue(any("29th February" in message and "Check for leap year" in message for message in messages))
        self.assertTrue(any("0th January" in message and "maximum 31 days" in message for message in messages))
        self.assertFalse(any("Avoid ordinal dates; use '31 April'" in message for message in messages))

    def test_century_format_matches_hyphenated_and_unhyphenated_forms(self) -> None:
        text = (
            "The record spans the twenty-first century. "
            "A second note refers to the twenty first century."
        )

        violations = DateChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        self.assertEqual(len(violations), 2)
        self.assertTrue(all("21st century" in message for message in messages))

    def test_date_rules_ignore_simple_style_tags(self) -> None:
        text = (
            "The review was completed on <i>11th</i> <b>January</b>. "
            "A note refers to the <i>twenty-first</i> <b>century</b>. "
            "Another draft mentions <i>1980</i><b>'s</b> records."
        )

        violations = DateChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        self.assertTrue(any("11 January" in message for message in messages))
        self.assertTrue(any("21st century" in message for message in messages))
        self.assertTrue(any("1980s" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
