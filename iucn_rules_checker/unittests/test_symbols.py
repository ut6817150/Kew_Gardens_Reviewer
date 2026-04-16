"""Regression tests for symbol and unit formatting behavior."""

import unittest

from iucn_rules_checker.checkers.symbols import SymbolChecker


class SymbolCheckerTests(unittest.TestCase):
    """
    Check the current symbol rules.

    Purpose:
        This test case groups regression checks for the current behavior covered by the enclosed tests.
    """

    def test_ampersand_usage_flags_literal_ampersands(self) -> None:
        """
        Test that ampersand usage flags literal ampersands.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = SymbolChecker()
        violations = checker.check_text(
            "Assessment > Rationale [paragraph 1]",
            "The habitat includes forest & woodland and grassland & wetland.",
        )

        ampersand_violations = [
            violation for violation in violations
            if violation.message == "Use 'and' not '&'"
        ]

        self.assertEqual(len(ampersand_violations), 2)
        self.assertTrue(all(v.suggested_fix == "and" for v in ampersand_violations))

    def test_ampersand_usage_ignores_simple_style_tags(self) -> None:
        """
        Test that ampersand usage ignores simple style tags.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = SymbolChecker()
        violations = checker.check_text(
            "Assessment > Rationale [paragraph 1]",
            "<i>forest</i> <b>&</b> woodland",
        )

        ampersand_violations = [
            violation for violation in violations
            if violation.message == "Use 'and' not '&'"
        ]

        self.assertEqual(len(ampersand_violations), 1)
        self.assertEqual(ampersand_violations[0].suggested_fix, "and")

    def test_ampersand_usage_skips_assessment_information_sections(self) -> None:
        """
        Test that ampersand usage skips assessment information sections.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = SymbolChecker()
        violations = checker.check_text(
            "Assessment > Assessment Information [paragraph 1]",
            "The habitat includes forest & woodland.",
        )

        ampersand_violations = [
            violation for violation in violations
            if violation.message == "Use 'and' not '&'"
        ]

        self.assertEqual(ampersand_violations, [])

    def test_area_units_ignore_simple_style_tags(self) -> None:
        """
        Test that area units ignore simple style tags.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = SymbolChecker()
        violations = checker.check_text(
            "Assessment > Rationale [paragraph 1]",
            "Area measured <i>sq</i> <i>km</i>, <b>km2</b>, <i>m</i><i>2</i>, <b>cm2</b>, and <i>mm</i><i>2</i>."
        )

        area_messages = [
            violation.message for violation in violations
            if "sqkm" in violation.message
            or "'km2'" in violation.message
            or "'m2'" in violation.message
            or "'cm2'" in violation.message
            or "'mm2'" in violation.message
        ]
        fixes = [violation.suggested_fix for violation in violations]

        self.assertEqual(len(area_messages), 5)
        self.assertIn("km²", fixes)
        self.assertIn("m²", fixes)
        self.assertIn("cm²", fixes)
        self.assertIn("mm²", fixes)

    def test_area_units_respect_existing_correct_forms_inside_style_tags(self) -> None:
        """
        Test that area units respect existing correct forms inside style tags.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = SymbolChecker()
        violations = checker.check_text(
            "Assessment > Rationale [paragraph 1]",
            "Area measured <i>km²</i> and <b>m²</b>."
        )

        area_messages = [
            violation.message for violation in violations
            if "sqkm" in violation.message
            or "'km2'" in violation.message
            or "'m2'" in violation.message
            or "'cm2'" in violation.message
            or "'mm2'" in violation.message
        ]

        self.assertEqual(area_messages, [])

    def test_degree_text_ignores_simple_style_tags_and_allows_decimals(self) -> None:
        """
        Test that degree text ignores simple style tags and allows decimals.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = SymbolChecker()
        violations = checker.check_degree_text(
            "Assessment > Rationale [paragraph 1]",
            "Coordinates were <i>12</i><i>.5</i> degrees n and temperature was <b>20.75 degrees Celsius</b>.",
        )

        degree_messages = [
            violation.message for violation in violations
            if "Use degree symbol" in violation.message
        ]
        fixes = [violation.suggested_fix for violation in violations]

        self.assertEqual(len(degree_messages), 2)
        self.assertIn("12.5°N", fixes)
        self.assertIn("20.75°C", fixes)

    def test_degree_text_skips_existing_correct_decimal_forms(self) -> None:
        """
        Test that degree text skips existing correct decimal forms.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = SymbolChecker()
        violations = checker.check_degree_text(
            "Assessment > Rationale [paragraph 1]",
            "Coordinates were <i>12.5°N</i> and temperature was <b>20.75°C</b>.",
        )

        degree_messages = [
            violation.message for violation in violations
            if "Use degree symbol" in violation.message
        ]

        self.assertEqual(degree_messages, [])

    def test_degree_symbol_spacing_flags_spacing_around_existing_degree_symbol(self) -> None:
        """
        Test that degree symbol spacing flags spacing around existing degree symbol.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = SymbolChecker()
        violations = checker.check_degree_symbol_spacing(
            "Assessment > Rationale [paragraph 1]",
            "Temperature was 12 °C and coordinates were 12.5 ° N.",
        )

        spacing_violations = [
            violation for violation in violations
            if "No spaces around degree symbol" in violation.message
        ]
        fixes = [violation.suggested_fix for violation in spacing_violations]

        self.assertEqual(len(spacing_violations), 2)
        self.assertIn("12°C", fixes)
        self.assertIn("12.5°N", fixes)

    def test_degree_symbol_spacing_skips_existing_no_space_degree_symbol_forms(self) -> None:
        """
        Test that degree symbol spacing skips existing no space degree symbol forms.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = SymbolChecker()
        violations = checker.check_degree_symbol_spacing(
            "Assessment > Rationale [paragraph 1]",
            "Temperature was 12°C and coordinates were 12.5°N.",
        )

        spacing_violations = [
            violation for violation in violations
            if "No spaces around degree symbol" in violation.message
        ]

        self.assertEqual(spacing_violations, [])

    def test_degree_symbol_spacing_skips_range_preceded_shared_unit_forms(self) -> None:
        """
        Test that degree symbol spacing skips range preceded shared unit forms.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = SymbolChecker()
        violations = checker.check_degree_symbol_spacing(
            "Assessment > Rationale [paragraph 1]",
            "Temperature ranged from 14–26 °C, from 14—26 °C, and bearings ranged from 10-12 ° N.",
        )

        spacing_violations = [
            violation for violation in violations
            if "No spaces around degree symbol" in violation.message
        ]

        self.assertEqual(spacing_violations, [])

    def test_percentage_ignores_simple_style_tags_and_allows_decimals(self) -> None:
        """
        Test that percentage ignores simple style tags and allows decimals.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = SymbolChecker()
        violations = checker.check_text(
            "Assessment > Rationale [paragraph 1]",
            "Success was <i>12</i><i>.5</i> percent and failure was <b>7.25 per cent</b>."
        )

        percentage_violations = [
            violation for violation in violations
            if "Use '%' symbol" in violation.message
        ]
        fixes = [violation.suggested_fix for violation in percentage_violations]

        self.assertEqual(len(percentage_violations), 2)
        self.assertIn("12.5%", fixes)
        self.assertIn("7.25%", fixes)

    def test_percentage_skips_existing_percent_symbol_forms(self) -> None:
        """
        Test that percentage skips existing percent symbol forms.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = SymbolChecker()
        violations = checker.check_text(
            "Assessment > Rationale [paragraph 1]",
            "Success was <i>12.5%</i> and failure was <b>7.25%</b>."
        )

        percentage_violations = [
            violation for violation in violations
            if "Use '%' symbol" in violation.message
        ]

        self.assertEqual(percentage_violations, [])

    def test_percentage_symbol_spacing_ignores_simple_style_tags(self) -> None:
        """
        Test that percentage symbol spacing ignores simple style tags.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = SymbolChecker()
        violations = checker.check_text(
            "Assessment > Rationale [paragraph 1]",
            "Success was <i>12</i> <b>%</b>."
        )

        spacing_violations = [
            violation for violation in violations
            if violation.message == "No space before '%'"
        ]

        self.assertEqual(len(spacing_violations), 1)
        self.assertEqual(spacing_violations[0].suggested_fix, "12%")

    def test_percentage_symbol_spacing_skips_range_preceded_percent_and_unit_forms(self) -> None:
        """
        Test that percentage symbol spacing skips range preceded percent and unit forms.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = SymbolChecker()
        violations = checker.check_percentage_symbol_spacing(
            "Assessment > Rationale [paragraph 1]",
            "Success ranged from 12–15 %, from 12—15 %, and elevation ranged from 600-1200m.",
        )

        spacing_violations = [
            violation for violation in violations
            if violation.message == "No space before '%'"
            or "Add space between number and unit" in violation.message
        ]

        self.assertEqual(spacing_violations, [])

    def test_unit_spacing_ignores_simple_style_tags(self) -> None:
        """
        Test that unit spacing ignores simple style tags.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = SymbolChecker()
        violations = checker.check_text(
            "Assessment > Rationale [paragraph 1]",
            "Distance was <i>12</i><b>km</b>."
        )

        unit_violations = [
            violation for violation in violations
            if "Add space between number and unit" in violation.message
        ]

        self.assertEqual(len(unit_violations), 1)
        self.assertEqual(unit_violations[0].suggested_fix, "12 km")


if __name__ == "__main__":
    unittest.main()
