"""Regression tests for geography-country matching behavior."""

import unittest

from iucn_rules_checker.checkers.geography import GeographyChecker


class GeographyCheckerTests(unittest.TestCase):
    """Check the current ISO-country correction behavior."""

    def test_country_names_match_known_non_preferred_forms(self) -> None:
        text = (
            "The species occurs in Vietnam and Laos. "
            "Older records also mention Burma and Holland."
        )

        violations = GeographyChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        self.assertIn("Use ISO 3166 country name: 'Viet Nam' instead of 'Vietnam'", messages)
        self.assertIn("Use ISO 3166 country name: 'Lao PDR' instead of 'Laos'", messages)
        self.assertIn("Use ISO 3166 country name: 'Myanmar' instead of 'Burma'", messages)
        self.assertIn("Use ISO 3166 country name: 'Netherlands' instead of 'Holland'", messages)

    def test_country_names_catch_common_misspellings_in_geographic_context(self) -> None:
        text = (
            "The species was found in Afganistan. "
            "It was later recorded from Argetina and distributed in Phillipines."
        )

        violations = GeographyChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        self.assertIn("Use ISO 3166 country name: 'Afghanistan' instead of 'Afganistan'", messages)
        self.assertIn("Use ISO 3166 country name: 'Argentina' instead of 'Argetina'", messages)
        self.assertIn("Use ISO 3166 country name: 'Philippines' instead of 'Phillipines'", messages)

    def test_exact_iso_country_names_are_not_flagged(self) -> None:
        text = (
            "The species is found in Afghanistan, Argentina, Viet Nam, and Timor-Leste. "
            "It also occurs in the Republic of the Congo."
        )

        violations = GeographyChecker().check(("Test Section", text))

        self.assertEqual(violations, [])

    def test_country_names_strip_simple_style_markers_before_matching(self) -> None:
        text = (
            "The species occurs in <b>Viet</b><i>nam</i>. "
            "Older records also mention <sup>La</sup><sub>os</sub>."
        )

        violations = GeographyChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        self.assertIn("Use ISO 3166 country name: 'Viet Nam' instead of 'Vietnam'", messages)
        self.assertIn("Use ISO 3166 country name: 'Lao PDR' instead of 'Laos'", messages)

    def test_directional_capitalization_flags_non_region_direction_phrases(self) -> None:
        text = (
            "The species occurs in Eastern Ecuador and Northern Peru. "
            "It is also found across Western Colombia."
        )

        violations = GeographyChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        self.assertIn("'Eastern Ecuador' should be lower case (unless it is a proper region name)", messages)
        self.assertIn("'Northern Peru' should be lower case (unless it is a proper region name)", messages)
        self.assertIn("'Western Colombia' should be lower case (unless it is a proper region name)", messages)

    def test_directional_capitalization_skips_iso_countries_and_recognised_regions(self) -> None:
        text = (
            "The review mentions North Korea, South Africa, North America, East Asia, "
            "South & Southeast Asia, and West & Central Asia."
        )

        violations = GeographyChecker().check(("Test Section", text))
        direction_messages = [
            violation.message for violation in violations
            if "should be lower case" in violation.message
        ]

        self.assertEqual(direction_messages, [])

    def test_directional_capitalization_strips_simple_style_markers_before_matching(self) -> None:
        text = (
            "The species occurs in <b>East</b><i>ern</i> Ecuador. "
            "It is also found in <i>North</i> America."
        )

        violations = GeographyChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        self.assertIn("'Eastern Ecuador' should be lower case (unless it is a proper region name)", messages)
        self.assertNotIn("'North America' should be lower case (unless it is a proper region name)", messages)


if __name__ == "__main__":
    unittest.main()
