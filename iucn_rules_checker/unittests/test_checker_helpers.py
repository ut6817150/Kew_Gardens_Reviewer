"""Focused helper and direct-method tests for checker modules."""

import unittest

from iucn_rules_checker.checkers.abbreviations import AbbreviationChecker
from iucn_rules_checker.checkers.bibliography import BibliographyChecker
from iucn_rules_checker.checkers.dates import DateChecker
from iucn_rules_checker.checkers.formatting import FormattingChecker
from iucn_rules_checker.checkers.geography import GeographyChecker
from iucn_rules_checker.checkers.iucn_terms import IUCNTermsChecker
from iucn_rules_checker.checkers.numbers import NumberChecker
from iucn_rules_checker.checkers.punctuation import PunctuationChecker
from iucn_rules_checker.checkers.references import ReferenceChecker
from iucn_rules_checker.checkers.scientific import ScientificNameChecker
from iucn_rules_checker.checkers.spelling import SpellingChecker
from iucn_rules_checker.checkers.symbols import SymbolChecker


class CheckerHelperTests(unittest.TestCase):
    """
    Exercise helper methods and direct per-rule methods explicitly.

    Purpose:
        This test case groups regression checks for the current behavior covered by the enclosed tests.
    """

    def test_abbreviation_helper_methods_and_direct_rules(self) -> None:
        """
        Test that abbreviation helper methods and direct rules.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = AbbreviationChecker()

        italic_text = "Use <i>et al</i>. and <i>in situ</i>."
        et_start = italic_text.index("et")
        latin_start = italic_text.index("in situ")

        self.assertTrue(checker.is_inside_italic(italic_text, et_start, et_start + 5))
        self.assertTrue(checker.is_inside_italic(italic_text, latin_start, latin_start + 7))
        self.assertFalse(checker.is_inside_italic("plain text", 0, 5))
        self.assertEqual(
            checker.strip_italic_markup_around_term("<i>in situ</i>.", 3, 10),
            "in situ.",
        )

        self.assertEqual(
            len(checker.check_eg_and_ie("Test Section", "E.g. one note and ie another.")),
            2,
        )
        self.assertEqual(
            len(checker.check_latin_terms_without_period("Test Section", "Plain in situ only.")),
            1,
        )
        self.assertEqual(
            len(checker.check_title_abbreviations("Test Section", "Dr. Green met Mr. Brown.")),
            2,
        )

    def test_date_helper_methods_and_direct_rules(self) -> None:
        """
        Test that date helper methods and direct rules.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = DateChecker()

        self.assertEqual(checker.normalize_month_token("Sept."), "sept")
        self.assertTrue(checker.is_valid_day_month(30, "Apr."))
        self.assertFalse(checker.is_valid_day_month(31, "Apr."))

        self.assertEqual(
            len(checker.check_ordinal_dates("Test Section", "The date was 11th January.")),
            1,
        )
        self.assertEqual(
            len(checker.check_century_format("Test Section", "It spans the twenty first century.")),
            1,
        )
        self.assertEqual(
            len(checker.check_decade_format("Test Section", "The 1980's records were reviewed.")),
            1,
        )

    def test_formatting_helper_methods_and_state_reset(self) -> None:
        """
        Test that formatting helper methods and state reset.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = FormattingChecker()

        self.assertTrue(
            checker.collect_taxonomy_names_from_ladder(
                "PLANTAE - TRACHEOPHYTA - LILIOPSIDA - PANDANALES - PANDANACEAE - Benstonea - celebica"
            )
        )
        self.assertIn("Plantae", checker._collected_higher_taxonomy_names)
        self.assertIn("Liliopsida", checker._collected_higher_taxonomy_names)
        self.assertEqual(checker._collected_genus_name, "Benstonea")
        self.assertEqual(checker._collected_species_name, "celebica")
        self.assertTrue(checker.is_inside_italic("<i>Benstonea</i>", 3, 12))

        taxonomy_violations = checker.find_taxonomy_name_violations(
            "plantae and <i>Plantae</i> and Plantae",
            "Plantae",
        )
        self.assertEqual(len(taxonomy_violations), 2)
        self.assertTrue(all(violation[2] == "Plantae" for violation in taxonomy_violations))

        checker.end_sweep()
        self.assertEqual(checker._collected_higher_taxonomy_names, set())
        self.assertIsNone(checker._collected_genus_name)
        self.assertIsNone(checker._collected_species_name)

        checker.begin_sweep()
        self.assertEqual(checker._collected_higher_taxonomy_names, set())

    def test_geography_helper_methods_and_direct_rules(self) -> None:
        """
        Test that geography helper methods and direct rules.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = GeographyChecker()

        self.assertTrue(checker.is_within_known_country_or_region("North America", (0, 5)))
        self.assertFalse(
            checker.is_within_known_country_or_region("North America", (0, len("North America")))
        )

        country_violations = checker.check_country_names("Test Section", "Vietnam and Laos are listed.")
        self.assertEqual(len(country_violations), 2)

        self.assertEqual(
            checker.check_directional_capitalization(
                "Test Section",
                "Eastern Ecuador contains suitable habitat.",
            ),
            [],
        )
        self.assertEqual(
            len(
                checker.check_directional_capitalization(
                    "Test Section",
                    "The species occurs in Eastern Ecuador.",
                )
            ),
            1,
        )

    def test_iucn_terms_direct_rule_methods(self) -> None:
        """
        Test that IUCN terms direct rule methods.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = IUCNTermsChecker()

        self.assertEqual(len(checker.check_the_iucn("Test Section", "the IUCN guidance")), 1)
        self.assertEqual(len(checker.check_CE_abbreviation("Test Section", "Status CE.")), 1)

        abbreviation_violations = checker.check_category_abbreviation_capitalization(
            "Test Section",
            "cr plants and ex situ conservation.",
        )
        self.assertEqual(len(abbreviation_violations), 1)
        self.assertEqual(abbreviation_violations[0].suggested_fix, "CR")

        full_name_violations = checker.check_category_full_name_capitalization(
            "Test Section",
            "critically endangered and extinct in the wild taxa",
        )
        self.assertEqual(len(full_name_violations), 2)
        self.assertTrue(all("Red List category should be capitalized" in v.message for v in full_name_violations))

        threatened_violations = checker.check_threatened_case(
            "Test Section",
            "many Threatened species and many Near Threatened species remain.",
        )
        self.assertEqual(len(threatened_violations), 1)

    def test_number_helper_methods_and_direct_rules(self) -> None:
        """
        Test that number helper methods and direct rules.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = NumberChecker()

        self.assertTrue(checker.should_exclude_small_number("3 May", 0, 1))
        plain_text = "There were 3 sites."
        start = plain_text.index("3")
        self.assertFalse(checker.should_exclude_small_number(plain_text, start, start + 1))

        span_text = "The DOI is https://doi.org/10.1038/s41598-020-64668-z and 1234 appears later."
        spans = checker.find_doi_or_url_spans(span_text)
        self.assertEqual(len(spans), 1)
        doi_start = span_text.index("64668")
        prose_start = span_text.index("1234")
        self.assertTrue(checker.is_within_spans(doi_start, doi_start + 5, spans))
        self.assertFalse(checker.is_within_spans(prose_start, prose_start + 4, spans))

        self.assertEqual(
            len(checker.check_large_numbers("Test Section", span_text)),
            1,
        )
        self.assertEqual(checker.check_sentence_start("Test Section", "c. 190"), [])
        self.assertEqual(len(checker.check_very_large_numbers("Test Section", "1000000")), 1)

        self.assertTrue(checker.is_very_large_number_candidate(1_500_000))
        self.assertFalse(checker.is_very_large_number_candidate(1_234_567))
        self.assertEqual(checker.format_large_number(1_500_000, 1_000_000, "million"), "1.5 million")
        self.assertEqual(checker.format_large_number(3_000_000_000, 1_000_000_000, "billion"), "3 billion")

    def test_punctuation_helper_methods_and_direct_rules(self) -> None:
        """
        Test that punctuation helper methods and direct rules.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = PunctuationChecker()

        self.assertTrue(checker.is_date_like_numeric_chain("2022-08-01", 0, 7))
        self.assertFalse(checker.is_date_like_numeric_chain("10-20 km", 0, 5))

        self.assertEqual(len(checker.check_colon_spacing("Test Section", "Altitude : 200 m")), 1)
        self.assertEqual(len(checker.check_semicolon_spacing("Test Section", "Peru ; Ecuador")), 1)

    def test_reference_and_scientific_direct_rule_methods(self) -> None:
        """
        Test that reference and scientific direct rule methods.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        reference_checker = ReferenceChecker()
        scientific_checker = ScientificNameChecker()

        citation_violations = reference_checker.check_citation_comma(
            "Test Section",
            "Examples include (Smith, 2020).",
        )
        self.assertEqual(len(citation_violations), 1)
        self.assertEqual(citation_violations[0].suggested_fix, "(Smith 2020)")

        species_violations = scientific_checker.check_species_abbreviations(
            "Test Section",
            "One specimen was sp and several were spp while species stayed unchanged.",
        )
        self.assertEqual(len(species_violations), 2)
        self.assertEqual({violation.suggested_fix for violation in species_violations}, {"sp.", "spp."})

    def test_spelling_helper_method(self) -> None:
        """
        Test that spelling helper method.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = SpellingChecker()

        self.assertEqual(checker.apply_case_pattern("colour", "color"), "colour")
        self.assertEqual(checker.apply_case_pattern("colour", "Color"), "Colour")
        self.assertEqual(checker.apply_case_pattern("colour", "COLOR"), "COLOUR")

    def test_symbol_helper_methods_and_direct_rules(self) -> None:
        """
        Test that symbol helper methods and direct rules.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = SymbolChecker()

        range_text = "Temperature ranged from 14–26 °C."
        self.assertTrue(checker.is_preceded_by_range(range_text, range_text.index("26")))
        self.assertFalse(checker.is_preceded_by_range("Temperature was 12 °C.", "Temperature was 12 °C.".index("12")))

        self.assertEqual(
            len(checker.check_ampersand_usage("Test Section", "Forest & woodland")),
            1,
        )
        self.assertEqual(
            len(checker.check_area_units("Test Section", "Area was sq km and km2.")),
            2,
        )

    def test_bibliography_direct_methods_and_lifecycle(self) -> None:
        """
        Test that bibliography direct methods and lifecycle.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = BibliographyChecker()
        checker.begin_sweep()
        checker.end_sweep()

        ampersand_violations = checker.check_ampersand_usage(
            "Assessment > Bibliography [paragraph 1]",
            "Smith & Jones 2020. Example reference.",
        )
        self.assertEqual(len(ampersand_violations), 1)

        self.assertEqual(
            checker.check_text(
                "Assessment > Rationale [paragraph 1]",
                "Smith & Jones 2020. Example reference.",
            ),
            [],
        )

        bibliography_violations = checker.check_text(
            "Assessment > Bibliography [paragraph 1]",
            "Smith & Jones 2020. Journal 10-20. Mishra et al. 2015.",
        )
        self.assertEqual(
            {violation.rule_class for violation in bibliography_violations},
            {"BibliographyChecker", "AbbreviationChecker", "PunctuationChecker"},
        )


if __name__ == "__main__":
    unittest.main()
