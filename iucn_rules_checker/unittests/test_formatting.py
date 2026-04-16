"""Regression tests for formatting-checker behavior."""

import unittest

from iucn_rules_checker.checkers.formatting import FormattingChecker


class FormattingCheckerTests(unittest.TestCase):
    """
    Check the current capitalization rules for spelled-out EOO/AOO terms.

    Purpose:
        This test case groups regression checks for the current behavior covered by the enclosed tests.
    """

    def test_eoo_aoo_capitalization_catches_partial_caps_and_punctuation_positions(self) -> None:
        """
        Test that EOO AOO capitalization catches partial caps and punctuation positions.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        text = (
            "The Extent of Occurrence is restricted. "
            "Area of Occupancy is also small. "
            "(Extent of Occurrence) remained unchanged. "
            "Field note: Area of Occupancy still applies. "
            "Summary, Extent of occurrence was revised. "
            "Only lowercase extent of occurrence and area of occupancy should pass."
        )

        violations = FormattingChecker().check_eoo_aoo_capitalization("Formatting Section", text)
        messages = [violation.message for violation in violations]

        self.assertEqual(len(violations), 5)
        self.assertTrue(all(violation.section_name == "Formatting Section" for violation in violations))
        self.assertIn(
            "Use lowercase: 'extent of occurrence' not 'Extent of Occurrence'",
            messages,
        )
        self.assertIn(
            "Use lowercase: 'area of occupancy' not 'Area of Occupancy'",
            messages,
        )
        self.assertIn(
            "Use lowercase: 'extent of occurrence' not 'Extent of occurrence'",
            messages,
        )

    def test_eoo_aoo_capitalization_allows_sentence_start_after_period_question_colon_and_paragraph_start(self) -> None:
        """
        Test that EOO AOO capitalization allows sentence start after period question colon and paragraph start.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        text = (
            "Extent of occurrence remained restricted. "
            "Area of occupancy stayed small. "
            "This was revised. Extent of occurrence remained restricted. "
            "Was this revised? Extent of occurrence was updated. "
            "Summary: Area of occupancy was recalculated."
        )

        violations = FormattingChecker().check_eoo_aoo_capitalization("Formatting Section", text)

        self.assertEqual(violations, [])

    def test_eoo_aoo_capitalization_flags_first_word_caps_outside_period_space_exception(self) -> None:
        """
        Test that EOO AOO capitalization flags first word caps outside period space exception.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        text = (
            "The Extent of occurrence remained restricted. "
            "Summary, Area of occupancy was recalculated. "
            "(Extent of occurrence) was revised."
        )

        violations = FormattingChecker().check_eoo_aoo_capitalization("Formatting Section", text)
        messages = [violation.message for violation in violations]

        self.assertEqual(len(violations), 3)
        self.assertIn(
            "Use lowercase: 'extent of occurrence' not 'Extent of occurrence'",
            messages,
        )
        self.assertIn(
            "Use lowercase: 'area of occupancy' not 'Area of occupancy'",
            messages,
        )

    def test_eoo_aoo_capitalization_ignores_simple_style_tags(self) -> None:
        """
        Test that EOO AOO capitalization ignores simple style tags.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        text = (
            "The <b>Extent of occurrence</b> remained restricted. "
            "This was revised. <i>Extent of occurrence</i> remained restricted. "
            "Summary: <b>Area of occupancy</b> was recalculated. "
            "Summary, <i>Area of occupancy</i> was recalculated."
        )

        violations = FormattingChecker().check_eoo_aoo_capitalization("Formatting Section", text)
        messages = [violation.message for violation in violations]

        self.assertEqual(len(violations), 2)
        self.assertIn(
            "Use lowercase: 'extent of occurrence' not 'Extent of occurrence'",
            messages,
        )
        self.assertIn(
            "Use lowercase: 'area of occupancy' not 'Area of occupancy'",
            messages,
        )

    def test_higher_order_taxonomy_formatting_ignores_non_harvested_family_like_names(self) -> None:
        """
        Test that higher order taxonomy formatting ignores non harvested family like names.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = FormattingChecker()
        text = (
            "orchidaceae is mentioned in plain text. "
            "The draft also contains <i>Orchidaceae</i> and <i>felidae</i>."
        )

        violations = checker.check_higher_order_taxonomy_formatting("Formatting Section", text)

        self.assertEqual(violations, [])

    def test_higher_order_taxonomy_formatting_checks_supplemental_names_only_after_ladder_harvest(self) -> None:
        """
        Test that higher order taxonomy formatting checks supplemental names only after ladder harvest.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = FormattingChecker()

        pre_harvest_violations = checker.check_higher_order_taxonomy_formatting(
            "Formatting Section",
            "plantae appears before any taxonomy ladder.",
        )
        self.assertEqual(pre_harvest_violations, [])

        checker.begin_sweep()
        try:
            ladder_text = (
                "FUNGI - ASCOMYCOTA - SORDARIOMYCETES - HYPOCREALES - "
                "NECTRIACEAE - Fusarium - oxysporum"
            )
            later_text = "plantae appears after a non-plantae ladder entry."

            ladder_violations = checker.check_higher_order_taxonomy_formatting(
                "Formatting Section",
                ladder_text,
            )
            later_violations = checker.check_higher_order_taxonomy_formatting(
                "Formatting Section",
                later_text,
            )

            self.assertEqual(ladder_violations, [])
            later_messages = [violation.message for violation in later_violations]
            self.assertIn(
                "Family/taxonomy names should be capitalized and not italicized: 'Plantae'",
                later_messages,
            )
        finally:
            checker.end_sweep()

    def test_higher_order_taxonomy_formatting_strips_non_italic_style_markers(self) -> None:
        """
        Test that higher order taxonomy formatting strips non italic style markers.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = FormattingChecker()
        checker.begin_sweep()
        try:
            ladder_text = (
                "<b>PLANTAE - TRACHEOPHYTA - MAGNOLIOPSIDA - FABALES - FABACEAE - "
                "Acrocarpus - fraxinifolius</b>"
            )
            later_text = (
                "<b>plantae</b> appears after the ladder entry."
                " <b><i>Magnoliopsida</i></b> appears in bold italics."
            )

            ladder_violations = checker.check_higher_order_taxonomy_formatting(
                "Formatting Section",
                ladder_text,
            )
            later_violations = checker.check_higher_order_taxonomy_formatting(
                "Formatting Section",
                later_text,
            )

            self.assertEqual(ladder_violations, [])
            later_messages = [violation.message for violation in later_violations]
            self.assertIn(
                "Family/taxonomy names should be capitalized and not italicized: 'Plantae'",
                later_messages,
            )
            self.assertIn(
                "Family/taxonomy names should be capitalized and not italicized: 'Magnoliopsida'",
                later_messages,
            )
        finally:
            checker.end_sweep()

    def test_higher_order_taxonomy_formatting_harvests_names_from_ladder_entry(self) -> None:
        """
        Test that higher order taxonomy formatting harvests names from ladder entry.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = FormattingChecker()
        checker.begin_sweep()
        try:
            ladder_text = "PLANTAE - TRACHEOPHYTA - MAGNOLIOPSIDA - FABALES - FABACEAE - Acrocarpus - fraxinifolius"
            later_text = "plantae and <i>Magnoliopsida</i> appear here, as does <i>Fabaceae</i>."

            ladder_violations = checker.check_higher_order_taxonomy_formatting(
                "Formatting Section",
                ladder_text,
            )
            later_violations = checker.check_higher_order_taxonomy_formatting(
                "Formatting Section",
                later_text,
            )

            self.assertEqual(ladder_violations, [])
            later_messages = [violation.message for violation in later_violations]
            self.assertIn(
                "Family/taxonomy names should be capitalized and not italicized: 'Plantae'",
                later_messages,
            )
            self.assertIn(
                "Family/taxonomy names should be capitalized and not italicized: 'Magnoliopsida'",
                later_messages,
            )
            self.assertIn(
                "Family/taxonomy names should be capitalized and not italicized: 'Fabaceae'",
                later_messages,
            )
        finally:
            checker.end_sweep()

        cleared_violations = checker.check_higher_order_taxonomy_formatting(
            "Formatting Section",
            "plantae and Magnoliopsida appear here.",
        )
        self.assertEqual(cleared_violations, [])

    def test_genus_and_species_rule_harvests_from_ladder_and_checks_italics_and_case(self) -> None:
        """
        Test that genus and species rule harvests from ladder and checks italics and case.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = FormattingChecker()
        checker.begin_sweep()
        try:
            ladder_text = "PLANTAE - TRACHEOPHYTA - MAGNOLIOPSIDA - FABALES - FABACEAE - Acrocarpus - fraxinifolius"
            later_text = (
                "Acrocarpus appears in plain text. "
                "acrocarpus also appears in plain text. "
                "fraxinifolius appears in plain text. "
                "Fraxinifolius appears in plain text. "
                "<i>Acrocarpus</i> is fine. "
                "<i>fraxinifolius</i> is fine. "
                "<i>Fraxinifolius</i> is not fine."
            )

            ladder_violations = checker.check_genus_and_species("Formatting Section", ladder_text)
            later_violations = checker.check_genus_and_species("Formatting Section", later_text)

            self.assertEqual(ladder_violations, [])
            later_messages = [violation.message for violation in later_violations]
            self.assertIn(
                "Scientific names should be italicized and use correct case: '<i>Acrocarpus</i>'",
                later_messages,
            )
            self.assertIn(
                "Scientific names should be italicized and use correct case: '<i>fraxinifolius</i>'",
                later_messages,
            )
            self.assertEqual(
                later_messages.count("Scientific names should be italicized and use correct case: '<i>Acrocarpus</i>'"),
                2,
            )
            self.assertEqual(
                later_messages.count("Scientific names should be italicized and use correct case: '<i>fraxinifolius</i>'"),
                3,
            )
        finally:
            checker.end_sweep()

        cleared_violations = checker.check_genus_and_species(
            "Formatting Section",
            "Acrocarpus and fraxinifolius appear here.",
        )
        self.assertEqual(cleared_violations, [])

    def test_genus_and_species_rule_checks_supplemental_names_only_after_ladder_harvest(self) -> None:
        """
        Test that genus and species rule checks supplemental names only after ladder harvest.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = FormattingChecker()

        pre_harvest_violations = checker.check_genus_and_species(
            "Formatting Section",
            "Chlidanthus and ariruma appear before any taxonomy ladder.",
        )
        self.assertEqual(pre_harvest_violations, [])

        checker.begin_sweep()
        try:
            ladder_text = (
                "FUNGI - ASCOMYCOTA - SORDARIOMYCETES - HYPOCREALES - "
                "NECTRIACEAE - Fusarium - oxysporum"
            )
            later_text = (
                "Chlidanthus appears in plain text. "
                "chlidanthus also appears in plain text. "
                "ariruma appears in plain text. "
                "Ariruma appears in plain text."
            )

            ladder_violations = checker.check_genus_and_species("Formatting Section", ladder_text)
            later_violations = checker.check_genus_and_species("Formatting Section", later_text)

            self.assertEqual(ladder_violations, [])
            later_messages = [violation.message for violation in later_violations]
            self.assertEqual(
                later_messages.count("Scientific names should be italicized and use correct case: '<i>Chlidanthus</i>'"),
                2,
            )
            self.assertEqual(
                later_messages.count("Scientific names should be italicized and use correct case: '<i>ariruma</i>'"),
                2,
            )
        finally:
            checker.end_sweep()

    def test_genus_and_species_rule_strips_non_italic_style_markers(self) -> None:
        """
        Test that genus and species rule strips non italic style markers.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = FormattingChecker()
        checker.begin_sweep()
        try:
            ladder_text = (
                "<b>PLANTAE - TRACHEOPHYTA - MAGNOLIOPSIDA - FABALES - FABACEAE - "
                "Acrocarpus - fraxinifolius</b>"
            )
            later_text = (
                "<b>Acrocarpus</b> appears in bold plain text. "
                "<b>acrocarpus</b> also appears in bold plain text. "
                "<b>fraxinifolius</b> appears in bold plain text. "
                "<b><i>Fraxinifolius</i></b> appears in bold italics with the wrong case."
            )

            ladder_violations = checker.check_genus_and_species("Formatting Section", ladder_text)
            later_violations = checker.check_genus_and_species("Formatting Section", later_text)

            self.assertEqual(ladder_violations, [])
            later_messages = [violation.message for violation in later_violations]
            self.assertEqual(
                later_messages.count("Scientific names should be italicized and use correct case: '<i>Acrocarpus</i>'"),
                2,
            )
            self.assertEqual(
                later_messages.count("Scientific names should be italicized and use correct case: '<i>fraxinifolius</i>'"),
                2,
            )
        finally:
            checker.end_sweep()

    def test_genus_and_species_rule_allows_words_inside_one_italicized_binomial(self) -> None:
        """
        Test that genus and species rule allows words inside one italicized binomial.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = FormattingChecker()
        checker.begin_sweep()
        try:
            ladder_text = "PLANTAE - TRACHEOPHYTA - MAGNOLIOPSIDA - FABALES - FABACEAE - Acrocarpus - fraxinifolius"
            later_text = "<b><i>Acrocarpus fraxinifolius</i></b> appears in the text."

            ladder_violations = checker.check_genus_and_species("Formatting Section", ladder_text)
            later_violations = checker.check_genus_and_species("Formatting Section", later_text)

            self.assertEqual(ladder_violations, [])
            self.assertEqual(later_violations, [])
        finally:
            checker.end_sweep()


if __name__ == "__main__":
    unittest.main()
