"""Regression tests for IUCN terminology behavior."""

import unittest

from iucn_rules_checker.checkers.iucn_terms import IUCNTermsChecker


class IUCNTermsCheckerTests(unittest.TestCase):
    """
    Check the current IUCN terminology rules.

    Purpose:
        This test case groups regression checks for the current behavior covered by the enclosed tests.
    """

    def test_the_iucn_strips_simple_style_markers_before_matching(self) -> None:
        """
        Test that the IUCN strips simple style markers before matching.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        text = (
            "This follows <i>the</i> <b>IUCN</b> guidance. "
            "Another line mentions <sup>the</sup> <sub>IUCN</sub> categories."
        )

        violations = IUCNTermsChecker().check_text("Test Section", text)
        messages = [violation.message for violation in violations]

        self.assertEqual(messages.count("Use 'IUCN' not 'the IUCN'"), 2)

    def test_CE_abbreviation_is_case_insensitive_and_whole_word_only(self) -> None:
        """
        Test that CE abbreviation is case insensitive and whole word only.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        text = (
            "The status was CE in one draft, ce in another, and (Ce) in notes. "
            "Correct text may use CR. "
            "Words like concern, species, and icefield should not match."
        )

        violations = IUCNTermsChecker().check_text("Test Section", text)
        messages = [violation.message for violation in violations]

        self.assertEqual(
            messages.count("Use 'CR' not 'CE' for Critically Endangered"),
            3,
        )

    def test_CE_abbreviation_strips_simple_style_markers_before_matching(self) -> None:
        """
        Test that CE abbreviation strips simple style markers before matching.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        text = (
            "Styled forms like <i>C</i><b>E</b>, <sup>c</sup><sub>e</sub>, and "
            "(<b>C</b><i>e</i>) should be caught. "
            "But species and icefield should still not match."
        )

        violations = IUCNTermsChecker().check_text("Test Section", text)
        messages = [violation.message for violation in violations]

        self.assertEqual(
            messages.count("Use 'CR' not 'CE' for Critically Endangered"),
            3,
        )

    def test_category_capitalization_checks_full_names_and_abbreviations(self) -> None:
        """
        Test that category capitalization checks full names and abbreviations.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        text = (
            "The draft listed critically endangered species as cr. "
            "Another section referred to Near threatened taxa and Vu populations. "
            "A final note mentioned extinct in the wild as ew."
        )

        violations = IUCNTermsChecker().check_text("Test Section", text)
        messages = [violation.message for violation in violations]

        self.assertIn(
            "Red List category should be capitalized: 'Critically Endangered'",
            messages,
        )
        self.assertIn(
            "Red List category abbreviation should be capitalized: 'CR'",
            messages,
        )
        self.assertIn(
            "Red List category should be capitalized: 'Near Threatened'",
            messages,
        )
        self.assertIn(
            "Red List category abbreviation should be capitalized: 'VU'",
            messages,
        )
        self.assertIn(
            "Red List category should be capitalized: 'Extinct in the Wild'",
            messages,
        )
        self.assertIn(
            "Red List category abbreviation should be capitalized: 'EW'",
            messages,
        )

    def test_category_capitalization_strips_simple_style_markers_before_matching(self) -> None:
        """
        Test that category capitalization strips simple style markers before matching.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        text = (
            "The draft listed <i>critically</i> <b>endangered</b> species as "
            "<sup>c</sup><sub>r</sub>. Another line referred to <b>Near</b> "
            "<i>threatened</i> taxa and <sup>V</sup><sub>u</sub> populations."
        )

        violations = IUCNTermsChecker().check_text("Test Section", text)
        messages = [violation.message for violation in violations]

        self.assertIn(
            "Red List category should be capitalized: 'Critically Endangered'",
            messages,
        )
        self.assertIn(
            "Red List category abbreviation should be capitalized: 'CR'",
            messages,
        )
        self.assertIn(
            "Red List category should be capitalized: 'Near Threatened'",
            messages,
        )
        self.assertIn(
            "Red List category abbreviation should be capitalized: 'VU'",
            messages,
        )

    def test_category_capitalization_ignores_non_category_ex_situ_and_hyphenated_terms(self) -> None:
        """
        Test that category capitalization ignores non category ex situ and hyphenated terms.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        text = (
            "Ex-situ conservation, ex situ propagation, and neotropical habitat "
            "notes should not trigger category abbreviation matches."
        )

        violations = IUCNTermsChecker().check_text("Test Section", text)
        messages = [violation.message for violation in violations]

        self.assertNotIn(
            "Red List category abbreviation should be capitalized: 'EX'",
            messages,
        )
        self.assertNotIn(
            "Red List category abbreviation should be capitalized: 'NE'",
            messages,
        )

    def test_category_full_name_capitalization_ignores_endangered_when_preceded_by_critically(self) -> None:
        """
        Test that category full name capitalization ignores endangered when preceded by critically.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        text = (
            "One draft described critically endangered plants. "
            "Another note mentioned Critically endangered taxa."
        )

        violations = IUCNTermsChecker().check_category_full_name_capitalization("Test Section", text)
        messages = [violation.message for violation in violations]

        self.assertNotIn(
            "Red List category should be capitalized: 'Endangered'",
            messages,
        )
        self.assertIn(
            "Red List category should be capitalized: 'Critically Endangered'",
            messages,
        )

    def test_category_full_name_capitalization_ignores_extinct_when_followed_by_in_the_wild(self) -> None:
        """
        Test that category full name capitalization ignores extinct when followed by in the wild.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        text = "A note described extinct in the wild populations."

        violations = IUCNTermsChecker().check_category_full_name_capitalization("Test Section", text)
        messages = [violation.message for violation in violations]

        self.assertNotIn(
            "Red List category should be capitalized: 'Extinct'",
            messages,
        )
        self.assertIn(
            "Red List category should be capitalized: 'Extinct in the Wild'",
            messages,
        )

    def test_threatened_case_strips_simple_style_markers_before_matching(self) -> None:
        """
        Test that threatened case strips simple style markers before matching.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        text = (
            "many <i>Threat</i><b>ened</b> species remain. "
            "Threatened species were also reviewed at sentence start."
        )

        violations = IUCNTermsChecker().check_text("Test Section", text)
        messages = [violation.message for violation in violations]

        self.assertEqual(
            messages.count(
                "Use lowercase 'threatened' when referring to CR/EN/VU species collectively"
            ),
            1,
        )

    def test_threatened_case_ignores_near_threatened(self) -> None:
        """
        Test that threatened case ignores near threatened.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        text = (
            "many Near Threatened species remain. "
            "many Threatened species also remain."
        )

        violations = IUCNTermsChecker().check_text("Test Section", text)
        messages = [violation.message for violation in violations]

        self.assertEqual(
            messages.count(
                "Use lowercase 'threatened' when referring to CR/EN/VU species collectively"
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
