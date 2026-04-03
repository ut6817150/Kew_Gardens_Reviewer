"""Regression tests for IUCN terminology behavior."""

import unittest

from iucn_rules_checker.checkers.iucn_terms import IUCNTermsChecker


class IUCNTermsCheckerTests(unittest.TestCase):
    """Check the current IUCN terminology rules."""

    def test_the_iucn_strips_simple_style_markers_before_matching(self) -> None:
        text = (
            "This follows <i>the</i> <b>IUCN</b> guidance. "
            "Another line mentions <sup>the</sup> <sub>IUCN</sub> categories."
        )

        violations = IUCNTermsChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        self.assertEqual(messages.count("Use 'IUCN' not 'the IUCN'"), 2)

    def test_CE_abbreviation_is_case_insensitive_and_whole_word_only(self) -> None:
        text = (
            "The status was CE in one draft, ce in another, and (Ce) in notes. "
            "Correct text may use CR. "
            "Words like concern, species, and icefield should not match."
        )

        violations = IUCNTermsChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        self.assertEqual(
            messages.count("Use 'CR' not 'CE' for Critically Endangered"),
            3,
        )

    def test_CE_abbreviation_strips_simple_style_markers_before_matching(self) -> None:
        text = (
            "Styled forms like <i>C</i><b>E</b>, <sup>c</sup><sub>e</sub>, and "
            "(<b>C</b><i>e</i>) should be caught. "
            "But species and icefield should still not match."
        )

        violations = IUCNTermsChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        self.assertEqual(
            messages.count("Use 'CR' not 'CE' for Critically Endangered"),
            3,
        )

    def test_category_capitalization_checks_full_names_and_abbreviations(self) -> None:
        text = (
            "The draft listed critically endangered species as cr. "
            "Another section referred to Near threatened taxa and Vu populations. "
            "A final note mentioned extinct in the wild as ew."
        )

        violations = IUCNTermsChecker().check(("Test Section", text))
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
        text = (
            "The draft listed <i>critically</i> <b>endangered</b> species as "
            "<sup>c</sup><sub>r</sub>. Another line referred to <b>Near</b> "
            "<i>threatened</i> taxa and <sup>V</sup><sub>u</sub> populations."
        )

        violations = IUCNTermsChecker().check(("Test Section", text))
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

    def test_category_capitalization_ignores_hyphenated_non_category_terms(self) -> None:
        text = "Ex-situ conservation and neotropical habitat notes should not trigger category abbreviation matches."

        violations = IUCNTermsChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        self.assertNotIn(
            "Red List category abbreviation should be capitalized: 'EX'",
            messages,
        )
        self.assertNotIn(
            "Red List category abbreviation should be capitalized: 'NE'",
            messages,
        )

    def test_threatened_case_strips_simple_style_markers_before_matching(self) -> None:
        text = (
            "many <i>Threat</i><b>ened</b> species remain. "
            "Threatened species were also reviewed at sentence start."
        )

        violations = IUCNTermsChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        self.assertEqual(
            messages.count(
                "Use lowercase 'threatened' when referring to CR/EN/VU species collectively"
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
