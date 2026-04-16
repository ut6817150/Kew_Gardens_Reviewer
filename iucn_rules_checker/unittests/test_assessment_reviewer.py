"""Regression tests for reviewer checker configuration."""

import unittest

from iucn_rules_checker.assessment_reviewer import IUCNAssessmentReviewer


class AssessmentReviewerTests(unittest.TestCase):
    """
    Check which checker classes the reviewer wires in.

    Purpose:
        This test case groups regression checks for the current behavior covered by the enclosed tests.
    """

    def test_reviewer_includes_all_checkers_except_language(self) -> None:
        """
        Test that reviewer includes all checkers except language.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        reviewer = IUCNAssessmentReviewer()
        configured_checkers = [type(checker).__name__ for checker in reviewer.checkers]

        self.assertEqual(
            configured_checkers,
            [
                "AbbreviationChecker",
                "DateChecker",
                "FormattingChecker",
                "GeographyChecker",
                "IUCNTermsChecker",
                "NumberChecker",
                "PunctuationChecker",
                "ReferenceChecker",
                "ScientificNameChecker",
                "SpellingChecker",
                "SymbolChecker",
            ],
        )
        self.assertEqual(type(reviewer.table_checker).__name__, "TableChecker")
        self.assertEqual(type(reviewer.bibliography_checker).__name__, "BibliographyChecker")
        self.assertNotIn("LanguageChecker", configured_checkers)
        self.assertFalse(hasattr(reviewer, "assessment_parser"))
        self.assertFalse(hasattr(reviewer, "review_assessment"))

    def test_reviewer_routes_table_sections_to_table_checker_only(self) -> None:
        """
        Test that reviewer routes table sections to table checker only.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        reviewer = IUCNAssessmentReviewer()
        full_report = {
            "Assessment > Notes [paragraph 1]": "Examples occur e.g. in text.",
            "Assessment > Notes [table 1] [row 1]": "Examples include Smith et al. 2020.",
        }

        violations = reviewer.review_full_report(full_report)
        messages = [violation.message for violation in violations]
        sections = [violation.section_name for violation in violations]

        self.assertIn("Avoid 'e.g.' in body text; use 'for example' instead", messages)
        self.assertIn("Assessment > Notes", sections)
        self.assertIn("Use italicized 'et al.'", messages)
        self.assertIn("Assessment > Notes [table 1] [row 1]", sections)
        self.assertEqual(
            [
                violation.rule_method for violation in violations
                if violation.section_name == "Assessment > Notes [table 1] [row 1]"
            ],
            ["AbbreviationChecker.check_et_al"],
        )

    def test_reviewer_runs_only_bibliography_checker_in_bibliography_sections(self) -> None:
        """
        Test that reviewer runs only bibliography checker in bibliography sections.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        reviewer = IUCNAssessmentReviewer()
        full_report = {
            "Assessment > Bibliography [paragraph 1]": (
                "Smith & Jones. Journal 10-20. Examples include (Mishra et al., 2015). Flora 5000 species."
            ),
        }

        violations = reviewer.review_full_report(full_report)
        rule_classes = [violation.rule_class for violation in violations]
        messages = [violation.message for violation in violations]

        self.assertEqual(
            rule_classes,
            [
                "BibliographyChecker",
                "AbbreviationChecker",
                "PunctuationChecker",
            ],
        )
        self.assertEqual(messages[:2], [
            "Use 'and' not '&' in bibliography author entries",
            "Use italicized 'et al.'",
        ])
        self.assertIn("Use an unspaced en dash", messages[2])
        self.assertEqual(len(messages), 3)

    def test_clean_up_violations_strips_style_markup_from_context_and_message(self) -> None:
        """
        Test that clean up violations strips style markup from context and message.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        reviewer = IUCNAssessmentReviewer()
        full_report = {
            "Assessment > Taxonomy [paragraph 1]": (
                "PLANTAE - TRACHEOPHYTA - MAGNOLIOPSIDA - FABALES - FABACEAE - Acrocarpus - fraxinifolius"
            ),
            "Assessment > Notes [paragraph 1]": (
                "The survey recorded <b><i>Fraxinifolius</i></b> seedlings."
            ),
        }

        violations = reviewer.review_full_report(full_report)
        cleaned_violations = reviewer.clean_up_violations(violations)
        species_violations = [
            violation for violation in cleaned_violations
            if violation.rule_method == "FormattingChecker.check_genus_and_species"
        ]

        self.assertEqual(len(species_violations), 1)
        self.assertEqual(
            species_violations[0].message,
            "Scientific names should be italicized and use correct case: 'fraxinifolius'",
        )
        self.assertEqual(
            species_violations[0].matched_snippet,
            "survey recorded Fraxinifolius seedlings.",
        )
        self.assertNotIn("<", species_violations[0].message)
        self.assertNotIn(">", species_violations[0].message)
        self.assertNotIn("<", species_violations[0].matched_snippet)
        self.assertNotIn(">", species_violations[0].matched_snippet)


if __name__ == "__main__":
    unittest.main()
