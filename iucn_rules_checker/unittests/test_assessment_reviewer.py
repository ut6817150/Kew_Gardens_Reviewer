"""Regression tests for reviewer checker configuration."""

import unittest

from iucn_rules_checker.assessment_reviewer import IUCNAssessmentReviewer


class AssessmentReviewerTests(unittest.TestCase):
    """Check which checker classes the reviewer wires in."""

    def test_reviewer_includes_all_checkers_except_language(self) -> None:
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
        self.assertEqual(type(reviewer.bibliography_checker).__name__, "BibliographyChecker")
        self.assertNotIn("LanguageChecker", configured_checkers)
        self.assertFalse(hasattr(reviewer, "assessment_parser"))
        self.assertFalse(hasattr(reviewer, "review_assessment"))

    def test_reviewer_skips_table_sections_but_checks_paragraph_sections(self) -> None:
        reviewer = IUCNAssessmentReviewer()
        full_report = {
            "Assessment > Notes [paragraph 1]": "Examples occur e.g. in text.",
            "Assessment > Notes [table 1] [row 1]": "Examples occur e.g. in text.",
        }

        violations = reviewer.review_full_report(full_report)
        messages = [violation.message for violation in violations]
        sections = [violation.section_name for violation in violations]

        self.assertIn("Avoid 'e.g.' in body text; use 'for example' instead", messages)
        self.assertIn("Assessment > Notes", sections)
        self.assertNotIn("Assessment > Notes [table 1] [row 1]", sections)

    def test_reviewer_runs_only_bibliography_checker_in_bibliography_sections(self) -> None:
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
                "NumberChecker",
            ],
        )
        self.assertEqual(messages[:2], [
            "Use 'and' not '&' in bibliography author entries",
            "Use italicized 'et al.'",
        ])
        self.assertIn("Use an unspaced en dash", messages[2])
        self.assertEqual(
            messages[3],
            "Use standard comma placement for numbers: '5000' should be '5,000'",
        )


if __name__ == "__main__":
    unittest.main()
