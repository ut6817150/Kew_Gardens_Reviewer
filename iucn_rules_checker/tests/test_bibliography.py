"""Regression tests for bibliography-formatting behavior."""

import unittest

from iucn_rules_checker.checkers.bibliography import BibliographyChecker


class BibliographyCheckerTests(unittest.TestCase):
    """Check the current bibliography rules."""

    def test_ampersand_usage_only_runs_in_bibliography_sections(self) -> None:
        checker = BibliographyChecker()

        bibliography_violations = checker.check((
            "Assessment > Bibliography [paragraph 1]",
            "Smith & Jones 2020. Example reference."
        ))
        body_violations = checker.check((
            "Assessment > Rationale [paragraph 1]",
            "Smith & Jones 2020 discussed the species."
        ))

        ampersand_messages = [
            violation.message for violation in bibliography_violations
            if "Use 'and' not '&'" in violation.message
        ]
        body_ampersand_messages = [
            violation.message for violation in body_violations
            if "Use 'and' not '&'" in violation.message
        ]

        self.assertEqual(len(ampersand_messages), 1)
        self.assertEqual(body_ampersand_messages, [])

    def test_ampersand_usage_flags_all_ampersands_in_bibliography(self) -> None:
        checker = BibliographyChecker()

        violations = checker.check((
            "Assessment > Bibliography [paragraph 1]",
            "<i>Smith</i> <b>&</b> <i>Jones</i> 2020 & Brown 2021."
        ))

        ampersand_violations = [
            violation for violation in violations
            if "Use 'and' not '&'" in violation.message
        ]

        self.assertEqual(len(ampersand_violations), 2)
        self.assertTrue(all(v.suggested_fix == "and" for v in ampersand_violations))

    def test_bibliography_checker_also_runs_et_al_rule(self) -> None:
        checker = BibliographyChecker()

        violations = checker.check((
            "Assessment > Bibliography [paragraph 1]",
            "Mishra et al. 2015. Example reference."
        ))

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].rule_class, "AbbreviationChecker")
        self.assertEqual(violations[0].rule_method, "AbbreviationChecker.check_et_al")
        self.assertEqual(violations[0].message, "Use italicized 'et al.'")

    def test_bibliography_checker_also_runs_range_dash_rule(self) -> None:
        checker = BibliographyChecker()

        violations = checker.check((
            "Assessment > Bibliography [paragraph 1]",
            "Smith 2020. Journal 10-20."
        ))

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].rule_class, "PunctuationChecker")
        self.assertEqual(violations[0].rule_method, "PunctuationChecker.check_range_dashes")
        self.assertIn("Use an unspaced en dash", violations[0].message)

    def test_bibliography_checker_also_runs_large_number_rule(self) -> None:
        checker = BibliographyChecker()

        violations = checker.check((
            "Assessment > Bibliography [paragraph 1]",
            "Smith 2020. Flora 5000 species."
        ))

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].rule_class, "NumberChecker")
        self.assertEqual(violations[0].rule_method, "NumberChecker.check_large_numbers")
        self.assertEqual(
            violations[0].message,
            "Use standard comma placement for numbers: '5000' should be '5,000'",
        )


if __name__ == "__main__":
    unittest.main()
