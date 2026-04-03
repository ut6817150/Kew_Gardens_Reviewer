"""Regression tests for abbreviation matching behavior."""

import unittest

from iucn_rules_checker.checkers.abbreviations import AbbreviationChecker


class AbbreviationCheckerTests(unittest.TestCase):
    """Check the current latin-abbreviation matching rules."""

    def test_latin_abbreviations_match_common_variants(self) -> None:
        text = (
            "E.g. this starts a sentence. "
            "Examples occur in running text, e.g. orchids and mosses. "
            "Some authors write EG at sentence end EG. "
            "Clarify the statement, i.e. explain it. "
            "Sometimes IE appears in uppercase. "
            "Short forms without dots like eg and ie should still match. "
            "Parenthetical usage (e.g. in notes) and (i.e. in glosses) should also match."
        )

        violations = AbbreviationChecker().check(("Test Section", text))

        eg_messages = [
            violation for violation in violations
            if "e.g." in violation.message
        ]
        ie_messages = [
            violation for violation in violations
            if "i.e." in violation.message
        ]

        self.assertEqual(len(eg_messages), 6)
        self.assertEqual(len(ie_messages), 4)
        self.assertTrue(all(v.section_name == "Test Section" for v in violations))

    def test_eg_and_ie_and_fixed_formats_ignore_simple_style_tags(self) -> None:
        text = (
            "Styled forms like <i>E</i><i>.g.</i>, <b>I</b><b>.e.</b>, and <sup>eg</sup> should still match. "
            "Fixed forms like <i>etc</i>, <b>in lit</b>, <i>Pers</i> <b>Comm</b>, and <sub>Prof</sub> should also match."
        )

        violations = AbbreviationChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        self.assertIn("Avoid 'e.g.' in body text; use 'for example' instead", messages)
        self.assertIn("Avoid 'i.e.' in body text; use 'that is' instead", messages)
        self.assertIn("Use 'etc.' with period", messages)
        self.assertIn("Use 'in lit.' not 'in lit', if referring to in published literature", messages)
        self.assertIn("Use 'pers. comm.' format", messages)
        self.assertIn("Use 'Prof.' with period", messages)

    def test_abbreviation_formats_are_case_insensitive(self) -> None:
        text = (
            "Lists may end with ETC in notes. "
            "References often cite ET AL without the final period. "
            "Older comments may say IN LIT in uppercase. "
            "Sources may mention Pers Comm from a field botanist. "
            "Drafts may also include Pers Obs without the first period. "
            "A later note ends with pers comm. "
            "Another line ends with pers obs. "
            "A heading might use PROF Smith without punctuation."
        )

        violations = AbbreviationChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        self.assertIn("Use 'etc.' with period", messages)
        self.assertIn("Use italicized 'et al.'", messages)
        self.assertIn("Use 'in lit.' not 'in lit', if referring to in published literature", messages)
        self.assertIn("Use 'pers. comm.' format", messages)
        self.assertIn("Use 'pers. obs.' format", messages)
        self.assertIn("Use 'Prof.' with period", messages)
        self.assertEqual(messages.count("Use 'pers. comm.' format"), 2)
        self.assertEqual(messages.count("Use 'pers. obs.' format"), 2)

    def test_et_al_requires_italicized_canonical_form(self) -> None:
        text = (
            "References may cite et al in plain text. "
            "Plain-text usage such as et al. is still wrong here. "
            "Others write <i>et al</i> without the final period. "
            "Some also write <i>et. al.</i> with the wrong internal period. "
            "Correct italicized usage appears as <i>et al.</i>."
        )

        violations = AbbreviationChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        combined_et_al = "Use italicized 'et al.'"

        self.assertEqual(messages.count(combined_et_al), 4)

    def test_latin_terms_strip_non_italic_style_markers_but_preserve_italics(self) -> None:
        text = (
            "Field notes mention <b>in situ</b> in bold plain text. "
            "One citation uses <b><i>et al</i></b> without the final period. "
            "Correct usage appears as <b><i>et al.</i></b>."
        )

        violations = AbbreviationChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        self.assertIn(
            "Latin term 'in situ' must be italicized and not contain periods",
            messages,
        )
        self.assertIn(
            "Use italicized 'et al.'",
            messages,
        )
        self.assertEqual(
            messages.count("Use italicized 'et al.'"),
            1,
        )

    def test_latin_terms_check_periods_and_missing_italics(self) -> None:
        text = (
            "Conservation may happen in situ and <i>ex situ</i>. "
            "Some drafts still write in. situ. or sensu. lato. "
            "A report may also mention de facto in plain text."
        )

        violations = AbbreviationChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        combined_in_situ = "Latin term 'in situ' must be italicized and not contain periods"
        combined_sensu_lato = "Latin term 'sensu lato' must be italicized and not contain periods"
        combined_de_facto = "Latin term 'de facto' must be italicized and not contain periods"
        combined_ex_situ = "Latin term 'ex situ' must be italicized and not contain periods"

        self.assertIn(combined_in_situ, messages)
        self.assertIn(combined_sensu_lato, messages)
        self.assertIn(combined_de_facto, messages)
        self.assertIn(combined_ex_situ, messages)
        self.assertEqual(sum("Latin term" in message for message in messages), 5)
        self.assertEqual(messages.count(combined_in_situ), 2)

    def test_title_abbreviations_cover_dr_mr_mrs_and_ms(self) -> None:
        text = (
            "Dr. Green met Mr. Brown. "
            "Later, Mrs. White and Ms. Black joined the survey."
        )

        violations = AbbreviationChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        self.assertIn("Use 'Dr' without period (UK style)", messages)
        self.assertIn("Use 'Mr' without period (UK style)", messages)
        self.assertIn("Use 'Mrs' without period (UK style)", messages)
        self.assertIn("Use 'Ms' without period (UK style)", messages)

    def test_title_abbreviations_ignore_all_simple_style_markers(self) -> None:
        text = (
            "<b>Dr.</b> Green met <i>Mr.</i> Brown. "
            "Later, <sup>Mrs.</sup> White and <sub>Ms.</sub> Black joined."
        )

        violations = AbbreviationChecker().check(("Test Section", text))
        messages = [violation.message for violation in violations]

        self.assertIn("Use 'Dr' without period (UK style)", messages)
        self.assertIn("Use 'Mr' without period (UK style)", messages)
        self.assertIn("Use 'Mrs' without period (UK style)", messages)
        self.assertIn("Use 'Ms' without period (UK style)", messages)

    def test_violation_section_name_hides_paragraph_suffix_only(self) -> None:
        checker = AbbreviationChecker()

        paragraph_violations = checker.check((
            "Assessment > Notes [paragraph 2]",
            "Examples occur e.g. in text.",
        ))
        table_violations = checker.check((
            "Assessment > Notes [table 1]",
            "Examples occur e.g. in text.",
        ))

        self.assertTrue(paragraph_violations)
        self.assertTrue(table_violations)
        self.assertEqual(paragraph_violations[0].section_name, "Assessment > Notes")
        self.assertEqual(table_violations[0].section_name, "Assessment > Notes [table 1]")


if __name__ == "__main__":
    unittest.main()
