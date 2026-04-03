"""Regression tests for shared BaseChecker helpers."""

import unittest

from iucn_rules_checker.checkers.base import BaseChecker


class DummyChecker(BaseChecker):
    """Minimal concrete checker for testing BaseChecker utilities."""

    def __init__(self) -> None:
        super().__init__()

    def check_text(self, section_name: str, text: str):  # pragma: no cover - not used here
        return []

    def check_demo_rule(self, section_name: str, text: str):
        return self.create_violation(
            section_name=section_name,
            text=text,
            span=(0, 5),
            message="Demo violation",
            suggested_fix="Demo",
        )


class BaseCheckerTests(unittest.TestCase):
    """Lock in shared helper behavior."""

    def test_strip_style_markers_defaults_strip_italics_and_bold_only(self) -> None:
        checker = DummyChecker()
        text = "<i>A</i><b>B</b><sup>2</sup><sub>3</sub>"

        cleaned_text, index_map = checker.strip_style_markers(text)

        self.assertEqual(cleaned_text, "AB<sup>2</sup><sub>3</sub>")
        self.assertEqual(len(index_map), len(cleaned_text))
        self.assertEqual(cleaned_text[index_map.index(text.index("A"))], "A")
        self.assertEqual(cleaned_text[index_map.index(text.index("B"))], "B")

    def test_strip_style_markers_can_also_strip_superscript_and_subscript(self) -> None:
        checker = DummyChecker()
        text = "<i>A</i><b>B</b><sup>2</sup><sub>3</sub>"

        cleaned_text, index_map = checker.strip_style_markers(
            text,
            superscript=True,
            subscript=True,
        )

        self.assertEqual(cleaned_text, "AB23")
        self.assertEqual(len(index_map), len(cleaned_text))
        self.assertEqual(text[index_map[2]], "2")
        self.assertEqual(text[index_map[3]], "3")

    def test_strip_style_markers_can_leave_all_markup_untouched(self) -> None:
        checker = DummyChecker()
        text = "<i>A</i><b>B</b><sup>2</sup><sub>3</sub>"

        cleaned_text, index_map = checker.strip_style_markers(
            text,
            italics=False,
            bold=False,
            superscript=False,
            subscript=False,
        )

        self.assertEqual(cleaned_text, text)
        self.assertEqual(index_map, list(range(len(text))))

    def test_create_violation_records_rule_method(self) -> None:
        checker = DummyChecker()

        violation = checker.check_demo_rule("Section [paragraph 1]", "Alpha beta gamma")

        self.assertEqual(violation.rule_class, "DummyChecker")
        self.assertEqual(violation.rule_method, "DummyChecker.check_demo_rule")
        self.assertEqual(violation.matched_text, "Alpha")
        self.assertEqual(violation.matched_snippet, "Alpha beta gamma")
        self.assertEqual(violation.section_name, "Section")


if __name__ == "__main__":
    unittest.main()
