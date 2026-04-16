"""Regression tests for shared BaseChecker helpers."""

import unittest

from iucn_rules_checker.checkers.base import BaseChecker


class DummyChecker(BaseChecker):
    """
    Minimal concrete checker for testing BaseChecker utilities.

    Purpose:
        This test double provides a minimal concrete implementation of ``BaseChecker`` for shared-helper tests.
    """

    def __init__(self) -> None:
        """
        Initialise the dummy checker test double.

        Args:
            None.

        Returns:
            None: None.
        """
        super().__init__()

    def check_text(self, section_name: str, text: str):
        """
        Dispatch the dummy checker rules used by the base-checker tests.

        Args:
            section_name (str): Parsed section key supplied by the caller.
            text (str): Parsed section text supplied by the caller.

        Returns:
            Any: Value produced by this method.
        """
        return [self.create_violation(
            section_name=section_name,
            text=text,
            span=(0, min(5, len(text))),
            message="Dispatched violation",
            suggested_fix=None,
        )]

    def check_demo_rule(self, section_name: str, text: str):
        """
        Return a placeholder violation for the demo-rule tests.

        Args:
            section_name (str): Parsed section key supplied by the caller.
            text (str): Parsed section text supplied by the caller.

        Returns:
            Any: Value described by the summary line above.
        """
        return self.create_violation(
            section_name=section_name,
            text=text,
            span=(0, 5),
            message="Demo violation",
            suggested_fix="Demo",
        )

    def check_method_name_rule(self) -> str:
        """
        Return a placeholder violation for the rule-method-name tests.

        Args:
            None.

        Returns:
            str: String value produced by this method.
        """
        return self.get_rule_method_name()

    def call_get_rule_method_name_through_wrapper(self) -> str:
        """
        Expose ``get_rule_method_name`` through a wrapper for testing.

        Args:
            None.

        Returns:
            str: String value produced by this method.
        """
        return self._helper_calls_get_rule_method_name()

    def _helper_calls_get_rule_method_name(self) -> str:
        """
        Call ``get_rule_method_name`` from a nested helper frame for testing.

        Args:
            None.

        Returns:
            str: String value produced by this method.
        """
        return self.get_rule_method_name()


class BaseCheckerTests(unittest.TestCase):
    """
    Lock in shared helper behavior.

    Purpose:
        This test case groups regression checks for the current behavior covered by the enclosed tests.
    """

    def test_strip_style_markers_defaults_strip_italics_and_bold_only(self) -> None:
        """
        Test that strip style markers defaults strip italics and bold only.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = DummyChecker()
        text = "<i>A</i><b>B</b><sup>2</sup><sub>3</sub>"

        cleaned_text, index_map = checker.strip_style_markers(text)

        self.assertEqual(cleaned_text, "AB<sup>2</sup><sub>3</sub>")
        self.assertEqual(len(index_map), len(cleaned_text))
        self.assertEqual(cleaned_text[index_map.index(text.index("A"))], "A")
        self.assertEqual(cleaned_text[index_map.index(text.index("B"))], "B")

    def test_strip_style_markers_can_also_strip_superscript_and_subscript(self) -> None:
        """
        Test that strip style markers can also strip superscript and subscript.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
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
        """
        Test that strip style markers can leave all markup untouched.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
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
        """
        Test that create violation records rule method.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = DummyChecker()

        violation = checker.check_demo_rule("Section [paragraph 1]", "Alpha beta gamma")

        self.assertEqual(violation.rule_class, "DummyChecker")
        self.assertEqual(violation.rule_method, "DummyChecker.check_demo_rule")
        self.assertEqual(violation.matched_text, "Alpha")
        self.assertEqual(violation.matched_snippet, "Alpha beta gamma")
        self.assertEqual(violation.section_name, "Section")

    def test_check_text_returns_violation_directly(self) -> None:
        """
        Test that check text returns violation directly.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = DummyChecker()

        violations = checker.check_text("Section [paragraph 1]", "Alpha beta gamma")

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].message, "Dispatched violation")
        self.assertEqual(violations[0].rule_method, "DummyChecker.check_text")

    def test_normalize_section_name_removes_only_paragraph_suffix(self) -> None:
        """
        Test that normalize section name removes only paragraph suffix.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = DummyChecker()

        self.assertEqual(
            checker.normalize_section_name("Assessment > Notes [paragraph 3]"),
            "Assessment > Notes",
        )
        self.assertEqual(
            checker.normalize_section_name("Assessment > Notes [table 1] [row 2]"),
            "Assessment > Notes [table 1] [row 2]",
        )

    def test_get_rule_method_name_returns_immediate_caller(self) -> None:
        """
        Test that get rule method name returns immediate caller.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = DummyChecker()

        self.assertEqual(
            checker.call_get_rule_method_name_through_wrapper(),
            "DummyChecker.call_get_rule_method_name_through_wrapper",
        )

    def test_begin_and_end_sweep_base_noop(self) -> None:
        """
        Test that begin and end sweep base noop.

        Args:
            None.

        Returns:
            None. The assertions inside the test body enforce the expected behavior.
        """
        checker = DummyChecker()

        checker.begin_sweep()
        checker.end_sweep()


if __name__ == "__main__":
    unittest.main()
