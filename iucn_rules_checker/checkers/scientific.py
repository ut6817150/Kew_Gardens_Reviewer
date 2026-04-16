"""Scientific name formatting checker for IUCN assessments."""

import re
from typing import List

from ..violation import Violation
from .base import BaseChecker


class ScientificNameChecker(BaseChecker):
    """
    Checker for scientific name formatting rules.

    Purpose:
        This class groups related rules within the rules-based assessment workflow.
    """

    def __init__(self):
        """
        Initialise the scientific-name checker.

        Args:
            None.

        Returns:
            None.
        """
        super().__init__()

    def check_text(self, section_name: str, text: str) -> List[Violation]:
        """
        Check for scientific name formatting violations.

        Args:
            section_name (str): Parsed section key supplied by the caller.
            text (str): Parsed section text supplied by the caller.

        Returns:
            List[Violation]: Violations produced by this method.
        """
        violations = []
        violations.extend(self.check_species_abbreviations(section_name, text))
        return violations

    def check_species_abbreviations(self, section_name: str, text: str) -> List[Violation]:
        """
        Check `sp` / `spp` abbreviations after removing simple style tags.

        This method strips simple inline style tags first:
        `<i>`, `<em>`, `<b>`, `<strong>`, `<sup>`, and `<sub>`.
        It then checks the cleaned text for:
        - `spp` used without the final period
        - `sp` used without the final period

        Examples flagged:
        `sp` -> suggests `sp.`
        `spp` -> suggests `spp.`
        `<i>sp</i>` -> suggests `sp.`
        `<b>spp</b>` -> suggests `spp.`
        `<i>s</i><i>p</i>` -> suggests `sp.`
        `<sup>s</sup><sub>p</sub>` -> suggests `sp.`

        Examples not flagged:
        `sp.`
        `spp.`
        `<i>sp.</i>`
        `<b>spp.</b>`
        `<sup>sp.</sup>`
        `species`
        `sppx`

        What it misses:
        it only strips those simple style tags and does not parse arbitrary
        markup.
        It only checks the abbreviations `sp` and `spp`.
        It does not validate whether the surrounding scientific-name usage is
        correct beyond the missing-period rule.

        Args:
            section_name (str): Parsed section key supplied by the caller.
            text (str): Parsed section text supplied by the caller.

        Returns:
            List[Violation]: Violations produced by this method.
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )

        spp_pattern = re.compile(r'\b(spp)(?!\.)\b')
        for match in spp_pattern.finditer(cleaned_text):
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message="Use 'spp.' (with period) for multiple species",
                suggested_fix="spp.",
            ))

        sp_pattern = re.compile(r'\b(sp)(?!p|\.|\w)')
        for match in sp_pattern.finditer(cleaned_text):
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message="Use 'sp.' (with period) for single unspecified species",
                suggested_fix="sp.",
            ))

        return violations
