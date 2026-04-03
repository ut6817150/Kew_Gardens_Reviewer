"""IUCN-specific terminology checker."""

import re
from typing import List

from ..violation import Violation
from .base import BaseChecker


class IUCNTermsChecker(BaseChecker):
    """Checker for IUCN-specific terminology and formatting."""

    CATEGORIES = [
        ('Critically Endangered', 'CR'),
        ('Endangered', 'EN'),
        ('Vulnerable', 'VU'),
        ('Near Threatened', 'NT'),
        ('Least Concern', 'LC'),
        ('Data Deficient', 'DD'),
        ('Not Evaluated', 'NE'),
        ('Extinct', 'EX'),
        ('Extinct in the Wild', 'EW'),
    ]

    def __init__(self):
        super().__init__()

    def check_text(self, section_name: str, text: str) -> List[Violation]:
        """Check for IUCN terminology violations."""
        violations = []
        violations.extend(self.check_the_iucn(section_name, text))
        violations.extend(self.check_CE_abbreviation(section_name, text))
        violations.extend(self.check_category_capitalization(section_name, text))
        violations.extend(self.check_threatened_case(section_name, text))
        return violations

    def check_the_iucn(self, section_name: str, text: str) -> List[Violation]:
        """Check for the non-preferred phrase ``the IUCN``.

        This method first strips simple inline style markers such as italics,
        bold, superscript, and subscript tags, then checks the cleaned text
        case-insensitively for ``the IUCN`` and maps any match span back to
        the original rich-text input.

        This method flags ``the IUCN`` case-insensitively and suggests using
        ``IUCN`` instead.

        Examples flagged:
        ``the IUCN Red List``
        ``The IUCN categories``
        ``THE IUCN framework``

        Examples not flagged:
        ``IUCN Red List``
        ``IUCN categories``
        text that does not contain the article ``the`` directly before ``IUCN``

        This is a simple style rule only. It does not rewrite surrounding
        grammar or treat quotations and citations differently.
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )
        pattern = re.compile(r'\bthe\s+IUCN\b', re.IGNORECASE)
        for match in pattern.finditer(cleaned_text):
            original_start = index_map[match.start()]
            original_end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(original_start, original_end),
                message="Use 'IUCN' not 'the IUCN'",
                suggested_fix="IUCN",
            ))
        return violations

    def check_CE_abbreviation(self, section_name: str, text: str) -> List[Violation]:
        """Check the incorrect Red List abbreviation ``CE``.

        This method first strips simple inline style markers such as italics,
        bold, superscript, and subscript tags, then checks the cleaned text
        for standalone ``CE`` case-insensitively and maps any match span back
        to the original rich-text input.

        This method flags standalone `CE` case-insensitively and suggests `CR`
        for `Critically Endangered`.

        Examples flagged:
        `CE`
        `ce`
        `(Ce)`

        Examples not flagged:
        `CR`
        `concern`
        `species`
        any longer word that merely contains the letters `ce`
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )
        ce_pattern = re.compile(r'\bCE\b', re.IGNORECASE)
        for match in ce_pattern.finditer(cleaned_text):
            original_start = index_map[match.start()]
            original_end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(original_start, original_end),
                message="Use 'CR' not 'CE' for Critically Endangered",
                suggested_fix="CR",
            ))
        return violations

    def check_category_capitalization(self, section_name: str, text: str) -> List[Violation]:
        """Check canonical case for Red List category names and abbreviations.

        This method first strips simple inline style markers such as italics,
        bold, superscript, and subscript tags, then checks the cleaned text
        against the canonical case defined in `CATEGORIES` and maps any match
        span back to the original rich-text input.

        This method loops through every `(full_name, abbreviation)` pair in
        `CATEGORIES` and checks both parts separately.

        For the full category name, it flags any case-insensitive whole-phrase
        match that is not written exactly in the canonical case from the list.
        Examples flagged:
        `critically endangered`
        `Near threatened`
        `extinct in the wild`

        For the abbreviation, it flags any case-insensitive standalone match
        that is not written exactly in the canonical case from the list.
        Examples flagged:
        `cr`
        `Vu`
        `ew`

        Examples not flagged:
        `Critically Endangered`
        `Near Threatened`
        `Extinct in the Wild`
        `CR`
        `VU`
        `EW`
        `Ex-situ conservation`

        The abbreviation check rejects matches that touch letters, digits, or
        hyphens, so it should not catch letter sequences inside larger words or
        hyphenated compounds such as:
        `crab`
        `species`
        `newt`
        `Ex-situ`

        The rule does not treat simple HTML bold or italics as an exception.
        Examples:
        `<i>cr</i>` is still flagged
        `<b>Vu</b>` is still flagged
        `<i>CR</i>` is not flagged because the abbreviation case is already correct
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )

        for category, abbrev in self.CATEGORIES:
            category_pattern = re.compile(rf'\b{re.escape(category)}\b', re.IGNORECASE)
            for match in category_pattern.finditer(cleaned_text):
                if match.group(0) != category:
                    original_start = index_map[match.start()]
                    original_end = index_map[match.end() - 1] + 1
                    violations.append(self.create_violation(
                        section_name=section_name,
                        text=text,
                        span=(original_start, original_end),
                        message=f"Red List category should be capitalized: '{category}'",
                        suggested_fix=category,
                    ))

            abbrev_pattern = re.compile(
                rf'(?<![\w-]){re.escape(abbrev)}(?![\w-])',
                re.IGNORECASE,
            )
            for match in abbrev_pattern.finditer(cleaned_text):
                if match.group(0) != abbrev:
                    original_start = index_map[match.start()]
                    original_end = index_map[match.end() - 1] + 1
                    violations.append(self.create_violation(
                        section_name=section_name,
                        text=text,
                        span=(original_start, original_end),
                        message=f"Red List category abbreviation should be capitalized: '{abbrev}'",
                        suggested_fix=abbrev,
                    ))

        return violations

    def check_threatened_case(self, section_name: str, text: str) -> List[Violation]:
        """Check mid-sentence capitalization of ``Threatened``.

        This method first strips simple inline style markers such as italics,
        bold, superscript, and subscript tags, then looks for the capitalized
        word `Threatened` in cleaned text when it appears after a lowercase
        letter and a space, and maps any match span back to the original
        rich-text input.

        This method looks for the capitalized word `Threatened` when it appears
        after a lowercase letter and a space, and suggests the lowercase form
        `threatened`.

        Examples flagged:
        `many Threatened species`
        `the Threatened categories include CR, EN and VU`

        Examples not flagged:
        `Threatened species were reviewed.` at sentence start
        `many threatened species`
        larger words that merely contain the same letters

        This is a narrow regex rule rather than a context-aware parser.
        One limitation is that it can also match the `Threatened` part of
        `Near Threatened` when that phrase appears mid-sentence.
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )
        pattern = re.compile(r'(?<=[a-z]\s)Threatened\b')
        for match in pattern.finditer(cleaned_text):
            original_start = index_map[match.start()]
            original_end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(original_start, original_end),
                message="Use lowercase 'threatened' when referring to CR/EN/VU species collectively",
                suggested_fix="threatened",
            ))
        return violations
