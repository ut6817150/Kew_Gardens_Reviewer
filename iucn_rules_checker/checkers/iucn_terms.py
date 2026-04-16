"""IUCN-specific terminology checker."""

import re
from typing import List

from ..violation import Violation
from .base import BaseChecker


class IUCNTermsChecker(BaseChecker):
    """
    Checker for IUCN-specific terminology and formatting.

    Purpose:
        This class groups related rules within the rules-based assessment workflow.
    """

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
        """
        Initialise the IUCN terminology checker.

        Args:
            None.

        Returns:
            None.
        """
        super().__init__()

    def check_text(self, section_name: str, text: str) -> List[Violation]:
        """
        Check for IUCN terminology violations.

        Args:
            section_name (str): Parsed section key supplied by the caller.
            text (str): Parsed section text supplied by the caller.

        Returns:
            List[Violation]: Violations produced by this method.
        """
        violations = []
        violations.extend(self.check_the_iucn(section_name, text))
        violations.extend(self.check_CE_abbreviation(section_name, text))
        violations.extend(self.check_category_full_name_capitalization(section_name, text))
        violations.extend(self.check_category_abbreviation_capitalization(section_name, text))
        violations.extend(self.check_threatened_case(section_name, text))
        return violations

    def check_the_iucn(self, section_name: str, text: str) -> List[Violation]:
        """
        Check for the non-preferred phrase ``the IUCN``.

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
        """
        Check the incorrect Red List abbreviation ``CE``.

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

    def check_category_full_name_capitalization(self, section_name: str, text: str) -> List[Violation]:
        """
        Check canonical case for Red List category full names.

        This method first strips simple inline style markers such as italics,
        bold, superscript, and subscript tags, then checks the cleaned text
        against the canonical full-name case defined in `CATEGORIES` and maps
        any match span back to the original rich-text input.

        This method loops through every full category name in `CATEGORIES`.
        It flags any case-insensitive whole-phrase match that is not written
        exactly in the canonical case from the list.
        Examples flagged:
        `critically endangered`
        `Near threatened`
        `extinct in the wild`

        Examples not flagged:
        `Critically Endangered`
        `Near Threatened`
        `Extinct in the Wild`
        `Critically Endangered`
        `critically endangered status` only if the category phrase itself is
        already correctly cased
        text that does not contain a full category phrase

        The rule does not treat simple HTML bold or italics as an exception.
        Examples:
        `<i>critically</i> <b>endangered</b>` is still flagged
        `<i>Critically</i> <b>Endangered</b>` is not flagged

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

        for category, _ in self.CATEGORIES:
            category_pattern = re.compile(rf'\b{re.escape(category)}\b', re.IGNORECASE)
            for match in category_pattern.finditer(cleaned_text):
                if category == "Endangered" and re.search(
                    r'critically\s+$',
                    cleaned_text[:match.start()],
                    re.IGNORECASE,
                ):
                    continue

                if category == "Extinct" and re.match(
                    r'\s+in\s+the\s+wild\b',
                    cleaned_text[match.end():],
                    re.IGNORECASE,
                ):
                    continue

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

        return violations

    def check_category_abbreviation_capitalization(self, section_name: str, text: str) -> List[Violation]:
        """
        Check canonical case for Red List category abbreviations.

        This method first strips simple inline style markers such as italics,
        bold, superscript, and subscript tags, then checks the cleaned text
        against the canonical abbreviation case defined in `CATEGORIES` and
        maps any match span back to the original rich-text input.

        This method loops through every category abbreviation in `CATEGORIES`.
        It flags any case-insensitive standalone match that is not written
        exactly in the canonical case from the list.
        Examples flagged:
        `cr`
        `Vu`
        `ew`

        Examples not flagged:
        `CR`
        `VU`
        `EW`
        `ex situ conservation`
        `Ex-situ conservation`

        The abbreviation check rejects matches that touch letters, digits, or
        hyphens, and it also skips the Latin phrase `ex situ`, so it should not
        catch letter sequences inside larger words or hyphenated compounds such
        as:
        `crab`
        `species`
        `newt`
        `ex situ`
        `Ex-situ`

        The rule does not treat simple HTML bold or italics as an exception.
        Examples:
        `<i>cr</i>` is still flagged
        `<b>Vu</b>` is still flagged
        `<i>CR</i>` is not flagged because the abbreviation case is already correct

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

        for _, abbrev in self.CATEGORIES:
            abbrev_pattern = re.compile(
                rf'(?<![\w-]){re.escape(abbrev)}(?![\w-])',
                re.IGNORECASE,
            )
            for match in abbrev_pattern.finditer(cleaned_text):
                if abbrev == "EX" and re.match(r'\s+situ\b', cleaned_text[match.end():], re.IGNORECASE):
                    continue

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
        """
        Check mid-sentence capitalization of ``Threatened``.

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
        `many Near Threatened species`
        larger words that merely contain the same letters

        This is a narrow regex rule rather than a context-aware parser.
        It skips the `Threatened` part of the category phrase `Near Threatened`.

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
        pattern = re.compile(r'(?<=[a-z]\s)Threatened\b')
        for match in pattern.finditer(cleaned_text):
            if cleaned_text[max(0, match.start() - 5):match.start()].lower() == "near ":
                continue

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
