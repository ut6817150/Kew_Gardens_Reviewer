"""Formatting checker for IUCN assessments (italics, bold, etc.).

This checker expects text to contain HTML formatting tags:
- <i>text</i> or <em>text</em> for italics
- <b>text</b> or <strong>text</strong> for bold
"""

import re
from typing import List, Optional, Set

from ..violation import Violation
from .base import BaseChecker


class FormattingChecker(BaseChecker):
    """Checker for formatting rules such as scientific-name italics."""

    def __init__(self):
        super().__init__()
        self._collected_higher_taxonomy_names: Set[str] = set()
        self._collected_genus_name: Optional[str] = None
        self._collected_species_name: Optional[str] = None

    def begin_sweep(self) -> None:
        """Reset temporary taxonomy names before processing a full report."""
        self._collected_higher_taxonomy_names.clear()
        self._collected_genus_name = None
        self._collected_species_name = None

    def end_sweep(self) -> None:
        """Clear temporary taxonomy names after processing a full report."""
        self._collected_higher_taxonomy_names.clear()
        self._collected_genus_name = None
        self._collected_species_name = None

    def check_text(self, section_name: str, text: str) -> List[Violation]:
        """Check for formatting violations."""
        violations = []
        violations.extend(self.check_genus_and_species(section_name, text))
        violations.extend(self.check_higher_order_taxonomy_formatting(section_name, text))
        violations.extend(self.check_eoo_aoo_capitalization(section_name, text))
        return violations

    def check_eoo_aoo_capitalization(self, section_name: str, text: str) -> List[Violation]:
        """Check capitalization of spelled-out EOO/AOO phrases.

        This method strips simple inline style tags first:
        ``<i>``, ``<em>``, ``<b>``, ``<strong>``, ``<sup>``, and ``<sub>``.

        This method checks the full phrases:
        - ``extent of occurrence``
        - ``area of occupancy``

        It treats the fully lowercase form as the default correct form and
        flags capitalized or partially capitalized variants such as:
        - ``Extent of Occurrence``
        - ``Area of Occupancy``
        - ``Extent of occurrence``
        - ``Area of occupancy``
        - ``extent of Occurrence``

        It can catch these forms:
        - mid-sentence, for example ``the Extent of Occurrence was revised``
        - after punctuation such as ``(``, `:` and `,`
        - at the start of a text block
        - after ``?`` and ``!``

        There is one explicit exception:
        if the phrase is written with only the first word capitalized and is
        at the start of a paragraph or immediately preceded by ``. ``, ``? ``,
        or ``: ``, it is treated as an allowed sentence start.
        Examples allowed:
        - ``Extent of occurrence remained restricted.``
        - ``This was revised. Extent of occurrence remained restricted.``
        - ``Was this revised? Extent of occurrence was updated.``
        - ``Summary: Area of occupancy was recalculated.``
        - ``That was updated. Area of occupancy stayed small.``

        Examples still flagged:
        - ``The Extent of occurrence was revised.``
        - ``Summary, Area of occupancy was recalculated.``
        - ``(Extent of occurrence) was revised.``

        Examples not checked:
        - ``EOO`` and ``AOO`` abbreviations
        - misspelled forms such as ``extent of occurence``
        - reworded phrases such as ``occupied area``
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )
        terms = (
            'extent of occurrence',
            'area of occupancy',
        )

        def is_allowed_sentence_start(index: int) -> bool:
            if index == 0:
                return True
            if index >= 2 and cleaned_text[index - 2:index] in {'. ', '? ', ': '}:
                return True
            return False

        for correct in terms:
            phrase_pattern = re.escape(correct).replace(r'\ ', r'\s+')
            pattern = re.compile(rf'\b{phrase_pattern}\b', re.IGNORECASE)
            for match in pattern.finditer(cleaned_text):
                matched_phrase = match.group(0)
                normalized_phrase = ' '.join(matched_phrase.split())
                sentence_start_phrase = correct.capitalize()

                if normalized_phrase == correct:
                    continue
                if normalized_phrase == sentence_start_phrase and is_allowed_sentence_start(match.start()):
                    continue

                original_start = index_map[match.start()]
                original_end = index_map[match.end() - 1] + 1
                violations.append(self.create_violation(
                    section_name=section_name,
                    text=text,
                    span=(original_start, original_end),
                    message=f"Use lowercase: '{correct}' not '{matched_phrase}'",
                    suggested_fix=correct,
                ))

        return violations

    def check_higher_order_taxonomy_formatting(self, section_name: str, text: str) -> List[Violation]:
        """Check harvested higher-order taxonomy names for capitalization/italics.

        This method strips non-italic inline style tags first:
        ``<b>``, ``<strong>``, ``<sup>``, and ``<sub>``.
        It preserves ``<i>`` / ``<em>`` because italicization is part of the
        rule being checked.

        During a full-report sweep, this method first looks for taxonomy-ladder
        entries such as:
        ``PLANTAE - TRACHEOPHYTA - MAGNOLIOPSIDA - FABALES - FABACEAE - Acrocarpus - fraxinifolius``.
        When it finds one, it does not report violations for that value.
        Instead, it harvests the all-uppercase higher-order taxonomy names,
        normalizes them to title case (for example ``FABACEAE`` -> ``Fabaceae``),
        stores them temporarily, and uses them while checking the remaining
        sections in the same sweep. The temporary list is cleared when the
        sweep ends.

        It then checks two things at once for those harvested names:
        - whether the name starts with a capital letter
        - whether the name is free of surrounding ``<i>...</i>`` or
          ``<em>...</em>`` markup

        Examples flagged:
        - after harvesting taxonomy names from a ladder entry:
          ``plantae`` -> suggests ``Plantae``
        - after harvesting taxonomy names from a ladder entry:
          ``<i>Magnoliopsida</i>`` -> suggests ``Magnoliopsida``
        - after harvesting taxonomy names from a ladder entry:
          ``<i>Fabaceae</i>`` -> suggests ``Fabaceae``

        Examples not flagged:
        - the taxonomy ladder entry that supplied the harvested names
        - harvested names already written in correct title case without italics
        - ``orchidaceae`` or ``Felidae`` before any ladder harvest
        - non-harvested taxonomy-like words, because this method no longer
          infers names from suffixes alone
        """
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=False,
            bold=True,
            superscript=True,
            subscript=True,
        )

        if self.collect_taxonomy_names_from_ladder(cleaned_text):
            return []

        violations = []
        seen_matches = set()

        for proper_name in sorted(self._collected_higher_taxonomy_names, key=len, reverse=True):
            for cleaned_span, message, suggested_fix in self.find_taxonomy_name_violations(cleaned_text, proper_name):
                original_span = (
                    index_map[cleaned_span[0]],
                    index_map[cleaned_span[1] - 1] + 1,
                )
                match_key = (original_span, suggested_fix)
                if match_key in seen_matches:
                    continue
                seen_matches.add(match_key)
                violations.append(self.create_violation(
                    section_name=section_name,
                    text=text,
                    span=original_span,
                    message=message,
                    suggested_fix=suggested_fix,
                ))

        return violations

    def check_genus_and_species(self, section_name: str, text: str) -> List[Violation]:
        """Check harvested genus/species names for italics and casing.

        This method strips non-italic inline style tags first:
        ``<b>``, ``<strong>``, ``<sup>``, and ``<sub>``.
        It preserves ``<i>`` / ``<em>`` because italicization is part of the
        rule being checked.

        During a full-report sweep, this method looks for taxonomy-ladder
        entries such as:
        ``PLANTAE - TRACHEOPHYTA - MAGNOLIOPSIDA - FABALES - FABACEAE - Acrocarpus - fraxinifolius``.
        From that ladder, it harvests:
        - the second-to-last segment as the genus
        - the last segment as the species

        The ladder entry itself is not checked for violations by this method.
        Instead, the harvested genus and species are stored temporarily and
        checked against the remaining sections in the same sweep.

        The applied rules are:
        - occurrences of the genus must be italicized
        - occurrences of the species must be italicized
        - the genus must start with a capital letter
        - the species must start with a lowercase letter

        Examples flagged after harvesting ``Acrocarpus`` / ``fraxinifolius``:
        - ``Acrocarpus`` -> suggests ``<i>Acrocarpus</i>``
        - ``acrocarpus`` -> suggests ``<i>Acrocarpus</i>``
        - ``fraxinifolius`` -> suggests ``<i>fraxinifolius</i>``
        - ``Fraxinifolius`` -> suggests ``<i>fraxinifolius</i>``
        - ``<i>Fraxinifolius</i>`` -> suggests ``<i>fraxinifolius</i>``

        Examples not flagged:
        - the taxonomy ladder entry that provided the genus/species names
        - ``<i>Acrocarpus</i>``
        - ``<i>fraxinifolius</i>``
        - names before any taxonomy ladder has been harvested in the current sweep
        """
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=False,
            bold=True,
            superscript=True,
            subscript=True,
        )

        if self.collect_taxonomy_names_from_ladder(cleaned_text):
            return []

        violations = []
        name_rules = (
            (self._collected_genus_name, True),
            (self._collected_species_name, False),
        )

        for proper_name, should_be_capitalized in name_rules:
            if not proper_name:
                continue

            name_pattern = re.escape(proper_name).replace(r'\ ', r'\s+')
            pattern = re.compile(
                rf'(?P<markup><(?:i|em)>)?(?P<name>\b{name_pattern}\b)(?(markup)</(?:i|em)>)',
                re.IGNORECASE,
            )

            for match in pattern.finditer(cleaned_text):
                matched_name = match.group('name')
                normalized_name = (
                    proper_name if should_be_capitalized else proper_name.lower()
                )
                is_italicized = (
                    match.group('markup') is not None
                    or self.is_inside_italic(cleaned_text, match.start(), match.end())
                )
                has_expected_case = matched_name == normalized_name

                if is_italicized and has_expected_case:
                    continue

                message = (
                    f"Scientific names should be italicized and use correct case: "
                    f"'<i>{normalized_name}</i>'"
                )
                original_span = (
                    index_map[match.start()],
                    index_map[match.end() - 1] + 1,
                )
                violations.append(self.create_violation(
                    section_name=section_name,
                    text=text,
                    span=original_span,
                    message=message,
                    suggested_fix=f"<i>{normalized_name}</i>",
                ))

        return violations

    def collect_taxonomy_names_from_ladder(self, text: str) -> bool:
        """Harvest higher taxonomy names plus genus/species from a ladder entry."""
        segments = [segment.strip() for segment in text.split(' - ') if segment.strip()]
        if len(segments) < 6:
            return False

        uppercase_segments = [
            segment for segment in segments
            if re.fullmatch(r'[A-Z][A-Z]+', segment)
        ]
        if len(uppercase_segments) < 4:
            return False

        genus_segment = segments[-2]
        species_segment = segments[-1]
        if not re.fullmatch(r'[A-Za-z][A-Za-z-]*', genus_segment):
            return False
        if not re.fullmatch(r'[A-Za-z][A-Za-z-]*', species_segment):
            return False

        for segment in uppercase_segments:
            self._collected_higher_taxonomy_names.add(segment.title())
        self._collected_genus_name = genus_segment.capitalize()
        self._collected_species_name = species_segment.lower()
        return True

    def find_taxonomy_name_violations(self, text: str, proper_name: str) -> List[tuple]:
        """Return violations for a harvested higher-order taxonomy name."""
        violations = []
        name_pattern = re.escape(proper_name).replace(r'\ ', r'\s+')
        pattern = re.compile(
            rf'(?P<markup><(?:i|em)>)?(?P<name>\b{name_pattern}\b)(?(markup)</(?:i|em)>)',
            re.IGNORECASE,
        )

        for match in pattern.finditer(text):
            matched_name = match.group('name')
            is_italicized = match.group('markup') is not None
            if matched_name == proper_name and not is_italicized:
                continue

            violations.append((
                match.span(),
                f"Family/taxonomy names should be capitalized and not italicized: '{proper_name}'",
                proper_name,
            ))

        return violations
    def is_inside_italic(self, text: str, start: int, end: int) -> bool:
        """Check if a position is inside italic tags."""
        before = text[:start]
        after = text[end:]

        open_i = before.rfind('<i>')
        open_em = before.rfind('<em>')
        last_open = max(open_i, open_em)
        if last_open == -1:
            return False

        close_i = before.rfind('</i>')
        close_em = before.rfind('</em>')
        last_close = max(close_i, close_em)

        if last_open > last_close:
            close_after_i = after.find('</i>')
            close_after_em = after.find('</em>')
            if close_after_i != -1 or close_after_em != -1:
                return True

        return False
