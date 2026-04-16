"""Abbreviation-focused rules for IUCN assessments."""

import re
from typing import List

from ..violation import Violation
from .base import BaseChecker


class AbbreviationChecker(BaseChecker):
    """
    Apply the package's abbreviation and Latin-term style rules.

    Purpose:
        This class groups related rules within the rules-based assessment workflow.
    """

    def __init__(self):
        """
        Initialise the abbreviation checker.

        Args:
            None.

        Returns:
            None.
        """
        super().__init__()

    def check_text(self, section_name: str, text: str) -> List[Violation]:
        """
        Run all abbreviation checks against one parsed section.

        The dispatcher currently combines:
        - ``check_eg_and_ie(...)``
        - ``check_abbreviation_formats(...)``
        - ``check_latin_terms_without_period(...)``
        - ``check_et_al(...)``
        - ``check_title_abbreviations(...)``

        Args:
            section_name (str): Parsed section key supplied by the caller.
            text (str): Parsed section text supplied by the caller.

        Returns:
            List[Violation]: Violations produced by this method.
        """
        violations = []
        violations.extend(self.check_eg_and_ie(section_name, text))
        violations.extend(self.check_abbreviation_formats(section_name, text))
        violations.extend(self.check_latin_terms_without_period(section_name, text))
        violations.extend(self.check_et_al(section_name, text))
        violations.extend(self.check_title_abbreviations(section_name, text))
        return violations

    def check_eg_and_ie(self, section_name: str, text: str) -> List[Violation]:
        """
        Flag standalone ``e.g.`` / ``i.e.`` variants in body text.

        The rule strips simple inline italic, bold, superscript, and subscript
        tags before matching. It then looks for standalone case-insensitive
        variants of ``e.g.`` and ``i.e.``, including dotted, partly dotted,
        and undotted forms such as ``E.g.``, ``eg``, ``I.e.``, and ``ie``.

        Matches are only reported when the abbreviation appears as its own
        token. Letter sequences inside larger words, such as ``segment``,
        ``diet``, ``species``, or ``siege``, are intentionally ignored.

        Suggested fixes are plain-English replacements:
        - ``e.g.`` -> ``for example,``
        - ``i.e.`` -> ``that is,``

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

        eg_pattern = re.compile(
            r'(?<!\w)e\.?\s*g\.?(?=(?:\W|$))',
            re.IGNORECASE,
        )
        for match in eg_pattern.finditer(cleaned_text):
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message="Avoid 'e.g.' in body text; use 'for example' instead",
                suggested_fix="for example,",
            ))

        ie_pattern = re.compile(
            r'(?<!\w)i\.?\s*e\.?(?=(?:\W|$))',
            re.IGNORECASE,
        )
        for match in ie_pattern.finditer(cleaned_text):
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message="Avoid 'i.e.' in body text; use 'that is' instead",
                suggested_fix="that is,",
            ))

        return violations

    def check_abbreviation_formats(self, section_name: str, text: str) -> List[Violation]:
        """
        Normalize a small hard-coded set of abbreviation formats.

        The rule strips simple inline italic, bold, superscript, and subscript
        tags before matching. It then checks these specific patterns only:
        - ``etc`` -> ``etc.``
        - ``in litt`` -> ``in litt.``
        - ``pers comm`` / ``pers comm.`` / ``pers. comm`` -> ``pers. comm.``
        - ``pers obs`` / ``pers obs.`` / ``pers. obs`` -> ``pers. obs.``
        - ``Prof`` -> ``Prof.``

        Already-correct forms such as ``etc.``, ``in litt.``, ``pers. comm.``,
        ``pers. obs.``, and ``Prof.`` are ignored. Abbreviations outside this
        fixed shortlist, such as ``Rev``, are also ignored.

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

        etc_pattern = re.compile(r'\betc\b(?!\.)', re.IGNORECASE)
        for match in etc_pattern.finditer(cleaned_text):
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message="Use 'etc.' with period",
                suggested_fix="etc.",
            ))

        inlitt_pattern = re.compile(r'\bin\s+litt\b(?!\.)', re.IGNORECASE)
        for match in inlitt_pattern.finditer(cleaned_text):
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message="Use 'in litt.' not 'in litt', if referring to in published literature",
                suggested_fix="in litt.",
            ))

        perscomm_pattern = re.compile(
            r'\b(?:pers\.\s*comm\b(?!\.)|pers\s+comm\.?(?=\W|$))',
            re.IGNORECASE,
        )
        for match in perscomm_pattern.finditer(cleaned_text):
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message="Use 'pers. comm.' format",
                suggested_fix="pers. comm.",
            ))

        persobs_pattern = re.compile(
            r'\b(?:pers\.\s*obs\b(?!\.)|pers\s+obs\.?(?=\W|$))',
            re.IGNORECASE,
        )
        for match in persobs_pattern.finditer(cleaned_text):
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message="Use 'pers. obs.' format",
                suggested_fix="pers. obs.",
            ))

        prof_pattern = re.compile(r'\bProf\b(?!\.)', re.IGNORECASE)
        for match in prof_pattern.finditer(cleaned_text):
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message="Use 'Prof.' with period",
                suggested_fix="Prof.",
            ))

        return violations

    def check_latin_terms_without_period(self, section_name: str, text: str) -> List[Violation]:
        """
        Enforce italics and no-period formatting for selected Latin terms.

        The rule checks only a fixed list of Latin expressions:
        ``in situ``, ``ex situ``, ``ad hoc``, ``in vivo``, ``in vitro``,
        ``sensu lato``, ``sensu stricto``, ``per se``, ``de facto``, and
        ``vice versa``.

        Matching is done in two stages:
        - non-italic style tags are stripped while ``<i>`` / ``<em>`` are kept
        - the matched term must appear inside simple HTML italics
        - if italicized, the term is then checked for internal periods

        The same message is used for both failure paths:
        ``Latin term '<term>' must be italicized and not contain periods``

        Examples flagged include plain-text ``in situ``, plain-text
        ``de facto``, ``<i>in. situ</i>``, ``<i>in.situ</i>``, and
        ``<i>in situ</i>.``.

        Correct italicized undotted forms such as ``<i>in situ</i>`` and
        ``<em>de facto</em>`` are ignored. Terms outside the fixed list, such
        as ``status quo``, are not checked. Markdown italics are not parsed.

        Args:
            section_name (str): Parsed section key supplied by the caller.
            text (str): Parsed section text supplied by the caller.

        Returns:
            List[Violation]: Violations produced by this method.
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=False,
            bold=True,
            superscript=True,
            subscript=True,
        )

        latin_terms = [
            'in situ',
            'ex situ',
            'ad hoc',
            'in vivo',
            'in vitro',
            'sensu lato',
            'sensu stricto',
            'per se',
            'de facto',
            'vice versa',
        ]

        for term in latin_terms:
            escaped_term = re.escape(term)
            core_pattern = escaped_term.replace(r'\ ', r'(?:\.\s*|\s+)')
            period_regex = re.compile(
                rf'(?<!\w)(?P<term>{core_pattern})(?P<trailing>[.!?,;:]?)(?=\W|$)',
                re.IGNORECASE,
            )

            for match in period_regex.finditer(cleaned_text):
                original_start = index_map[match.start('term')]
                original_end = index_map[match.end('term') - 1] + 1

                if not self.is_inside_italic(text, original_start, original_end):
                    violations.append(self.create_violation(
                        section_name=section_name,
                        text=text,
                        span=(original_start, original_end),
                        message=f"Latin term '{term}' must be italicized and not contain periods",
                        suggested_fix=f"Italicized: {term}",
                    ))
                    continue

                stripped_term_text = self.strip_italic_markup_around_term(
                    text,
                    original_start,
                    original_end,
                )
                if '.' in stripped_term_text:
                    violations.append(self.create_violation(
                        section_name=section_name,
                        text=text,
                        span=(original_start, original_end),
                        message=f"Latin term '{term}' must be italicized and not contain periods",
                        suggested_fix=f"Italicized: {term}",
                    ))

        return violations

    def check_et_al(self, section_name: str, text: str) -> List[Violation]:
        """
        Require the citation term ``et al.`` to use the preferred styling.

        This rule checks only ``et al`` / ``et al.``-style text. It strips
        simple inline italic, bold, superscript, and subscript tags for
        matching, then maps the match back to the original text.

        The preferred form is italicized ``et al.``. Italics are checked on the
        letters ``et al`` only, not on the trailing period, so both of these
        are accepted:
        - ``<i>et al.</i>``
        - ``<i>et al</i>.``

        The rule flags:
        - missing final periods, such as ``et al``
        - internal dotted variants, such as ``et. al.``
        - plain-text forms such as ``et al.`` that are missing italics

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

        etal_pattern = re.compile(
            r'(?<!\w)(?P<term>et(?:\.\s*|\s+)al)(?P<trailing>\.?)(?=\W|$)',
            re.IGNORECASE,
        )
        for match in etal_pattern.finditer(cleaned_text):
            original_term_start = index_map[match.start('term')]
            original_term_end = index_map[match.end('term') - 1] + 1
            original_match_start = index_map[match.start()]
            original_match_end = index_map[match.end() - 1] + 1

            is_italicized = self.is_inside_italic(text, original_term_start, original_term_end)
            has_canonical_term = re.fullmatch(r'et\s+al', match.group('term'), re.IGNORECASE) is not None
            has_final_period = match.group('trailing') == "."
            if (not is_italicized) or (not has_canonical_term) or (not has_final_period):
                violations.append(self.create_violation(
                    section_name=section_name,
                    text=text,
                    span=(original_match_start, original_match_end),
                    message="Use italicized 'et al.'",
                    suggested_fix="Italicized: et al.",
                ))

        return violations

    def check_title_abbreviations(self, section_name: str, text: str) -> List[Violation]:
        """
        Enforce periodless UK-style courtesy-title abbreviations.

        The rule strips simple inline italic, bold, superscript, and subscript
        tags before matching and checks only this fixed set:
        - ``Dr.``
        - ``Mr.``
        - ``Mrs.``
        - ``Ms.``

        Matching is case-insensitive and works before whitespace, punctuation,
        or end of text, so forms such as ``Dr. Green``, ``mr.``, ``Mrs.,``,
        and ``Ms.)`` are all flagged.

        Already-correct forms without a period, such as ``Dr`` and ``Mr``, are
        ignored. Titles outside the fixed set, such as ``Prof.``, ``Rev.``, and
        ``Assoc. Prof.``, are not handled here.

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

        for title in ['Dr', 'Mr', 'Mrs', 'Ms']:
            title_pattern = re.compile(
                rf'\b{title}\.(?=(?:\W|$))',
                re.IGNORECASE,
            )
            for match in title_pattern.finditer(cleaned_text):
                start = index_map[match.start()]
                end = index_map[match.end() - 1] + 1
                violations.append(self.create_violation(
                    section_name=section_name,
                    text=text,
                    span=(start, end),
                    message=f"Use '{title}' without period (UK style)",
                    suggested_fix=title,
                ))

        return violations

    def is_inside_italic(self, text: str, start: int, end: int) -> bool:
        """
        Return whether a span sits inside simple ``<i>`` or ``<em>`` markup.

        This is a lightweight heuristic rather than a full HTML parser. It
        checks whether the most recent opening ``<i>`` / ``<em>`` tag before the
        span is later than the most recent closing tag and whether a matching
        closing tag appears after the span.

        Args:
            text (str): Parsed section text supplied by the caller.
            start (int): Input value used by this method.
            end (int): Input value used by this method.

        Returns:
            bool: Boolean result described by the summary line above.
        """
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
            return ('</i>' in after) or ('</em>' in after)

        return False

    def strip_italic_markup_around_term(self, text: str, start: int, end: int) -> str:
        """
        Return the enclosing italic fragment with ``<i>`` / ``<em>`` removed.

        This helper expands the matched span outward to the surrounding simple
        italic block, removes only ``<i>`` / ``<em>`` tags, and preserves any
        trailing punctuation that should still count when period use is being
        checked.

        Args:
            text (str): Parsed section text supplied by the caller.
            start (int): Input value used by this method.
            end (int): Input value used by this method.

        Returns:
            str: String value produced by this method.
        """
        open_i = text.rfind('<i>', 0, start)
        open_em = text.rfind('<em>', 0, start)

        block_start = start
        close_tag = None
        if open_i > open_em:
            block_start = open_i
            close_tag = '</i>'
        elif open_em > open_i:
            block_start = open_em
            close_tag = '</em>'

        block_end = end
        if close_tag is not None:
            close_index = text.find(close_tag, end)
            if close_index != -1:
                block_end = close_index + len(close_tag)
                if block_end < len(text) and text[block_end] in '.!?,;:':
                    block_end += 1

        fragment = text[block_start:block_end]
        return re.sub(r'</?(?:i|em)>', '', fragment, flags=re.IGNORECASE)
