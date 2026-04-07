"""Abbreviation checker for IUCN assessments."""

import re
from typing import List

from ..violation import Violation
from .base import BaseChecker


class AbbreviationChecker(BaseChecker):
    """Checker for abbreviation rules."""

    def __init__(self):
        super().__init__()

    def check_text(self, section_name: str, text: str) -> List[Violation]:
        """Check for abbreviation violations."""
        violations = []
        violations.extend(self.check_eg_and_ie(section_name, text))
        violations.extend(self.check_abbreviation_formats(section_name, text))
        violations.extend(self.check_latin_terms_without_period(section_name, text))
        violations.extend(self.check_et_al(section_name, text))
        violations.extend(self.check_title_abbreviations(section_name, text))
        return violations

    def check_eg_and_ie(self, section_name: str, text: str) -> List[Violation]:
        """Check for ``e.g.`` and ``i.e.``-style abbreviations in body text.

        This method strips simple inline bold/italic/superscript/subscript
        HTML tags first:
        ``<i>``, ``<em>``, ``<b>``, ``<strong>``, ``<sup>``, and ``<sub>``.

        Catches case-insensitive standalone variants of ``e.g.`` and ``i.e.``,
        including dotted, partly dotted, and undotted forms.
        Examples caught:
        - ``E.g.``
        - ``Eg.``
        - ``e.g.``
        - ``E.G.``
        - ``eg``
        - ``I.e.``
        - ``Ie.``
        - ``i.e.``
        - ``IE``
        - ``ie``

        It can match these at sentence start, mid-sentence, sentence end, and
        inside parentheses, as long as they appear as standalone abbreviations.

        Misses cases where the same letters occur inside larger words or where
        the surrounding context would require a more semantic judgement, such as
        quotations, copied headings, or bibliography-specific exceptions.
        Examples of missed cases:
        - ``segment``
        - ``diet``
        - ``species``
        - ``siege``
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
        """Check fixed abbreviation forms that should end with a period.

        This method strips simple inline bold/italic/superscript/subscript
        HTML tags first:
        ``<i>``, ``<em>``, ``<b>``, ``<strong>``, ``<sup>``, and ``<sub>``.

        Catches case-insensitive standalone uses of:
        - ``etc`` -> ``etc.``
        - ``in lit`` -> ``in lit.``
        - ``pers comm`` / ``pers comm.`` / ``pers. comm`` -> ``pers. comm.``
        - ``pers obs`` / ``pers obs.`` / ``pers. obs`` -> ``pers. obs.``
        - ``Prof`` -> ``Prof.``

        Misses forms outside those hardcoded patterns, such as heavily broken
        spacing/punctuation variants, different abbreviations not listed here,
        and already-correct forms that already include the expected period.
        Examples of missed cases:
        - ``p e r s comm``
        - ``in-lot`` or other non-matching typos
        - ``Rev`` or ``Assoc Prof`` because those abbreviations are not covered
        - ``etc.`` and ``pers. comm.`` because they are already in the expected form
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

        inlit_pattern = re.compile(r'\bin\s+lit\b(?!t)', re.IGNORECASE)
        for match in inlit_pattern.finditer(cleaned_text):
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message="Use 'in lit.' not 'in lit', if referring to in published literature",
                suggested_fix="in lit.",
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
        """Check Latin terms for italicization first, then for periods.

        This method checks a fixed list of Latin expressions in two stages:
        - first, it strips non-italic style tags such as bold, superscript, and
          subscript tags while preserving ``<i>...</i>`` and ``<em>...</em>``
        - first, it checks whether the matched term is wrapped in ``<i>...</i>``
          or ``<em>...</em>``
        - only if the term is italicized does it then strip the surrounding
          italics markup and check the resulting text for periods

        Both failure paths use the same message:
        - ``Latin term '<term>' must be italicized and not contain periods``

        Examples caught:
        - ``in situ`` in plain text
        - ``de facto`` in plain text
        - ``sensu. lato`` in plain text
        - ``<i>in. situ</i>``
        - ``<i>in.situ</i>``
        - ``<i>in.situ.</i>``
        - ``<i>in situ</i>.``

        Examples not caught:
        - ``<i>in situ</i>`` because it is italicized and undotted
        - ``<em>de facto</em>`` because it is italicized and undotted
        - ``status quo`` because it is not in the hardcoded list
        - ``*in situ*`` because Markdown italics are not parsed here
        - unusual malformed dotted forms such as ``in .. situ`` or ``in . situ``
        - hyphenated forms such as ``in-situ``
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
                        suggested_fix=f"<i>{term}</i>",
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
                        suggested_fix=f"<i>{term}</i>",
                    ))

        return violations

    def check_et_al(self, section_name: str, text: str) -> List[Violation]:
        """Check that ``et al.`` is italicized and uses the canonical punctuation.

        This method checks only ``et al`` / ``et al.`` and enforces the
        canonical wording ``et al.`` while checking italics on the letters
        ``et al`` only, not on the trailing period. That means both of these
        are accepted:
        - ``<i>et al.</i>``
        - ``<i>et al</i>.``

        It strips all simple style tags for matching, then maps the matched
        text back to the original rich text and checks whether the ``et al``
        portion sits inside ``<i>...</i>`` / ``<em>...</em>``.

        The rule flags:
        - missing final periods, such as ``et al``
        - internal dotted variants such as ``et. al.``
        - plain-text forms such as ``et al.`` that are missing italics

        Examples caught:
        - ``et al``
        - ``ET AL``
        - ``et al.``
        - ``et. al.``
        - ``<i>et al</i>``

        Examples not caught:
        - ``<i>et al.</i>`` because it is already in the expected form
        - ``<em>et al.</em>`` because it is already in the expected form
        - ``<i>et al</i>.`` because only ``et al`` needs to be italicized
        - unrelated abbreviations because this method only checks ``et al``
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
                    suggested_fix="<i>et al</i>.",
                ))

        return violations

    def check_title_abbreviations(self, section_name: str, text: str) -> List[Violation]:
        """Check UK-style title abbreviations that should not take periods.

        This method strips all simple inline style tags first:
        ``<i>``, ``<em>``, ``<b>``, ``<strong>``, ``<sup>``, and ``<sub>``.

        Catches case-insensitive uses of:
        - ``Dr.``
        - ``Mr.``
        - ``Mrs.``
        - ``Ms.``

        These are matched before whitespace, punctuation, or end of text, so
        examples such as ``Dr. Green``, ``mr.``, ``Mrs.,`` and ``Ms.)`` are
        all caught and suggested in their periodless UK-style forms.

        Misses titles that are not covered by this hardcoded list, as well as
        already-correct forms without a period.
        Examples of missed cases:
        - ``Prof.`` because it is checked elsewhere
        - ``Rev.`` and ``Assoc. Prof.`` because they are not covered here
        - ``Dr`` and ``Mr`` because they are already in the expected form
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
        """Return whether a matched range appears inside simple HTML italics."""
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
        """Strip surrounding HTML italics tags before checking a term for periods."""
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
