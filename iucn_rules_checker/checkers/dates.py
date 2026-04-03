"""Date formatting checker for IUCN assessments."""

import re
from typing import List

from ..violation import Violation
from .base import BaseChecker


class DateChecker(BaseChecker):
    """Checker for date formatting rules."""

    MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    MONTH_ABBREVIATIONS = [
        'Jan\\.?', 'Feb\\.?', 'Mar\\.?', 'Apr\\.?', 'May',
        'Jun\\.?', 'Jul\\.?', 'Aug\\.?', 'Sep\\.?', 'Sept\\.?',
        'Oct\\.?', 'Nov\\.?', 'Dec\\.?',
    ]
    MONTH_DAY_LIMITS = {
        'january': 31,
        'jan': 31,
        'february': 28,
        'feb': 28,
        'march': 31,
        'mar': 31,
        'april': 30,
        'apr': 30,
        'may': 31,
        'june': 30,
        'jun': 30,
        'july': 31,
        'jul': 31,
        'august': 31,
        'aug': 31,
        'september': 30,
        'sep': 30,
        'sept': 30,
        'october': 31,
        'oct': 31,
        'november': 30,
        'nov': 30,
        'december': 31,
        'dec': 31,
    }

    CENTURY_WORDS = {
        'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th',
        'fifth': '5th', 'sixth': '6th', 'seventh': '7th', 'eighth': '8th',
        'ninth': '9th', 'tenth': '10th', 'eleventh': '11th', 'twelfth': '12th',
        'thirteenth': '13th', 'fourteenth': '14th', 'fifteenth': '15th',
        'sixteenth': '16th', 'seventeenth': '17th', 'eighteenth': '18th',
        'nineteenth': '19th', 'twentieth': '20th', 'twenty-first': '21st'
    }

    def __init__(self):
        super().__init__()

    def check_text(self, section_name: str, text: str) -> List[Violation]:
        """Check for date formatting violations."""
        violations = []
        violations.extend(self.check_ordinal_dates(section_name, text))
        violations.extend(self.check_century_format(section_name, text))
        violations.extend(self.check_decade_format(section_name, text))
        return violations

    def check_ordinal_dates(self, section_name: str, text: str) -> List[Violation]:
        """Check ordinal day-month dates and basic calendar validity.

        This method strips simple inline style tags first:
        ``<i>``, ``<em>``, ``<b>``, ``<strong>``, ``<sup>``, and ``<sub>``.

        This method matches dates written as:
        - one or two digits
        - followed by an ordinal suffix such as ``st``, ``nd``, ``rd``, or ``th``
        - optionally followed by ``of``
        - followed by a full month name or a covered month abbreviation

        It flags two kinds of problems:
        - style issues where an ordinal date should be rewritten without the suffix
        - invalid day/month combinations such as ``31st April``

        Examples caught as style issues:
        - ``11th January`` -> suggests ``11 January``
        - ``3rd of May`` -> suggests ``3 May``
        - ``21st Jan`` -> suggests ``21 Jan``
        - ``4th Sept.`` -> suggests ``4 Sept.``

        Examples caught as invalid dates:
        - ``31st April`` because April has only 30 days
        - ``30th Feb.`` because February is treated as having 28 days by default
        - ``0th January`` because day 0 is impossible
        - ``29th February`` with a separate message telling the reader to check for leap year

        Examples not caught:
        - ``11 January`` because it is already cardinal rather than ordinal
        - ``January 11th`` because the method only matches day-month order
        - ``11/01/2024`` and ``2024-05-15`` because numeric and ISO dates are out of scope
        - ``11th Jnauary`` because misspelled months do not match the month list
        - ``29th February 2024`` is not confirmed as valid from the year, because the year is not used
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )
        months_pattern = '|'.join(self.MONTHS + self.MONTH_ABBREVIATIONS)
        pattern = re.compile(
            rf'\b(\d{{1,2}})(st|nd|rd|th)\s+(?:of\s+)?({months_pattern})(?=\W|$)',
            re.IGNORECASE
        )

        for match in pattern.finditer(cleaned_text):
            day = match.group(1)
            day_num = int(day)
            month = match.group(3)
            original_start = index_map[match.start()]
            original_end = index_map[match.end() - 1] + 1
            if not self.is_valid_day_month(day_num, month):
                normalized_month = self.normalize_month_token(month)
                if normalized_month in {'february', 'feb'} and day_num == 29:
                    violations.append(self.create_violation(
                        section_name=section_name,
                        text=text,
                        span=(original_start, original_end),
                        message=f"Check for leap year: '{match.group(0)}' is only valid in a leap year",
                        suggested_fix=None,
                    ))
                    continue

                max_day = self.MONTH_DAY_LIMITS[normalized_month]
                violations.append(self.create_violation(
                    section_name=section_name,
                    text=text,
                    span=(original_start, original_end),
                    message=f"Invalid calendar date: '{match.group(0)}' is not valid for {month} (maximum {max_day} days)",
                    suggested_fix=None,
                ))
                continue

            fix = f"{day} {month}"
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(original_start, original_end),
                message=f"Avoid ordinal dates; use '{fix}' instead of '{match.group(0)}'",
                suggested_fix=fix,
            ))

        return violations

    def normalize_month_token(self, month: str) -> str:
        """Normalize a month token for lookup in the day-limit table."""
        return month.strip().rstrip('.').lower()

    def is_valid_day_month(self, day: int, month: str) -> bool:
        """Return whether a day/month combination is calendar-valid without a year."""
        normalized_month = self.normalize_month_token(month)
        max_day = self.MONTH_DAY_LIMITS.get(normalized_month)
        if max_day is None:
            return False
        return 1 <= day <= max_day

    def check_century_format(self, section_name: str, text: str) -> List[Violation]:
        """Check written-out centuries and prefer numeric forms.

        This method strips simple inline style tags first:
        ``<i>``, ``<em>``, ``<b>``, ``<strong>``, ``<sup>``, and ``<sub>``.

        This method scans for century phrases from the hardcoded
        ``CENTURY_WORDS`` mapping and rewrites them to numeric forms such as
        ``19th century``.

        It matches:
        - simple written forms such as ``first century`` and ``nineteenth century``
        - compound written forms such as ``twentieth century``
        - both hyphenated and space-separated compounds when the mapped word
          contains a hyphen, for example ``twenty-first century`` and
          ``twenty first century``

        Examples caught:
        - ``first century`` -> suggests ``1st century``
        - ``ninth century`` -> suggests ``9th century``
        - ``nineteenth century`` -> suggests ``19th century``
        - ``twentieth century`` -> suggests ``20th century``
        - ``twenty-first century`` -> suggests ``21st century``
        - ``twenty first century`` -> suggests ``21st century``

        The method sorts longer forms first and skips overlapping shorter
        matches, so ``twenty first century`` is reported once rather than also
        flagging ``first century`` inside it.

        Examples not caught:
        - ``19th century`` because it is already numeric
        - ``XIX century`` because Roman numerals are not checked
        - ``twenty-second century`` because it is not in the hardcoded mapping
        - ``nineteenth-century art`` because adjectival hyphenated compounds are out of scope
        - ``nineteenth centuries`` because plural forms are not checked
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )
        matched_spans = []

        for word, num in sorted(self.CENTURY_WORDS.items(), key=lambda item: len(item[0]), reverse=True):
            word_pattern = re.escape(word).replace(r'\-', r'(?:-|\s+)')
            pattern = re.compile(rf'(?<![\w-]){word_pattern}\s+century\b', re.IGNORECASE)
            for match in pattern.finditer(cleaned_text):
                if any(match.start() < end and match.end() > start for start, end in matched_spans):
                    continue

                fix = f"{num} century"
                original_start = index_map[match.start()]
                original_end = index_map[match.end() - 1] + 1
                violations.append(self.create_violation(
                    section_name=section_name,
                    text=text,
                    span=(original_start, original_end),
                    message=f"Use '{fix}' instead of '{match.group(0)}'",
                    suggested_fix=fix,
                ))
                matched_spans.append(match.span())

        return violations

    def check_decade_format(self, section_name: str, text: str) -> List[Violation]:
        """Check decade formatting for four-digit decades.

        This method strips simple inline style tags first:
        ``<i>``, ``<em>``, ``<b>``, ``<strong>``, ``<sup>``, and ``<sub>``.

        This method looks for decade expressions written with:
        - an apostrophe before the trailing ``s``, such as ``1980's``
        - one or more spaces before the trailing ``s``, such as ``1980 s``

        It then suggests the compact preferred form with no apostrophe or
        space, such as ``1980s``.

        Examples caught:
        - ``1980's`` -> suggests ``1980s``
        - ``1980 s`` -> suggests ``1980s``
        - ``1990   s`` -> suggests ``1990s``
        - ``2000' s`` -> suggests ``2000s``

        Examples not caught:
        - ``1980s`` because it is already in the preferred form
        - ``'80s`` because shorthand decades are not checked
        - ``80's`` because the rule only matches four-digit decades
        - ``the eighties`` because written-out decades are out of scope
        - ``1980s-1990s`` because already-correct compact decade ranges are not rewritten here
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )
        pattern = re.compile(r"(\d{4})[\'\'\s]+s\b")

        for match in pattern.finditer(cleaned_text):
            fix = f"{match.group(1)}s"
            original_start = index_map[match.start()]
            original_end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(original_start, original_end),
                message=f"Use '{fix}' for decades (no apostrophe or space)",
                suggested_fix=fix,
            ))

        return violations
