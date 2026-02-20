"""Date formatting checker for IUCN assessments."""

import re
from typing import List

from .base import BaseChecker
from ..models import Violation, Severity


class DateChecker(BaseChecker):
    """Checker for date formatting rules."""

    MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']

    CENTURY_WORDS = {
        'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th',
        'fifth': '5th', 'sixth': '6th', 'seventh': '7th', 'eighth': '8th',
        'ninth': '9th', 'tenth': '10th', 'eleventh': '11th', 'twelfth': '12th',
        'thirteenth': '13th', 'fourteenth': '14th', 'fifteenth': '15th',
        'sixteenth': '16th', 'seventeenth': '17th', 'eighteenth': '18th',
        'nineteenth': '19th', 'twentieth': '20th', 'twenty-first': '21st'
    }

    def __init__(self):
        super().__init__(
            rule_id="dates_format",
            rule_name="Date formatting rules",
            category="Dates",
            severity=Severity.WARNING,
            assessment_section="Whole Document"
        )

    def check(self, text: str) -> List[Violation]:
        """Check for date formatting violations."""
        violations = []

        # Rule: Avoid ordinal dates (11th January -> 11 January)
        violations.extend(self._check_ordinal_dates(text))

        # Rule: Written-out century (nineteenth century -> 19th century)
        violations.extend(self._check_century_format(text))

        # Rule: Decade format (1980's or 1980 s -> 1980s)
        violations.extend(self._check_decade_format(text))

        return violations

    def _check_ordinal_dates(self, text: str) -> List[Violation]:
        """Check for ordinal dates that should use cardinal numbers."""
        violations = []

        # Pattern: 11th January, 1st of March, etc.
        months_pattern = '|'.join(self.MONTHS)
        pattern = re.compile(
            rf'\b(\d{{1,2}})(st|nd|rd|th)\s+(of\s+)?({months_pattern})\b',
            re.IGNORECASE
        )

        for match in pattern.finditer(text):
            day = match.group(1)
            month = match.group(4)
            fix = f"{day} {month}"

            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message=f"Avoid ordinal dates; use '{fix}' instead of '{match.group(0)}'",
                suggested_fix=fix
            ))

        return violations

    def _check_century_format(self, text: str) -> List[Violation]:
        """Check that centuries use numerals (19th century, not nineteenth century)."""
        violations = []

        for word, num in self.CENTURY_WORDS.items():
            pattern = re.compile(rf'\b{word}\s+century\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                fix = f"{num} century"
                violations.append(self._create_violation(
                    text=text,
                    matched_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    message=f"Use '{fix}' instead of '{match.group(0)}'",
                    suggested_fix=fix
                ))

        return violations

    def _check_decade_format(self, text: str) -> List[Violation]:
        """Check that decades are formatted correctly (1980s, not 1980's or 1980 s)."""
        violations = []

        # Pattern: 1980's or 1980 s or 1980's
        pattern = re.compile(r"(\d{4})[\'\'\s]+s\b")

        for match in pattern.finditer(text):
            fix = f"{match.group(1)}s"
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message=f"Use '{fix}' for decades (no apostrophe or space)",
                suggested_fix=fix
            ))

        return violations
