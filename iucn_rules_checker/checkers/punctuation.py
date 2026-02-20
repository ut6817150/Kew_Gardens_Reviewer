"""Punctuation checker for IUCN assessments (Section 3.5)."""

import re
from typing import List

from .base import BaseChecker
from ..models import Violation, Severity


class PunctuationChecker(BaseChecker):
    """Checker enforcing punctuation formatting rules."""

    def __init__(self):
        super().__init__(
            rule_id="punctuation_format",
            rule_name="Punctuation formatting rules",
            category="Punctuation",
            severity=Severity.WARNING,
            assessment_section="Whole Document"
        )

    # Public API

    def check(self, text: str) -> List[Violation]:
        """Run all punctuation checks."""
        violations = []

        violations.extend(self._check_range_dashes(text))
        violations.extend(self._check_sentence_dashes(text))
        violations.extend(self._check_em_dash_spacing(text))
        violations.extend(self._check_for_example_commas(text))
        violations.extend(self._check_colon_spacing(text))
        violations.extend(self._check_semicolon_spacing(text))
        violations.extend(self._check_oxford_comma(text))

        return violations

    # 3.5.2 — En dashes for ranges

    def _check_range_dashes(self, text: str) -> List[Violation]:
        """Ensure numeric ranges use en dash (–) not hyphen (-)."""
        violations = []

        # Conservative: only match 1–4 digit numbers
        pattern = re.compile(r'\b(\d{1,4})\s*-\s*(\d{1,4})\b')

        for match in pattern.finditer(text):
            # Skip obvious phone numbers (123-555-0123)
            if re.match(r'\d{3}-\d{3}-\d{4}', match.group(0)):
                continue

            # Skip mathematical expressions (e.g., 4-3=1)
            after = text[match.end():match.end()+2]
            if after.startswith("="):
                continue

            num1, num2 = match.group(1), match.group(2)

            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message="Use en dash (–) for numeric ranges, not hyphen (-)",
                suggested_fix=f"{num1}–{num2}"
            ))

        return violations

    # 3.5.2 — Hyphen used instead of en dash in sentence breaks

    def _check_sentence_dashes(self, text: str) -> List[Violation]:
        """Detect spaced hyphen used as sentence dash (' - ')."""
        violations = []

        pattern = re.compile(r'\s-\s')

        for match in pattern.finditer(text):
            violations.append(self._create_violation(
                text=text,
                matched_text=" - ",
                start=match.start(),
                end=match.end(),
                message="Use en dash (–) or em dash (—), not spaced hyphen (-), in sentence breaks",
                suggested_fix=" – "
            ))

        return violations

    # 3.5.2 — Em dash spacing

    def _check_em_dash_spacing(self, text: str) -> List[Violation]:
        """Ensure em dashes (—) are not surrounded by spaces."""
        violations = []

        pattern = re.compile(r'\s—|—\s')

        for match in pattern.finditer(text):
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message="Em dashes (—) should not have spaces around them",
                suggested_fix=None
            ))

        return violations

    
    # 3.5.6 — Commas around 'for example'

    def _check_for_example_commas(self, text: str) -> List[Violation]:
        """Ensure 'for example' is enclosed by commas when mid-sentence."""
        violations = []

        pattern = re.compile(r'\bfor example\b', re.IGNORECASE)

        for match in pattern.finditer(text):
            start, end = match.start(), match.end()

            before = text[max(0, start-2):start]
            after = text[end:end+1]

            # Skip if sentence-initial
            if start == 0 or text[start-1] in ".!?":
                continue

            if not before.strip().endswith(","):
                violations.append(self._create_violation(
                    text=text,
                    matched_text="for example",
                    start=start,
                    end=end,
                    message="'for example' should be preceded by a comma",
                    suggested_fix=None
                ))

            if after != ",":
                violations.append(self._create_violation(
                    text=text,
                    matched_text="for example",
                    start=start,
                    end=end,
                    message="'for example' should be followed by a comma",
                    suggested_fix="for example,"
                ))

        return violations

    # 3.5.4 — Colon spacing

    def _check_colon_spacing(self, text: str) -> List[Violation]:
        """Detect space before colon."""
        violations = []

        pattern = re.compile(r'\s+:')

        for match in pattern.finditer(text):
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message="Do not put a space before a colon",
                suggested_fix=":"
            ))

        return violations

    # 3.5.5 — Semicolon spacing

    def _check_semicolon_spacing(self, text: str) -> List[Violation]:
        """Detect space before semicolon."""
        violations = []

        pattern = re.compile(r'\s+;')

        for match in pattern.finditer(text):
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message="Do not put a space before a semicolon",
                suggested_fix=";"
            ))

        return violations

    # 3.5.6 — Oxford comma (conservative heuristic)

    def _check_oxford_comma(self, text: str) -> List[Violation]:
        """
        Detect likely missing Oxford comma in 3+ item lists.
        Conservative heuristic to reduce false positives.
        """
        violations = []

        pattern = re.compile(
            r'\b(\w+,\s+\w+,\s+\w+)\s+and\s+\w+'
        )

        for match in pattern.finditer(text):
            phrase = match.group(0)

            if ", and" not in phrase:
                violations.append(self._create_violation(
                    text=text,
                    matched_text=phrase,
                    start=match.start(),
                    end=match.end(),
                    message="Use a comma before 'and' in lists of four or more elements (Oxford comma rule)",
                    suggested_fix=None
                ))

        return violations
