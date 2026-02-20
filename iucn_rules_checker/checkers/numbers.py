"""Number formatting checker for IUCN assessments."""

import re
from typing import List

from .base import BaseChecker
from ..models import Violation, Severity


class NumberChecker(BaseChecker):
    """Checker for number formatting rules."""

    NUMBER_WORDS = {
        1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine"
    }

    WORD_TO_NUMBER = {v: k for k, v in NUMBER_WORDS.items()}

    def __init__(self):
        super().__init__(
            rule_id="numbers_format",
            rule_name="Number formatting rules",
            category="Numbers",
            severity=Severity.WARNING,
            assessment_section="Whole Document"
        )

    def check(self, text: str) -> List[Violation]:
        """Check for number formatting violations."""
        violations = []

        # Rule: Numbers 1-9 should be written out (with exceptions)
        violations.extend(self._check_small_numbers(text))

        # Rule: Numbers >= 1000 should have commas
        violations.extend(self._check_large_numbers(text))

        # Rule: Don't start sentences with numerals
        violations.extend(self._check_sentence_start(text))

        # Rule: Large numbers (millions, billions) format
        violations.extend(self._check_very_large_numbers(text))

        return violations

    def _check_small_numbers(self, text: str) -> List[Violation]:
        """Check that numbers 1-9 are written as words."""
        violations = []

        # Pattern matches standalone digits 1-9
        # Excludes: before/after decimals, in ranges with dash, before units/symbols
        pattern = re.compile(
            r'(?<![0-9,.\-–])(?<!\d)\b([1-9])\b(?![0-9,.\-–]|%|°|km|m\b|ha\b|cm|mm|kg|g\b|ml|l\b|\s*[-–]\s*\d)'
        )

        for match in pattern.finditer(text):
            num = int(match.group(1))
            # Skip if it's part of a date pattern (e.g., "5 January")
            after_match = text[match.end():match.end()+20] if match.end() < len(text) else ""
            months = ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']
            if any(after_match.strip().startswith(m) for m in months):
                continue

            word = self.NUMBER_WORDS.get(num)
            if word:
                violations.append(self._create_violation(
                    text=text,
                    matched_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    message=f"Numbers 1-9 should be written out: '{num}' should be '{word}'",
                    suggested_fix=word
                ))

        return violations

    def _check_large_numbers(self, text: str) -> List[Violation]:
        """Check that numbers >= 1000 have comma separators."""
        violations = []

        # Match numbers with 4+ digits that don't have commas
        # Exclude: years (1900-2100), decimals, already formatted
        pattern = re.compile(r'\b(\d{4,})\b')

        for match in pattern.finditer(text):
            num_str = match.group(1)
            num = int(num_str)

            # Skip years (1800-2100)
            if 1800 <= num <= 2100:
                continue

            # Skip if already has commas (shouldn't match but just in case)
            if ',' in num_str:
                continue

            # Skip if it's a decimal
            before = text[max(0, match.start()-1):match.start()]
            after = text[match.end():match.end()+1] if match.end() < len(text) else ""
            if before == '.' or after == '.':
                continue

            # Format with commas
            formatted = "{:,}".format(num)
            if formatted != num_str:
                violations.append(self._create_violation(
                    text=text,
                    matched_text=num_str,
                    start=match.start(),
                    end=match.end(),
                    message=f"Use commas for numbers >= 1,000: '{num_str}' should be '{formatted}'",
                    suggested_fix=formatted
                ))

        return violations

    def _check_sentence_start(self, text: str) -> List[Violation]:
        """Check that sentences don't start with numerals."""
        violations = []

        # Match numeral at start of sentence
        # Start of text or after sentence-ending punctuation
        pattern = re.compile(r'(?:^|[.!?]\s+)(\d+)\b')

        for match in pattern.finditer(text):
            num_str = match.group(1)
            violations.append(self._create_violation(
                text=text,
                matched_text=num_str,
                start=match.start(1),
                end=match.end(1),
                message=f"Do not start sentences with numerals; write the number out or rephrase",
                suggested_fix=None  # Complex - may need rephrasing
            ))

        return violations

    def _check_very_large_numbers(self, text: str) -> List[Violation]:
        """Check formatting of very large numbers (millions, billions)."""
        violations = []

        # Pattern: numbers like 1000000 should be "1 million"
        # Match 7+ digit numbers (millions)
        pattern = re.compile(r'\b(\d{7,})\b')

        for match in pattern.finditer(text):
            num_str = match.group(1)
            num = int(num_str)

            # Skip if already properly formatted with commas
            if ',' in text[match.start():match.end()]:
                continue

            # Suggest conversion to "X million/billion" format
            if num >= 1_000_000_000:
                billions = num / 1_000_000_000
                if billions == int(billions):
                    suggested = f"{int(billions)} billion"
                else:
                    suggested = f"{billions:.1f} billion"
            elif num >= 1_000_000:
                millions = num / 1_000_000
                if millions == int(millions):
                    suggested = f"{int(millions)} million"
                else:
                    suggested = f"{millions:.1f} million"
            else:
                continue

            violations.append(self._create_violation(
                text=text,
                matched_text=num_str,
                start=match.start(),
                end=match.end(),
                message=f"For numbers >= 1 million, use 'X million/billion': '{num_str}' could be '{suggested}'",
                suggested_fix=suggested
            ))

        return violations
