"""Number formatting checker for IUCN assessments."""

import re
from typing import List

from ..violation import Violation
from .base import BaseChecker


class NumberChecker(BaseChecker):
    """Checker for number formatting rules."""

    NUMBER_WORDS = {
        1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine",
    }

    WORD_TO_NUMBER = {v: k for k, v in NUMBER_WORDS.items()}

    MONTHS = (
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    )

    SMALL_NUMBER_UNITS = (
        "km", "m", "cm", "mm", "ha",
        "km2", "m2", "cm2", "mm2", "ha2", "sq km", "sqkm", "sq m", "sqm", "sq cm", "sqcm", "sq mm", "sqmm", "sq ha", "sqha",
        "km3", "m3", "cm3", "mm3", "l", "ml", "kg", "g", "t",
        "m asl", "m bsl", "°", "%",
    )

    def __init__(self):
        super().__init__()

    def check_text(self, section_name: str, text: str) -> List[Violation]:
        """Check for number formatting violations."""
        violations = []
        violations.extend(self.check_small_numbers(section_name, text))
        violations.extend(self.check_large_numbers(section_name, text))
        violations.extend(self.check_sentence_start(section_name, text))
        violations.extend(self.check_very_large_numbers(section_name, text))
        return violations

    def check_small_numbers(self, section_name: str, text: str) -> List[Violation]:
        """Check that standalone numerals 1-9 are written as words.

        This method first strips simple inline style markers such as italics,
        bold, superscript, and subscript tags, then checks the cleaned text
        for standalone small numerals while mapping any match span back to the
        original rich-text input.

        This method finds single-digit numerals from `1` to `9` and suggests
        the spelled-out form from `NUMBER_WORDS`, for example:
        `3` -> `three`
        `7` -> `seven`

        It only flags a numeral when it appears as a standalone prose number.
        Before creating a violation, it runs deterministic exclusions for
        contexts where numerals are normally allowed. The excluded contexts are:
        - dates such as `3 May`
        - measurements such as `5 km`, `7 ha`, `8 cm`
        - percentages and degrees such as `6%`, `8°`
        - decimals and comma/period-adjacent numeric contexts such as `1.5`
        - numeric ranges such as `4-5` or `2 – 3`
        - any digit that is immediately adjacent to another digit

        Examples flagged:
        `There were 3 sites`
        `The species survives in 2 valleys`

        Examples not flagged:
        `3 May`
        `5 km`
        `7 ha`
        `6%`
        `8°`
        `1.5`
        `4-5`
        `2 – 3`
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )
        pattern = re.compile(r"\b([1-9])\b")

        for match in pattern.finditer(cleaned_text):
            if self.should_exclude_small_number(cleaned_text, match.start(), match.end()):
                continue

            num = int(match.group(1))
            word = self.NUMBER_WORDS.get(num)
            if word:
                original_start = index_map[match.start()]
                original_end = index_map[match.end() - 1] + 1
                violations.append(self.create_violation(
                    section_name=section_name,
                    text=text,
                    span=(original_start, original_end),
                    message=f"Numbers 1-9 should be written out: '{num}' should be '{word}'",
                    suggested_fix=word,
                ))

        return violations

    def should_exclude_small_number(self, text: str, start: int, end: int) -> bool:
        """Return True when a small numeral appears in an excluded context."""
        before_char = text[start - 1] if start > 0 else ""
        after_char = text[end] if end < len(text) else ""
        before_text = text[max(0, start - 10):start]
        after_text = text[end:end + 20]
        stripped_after = after_text.lstrip()

        range_separators = {"-", "\u2013", "\u2014"}

        if before_char.isdigit() or after_char.isdigit():
            return True
        if before_char in {",", "."} or after_char in {",", "."}:
            return True
        if before_char in range_separators or after_char in range_separators:
            return True
        if re.match(r"^\s*[%\u00B0]", after_text):
            return True
        if re.match(
            rf"^\s*(?:{'|'.join(re.escape(unit) for unit in sorted(self.SMALL_NUMBER_UNITS, key=len, reverse=True))})\b",
            after_text,
            re.IGNORECASE,
        ):
            return True
        if re.match(r"^\s*[-\u2013\u2014]\s*\d", after_text):
            return True
        if re.search(r"\d\s*[-\u2013\u2014]\s*$", before_text):
            return True
        if any(stripped_after.startswith(month) for month in self.MONTHS):
            return True
        return False

    def check_large_numbers(self, section_name: str, text: str) -> List[Violation]:
        """Check large numbers for standard comma placement.

        This method first strips simple inline style markers such as italics,
        bold, superscript, and subscript tags, then checks cleaned numeric
        tokens for:
        - missing commas in large integers such as `1000` -> `1,000`
        - incorrect comma grouping such as `12,34` -> `1,234`
        - decimal forms whose integer part should be comma-grouped, such as
          `1234.56` -> `1,234.56` and `12,34.56` -> `1,234.56`

        It intentionally skips:
        - likely years from `1800` to `2100`
        - plain rounded million/billion candidates such as `1500000`, which
          are handled by `check_very_large_numbers(...)`

        Examples flagged:
        `1000`
        `12,34`
        `1234.56`
        `12,34.56`

        Examples not flagged:
        `2024`
        `1998`
        `1,000`
        `1,234.56`
        `1500000`
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )
        pattern = re.compile(r"(?<!\w)(\d(?:[\d,]*\d)?(?:\.\d+)?)(?!\w)")

        for match in pattern.finditer(cleaned_text):
            num_str = match.group(1)
            integer_part, decimal_point, decimal_part = num_str.partition(".")
            integer_digits = integer_part.replace(",", "")

            if not integer_digits.isdigit():
                continue

            num = int(integer_digits)
            if not decimal_point and 1800 <= num <= 2100:
                continue
            if (
                not decimal_point
                and "," not in num_str
                and self.is_very_large_number_candidate(num)
            ):
                continue

            if len(integer_digits) < 4 and "," not in integer_part:
                continue

            formatted_integer = f"{num:,}" if len(integer_digits) >= 4 else integer_digits
            formatted = (
                f"{formatted_integer}.{decimal_part}"
                if decimal_point
                else formatted_integer
            )

            if formatted == num_str:
                continue

            original_start = index_map[match.start()]
            original_end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(original_start, original_end),
                message=f"Use standard comma placement for numbers: '{num_str}' should be '{formatted}'",
                suggested_fix=formatted,
            ))

        return violations

    def check_sentence_start(self, section_name: str, text: str) -> List[Violation]:
        """Check that sentences do not start with numerals.

        This method first strips simple inline style markers such as italics,
        bold, superscript, and subscript tags, then looks for digit strings
        in the cleaned text that appear:
        - at the very start of the text block
        - immediately after `. `, `! `, or `? `

        Examples flagged:
        `3 sites were surveyed.`
        `The assessment was updated. 12 records were added.`
        `Was it revised? 4 locations remained.`
        `<b>3</b> sites were surveyed.`

        Examples not flagged:
        numbers that occur mid-sentence
        already written-out sentence starts such as `Three sites were surveyed.`
        bibliography-style years immediately following `et al. `
        sentence-like starts after punctuation not covered by the regex

        This rule does not try to generate an automatic rewrite. It only
        reports that the sentence should be rephrased or the number written out.
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )
        pattern = re.compile(r"(?:^|[.!?]\s+)(\d+)\b")

        for match in pattern.finditer(cleaned_text):
            if cleaned_text[max(0, match.start(1) - 16):match.start(1)].lower().endswith("et al. "):
                continue
            original_start = index_map[match.start(1)]
            original_end = index_map[match.end(1) - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(original_start, original_end),
                message="Do not start sentences with numerals; write the number out or rephrase",
                suggested_fix=None,
            ))

        return violations

    def check_very_large_numbers(self, section_name: str, text: str) -> List[Violation]:
        """Check rounded very large numbers for ``million``/``billion`` style.

        This method first strips simple inline style markers such as italics,
        bold, superscript, and subscript tags, then checks the cleaned text
        for rounded million/billion-sized digit strings while mapping any
        match span back to the original rich-text input.

        This method looks for standalone digit strings with 7 or more digits,
        but it only flags values that are already rounded enough to be written
        cleanly as millions or billions.

        In practice, that means:
        - exact whole millions such as `1000000` -> `1 million`
        - clean one- or two-decimal millions such as `1500000` -> `1.5 million`
          and `1570000` -> `1.57 million`
        - exact whole billions such as `3000000000` -> `3 billion`
        - clean one- or two-decimal billions such as `1500000000` -> `1.5 billion`
          and `1570000000` -> `1.57 billion`
        - more precise values are ignored rather than rounded

        Examples flagged:
        `1000000` -> `1 million`
        `1500000` -> `1.5 million`
        `1570000` -> `1.57 million`
        `25000000` -> `25 million`
        `1500000000` -> `1.5 billion`
        `1570000000` -> `1.57 billion`
        `3000000000` -> `3 billion`

        Examples not flagged:
        `1234567`
        `9876543210`
        values already written with commas

        This keeps the rule limited to clearly rounded large counts and avoids
        rewriting precise values into approximate `million`/`billion` wording.
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )
        pattern = re.compile(r"\b(\d{7,})\b")

        for match in pattern.finditer(cleaned_text):
            num_str = match.group(1)
            num = int(num_str)

            if "," in cleaned_text[match.start():match.end()]:
                continue

            if num >= 1_000_000_000 and num % 10_000_000 == 0:
                suggested = self.format_large_number(num, 1_000_000_000, "billion")
            elif num >= 1_000_000 and num % 10_000 == 0:
                suggested = self.format_large_number(num, 1_000_000, "million")
            else:
                continue

            original_start = index_map[match.start()]
            original_end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(original_start, original_end),
                message=f"For rounded numbers >= 1 million, use 'X million/billion': '{num_str}' could be '{suggested}'",
                suggested_fix=suggested,
            ))

        return violations

    def is_very_large_number_candidate(self, num: int) -> bool:
        """Return True when a plain integer is handled by the million/billion rule."""
        return (
            (num >= 1_000_000_000 and num % 10_000_000 == 0)
            or (num >= 1_000_000 and num % 10_000 == 0)
        )

    def format_large_number(self, num: int, scale: int, unit: str) -> str:
        """Format a rounded large number with up to two decimal places."""
        scaled = num / scale
        if scaled.is_integer():
            scaled_str = str(int(scaled))
        else:
            scaled_str = f"{scaled:.2f}".rstrip("0").rstrip(".")
        return f"{scaled_str} {unit}"
