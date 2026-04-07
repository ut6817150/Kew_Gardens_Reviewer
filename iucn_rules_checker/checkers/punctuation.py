"""Punctuation checker for IUCN assessments (Section 3.5)."""

import re
from typing import List

from ..violation import Violation
from .base import BaseChecker


class PunctuationChecker(BaseChecker):
    """Checker enforcing punctuation formatting rules."""

    EN_DASH = "–"
    RANGE_UNITS = (
        # Distance & Area
        "km", "m", "cm", "mm", "ha",
        "km2", "m2", "cm2", "mm2", "ha2",
        "km<sup>2</sup>", "m<sup>2</sup>", "cm<sup>2</sup>", "mm<sup>2</sup>", "ha<sup>2</sup>",
        "sq km", "sqkm", "sq m", "sqm", "sq cm", "sqcm", "sq mm", "sqmm", "sq ha", "sqha",
        # Volume & Mass
        "km3", "m3", "cm3", "mm3",
        "km<sup>3</sup>", "m<sup>3</sup>", "cm<sup>3</sup>", "mm<sup>3</sup>",
        "l", "ml", "kg", "g", "t",
        # Elevation & Angles
        "m asl", "m bsl", "°", "%",
        # Time (Critical for Generation Length)
        "yr", "mo", "days",
    )

    def __init__(self):
        super().__init__()

    def check_text(self, section_name: str, text: str) -> List[Violation]:
        """Run all punctuation checks for one parsed report entry.

        This is just the dispatcher for the punctuation checker. It runs:
        `check_range_dashes(...)`
        `check_for_example_commas(...)`
        `check_colon_spacing(...)`
        `check_semicolon_spacing(...)`

        It does not deduplicate overlapping findings between those methods; it
        simply concatenates the returned `Violation` lists in that order.
        """
        violations = []
        violations.extend(self.check_range_dashes(section_name, text))
        violations.extend(self.check_for_example_commas(section_name, text))
        violations.extend(self.check_colon_spacing(section_name, text))
        violations.extend(self.check_semicolon_spacing(section_name, text))
        return violations

    def check_range_dashes(self, section_name: str, text: str) -> List[Violation]:
        """Ensure numeric ranges use an unspaced en dash.

        This method strips italic and bold markers, then looks for a plain
        numeric pair such as `10-20`, `10 - 20`, or `10 – 20`. It rewrites the
        separator to a bare en dash (`–`).

        Shared-unit forms are still checked, so `10-20 km`, `600-1200 m`,
        `500-3000 mm`, and `14-26 °C` are flagged. However, expressions where
        each endpoint carries its own unit are still ignored, such as
        `10 km - 20 km`, `5% - 7%`, and `10 km<sup>2</sup> - 20 km<sup>2</sup>`.

        It also skips date-like three-part numeric chains such as
        `2022-08-01`, `08-01-2022`, and `08-2022-01`.
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=False,
            subscript=False,
        )
        pattern = re.compile(
            rf"(?<!\w)(?P<left>\d{{1,4}})"
            rf"(?P<sep>\s*(?:-|{self.EN_DASH})\s*)"
            rf"(?P<right>\d{{1,4}})(?!\w)"
        )

        for match in pattern.finditer(cleaned_text):
            snippet = cleaned_text[match.start():match.end() + 5]
            if re.match(r"^\d{3}-\d{3}-\d{4}\b", snippet):
                continue
            if self.is_date_like_numeric_chain(cleaned_text, match.start(), match.end()):
                continue

            after = cleaned_text[match.end():match.end() + 2]
            if after.startswith("="):
                continue

            left = match.group("left")
            right = match.group("right")
            sep = match.group("sep")
            if sep == self.EN_DASH:
                continue

            original_start = index_map[match.start()]
            original_end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(original_start, original_end),
                message=f"Use an unspaced en dash ({self.EN_DASH}) for numeric ranges",
                suggested_fix=f"{left}{self.EN_DASH}{right}",
            ))

        return violations

    def is_date_like_numeric_chain(self, text: str, start: int, end: int) -> bool:
        """Return True when a matched pair is part of a three-part hyphenated date."""
        chain_start = start
        while chain_start > 0 and (
            text[chain_start - 1].isdigit() or text[chain_start - 1] in f"-{self.EN_DASH}"
        ):
            chain_start -= 1

        chain_end = end
        while chain_end < len(text) and (
            text[chain_end].isdigit() or text[chain_end] in f"-{self.EN_DASH}"
        ):
            chain_end += 1

        chain = text[chain_start:chain_end]
        if not re.fullmatch(r"\d{1,4}(?:[-–]\d{1,4}){2}", chain):
            return False

        segments = re.split(r"[-–]", chain)
        year_segment_count = sum(len(segment) == 4 for segment in segments)
        return year_segment_count == 1 and all(
            1 <= len(segment) <= 2
            for segment in segments
            if len(segment) != 4
        )

    def check_for_example_commas(self, section_name: str, text: str) -> List[Violation]:
        """Ensure `for example` is enclosed by commas when used mid-sentence.

        This method first strips simple inline style markers such as italics,
        bold, superscript, and subscript tags, then checks the cleaned text
        while mapping any match span back to the original rich-text input.

        This method searches case-insensitively for the exact phrase
        `for example` and then checks whether it has:
        - a comma immediately before it
        - a comma immediately after it

        It is designed for mid-sentence parenthetical use, such as:
        `The species, for example, occurs in cloud forest.`

        Examples flagged:
        `The species for example, occurs in cloud forest.`
        `The species, for example occurs in cloud forest.`
        `The species for example occurs in cloud forest.`

        Examples not flagged:
        `The species, for example, occurs in cloud forest.`
        `For example, the species occurs in cloud forest.`
        `This changed. For example, the species occurs in cloud forest.`

        What it does not check:
        it does not rewrite the whole sentence.
        It does not understand all punctuation contexts beyond this narrow
        comma rule.
        It does not ignore simple HTML markup around the phrase, so forms such
        as `<i>for example</i>` can still be matched.
        It may miss split-markup forms such as `for <i>example</i>`.
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )
        pattern = re.compile(r"\bfor example\b", re.IGNORECASE)

        for match in pattern.finditer(cleaned_text):
            start, end = match.start(), match.end()
            before = cleaned_text[max(0, start - 2):start]
            after = cleaned_text[end:end + 1]
            original_start = index_map[start]
            original_end = index_map[end - 1] + 1

            if (
                start == 0
                or cleaned_text[:start].endswith(". ")
                or cleaned_text[:start].endswith("! ")
                or cleaned_text[:start].endswith("? ")
            ):
                continue

            if not before.strip().endswith(","):
                violations.append(self.create_violation(
                    section_name=section_name,
                    text=text,
                    span=(original_start, original_end),
                    message="'for example' should be preceded by a comma",
                    suggested_fix=None,
                ))

            if after != ",":
                violations.append(self.create_violation(
                    section_name=section_name,
                    text=text,
                    span=(original_start, original_end),
                    message="'for example' should be followed by a comma",
                    suggested_fix="for example,",
                ))

        return violations

    def check_colon_spacing(self, section_name: str, text: str) -> List[Violation]:
        """Detect spaces immediately before a colon.

        This method first strips simple inline style markers such as italics,
        bold, superscript, and subscript tags, then checks the cleaned text
        while mapping any match span back to the original rich-text input.

        This method flags any whitespace that appears directly before `:`.
        It is a simple formatting rule intended to normalize:
        `term : value` -> `term: value`

        Examples flagged:
        `Altitude : 200 m`
        `Countries  : Peru, Ecuador`

        Examples not flagged:
        `Altitude: 200 m`
        `Countries: Peru, Ecuador`

        What it does not check:
        it does not validate spacing after the colon.
        It does not distinguish prose from labels, tables, or copied text.
        It also does not reason about whether the colon itself is appropriate
        punctuation in the sentence.
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )
        pattern = re.compile(r"\s+:")
        for match in pattern.finditer(cleaned_text):
            original_start = index_map[match.start()]
            original_end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(original_start, original_end),
                message="Do not put a space before a colon",
                suggested_fix=":",
            ))
        return violations

    def check_semicolon_spacing(self, section_name: str, text: str) -> List[Violation]:
        """Detect spaces immediately before a semicolon.

        This method first strips simple inline style markers such as italics,
        bold, superscript, and subscript tags, then checks the cleaned text
        while mapping any match span back to the original rich-text input.

        This method flags any whitespace that appears directly before `;`.
        It is a simple formatting rule intended to normalize:
        `Peru ; Ecuador` -> `Peru; Ecuador`

        Examples flagged:
        `Peru ; Ecuador`
        `The range is broad ; however, records are sparse.`

        Examples not flagged:
        `Peru; Ecuador`
        `The range is broad; however, records are sparse.`

        What it does not check:
        it does not validate spacing after the semicolon.
        It does not decide whether a semicolon is the right punctuation mark.
        It treats all contexts the same, including prose, labels, and copied
        text.
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )
        pattern = re.compile(r"\s+;")
        for match in pattern.finditer(cleaned_text):
            original_start = index_map[match.start()]
            original_end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(original_start, original_end),
                message="Do not put a space before a semicolon",
                suggested_fix=";",
            ))
        return violations
