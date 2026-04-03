"""Symbol and unit formatting checker for IUCN assessments."""

import re
from typing import List

from ..violation import Violation
from .base import BaseChecker


class SymbolChecker(BaseChecker):
    """Checker for symbol and unit formatting rules."""

    def __init__(self):
        super().__init__()

    def check_text(self, section_name: str, text: str) -> List[Violation]:
        """Check for symbol and unit formatting violations."""
        violations = []
        violations.extend(self.check_area_units(section_name, text))
        violations.extend(self.check_degree_symbol(section_name, text))
        violations.extend(self.check_percentage(section_name, text))
        violations.extend(self.check_percentage_symbol_spacing(section_name, text))
        return violations

    def check_area_units(self, section_name: str, text: str) -> List[Violation]:
        """Check area unit formatting after removing simple style tags.

        This method strips simple inline bold/italic HTML tags first:
        `<i>`, `<em>`, `<b>`, and `<strong>`.
        It then checks the cleaned text for a small set of non-preferred area
        unit forms:
        - `sq km` or `sqkm` -> `km²`
        - `km2` -> `km²`
        - `m2` -> `m²`
        - `cm2` -> `cm²`
        - `mm2` -> `mm²`

        Examples flagged:
        `sq km`
        `sqkm`
        `km2`
        `m2`
        `cm2`
        `mm2`
        `<i>sq</i> <i>km</i>`
        `<b>km2</b>`
        `<i>m</i><i>2</i>`

        Examples not flagged:
        `km²`
        `m²`
        `<i>km²</i>`
        `<b>m²</b>`

        What it misses:
        it only strips those simple style tags and does not parse arbitrary
        markup.
        It only checks the hardcoded forms above, so other area-unit variants
        are ignored.
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=False,
            subscript=False,
        )

        sqkm_pattern = re.compile(r'\bsq\s*km\b', re.IGNORECASE)
        for match in sqkm_pattern.finditer(cleaned_text):
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message="Use 'km²' not 'sqkm' or 'sq km'",
                suggested_fix="km²",
            ))

        km2_pattern = re.compile(r'\bkm2\b')
        for match in km2_pattern.finditer(cleaned_text):
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message="Use 'km²' not 'km2'",
                suggested_fix="km²",
            ))

        m2_pattern = re.compile(r'\bm2\b')
        for match in m2_pattern.finditer(cleaned_text):
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message="Use 'm²' not 'm2'",
                suggested_fix="m²",
            ))

        cm2_pattern = re.compile(r'\bcm2\b')
        for match in cm2_pattern.finditer(cleaned_text):
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message="Use 'cm²' not 'cm2'",
                suggested_fix="cm²",
            ))

        mm2_pattern = re.compile(r'\bmm2\b')
        for match in mm2_pattern.finditer(cleaned_text):
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message="Use 'mm²' not 'mm2'",
                suggested_fix="mm²",
            ))

        return violations

    def check_degree_symbol(self, section_name: str, text: str) -> List[Violation]:
        """Check written-out degree forms and spacing around `°`.

        This method strips simple inline bold/italic HTML tags first:
        `<i>`, `<em>`, `<b>`, and `<strong>`.
        It then applies three checks:

        1. Direction forms written with words:
        `12 degrees N` -> `12°N`
        `12.5 degree s` -> `12.5°S`

        2. Celsius forms written with words:
        `20 degrees C` -> `20°C`
        `20.75 degrees Celsius` -> `20.75°C`

        3. Existing degree-symbol forms with incorrect spacing:
        `12 °C` -> `12°C`
        `12° C` -> `12°C`
        `12.5 ° N` -> `12.5°N`

        Examples flagged:
        `12 degrees N`
        `12.5 degrees n`
        `20 degrees C`
        `20.75 degrees Celsius`
        `12 °C`
        `12° C`
        `12.5 ° N`

        Examples not flagged:
        `12°N`
        `12.5°N`
        `20°C`
        `20.75°C`
        `<i>12.5°N</i>`
        `<b>20.75°C</b>`

        What it misses:
        it only checks direction letters `N`, `S`, `E`, `W` and Celsius (`C`
        / `Celsius`).
        It does not handle Fahrenheit or more complex coordinate formats.
        It only strips the simple style tags listed above.
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )

        degrees_direction = re.compile(r'(\d+(?:\.\d+)?)\s*degrees?\b\s*([NSEW])\b', re.IGNORECASE)
        for match in degrees_direction.finditer(cleaned_text):
            fix = f"{match.group(1)}°{match.group(2).upper()}"
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message=f"Use degree symbol: '{fix}'",
                suggested_fix=fix,
            ))

        degrees_temp = re.compile(r'(\d+(?:\.\d+)?)\s*degrees?\b\s*(C|Celsius)\b', re.IGNORECASE)
        for match in degrees_temp.finditer(cleaned_text):
            fix = f"{match.group(1)}°C"
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message=f"Use degree symbol: '{fix}'",
                suggested_fix=fix,
            ))

        spaced_degree_symbol = re.compile(r'(\d+(?:\.\d+)?)\s*°\s*([NSEWC])\b', re.IGNORECASE)
        for match in spaced_degree_symbol.finditer(cleaned_text):
            fix = f"{match.group(1)}°{match.group(2).upper()}"
            if match.group(0) == fix:
                continue

            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message=f"No spaces around degree symbol: '{fix}'",
                suggested_fix=fix,
            ))

        return violations

    def check_percentage(self, section_name: str, text: str) -> List[Violation]:
        """Check written-out percentage forms after removing simple style tags.

        This method strips simple inline bold/italic HTML tags first:
        `<i>`, `<em>`, `<b>`, and `<strong>`.
        It then checks the cleaned text for:
        - number + `percent`
        - number + `per cent`

        Both integer and decimal values are supported.

        Examples flagged:
        `12 percent` -> `12%`
        `12.5 percent` -> `12.5%`
        `7 per cent` -> `7%`
        `7.25 per cent` -> `7.25%`
        `<i>12</i><i>.5</i> percent` -> `12.5%`
        `<b>7.25 per cent</b>` -> `7.25%`

        Examples not flagged:
        `12%`
        `12.5%`
        `<i>12.5%</i>`
        `<b>7.25%</b>`

        What it misses:
        it only strips those simple style tags and does not parse arbitrary
        markup.
        It only checks `percent` and `per cent`, not other paraphrases such as
        `percentage`.
        Spacing around an existing `%` symbol is handled separately by
        `check_percentage_symbol_spacing(...)`.
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )

        percent_word = re.compile(r'(?<![\d.])(\d+(?:\.\d+)?)\s*percent\b', re.IGNORECASE)
        for match in percent_word.finditer(cleaned_text):
            fix = f"{match.group(1)}%"
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message=f"Use '%' symbol: '{fix}'",
                suggested_fix=fix,
            ))

        per_cent = re.compile(r'(?<![\d.])(\d+(?:\.\d+)?)\s+per\s+cent\b', re.IGNORECASE)
        for match in per_cent.finditer(cleaned_text):
            fix = f"{match.group(1)}%"
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message=f"Use '%' symbol: '{fix}'",
                suggested_fix=fix,
            ))

        return violations

    def check_percentage_symbol_spacing(self, section_name: str, text: str) -> List[Violation]:
        """Check spacing around `%` and simple units after removing style tags.

        This method strips simple inline bold/italic HTML tags first:
        `<i>`, `<em>`, `<b>`, and `<strong>`.
        It then applies two spacing checks to the cleaned text:
        - remove spaces before `%`: `12 %` -> `12%`
        - add spaces between numbers and short units: `12km` -> `12 km`

        Wrapped or split-tag forms are still checked, for example:
        `<i>12</i> <b>%</b>` -> `12%`
        `<i>12</i><b>km</b>` -> `12 km`
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )

        space_percent = re.compile(r'(\d+)\s+%')
        for match in space_percent.finditer(cleaned_text):
            fix = f"{match.group(1)}%"
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message="No space before '%'",
                suggested_fix=fix,
            ))

        no_space_units = re.compile(r'(\d+)(km|ha|kg|m|cm|mm|ml|l)\b(?!Â²)')
        for match in no_space_units.finditer(cleaned_text):
            fix = f"{match.group(1)} {match.group(2)}"
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(start, end),
                message=f"Add space between number and unit: '{fix}'",
                suggested_fix=fix,
            ))

        return violations
