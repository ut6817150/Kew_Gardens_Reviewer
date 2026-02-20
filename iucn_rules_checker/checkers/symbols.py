"""Symbol and unit formatting checker for IUCN assessments."""

import re
from typing import List

from .base import BaseChecker
from ..models import Violation, Severity


class SymbolChecker(BaseChecker):
    """Checker for symbol and unit formatting rules."""

    def __init__(self):
        super().__init__(
            rule_id="symbols_format",
            rule_name="Symbol and unit formatting rules",
            category="Symbols",
            severity=Severity.WARNING,
            assessment_section="Whole Document"
        )

    def check(self, text: str) -> List[Violation]:
        """Check for symbol and unit formatting violations."""
        violations = []

        # km² format
        violations.extend(self._check_area_units(text))

        # Degree symbol
        violations.extend(self._check_degree_symbol(text))

        # Percentage format
        violations.extend(self._check_percentage(text))

        # Spacing around symbols
        violations.extend(self._check_symbol_spacing(text))

        return violations

    def _check_area_units(self, text: str) -> List[Violation]:
        """Check area unit formatting (km², not sqkm or km2)."""
        violations = []

        # sqkm or sq km -> km²
        sqkm_pattern = re.compile(r'\bsq\s*km\b', re.IGNORECASE)
        for match in sqkm_pattern.finditer(text):
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message="Use 'km²' not 'sqkm' or 'sq km'",
                suggested_fix="km²"
            ))

        # km2 -> km²
        km2_pattern = re.compile(r'\bkm2\b')
        for match in km2_pattern.finditer(text):
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message="Use 'km²' not 'km2'",
                suggested_fix="km²"
            ))

        # m2 -> m²
        m2_pattern = re.compile(r'\bm2\b')
        for match in m2_pattern.finditer(text):
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message="Use 'm²' not 'm2'",
                suggested_fix="m²"
            ))

        return violations

    def _check_degree_symbol(self, text: str) -> List[Violation]:
        """Check degree symbol usage."""
        violations = []

        # "X degrees N/S/E/W" should be "X°N/S/E/W"
        degrees_direction = re.compile(r'(\d+)\s*degrees?\s*([NSEW])\b', re.IGNORECASE)
        for match in degrees_direction.finditer(text):
            fix = f"{match.group(1)}°{match.group(2).upper()}"
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message=f"Use degree symbol: '{fix}'",
                suggested_fix=fix
            ))

        # "X degrees C/Celsius" should be "X°C"
        degrees_temp = re.compile(r'(\d+)\s*degrees?\s*(C|Celsius)\b', re.IGNORECASE)
        for match in degrees_temp.finditer(text):
            fix = f"{match.group(1)}°C"
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message=f"Use degree symbol: '{fix}'",
                suggested_fix=fix
            ))

        return violations

    def _check_percentage(self, text: str) -> List[Violation]:
        """Check percentage formatting."""
        violations = []

        # "X percent" should be "X%"
        percent_word = re.compile(r'(\d+)\s*percent\b', re.IGNORECASE)
        for match in percent_word.finditer(text):
            fix = f"{match.group(1)}%"
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message=f"Use '%' symbol: '{fix}'",
                suggested_fix=fix
            ))

        # "X per cent" should be "X%"
        per_cent = re.compile(r'(\d+)\s*per\s*cent\b', re.IGNORECASE)
        for match in per_cent.finditer(text):
            fix = f"{match.group(1)}%"
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message=f"Use '%' symbol: '{fix}'",
                suggested_fix=fix
            ))

        return violations

    def _check_symbol_spacing(self, text: str) -> List[Violation]:
        """Check spacing around symbols."""
        violations = []

        # Space before % is wrong (25 % should be 25%)
        space_percent = re.compile(r'(\d+)\s+%')
        for match in space_percent.finditer(text):
            fix = f"{match.group(1)}%"
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message="No space before '%'",
                suggested_fix=fix
            ))

        # No space between number and unit is wrong for most units
        # But % and ° are exceptions (no space)
        # Check for missing space: "5km" should be "5 km"
        no_space_units = re.compile(r'(\d+)(km|ha|kg|m|cm|mm|ml|l)\b(?!²)')
        for match in no_space_units.finditer(text):
            # Make sure it's not already "5 km"
            if match.group(0) == match.group(1) + match.group(2):
                fix = f"{match.group(1)} {match.group(2)}"
                violations.append(self._create_violation(
                    text=text,
                    matched_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    message=f"Add space between number and unit: '{fix}'",
                    suggested_fix=fix
                ))

        return violations
