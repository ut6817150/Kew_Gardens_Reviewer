"""Scientific name formatting checker for IUCN assessments."""

import re
from typing import List

from .base import BaseChecker
from ..models import Violation, Severity


class ScientificNameChecker(BaseChecker):
    """Checker for scientific name formatting rules."""

    def __init__(self):
        super().__init__(
            rule_id="scientific_names",
            rule_name="Scientific name formatting rules",
            category="Scientific Names",
            severity=Severity.WARNING,
            assessment_section="Whole Document"
        )

    def check(self, text: str) -> List[Violation]:
        """Check for scientific name formatting violations."""
        violations = []

        # spp./sp. formatting
        violations.extend(self._check_species_abbreviations(text))

        # Common name capitalization (should be capitalized)
        violations.extend(self._check_common_name_format(text))

        return violations

    def _check_species_abbreviations(self, text: str) -> List[Violation]:
        """Check spp. and sp. formatting."""
        violations = []

        # spp without period (for multiple species)
        spp_pattern = re.compile(r'\b(spp)(?!\.)\b')
        for match in spp_pattern.finditer(text):
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message="Use 'spp.' (with period) for multiple species",
                suggested_fix="spp."
            ))

        # sp without period (for single unspecified species)
        # Make sure we don't match "spp" or "species"
        sp_pattern = re.compile(r'\b(sp)(?!p|\.|\w)')
        for match in sp_pattern.finditer(text):
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message="Use 'sp.' (with period) for single unspecified species",
                suggested_fix="sp."
            ))

        return violations

    def _check_common_name_format(self, text: str) -> List[Violation]:
        """Check common name formatting."""
        violations = []

        # Common names should be capitalized when they are proper nouns
        # This is complex to detect automatically, so we just check for
        # the pattern "common name, scientific name" and ensure capitalization

        # Pattern: lowercase word followed by (Genus species)
        # e.g., "the african elephant (Loxodonta africana)"
        pattern = re.compile(r'\b([a-z][a-z]+(?:\s+[a-z]+)*)\s*\([A-Z][a-z]+\s+[a-z]+\)')

        for match in pattern.finditer(text):
            common_name = match.group(1)
            # Check if it looks like a common name that should be capitalized
            # (This is a heuristic - common names typically have 1-3 words)
            words = common_name.split()
            if len(words) <= 3 and words[0].islower():
                # Check if preceding text suggests this is a common name
                before = text[max(0, match.start()-10):match.start()].lower()
                if 'the ' in before or before.strip().endswith(('as', 'is', 'are')):
                    capitalized = ' '.join(w.capitalize() for w in words)
                    violations.append(self._create_violation(
                        text=text,
                        matched_text=common_name,
                        start=match.start(),
                        end=match.start() + len(common_name),
                        message=f"Common names should be capitalized: '{capitalized}'",
                        suggested_fix=capitalized
                    ))

        return violations
