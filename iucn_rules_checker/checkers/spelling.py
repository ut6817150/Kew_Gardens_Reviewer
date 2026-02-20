"""UK vs US spelling checker for IUCN assessments."""

import re
from typing import List

from .base import BaseChecker
from ..models import Violation, Severity


class SpellingChecker(BaseChecker):
    """Checker for UK vs US spelling rules."""

    # Comprehensive UK spelling mappings from IUCN rules
    # Format: US spelling -> UK spelling
    UK_SPELLING_MAP = {
        # -our vs -or
        "color": "colour",
        "colors": "colours",
        "colored": "coloured",
        "coloring": "colouring",
        "favorite": "favourite",
        "favorites": "favourites",
        "favor": "favour",
        "favors": "favours",
        "favored": "favoured",
        "favoring": "favouring",
        "behavior": "behaviour",
        "behaviors": "behaviours",
        "behavioral": "behavioural",
        "harbor": "harbour",
        "harbors": "harbours",
        "honor": "honour",
        "honors": "honours",
        "honored": "honoured",
        "labor": "labour",
        "labors": "labours",
        "labored": "laboured",
        "neighbor": "neighbour",
        "neighbors": "neighbours",
        "neighboring": "neighbouring",
        "neighborhood": "neighbourhood",
        "odor": "odour",
        "odors": "odours",
        "vigor": "vigour",
        "rigorous": "rigorous",  # Same in both
        "rumor": "rumour",
        "rumors": "rumours",
        "savior": "saviour",
        "tumor": "tumour",
        "tumors": "tumours",

        # -re vs -er
        "center": "centre",
        "centers": "centres",
        "centered": "centred",
        "centering": "centring",
        "meter": "metre",
        "meters": "metres",
        "kilometer": "kilometre",
        "kilometers": "kilometres",
        "centimeter": "centimetre",
        "centimeters": "centimetres",
        "millimeter": "millimetre",
        "millimeters": "millimetres",
        "liter": "litre",
        "liters": "litres",
        "theater": "theatre",
        "theaters": "theatres",
        "fiber": "fibre",
        "fibers": "fibres",

        # grey/gray
        "gray": "grey",
        "grays": "greys",
        "grayer": "greyer",
        "grayest": "greyest",
        "grayish": "greyish",

        # Note: IUCN prefers -ize over -ise (organization, not organisation)
        # So these are the corrections:
        "organisation": "organization",
        "organisations": "organizations",
        "organised": "organized",
        "organising": "organizing",
        "colonise": "colonize",
        "colonised": "colonized",
        "colonising": "colonizing",
        "colonisation": "colonization",
        "recognise": "recognize",
        "recognised": "recognized",
        "recognising": "recognizing",
        "recognition": "recognition",  # Same
        "realise": "realize",
        "realised": "realized",
        "realising": "realizing",
        "specialise": "specialize",
        "specialised": "specialized",
        "specialising": "specializing",
        "utilise": "utilize",
        "utilised": "utilized",
        "utilising": "utilizing",
        "utilisation": "utilization",
        "characterise": "characterize",
        "characterised": "characterized",
        "characterising": "characterizing",

        # programme/program (programme for schemes/plans, program for computer)
        # We'll flag 'program' only in non-computer contexts - this is complex
        # For now, we'll skip this as it requires context

        # Other common differences
        "analyze": "analyse",  # Actually IUCN may prefer analyze
        "catalogue": "catalog",  # US preferred
        "defense": "defence",
        "offense": "offence",
        "license": "licence",  # noun form
        "practice": "practise",  # verb form - complex
        "aging": "ageing",
        "artifact": "artefact",
        "judgement": "judgment",  # IUCN prefers judgment
        "acknowledgement": "acknowledgment",
    }

    def __init__(self):
        super().__init__(
            rule_id="spelling_uk",
            rule_name="UK English spelling required",
            category="Language",
            severity=Severity.WARNING,
            assessment_section="Whole Document"
        )
        # Build pattern from all incorrect spellings
        self.pattern = re.compile(
            r'\b(' + '|'.join(re.escape(k) for k in self.UK_SPELLING_MAP.keys()) + r')\b',
            re.IGNORECASE
        )

    def check(self, text: str) -> List[Violation]:
        """Check for US spellings that should be UK spellings."""
        violations = []

        for match in self.pattern.finditer(text):
            matched_text = match.group(0)
            matched_lower = matched_text.lower()

            if matched_lower in self.UK_SPELLING_MAP:
                fix = self.UK_SPELLING_MAP[matched_lower]

                # Preserve original case
                if matched_text[0].isupper():
                    fix = fix[0].upper() + fix[1:]
                if matched_text.isupper():
                    fix = fix.upper()

                violations.append(self._create_violation(
                    text=text,
                    matched_text=matched_text,
                    start=match.start(),
                    end=match.end(),
                    message=f"Use UK spelling '{fix}' instead of '{matched_text}'",
                    suggested_fix=fix
                ))

        return violations
