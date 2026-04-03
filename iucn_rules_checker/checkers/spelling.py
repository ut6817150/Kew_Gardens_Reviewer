"""UK vs US spelling checker for IUCN assessments."""

import re
from typing import List

from ..violation import Violation
from .base import BaseChecker


class SpellingChecker(BaseChecker):
    """Checker for UK vs US spelling rules."""

    UK_SPELLING_MAP = {
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
        "rigorous": "rigorous",
        "rumor": "rumour",
        "rumors": "rumours",
        "savior": "saviour",
        "tumor": "tumour",
        "tumors": "tumours",
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
        "gray": "grey",
        "grays": "greys",
        "grayer": "greyer",
        "grayest": "greyest",
        "grayish": "greyish",
        "analyze": "analyse",
        "catalog": "catalogue",
        "defense": "defence",
        "offense": "offence",
        "aging": "ageing",
        "artifact": "artefact",
        "judgment": "judgement",
        "acknowledgment": "acknowledgement",
    }

    IZE_WORDS = {
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
        "recognition": "recognition",
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
    }

    def __init__(self):
        super().__init__()
        self.uk_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(k) for k in self.UK_SPELLING_MAP.keys()) + r')\b',
            re.IGNORECASE
        )
        self.ize_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(k) for k in self.IZE_WORDS.keys()) + r')\b',
            re.IGNORECASE
        )

    def check_text(self, section_name: str, text: str) -> List[Violation]:
        """Check spelling preferences after removing simple style tags.

        This method strips simple inline style tags first:
        `<i>`, `<em>`, `<b>`, `<strong>`, `<sup>`, and `<sub>`.
        It then runs two passes over the cleaned text:
        - general spelling replacements from `UK_SPELLING_MAP`
        - dedicated `-ize` preferences from `IZE_WORDS`

        Examples flagged:
        `color` -> `colour`
        `organised` -> `organized`
        `<i>color</i>` -> `colour`
        `<b>organised</b>` -> `organized`
        `<i>col</i><i>or</i>` -> `colour`
        `<sup>org</sup><sub>anised</sub>` -> `organized`

        Examples not flagged:
        `colour`
        `organized`
        `<i>colour</i>`
        `<sup>organized</sup>`
        `<b>organized</b>`

        What it misses:
        it only strips those simple style tags and does not parse arbitrary
        markup.
        It may still miss words broken up by other tags or markup that leaves
        non-letter content between fragments.
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )

        for match in self.uk_pattern.finditer(cleaned_text):
            matched_text = match.group(0)
            matched_lower = matched_text.lower()

            if matched_lower in self.UK_SPELLING_MAP:
                fix = self.apply_case_pattern(self.UK_SPELLING_MAP[matched_lower], matched_text)

                if fix == matched_text:
                    continue

                violations.append(self.create_violation(
                    section_name=section_name,
                    text=text,
                    span=(index_map[match.start()], index_map[match.end() - 1] + 1),
                    message=f"Use UK spelling '{fix}' instead of '{matched_text}'",
                    suggested_fix=fix,
                ))

        for match in self.ize_pattern.finditer(cleaned_text):
            matched_text = match.group(0)
            matched_lower = matched_text.lower()

            if matched_lower in self.IZE_WORDS:
                fix = self.apply_case_pattern(self.IZE_WORDS[matched_lower], matched_text)

                if fix == matched_text:
                    continue

                violations.append(self.create_violation(
                    section_name=section_name,
                    text=text,
                    span=(index_map[match.start()], index_map[match.end() - 1] + 1),
                    message=f"IUCN prefers ize spelling '{fix}' instead of '{matched_text}'",
                    suggested_fix=fix,
                ))

        return violations

    def apply_case_pattern(self, fix: str, matched_text: str) -> str:
        """Apply the matched token's capitalization pattern to the fix."""
        if matched_text.isupper():
            return fix.upper()
        if matched_text[0].isupper():
            return fix[0].upper() + fix[1:]
        return fix
