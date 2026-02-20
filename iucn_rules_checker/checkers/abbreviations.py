"""Abbreviation checker for IUCN assessments."""

import re
from typing import List, Tuple

from .base import BaseChecker
from ..models import Violation, Severity


class AbbreviationChecker(BaseChecker):
    """Checker for abbreviation rules."""

    def __init__(self):
        super().__init__(
            rule_id="abbreviations_format",
            rule_name="Abbreviation formatting rules",
            category="Abbreviations",
            severity=Severity.WARNING,
            assessment_section="Whole Document"
        )

    def check(self, text: str) -> List[Violation]:
        """Check for abbreviation violations."""
        violations = []

        # e.g. and i.e. should be avoided in body text
        violations.extend(self._check_latin_abbreviations(text))

        # Correct format for common abbreviations
        violations.extend(self._check_abbreviation_formats(text))

        # Latin terms without periods and italicized
        violations.extend(self._check_latin_terms(text))

        # Title abbreviations
        violations.extend(self._check_title_abbreviations(text))

        return violations

    def _check_latin_abbreviations(self, text: str) -> List[Violation]:
        """Check for e.g. and i.e. which should be avoided in body text."""
        violations = []

        # e.g. - should use "for example"
        eg_pattern = re.compile(r'\be\.g\.(?:\s|,)', re.IGNORECASE)
        for match in eg_pattern.finditer(text):
            # Check if it's in brackets (more acceptable)
            before = text[max(0, match.start()-5):match.start()]
            if '(' in before:
                continue  # Skip if in brackets

            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0).strip(),
                start=match.start(),
                end=match.end()-1,
                message="Avoid 'e.g.' in body text; use 'for example' instead",
                suggested_fix="for example,"
            ))

        # i.e. - should use "that is"
        ie_pattern = re.compile(r'\bi\.e\.(?:\s|,)', re.IGNORECASE)
        for match in ie_pattern.finditer(text):
            before = text[max(0, match.start()-5):match.start()]
            if '(' in before:
                continue

            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0).strip(),
                start=match.start(),
                end=match.end()-1,
                message="Avoid 'i.e.' in body text; use 'that is' instead",
                suggested_fix="that is,"
            ))

        return violations

    def _check_abbreviation_formats(self, text: str) -> List[Violation]:
        """Check correct format for common abbreviations."""
        violations = []

        # etc without period
        etc_pattern = re.compile(r'\betc\b(?!\.)')
        for match in etc_pattern.finditer(text):
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message="Use 'etc.' with period",
                suggested_fix="etc."
            ))

        # et al without period
        etal_pattern = re.compile(r'\bet al\b(?!\.)')
        for match in etal_pattern.finditer(text):
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message="Use 'et al.' with period",
                suggested_fix="et al."
            ))

        # in lit instead of in litt.
        inlit_pattern = re.compile(r'\bin lit\b(?!t)')
        for match in inlit_pattern.finditer(text):
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message="Use 'in litt.' not 'in lit'",
                suggested_fix="in litt."
            ))

        # pers. comm without period after comm
        perscomm_pattern = re.compile(r'\bpers\.\s*comm\b(?!\.)')
        for match in perscomm_pattern.finditer(text):
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message="Use 'pers. comm.' format",
                suggested_fix="pers. comm."
            ))

        # pers. obs without period
        persobs_pattern = re.compile(r'\bpers\.\s*obs\b(?!\.)')
        for match in persobs_pattern.finditer(text):
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message="Use 'pers. obs.' format",
                suggested_fix="pers. obs."
            ))

        # Prof without period
        prof_pattern = re.compile(r'\bProf\b(?!\.)')
        for match in prof_pattern.finditer(text):
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message="Use 'Prof.' with period",
                suggested_fix="Prof."
            ))

        return violations

    def _check_latin_terms(self, text: str) -> List[Violation]:
        """Check Latin terms that should NOT have periods AND should be italicized."""
        violations = []

        # List of Latin terms to check
        latin_terms = [
            ('in situ', 'in natural habitat'),
            ('ex situ', 'outside natural habitat'),
            ('ad hoc', 'for this purpose'),
            ('in vivo', 'in living organism'),
            ('in vitro', 'in laboratory'),
            ('sensu lato', 'in the broad sense'),
            ('sensu stricto', 'in the strict sense'),
            ('per se', 'by itself'),
            ('de facto', 'in reality'),
            ('vice versa', 'the other way around'),
        ]

        for term, description in latin_terms:
            # 1. Check for version WITH periods (always wrong)
            escaped_term = re.escape(term)
            period_pattern = escaped_term.replace(r'\ ', r'\.?\s*') + r'\.?'
            period_regex = re.compile(rf'\b{period_pattern}\b', re.IGNORECASE)
            
            for match in period_regex.finditer(text):
                matched_text = match.group(0)
                
                # If it has periods, flag it
                if '.' in matched_text:
                    violations.append(self._create_violation(
                        text=text,
                        matched_text=matched_text,
                        start=match.start(),
                        end=match.end(),
                        message=f"Latin term '{term}' should not have periods and should be italicized",
                        suggested_fix=f"<i>{term}</i>"
                    ))
            
            # 2. Check for non-italicized version (without periods)
            plain_pattern = re.compile(rf'\b{escaped_term}\b', re.IGNORECASE)
            
            for match in plain_pattern.finditer(text):
                # Check if this is inside italic tags
                if not self._is_inside_italic(text, match.start(), match.end()):
                    violations.append(self._create_violation(
                        text=text,
                        matched_text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        message=f"Latin term should be italicized: <i>{term}</i>",
                        suggested_fix=f"<i>{term}</i>"
                    ))

        return violations

    def _is_inside_italic(self, text: str, start: int, end: int) -> bool:
        """Check if a position is inside italic tags."""
        # Look backwards for opening tag
        before = text[:start]
        after = text[end:]
        
        # Find the last opening italic tag before this position
        open_i = before.rfind('<i>')
        open_em = before.rfind('<em>')
        last_open = max(open_i, open_em)
        
        if last_open == -1:
            return False
        
        # Find the closing tag after the opening
        close_i = before.rfind('</i>')
        close_em = before.rfind('</em>')
        last_close = max(close_i, close_em)
        
        # If the last opening is after the last closing, we're inside italic
        if last_open > last_close:
            # Verify there's a closing tag after our position
            close_after_i = after.find('</i>')
            close_after_em = after.find('</em>')
            if close_after_i != -1 or close_after_em != -1:
                return True
        
        return False

    def _check_title_abbreviations(self, text: str) -> List[Violation]:
        """Check title abbreviation formats."""
        violations = []

        # Dr. should be Dr (UK style - no period)
        dr_pattern = re.compile(r'\bDr\.(?=\s)')
        for match in dr_pattern.finditer(text):
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message="Use 'Dr' without period (UK style)",
                suggested_fix="Dr"
            ))

        return violations






























