"""Language and style checker for IUCN assessments."""

import re
from typing import List

from .base import BaseChecker
from ..models import Violation, Severity


class LanguageChecker(BaseChecker):
    """Checker for language and style issues."""

    def __init__(self):
        super().__init__(
            rule_id="language_style",
            rule_name="Language and style rules",
            category="Language",
            severity=Severity.WARNING,
            assessment_section="Whole Document"
        )

    def check(self, text: str) -> List[Violation]:
        """Check for language violations."""
        violations = []

        # Check for incomplete sentences (fragments)
        violations.extend(self._check_incomplete_sentences(text))

        return violations

    def _check_incomplete_sentences(self, text: str) -> List[Violation]:
        """Check for incomplete sentences (single-word fragments).
        
        Common in Habitat & Ecology sections:
        "Tree. Humid forest. Altitude 1000-2000m."
        
        These should be full sentences instead.
        """
        violations = []

        # Pattern 1: Single word followed by period, then another word
        # Example: "Tree. Humid" or "Shrub. Rocky"
        pattern1 = re.compile(r'\b([A-Z][a-z]{2,})\.\s+([A-Z][a-z]+)')
        
        for match in pattern1.finditer(text):
            word1, word2 = match.group(1), match.group(2)
            
            # Skip if it's a valid abbreviation (Dr. Smith, Prof. Jones)
            if word1 in ['Dr', 'Prof', 'Mr', 'Ms', 'Mrs', 'St']:
                continue
            
            # Skip if word2 is a month (date format)
            months = ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']
            if word2 in months:
                continue
            
            violations.append(self._create_violation(
                text=text,
                matched_text=f"{word1}. {word2}",
                start=match.start(),
                end=match.end(),
                message=f"Avoid sentence fragments; use complete sentences instead of '{word1}. {word2}...'",
                suggested_fix=None  # User must rewrite properly
            ))

        # Pattern 2: Very short "sentences" (< 3 words) that look like fragments
        # Split by periods and check
        sentences = re.split(r'\.\s+', text)
        pos = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            words = sentence.split()
            
            # If sentence is 1-2 words and looks like a fragment
            if 1 <= len(words) <= 2 and sentence:
                # Skip numbers (e.g., "2000m.")
                if words[0][0].isdigit():
                    pos += len(sentence) + 2
                    continue
                
                # Skip common abbreviations
                if any(abbrev in sentence for abbrev in ['Fig', 'Table', 'Appendix', 'cf', 'ca', 'ibid']):
                    pos += len(sentence) + 2
                    continue
                
                # Check if it's likely a fragment (starts with lowercase or very short)
                if len(words) == 1 and len(words[0]) > 3 and not words[0].isupper():
                    # Find position in original text
                    fragment_pos = text.find(sentence, pos)
                    if fragment_pos != -1:
                        violations.append(self._create_violation(
                            text=text,
                            matched_text=sentence,
                            start=fragment_pos,
                            end=fragment_pos + len(sentence),
                            message=f"Possible sentence fragment: '{sentence}'. Use complete sentences.",
                            suggested_fix=None
                        ))
            
            pos += len(sentence) + 2  # Move position forward

        return violations