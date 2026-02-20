"""Formatting checker for IUCN assessments (italics, bold, etc.).

This checker expects text to contain HTML formatting tags:
- <i>text</i> or <em>text</em> for italics
- <b>text</b> or <strong>text</strong> for bold

The DOCX-to-JSON conversion should preserve formatting as HTML tags.
"""

import re
from typing import List, Tuple, Set

from .base import BaseChecker
from ..models import Violation, Severity


class FormattingChecker(BaseChecker):
    """Checker for formatting rules (italics for scientific names, et al., etc.)."""

    # Common taxonomic family suffixes (these should NOT be italicized)
    FAMILY_SUFFIXES = {
        'aceae',   # Plant families: Rosaceae, Orchidaceae
        'idae',    # Animal families: Felidae, Canidae
        'ales',    # Orders: Rosales, Asparagales
        'ineae',   # Subtribes
        'inae',    # Subfamilies
        'eae',     # Tribes
        'oideae',  # Subfamilies
    }
    
    KNOWN_FAMILIES = {
        # Major plant families
        'Orchidaceae', 'Rubiaceae', 'Fabaceae', 'Asteraceae', 'Poaceae',
        'Rosaceae', 'Euphorbiaceae', 'Lamiaceae', 'Malvaceae', 'Solanaceae',
        'Brassicaceae', 'Apiaceae', 'Cactaceae', 'Acanthaceae', 'Araceae',
        
        # Major animal families
        'Felidae', 'Canidae', 'Hominidae', 'Bovidae', 'Cervidae',
        'Accipitridae', 'Columbidae', 'Psittacidae', 'Salamandridae',
        
        # Orders
        'Rosales', 'Fabales', 'Asparagales', 'Lamiales', 'Solanales',
        'Carnivora', 'Primates', 'Rodentia',
    }

    # Higher taxonomy ranks that should NOT be italicized
    HIGHER_RANKS = [
        'phylum', 'class', 'order', 'family', 'superfamily',
        'suborder', 'infraorder', 'subclass', 'superorder'
    ]

    def __init__(self):
        super().__init__(
            rule_id="formatting_italics",
            rule_name="Formatting rules (italics)",
            category="Formatting",
            severity=Severity.WARNING,
            assessment_section="Whole Document"
        )
        self._html_processor = None

    def set_html_processor(self, processor):
        """Set the HTML processor for checking formatting."""
        self._html_processor = processor

    def _check_eoo_aoo_capitalization(self, text: str) -> List[Violation]:
        """Check that EOO/AOO are lowercase when mid-sentence."""
        violations = []
    
        # Terms to check
        terms = {
            'Extent of Occurrence': 'extent of occurrence',
            'Area of Occupancy': 'area of occupancy',
        }
    
        for incorrect, correct in terms.items():
            # Pattern: Not at start of sentence (after lowercase letter + space)
            pattern = re.compile(rf'(?<=[a-z]\s){re.escape(incorrect)}\b')
        
            for match in pattern.finditer(text):
                violations.append(self._create_violation(
                    text=text,
                    matched_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    message=f"Use lowercase mid-sentence: '{correct}' not '{incorrect}'",
                    suggested_fix=correct
                ))
    
        return violations


    def check(self, text: str) -> List[Violation]:
        """Check for formatting violations."""
        violations = []

        # Check that et al. is italicized
        violations.extend(self._check_et_al_italics(text))

        # Check that scientific names (binomial) are italicized
        violations.extend(self._check_scientific_name_italics(text))

        # Check that family/higher taxonomy names are NOT italicized
        violations.extend(self._check_family_not_italicized(text))

        # Check that spp./sp. after genus are NOT italicized
        violations.extend(self._check_spp_not_italicized(text))
        
        #capitalisation
        violations.extend(self._check_family_name_capitalization(text))

        #AOO/EOO capitalisation
        violations.extend(self._check_eoo_aoo_capitalization(text))

        return violations

    def _check_et_al_italics(self, text: str) -> List[Violation]:
        """Check that 'et al.' is italicized."""
        violations = []

        # Find all occurrences of "et al." not inside italic tags
        # First, find all et al. that ARE in italics (to exclude them)
        italicized_etal = set()
        italic_pattern = re.compile(r'<(?:i|em)>(.*?)</(?:i|em)>', re.IGNORECASE | re.DOTALL)
        for match in italic_pattern.finditer(text):
            content = match.group(1)
            if 'et al' in content.lower():
                # This et al. is properly italicized
                italicized_etal.add(match.start())

        # Now find all et al. occurrences
        etal_pattern = re.compile(r'\bet\s+al\.?\b', re.IGNORECASE)
        for match in etal_pattern.finditer(text):
            # Check if this position is inside an italic tag
            is_italicized = self._is_inside_italic(text, match.start(), match.end())
            if not is_italicized:
                violations.append(self._create_violation(
                    text=text,
                    matched_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    message="'et al.' should be italicized: <i>et al.</i>",
                    suggested_fix=f"<i>{match.group(0)}</i>"
                ))

        return violations

    def _check_scientific_name_italics(self, text: str) -> List[Violation]:
        """Check that scientific names (Genus species) are italicized.

        This is conservative to avoid false positives - it only flags patterns
        that strongly look like scientific names.
        """
        violations = []

        # Pattern for binomial names: Capitalized word + lowercase word
        # e.g., "Panthera leo", "Quercus robur"
        binomial_pattern = re.compile(r'\b([A-Z][a-z]+)\s+([a-z]{2,})\b')

        # Common English words that start sentences or phrases (skip these as genus)
        common_first_words = {
            'The', 'This', 'That', 'These', 'Those', 'Some', 'Many', 'Most',
            'Each', 'Every', 'Which', 'What', 'Where', 'When', 'While', 'While',
            'Although', 'Because', 'Since', 'After', 'Before', 'During', 'Until',
            'According', 'Based', 'Given', 'However', 'Therefore', 'Furthermore',
            'Moreover', 'Nevertheless', 'Consequently', 'Additionally', 'Finally',
            'Recent', 'Current', 'Previous', 'Several', 'Various', 'Different',
            'Similar', 'Other', 'Another', 'Such', 'Only', 'Both', 'Either',
            'Neither', 'All', 'Any', 'No', 'Not', 'But', 'And', 'Or', 'For',
            'With', 'From', 'Into', 'Over', 'Under', 'About', 'Between', 'Among',
            'Through', 'Within', 'Without', 'Along', 'Across', 'Around', 'Against',
            'Studies', 'Research', 'Data', 'Results', 'Analysis', 'Evidence',
            'Population', 'Species', 'Habitat', 'Range', 'Distribution', 'Status',
            'Conservation', 'Threat', 'Decline', 'Increase', 'Change', 'Loss',
            'Smith', 'Jones', 'Brown', 'Wilson', 'Johnson', 'Williams', 'Taylor',
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December',
            'Very', 'More', 'Less', 'Blue', 'Red', 'Green', 'Black', 'White',
            'Past', 'Present', 'Future', 'Long', 'Short', 'High', 'Low',
            'Large', 'Small', 'Good', 'Bad', 'New', 'Old', 'First', 'Last',
            'Whole', 'Part', 'Full', 'Empty', 'Wild', 'Domestic', 'Native',
            'Foreign', 'Local', 'Global', 'National', 'International', 'Regional',
            'Continued', 'Continuing', 'Systematic', 'Area', 'Invasive', 'Harvest',
            'Successfully', 'Genome', 'List', 'Endemic', 'Forest', 'Overwinter',
            'Biological', 'Intentional', 'Named', 'Shifting', 'Republic', 'Please',
            'Use', 'Geographic', 'Date', 'Map', 'Further', 'Extreme', 'Severely',
            'Australian', 'Maestra', 'Satellite', 'Jamaican', 'Park', 'High',
            'Targeted', 'Portland', 'Granma', 'Estimated', 'Lower', 'Mountain',
            'Is', 'Are', 'Was', 'Were', 'Has', 'Have', 'Had',
            'Santiago', 'Saint', 'Sierra', 'Monte',  # Place names
            'Blue', 'Green', 'Red', 'White', 'Black',  # Colors often in place names
        }

        # Common English second words (skip these as species epithet)
        common_second_words = {
            'the', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from',
            'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'shall', 'can', 'cannot', 'could', 'need',
            'that', 'which', 'who', 'whom', 'whose', 'what', 'where', 'when',
            'how', 'why', 'if', 'then', 'than', 'because', 'although', 'while',
            'and', 'or', 'but', 'nor', 'so', 'yet', 'both', 'either', 'neither',
            'not', 'no', 'yes', 'only', 'also', 'just', 'even', 'still', 'already',
            'et', 'al', 'etal',  # Skip "et al" patterns
            'studies', 'shows', 'found', 'reported', 'suggests', 'indicates',
            'between', 'among', 'within', 'across', 'along', 'through', 'under',
            'restricted', 'available', 'map', 'state', 'range', 'restriction',
            'created', 'fluctuations', 'fragmented', 'decline', 'mountains',
            'imagery', 'relative', 'detail', 'harvest', 'road', 'clearance',
            'tree', 'term', 'mountain', 'due', 'agriculture', 'resource',
            'use', 'species', 'based', 'plan', 'control', 'management',
            'reintroduced', 'monitoring', 'change', 'rates', 'establishment',
            'montane', 'factsheet', 'status', 'threat', 'paper',
            'surveys', 'parishes', 'provinces', 'area', 'extent', 'estimate',
            'subpopulation', 'there', 'here',
        }

        #checking if word looks like it could be latin
        def looks_like_latin(word):
            """Simple heuristic: Latin words often have certain endings."""
            latin_endings = ('us', 'a', 'um', 'is', 'e', 'ensis', 'oides', 'ella', 'ina', 'ana')
            return word.lower().endswith(latin_endings) and len(word) >= 4

        for match in binomial_pattern.finditer(text):
            genus = match.group(1)
            species = match.group(2)
            full_name = match.group(0)

            # Skip common English patterns
            if genus in common_first_words:
                continue
            if species in common_second_words:
                continue

            # Skip if genus ends with family suffix
            if any(genus.lower().endswith(suffix) for suffix in self.FAMILY_SUFFIXES):
                continue

            # Skip very short species epithets (likely not Latin)
            if len(species) < 4:
                continue

            # Additional check: at least one word should look Latin-ish
            if not (looks_like_latin(genus) or looks_like_latin(species)):
                # If neither looks Latin, be more cautious
                # Only flag if both are uncommon words (not in any common word list)
                if genus.lower() in [w.lower() for w in common_first_words]:
                    continue
                if species.lower() in common_second_words:
                    continue

            # Check if already italicized
            if not self._is_inside_italic(text, match.start(), match.end()):
                # This looks like an un-italicized scientific name
                violations.append(self._create_violation(
                    text=text,
                    matched_text=full_name,
                    start=match.start(),
                    end=match.end(),
                    message=f"Scientific name should be italicized: <i>{full_name}</i>",
                    suggested_fix=f"<i>{full_name}</i>"
                ))

        return violations

    def _check_family_not_italicized(self, text: str) -> List[Violation]:
        """Check that family/order/class names are NOT italicized."""
        violations = []

        # Find italicized text
        italic_pattern = re.compile(r'<(?:i|em)>(.*?)</(?:i|em)>', re.IGNORECASE | re.DOTALL)

        for match in italic_pattern.finditer(text):
            content = match.group(1).strip()

            # Check if content looks like a family/order name
            for suffix in self.FAMILY_SUFFIXES:
                if content.lower().endswith(suffix) and content[0].isupper():
                    # This looks like a family name that shouldn't be italicized
                    violations.append(self._create_violation(
                        text=text,
                        matched_text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        message=f"Family/order names should not be italicized: {content}",
                        suggested_fix=content
                    ))
                    break

        return violations
    
    def _check_family_name_capitalization(self, text: str) -> List[Violation]:
        """Check that family/higher taxonomy names are properly capitalized."""
        violations = []

        # Pattern: find words ending in family suffixes
        suffix_pattern = '|'.join(re.escape(s) for s in self.FAMILY_SUFFIXES)
        pattern = rf'\b([a-z][a-z]+)({suffix_pattern})\b'

        for match in re.finditer(pattern, text, re.IGNORECASE):
            full_name = match.group(0)
            stem = match.group(1)
            suffix = match.group(2)
    
            # Check if it's lowercase (incorrect)
            if full_name[0].islower():
                proper_name = stem.capitalize() + suffix
            
                # Only flag if it's a known family or clearly looks like one
                if proper_name in self.KNOWN_FAMILIES or len(stem) >= 4:
                    violations.append(self._create_violation(
                        text=text,
                        matched_text=full_name,
                        start=match.start(),
                        end=match.end(),
                        message=f"Family/taxonomy names should be capitalized: '{proper_name}'",
                        suggested_fix=proper_name
                    ))

        # Check for italicized family names (should NOT be italicized)
        italic_pattern = rf'<(?:i|em)>([A-Z][a-z]+(?:{suffix_pattern}))</(?:i|em)>'

        for match in re.finditer(italic_pattern, text):
            family_name = match.group(1)
    
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message=f"Family names should NOT be italicized: '{family_name}'",
                suggested_fix=family_name
            ))

        return violations
    
    def _check_spp_not_italicized(self, text: str) -> List[Violation]:
        """Check that spp./sp. after genus are NOT italicized (only genus is)."""
        violations = []

        # Pattern: <i>Genus spp.</i> or <i>Genus sp.</i>
        # The spp./sp. should be outside the italic tags
        pattern = re.compile(r'<(?:i|em)>([A-Z][a-z]+)\s+(spp?\.?)</(?:i|em)>', re.IGNORECASE)

        for match in pattern.finditer(text):
            genus = match.group(1)
            spp = match.group(2)
            violations.append(self._create_violation(
                text=text,
                matched_text=match.group(0),
                start=match.start(),
                end=match.end(),
                message=f"'{spp}' should not be italicized; only the genus: <i>{genus}</i> {spp}",
                suggested_fix=f"<i>{genus}</i> {spp}"
            ))

        return violations

    def _is_inside_italic(self, text: str, start: int, end: int) -> bool:
        """Check if a position is inside italic tags."""

        # Use HTML processor if available
        if self._html_processor:
            return self._html_processor.is_italicized(start, end)

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