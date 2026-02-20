"""Validate IUCN assessments from Streamlit's hierarchical JSON structure.

This validator works with the actual JSON format from Streamlit app,
which has a hierarchical structure with sections as 'children' and content in 'blocks'.
"""

from typing import List, Dict, Any, Tuple
import re
from .models import Violation, Severity, TextPosition, ViolationReport

class StreamlitAssessmentValidator:
    """Validate IUCN assessments from Streamlit JSON."""
    
    def __init__(self):
        self.violations = []
        self.sections = {}  # section_name -> text content
        self.tables = {}    # section_name -> table data
        self.metadata = {}  # extracted metadata (category, criteria, etc.)
        self.checked_rules = []


    def validate(self, assessment: Dict[str, Any]) -> Tuple[ViolationReport, str]:
        """
        Validate assessment from Streamlit JSON.
        
        Returns:
            Tuple of (ViolationReport, full_text)
        """
        self.violations = []
        self.sections = {}
        self.tables = {}
        self.metadata = {}
        
        # Parse the hierarchical structure
        self._parse_document(assessment)
        
        # Extract metadata from parsed content
        self._extract_metadata()
        
        # Run validations
        self._validate_metadata()
        self._validate_required_sections()
        self._validate_geographic_data()
        self._validate_criteria()
        self._validate_numeric_fields()
        self._validate_d2_criteria()
        self._check_eoo_aoo_discrepancy()
        
        # Build full text for legacy checkers
        full_text = self._build_full_text()

        total_violations = len(self.violations)

        violations_by_severity = {}
        violations_by_category = {}

        for v in self.violations:
        # assuming v has attributes severity and category
            violations_by_severity[v.severity] = (
                violations_by_severity.get(v.severity, 0) + 1
            )
            violations_by_category[v.category] = (
                violations_by_category.get(v.category, 0) + 1
            )
        
        return ViolationReport(
            text_length=len(full_text),
            total_violations=total_violations,
            violations_by_severity=violations_by_severity,
            violations_by_category=violations_by_category,
            violations=self.violations,
            checked_rules=self.checked_rules,   # make sure this exists
        ), full_text
    
    def _parse_document(self, node: Dict[str, Any], path: List[str] = None):
        """Recursively parse the hierarchical document structure."""
        if path is None:
            path = []
        
        # Get section title
        title = node.get('title', '')
        if title:
            path = path + [title]
        
        # Process blocks in this section
        blocks = node.get('blocks', [])
        section_text = []
        section_tables = []
        
        for block in blocks:
            if block.get('type') == 'paragraph':
                text = block.get('text', '').strip()
                if text:
                    section_text.append(text)
            
            elif block.get('type') == 'table':
                rows = block.get('rows', [])
                section_tables.append(rows)
        
        # Store section content
        if path:
            section_key = ' > '.join(path)
            if section_text:
                self.sections[section_key] = '\n'.join(section_text)
            if section_tables:
                self.tables[section_key] = section_tables
        
        # Recurse into children
        for child in node.get('children', []):
            self._parse_document(child, path)
    
    def _extract_metadata(self):
        """Extract key metadata from parsed sections."""
        # Extract category and criteria from Red List Status table
        for section_key, tables in self.tables.items():
            for table in tables:
                if len(table) >= 2 and any('Red List Status' in str(cell) for cell in table[0]):
                    # Parse status like "VU - Vulnerable, B2ab(iii) (IUCN version 3.1)"
                    status_text = str(table[1][0]) if len(table[1]) > 0 else ''
                    
                    # Extract category (CR, EN, VU, NT, LC, DD, EX, EW)
                    cat_match = re.match(r'([A-Z]{2,3})\s*-', status_text)
                    if cat_match:
                        self.metadata['category'] = cat_match.group(1)
                    
                    # Extract criteria
                    crit_match = re.search(r',\s*([A-E][0-9]?[a-z]*(?:\([^)]+\))?(?:\+[A-E][0-9]?[a-z]*(?:\([^)]+\))?)*)', status_text)
                    if crit_match:
                        self.metadata['criteria'] = crit_match.group(1)
        
        # Extract EOO and AOO from tables
        for section_key, tables in self.tables.items():
            if 'EOO' in section_key or 'Extent of Occurrence' in section_key:
                for table in tables:
                    if len(table) >= 2:
                        # EOO is usually in second row, first column
                        eoo_text = str(table[1][0]) if len(table[1]) > 0 else ''
                        if eoo_text and eoo_text.isdigit():
                            self.metadata['eoo'] = int(eoo_text)
            
            if 'AOO' in section_key or 'Area of Occupancy' in section_key:
                for table in tables:
                    if len(table) >= 2:
                        aoo_text = str(table[1][0]) if len(table[1]) > 0 else ''
                        if aoo_text and aoo_text.isdigit():
                            self.metadata['aoo'] = int(aoo_text)
        
        # Extract assessors and reviewers
        for section_key, text in self.sections.items():
            if 'assessor' in section_key.lower() or 'Assessor(s):' in text:
                match = re.search(r'Assessor\(s\):\s*(.+)', text)
                if match:
                    self.metadata['assessors'] = match.group(1).strip()
            
            if 'reviewer' in section_key.lower() or 'Reviewer(s):' in text:
                match = re.search(r'Reviewer\(s\):\s*(.+)', text)
                if match:
                    self.metadata['reviewers'] = match.group(1).strip()
    
    def _validate_metadata(self):
        """Validate basic metadata requirements."""
        # Category required
        if not self.metadata.get('category'):
            self._add_violation(
                rule_id="category_required",
                message="Red List category is required",
                severity=Severity.ERROR,
                category="Metadata"
            )
        
        # NE not submittable
        if self.metadata.get('category') == 'NE':
            self._add_violation(
                rule_id="ne_not_submittable",
                message="NE (Not Evaluated) assessments cannot be submitted",
                severity=Severity.ERROR,
                category="Metadata"
            )
        
        # Criteria required for threatened species
        cat = self.metadata.get('category')
        if cat in ['CR', 'EN', 'VU'] and not self.metadata.get('criteria'):
            self._add_violation(
                rule_id="criteria_required",
                message=f"Red List criteria required for {cat} assessment",
                severity=Severity.ERROR,
                category="Metadata"
            )
        
        # Assessors required
        if not self.metadata.get('assessors'):
            self._add_violation(
                rule_id="assessors_required",
                message="Assessor name(s) required",
                severity=Severity.ERROR,
                category="Metadata"
            )
        
        # Reviewers required
        if not self.metadata.get('reviewers'):
            self._add_violation(
                rule_id="reviewers_required",
                message="Reviewer name(s) required",
                severity=Severity.ERROR,
                category="Metadata"
            )
    
    def _validate_required_sections(self):
        """Check required narrative sections exist and have content."""
        category = self.metadata.get('category', '')
        
        # Rationale always required
        rationale_section = self._find_section('Assessment Rationale')
        if not rationale_section:
            self._add_violation(
                rule_id="rationale_required",
                message="Assessment Rationale section is required",
                severity=Severity.ERROR,
                category="Content"
            )
        elif len(rationale_section.split()) < 20:
            self._add_violation(
                rule_id="rationale_too_brief",
                message=f"Assessment Rationale too brief ({len(rationale_section.split())} words, minimum 20 words)",
                severity=Severity.ERROR,
                category="Content",
                matched_text=rationale_section[:100] + "..."
            )
        
        # For non-LC species
        if category not in ['LC', 'NE', '']:
            # Geographic Range required
            geo_section = self._find_section('Geographic Range')
            if not geo_section:
                self._add_violation(
                    rule_id="geographic_range_required",
                    message=f"Geographic Range section required for {category} species",
                    severity=Severity.ERROR,
                    category="Content"
                )
            elif len(geo_section.split()) < 15:
                self._add_violation(
                    rule_id="geographic_range_too_brief",
                    message=f"Geographic Range too brief ({len(geo_section.split())} words, minimum 15 words)",
                    severity=Severity.WARNING,
                    category="Content"
                )
            
            # Population section required
            pop_section = self._find_section('Population')
            if not pop_section:
                self._add_violation(
                    rule_id="population_required",
                    message=f"Population section required for {category} species",
                    severity=Severity.ERROR,
                    category="Content"
                )
            
            # Habitat & Ecology required
            habitat_section = self._find_section('Habitat and Ecology') or self._find_section('Habitats and Ecology')
            if not habitat_section:
                self._add_violation(
                    rule_id="habitat_ecology_required",
                    message=f"Habitat and Ecology section required for {category} species",
                    severity=Severity.ERROR,
                    category="Content"
                )
        
        # Threats required for threatened species
        if category in ['CR', 'EN', 'VU', 'NT']:
            threats_section = self._find_section('Threats')
            if not threats_section:
                self._add_violation(
                    rule_id="threats_required",
                    message=f"Threats section required for {category} species",
                    severity=Severity.ERROR,
                    category="Content"
                )
    
    def _validate_geographic_data(self):
        """Validate EOO and AOO data."""
        eoo = self.metadata.get('eoo')
        aoo = self.metadata.get('aoo')
        
        # EOO and AOO relationship
        if eoo and aoo:
            if eoo < aoo:
                self._add_violation(
                    rule_id="eoo_less_than_aoo",
                    message=f"EOO ({eoo:,} km²) cannot be less than AOO ({aoo:,} km²). If calculated EOO < AOO, set EOO = AOO.",
                    severity=Severity.ERROR,
                    category="Geographic Data",
                    matched_text=f"EOO: {eoo}, AOO: {aoo}"
                )
    
    def _validate_criteria(self):
        """Validate criteria-specific requirements."""
        criteria = self.metadata.get('criteria', '')
        category = self.metadata.get('category', '')
        eoo = self.metadata.get('eoo')
        aoo = self.metadata.get('aoo')
        
        if not criteria:
            return
        
        # B1 requires EOO
        if 'B1' in criteria and not eoo:
            self._add_violation(
                rule_id="b1_eoo_required",
                message="Criterion B1 requires estimated EOO value",
                severity=Severity.ERROR,
                category="Criteria"
            )
        
        # B2 requires AOO
        if 'B2' in criteria and not aoo:
            self._add_violation(
                rule_id="b2_aoo_required",
                message="Criterion B2 requires estimated AOO value",
                severity=Severity.ERROR,
                category="Criteria"
            )
        
        # D1/D2 only for VU
        if any(d in criteria for d in ['D1', 'D2']):
            if category != 'VU':
                self._add_violation(
                    rule_id="d1_d2_only_vu",
                    message=f"Criteria D1/D2 can only be used with VU category (current: {category})",
                    severity=Severity.ERROR,
                    category="Criteria"
                )
    
    def _validate_numeric_fields(self):
        """Check for excessive decimal places and formatting."""
        # Check EOO/AOO in table data
        for section_key, tables in self.tables.items():
            for table in tables:
                for row in table:
                    for cell in row:
                        cell_str = str(cell)
                        
                        # Check for excessive decimals in EOO/AOO
                        if re.search(r'(\d+\.\d{3,})\s*km', cell_str):
                            match = re.search(r'([\d,]+\.[\d]+)\s*km', cell_str)
                            if match:
                                value = match.group(1).replace(',', '')
                                rounded = str(int(round(float(value))))
                                self._add_violation(
                                    rule_id="excessive_decimals_eoo_aoo",
                                    message=f"Round EOO/AOO to whole number: {value} → {rounded} km²",
                                    severity=Severity.WARNING,
                                    category="Geographic Data",
                                    matched_text=cell_str,
                                    suggested_fix=f"{rounded} km²"
                                )
    
    
    def _validate_d2_criteria(self):
        """Check that VU D2 has a viable threat identified."""
        category = self.metadata.get('category', '')
        criteria = self.metadata.get('criteria', '')
    
        if category == 'VU' and 'D2' in criteria:
            # Check if threats section has substantial content
            threats_section = self._find_section('Threats')
        
            if not threats_section:
                self._add_violation(
                    rule_id="vu_d2_requires_threat",
                    message="VU D2 requires identification of a plausible threat (not just restricted AOO/locations)",
                    severity=Severity.ERROR,
                    category="Criteria"
                )

            elif len(threats_section.split()) < 20:
                self._add_violation(
                    rule_id="vu_d2_threat_too_brief",
                    message=f"VU D2 threat description too brief ({len(threats_section.split())} words). Describe a plausible future threat.",
                    severity=Severity.WARNING,
                    category="Criteria"
                )
            else:
                # Check for vague threat language
                vague_threats = [
                    'restricted range', 'small range', 'limited range',
                    'few locations', 'small aoo', 'restricted aoo'
                ]
            
                threats_lower = threats_section.lower()
                has_specific_threat = any(
                    threat in threats_lower for threat in [
                        'fire', 'drought', 'invasive', 'development', 'agriculture',
                        'logging', 'mining', 'climate', 'disease', 'predation',
                        'habitat loss', 'pollution', 'overexploitation'
                    ]
                )
            
                has_only_vague = any(v in threats_lower for v in vague_threats)
            
                if has_only_vague and not has_specific_threat:
                    self._add_violation(
                        rule_id="vu_d2_vague_threat",
                        message="VU D2 requires a specific plausible threat, not just restricted range/AOO",
                        severity=Severity.WARNING,
                        category="Criteria"
                    )
    
    def _check_locations_explained(self):
        """Check if number of locations is explained, not just stated."""
        rationale = self._find_section('Assessment Rationale')
        locations_section = self._find_section('Locations')
    
        # Check if locations are mentioned
        location_mention = re.search(r'(\d+)\s+locations?', rationale, re.IGNORECASE)
    
        if location_mention:
            num_locations = location_mention.group(1)
        
            # Look for explanation terms nearby (within 100 chars)
            context_start = max(0, location_mention.start() - 100)
            context_end = min(len(rationale), location_mention.end() + 100)
            context = rationale[context_start:context_end].lower()
        
            explanation_terms = [
                'threat', 'fire', 'invasive', 'logging', 'agriculture',
                'based on', 'defined by', 'because', 'due to', 'as',
                'where', 'considering', 'since'
            ]
        
            has_explanation = any(term in context for term in explanation_terms)
        
            if not has_explanation:
                self._add_violation(
                    rule_id="locations_not_explained",
                    message=f"Number of locations ({num_locations}) mentioned but not explained. Define what constitutes a 'location' based on threats.",
                    severity=Severity.WARNING,
                    category="Geographic Data"
                )
    
    
    def _check_eoo_aoo_discrepancy(self):
        """Warn if large EOO with small AOO without explanation."""
        eoo = self.metadata.get('eoo')
        aoo = self.metadata.get('aoo')
    
        if eoo and aoo and eoo > (aoo * 100):  # EOO > 100x AOO
            rationale = self._find_section('Assessment Rationale')
        
            explanation_terms = [
                'sampling', 'sample', 'uncertainty', 'uncertain',
                'disjunct', 'patchy', 'fragmented', 'isolated',
                'poorly known', 'under-recorded', 'sparse'
            ]
        
            has_explanation = any(term in rationale.lower() for term in explanation_terms)
        
            if not has_explanation:
                self._add_violation(
                    rule_id="eoo_aoo_discrepancy",
                    message=f"Large EOO ({eoo:,} km²) with small AOO ({aoo} km²) should mention sampling effort, uncertainty, or habitat patchiness",
                    severity=Severity.WARNING,
                    category="Geographic Data"
                )

    def _find_section(self, section_name: str) -> str:
        """Find section by name (case-insensitive, partial match)."""
        section_name_lower = section_name.lower()
        for key, text in self.sections.items():
            if section_name_lower in key.lower():
                return text
        return ''
    
    def _build_full_text(self) -> str:
        """Build full text from all sections for legacy text checkers."""
        all_text = []
        
        # Add sections in a logical order
        section_order = [
            'Assessment Rationale',
            'Geographic Range',
            'Population',
            'Habitat',
            'Threats',
            'Conservation',
            'Use and Trade'
        ]
        
        for section_name in section_order:
            text = self._find_section(section_name)
            if text:
                all_text.append(f"## {section_name}\n{text}")
        
        # Add any remaining sections
        for key, text in self.sections.items():
            if not any(s.lower() in key.lower() for s in section_order):
                all_text.append(f"## {key}\n{text}")
        
        return '\n\n'.join(all_text)
    
    def _add_violation(self, rule_id: str, message: str, severity: Severity,
                      category: str, matched_text: str = "", suggested_fix: str = None):
        """Helper to add a violation."""
        self.violations.append(Violation(
            rule_id=rule_id,
            rule_name=message,
            category=category,
            matched_text=matched_text,
            position=TextPosition(0, 0, 0, 0),
            severity=severity,
            message=message,
            suggested_fix=suggested_fix,
            context="",
            assessment_section="Whole Document",
            metadata={}
        ))
    
    def _check_criteria_thresholds(self):
        """Check that criteria thresholds match the category."""
        category = self.metadata.get('category', '')
        eoo = self.metadata.get('eoo')
        aoo = self.metadata.get('aoo')
    
        # Check EN category
        if category == 'EN':
            if eoo and eoo > 5000:  # Only meets VU threshold
                self._add_violation(
                    rule_id="en_with_vu_eoo",
                    message=f"Assessed as EN but EOO ({eoo:,} km²) only meets VU threshold (<5,000 km²). Verify other criteria justify EN category.",
                    severity=Severity.WARNING,
                    category="Criteria"
                )
        
            if aoo and aoo > 500:  # Only meets VU threshold
                self._add_violation(
                    rule_id="en_with_vu_aoo",
                    message=f"Assessed as EN but AOO ({aoo} km²) only meets VU threshold (<500 km²). Verify other criteria justify EN category.",
                    severity=Severity.WARNING,
                    category="Criteria"
                )
    
        # Check CR category
        if category == 'CR':
            if eoo and eoo > 100:  # Only meets EN/VU threshold
                if eoo > 5000:
                    level = "VU"
                else:
                    level = "EN"
                self._add_violation(
                    rule_id="cr_with_lower_eoo",
                    message=f"Assessed as CR but EOO ({eoo:,} km²) only meets {level} threshold. Verify other criteria justify CR category.",
                    severity=Severity.WARNING,
                    category="Criteria"
                )
        
        if aoo and aoo > 10:  # Only meets EN/VU threshold
            if aoo > 500:
                level = "VU"
            else:
                level = "EN"
            self._add_violation(
                rule_id="cr_with_lower_aoo",
                message=f"Assessed as CR but AOO ({aoo} km²) only meets {level} threshold. Verify other criteria justify CR category.",
                severity=Severity.WARNING,
                category="Criteria"
            )


# Example usage
if __name__ == '__main__':
    import json
    import sys
    
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            assessment_json = json.load(f)
        
        validator = StreamlitAssessmentValidator()
        report, full_text = validator.validate(assessment_json)
        
        print(f"Violations: {len(report.violations)}\n")
        for v in report.violations:
            print(f"[{v.severity.value.upper()}] {v.category}: {v.message}")
            if v.matched_text:
                print(f"  Found: {v.matched_text[:100]}")
            if v.suggested_fix:
                print(f"  Fix: {v.suggested_fix}")
            print()




















