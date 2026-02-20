"""Geographic naming checker for IUCN assessments with ISO 3166 validation."""

import re
from typing import List, Set

from .base import BaseChecker
from ..models import Violation, Severity


class GeographyChecker(BaseChecker):
    """Checker for geographic naming conventions (ISO 3166)."""

    # Official ISO 3166 country names (2024)
    # Source: https://www.iso.org/iso-3166-country-codes.html
    ISO_3166_COUNTRIES = {
        'Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda',
        'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain',
        'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan',
        'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria',
        'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada',
        'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros',
        'Congo', 'Costa Rica', 'Croatia', 'Cuba', 'Cyprus', 'Czechia',
        'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt',
        'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia',
        'Fiji', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana',
        'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti',
        'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland',
        'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati',
        'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya',
        'Liechtenstein', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia',
        'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico',
        'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique',
        'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua',
        'Niger', 'Nigeria', 'North Korea', 'North Macedonia', 'Norway', 'Oman', 'Pakistan',
        'Palau', 'Palestine', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines',
        'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'Saint Kitts and Nevis',
        'Saint Lucia', 'Saint Vincent and the Grenadines', 'Samoa', 'San Marino',
        'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles',
        'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia',
        'South Africa', 'South Korea', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan',
        'Suriname', 'Sweden', 'Switzerland', 'Syria', 'Tajikistan', 'Tanzania', 'Thailand',
        'Timor-Leste', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey',
        'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 'United Arab Emirates', 
        'United Kingdom', 'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Vatican City',
        'Venezuela', 'Viet Nam', 'Yemen', 'Zambia', 'Zimbabwe',
        
        # Common acceptable variations for IUCN
        'USA', 'UK', 'UAE', 'DRC', 'Republic of the Congo', 'Lao PDR',
        "Cote d'Ivoire", 'The Bahamas', 'The Gambia',
    }

    # Common incorrect names -> correct ISO 3166 names
    COUNTRY_CORRECTIONS = {
        "vietnam": "Viet Nam",
        "laos": "Lao PDR",
        "ivory coast": "Cote d'Ivoire",
        "burma": "Myanmar",
        "kazakstan": "Kazakhstan",
        "zaire": "DRC",
        "democratic republic of congo": "DRC",
        "democratic republic of the congo": "DRC",
        "congo-kinshasa": "DRC",
        "congo, democratic republic": "DRC",
        "republic of congo": "Republic of the Congo",
        "congo-brazzaville": "Republic of the Congo",
        "great britain": "United Kingdom",
        "holland": "Netherlands",
        "czech republic": "Czechia",
        "swaziland": "Eswatini",
        "cape verde": "Cabo Verde",
        "united states of america": "United States",
        "united states america": "United States",
        "america": "United States",  # When clearly referring to USA
        "north korea": "North Korea",
        "south korea": "South Korea",
        "timor leste": "Timor-Leste",
        "east timor": "Timor-Leste",
    }

    def __init__(self):
        super().__init__(
            rule_id="geography_names",
            rule_name="Geographic naming conventions (ISO 3166)",
            category="Geography",
            severity=Severity.WARNING,
            assessment_section="Geographic Range"
        )
        
        # Create lowercase lookup for case-insensitive matching
        self.iso_countries_lower = {c.lower(): c for c in self.ISO_3166_COUNTRIES}

    def check(self, text: str) -> List[Violation]:
        """Check for geographic naming violations."""
        violations = []

        # Check country name corrections (common mistakes)
        violations.extend(self._check_country_names(text))

        # Check for non-ISO country names
        violations.extend(self._check_iso_compliance(text))

        # Check directional capitalization
        violations.extend(self._check_directional_capitalization(text))

        return violations

    def _check_country_names(self, text: str) -> List[Violation]:
        """Check for incorrect country names from known corrections list."""
        violations = []

        for incorrect, correct in self.COUNTRY_CORRECTIONS.items():
            pattern = re.compile(rf'\b{re.escape(incorrect)}\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                # Don't flag if already correct
                if match.group(0) == correct:
                    continue

                violations.append(self._create_violation(
                    text=text,
                    matched_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    message=f"Use ISO 3166 country name: '{correct}' instead of '{match.group(0)}'",
                    suggested_fix=correct
                ))

        return violations

    def _check_iso_compliance(self, text: str) -> List[Violation]:
        """Check if country names match ISO 3166 standard.
        
        This is a more general check that looks for potential country names
        and validates them against the ISO 3166 list.
        """
        violations = []
        
        # Pattern to find potential country names (capitalized words, 2-4 words)
        # This is conservative to avoid false positives
        country_pattern = re.compile(
            r'\b([A-Z][a-z]+(?:\s+(?:and|of|the)\s+[A-Z][a-z]+|\s+[A-Z][a-z]+){0,3})\b'
        )
        
        # Words that are NOT country names (to reduce false positives)
        exclude_words = {
            'The', 'This', 'That', 'These', 'Those', 'Some', 'Many', 'Several',
            'January', 'February', 'March', 'April', 'May', 'June', 'July',
            'August', 'September', 'October', 'November', 'December',
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
            'Red', 'Blue', 'Green', 'Yellow', 'Black', 'White',
            'Table', 'Figure', 'Appendix', 'Section', 'Chapter',
        }
        
        # Context words that suggest this is a country name
        country_context = [
            r'in\s+', r'from\s+', r'to\s+', r'across\s+', r'throughout\s+',
            r'endemic\s+to\s+', r'found\s+in\s+', r'occurs\s+in\s+',
            r'native\s+to\s+', r'distributed\s+in\s+', r'recorded\s+from\s+',
        ]
        
        for match in country_pattern.finditer(text):
            potential_country = match.group(1)
            
            # Skip if in exclude list
            if potential_country in exclude_words:
                continue
            
            # Check if preceded by country context
            has_context = False
            before_text = text[max(0, match.start()-30):match.start()]
            for context in country_context:
                if re.search(context, before_text, re.IGNORECASE):
                    has_context = True
                    break
            
            # Only check names that have country context
            if has_context:
                # Check if it's a valid ISO name
                if potential_country not in self.ISO_3166_COUNTRIES:
                    # Check if there's a close match
                    suggestion = self._find_closest_country(potential_country)
                    if suggestion:
                        violations.append(self._create_violation(
                            text=text,
                            matched_text=potential_country,
                            start=match.start(),
                            end=match.end(),
                            message=f"'{potential_country}' may not be ISO 3166 compliant. Did you mean '{suggestion}'?",
                            suggested_fix=suggestion
                        ))
        
        return violations

    def _find_closest_country(self, name: str) -> str:
        """Find the closest matching ISO 3166 country name."""
        name_lower = name.lower()
        
        # Exact match (case-insensitive)
        if name_lower in self.iso_countries_lower:
            return self.iso_countries_lower[name_lower]
        
        # Check corrections first
        if name_lower in self.COUNTRY_CORRECTIONS:
            return self.COUNTRY_CORRECTIONS[name_lower]
        
        # Partial match
        for iso_name in self.ISO_3166_COUNTRIES:
            if name_lower in iso_name.lower() or iso_name.lower() in name_lower:
                return iso_name
        
        return None

    def _check_directional_capitalization(self, text: str) -> List[Violation]:
        """Check capitalization of directions (lowercase unless part of proper name)."""
        violations = []

        # Directions should be lowercase when describing parts of countries
        # e.g., "east Japan" not "East Japan" (but "East Africa" is a proper region name)
        directions = ['North', 'South', 'East', 'West', 'Northern', 'Southern', 'Eastern', 'Western']

        for direction in directions:
            # Pattern: Direction + location (not at sentence start)
            pattern = re.compile(rf'(?<=[a-z]\s){direction}\s+([A-Z][a-z]+)\b')
            for match in pattern.finditer(text):
                location = match.group(1)
                
                # Proper region names (should stay capitalized)
                proper_regions = [
                    'Africa', 'America', 'Asia', 'Europe', 'Pacific', 'Atlantic',
                    'Indies', 'Ireland', 'Korea', 'Carolina', 'Dakota', 'Virginia',
                    'Zealand', 'Guinea', 'Macedonia', 'Sudan', 'Hemisphere'
                ]
                
                if location in proper_regions:
                    continue

                lower_direction = direction.lower()
                fix = f"{lower_direction} {location}"
                violations.append(self._create_violation(
                    text=text,
                    matched_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    message=f"Use lowercase direction for parts of countries: '{fix}' (unless it's a proper region name)",
                    suggested_fix=fix
                ))

        return violations
