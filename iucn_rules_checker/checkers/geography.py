"""Geographic naming checker for ISO 3166 country-name usage."""

import re
from typing import List

from ..violation import Violation
from .base import BaseChecker


class GeographyChecker(BaseChecker):
    """
    Checker for geographic naming conventions (ISO 3166).

    Purpose:
        This class groups related rules within the rules-based assessment workflow.
    """

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
        'USA', 'UK', 'UAE', 'DRC', 'Republic of the Congo', 'Lao PDR',
        "Cote d'Ivoire", 'The Bahamas', 'The Gambia',
    }

    COUNTRY_CORRECTIONS = {
        "afganistan": "Afghanistan",
        "argentinia": "Argentina",
        "argentinaa": "Argentina",
        "argetina": "Argentina",
        "argintina": "Argentina",
        "bangledesh": "Bangladesh",
        "columbia": "Colombia",
        "ethopia": "Ethiopia",
        "madacascar": "Madagascar",
        "morroco": "Morocco",
        "vietnam": "Viet Nam",
        "laos": "Lao PDR",
        "ivory coast": "Cote d'Ivoire",
        "burma": "Myanmar",
        "kazakstan": "Kazakhstan",
        "phillipines": "Philippines",
        "phillippines": "Philippines",
        "portugual": "Portugal",
        "tunsia": "Tunisia",
        "turky": "Turkey",
        "uraguay": "Uruguay",
        "venezeula": "Venezuela",
        "zimbabwae": "Zimbabwe",
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
        "america": "United States",
        "north korea": "North Korea",
        "south korea": "South Korea",
        "timor leste": "Timor-Leste",
        "east timor": "Timor-Leste",
    }

    RECOGNISED_REGIONS = {
        # Countries and sovereign states.
        'East Timor',
        'Western Sahara',
        'Equatorial Guinea',
        'Central African Republic',
        'Northern Ireland',

        # Continents and major world regions.
        'Africa',
        'North Africa',
        'East Africa',
        'West Africa',
        'Southern Africa',
        'Central Africa',
        'Sub-Saharan Africa',
        'Antarctica',
        'Antarctic',
        'Asia',
        'East Asia',
        'Southeast Asia',
        'South Asia',
        'West Asia',
        'North Asia',
        'South & Southeast Asia',
        'West & Central Asia',
        'Europe',
        'Eastern Europe',
        'Western Europe',
        'Northern Europe',
        'Southern Europe',
        'Middle East',
        'Americas',
        'North America',
        'Mesoamerica',
        'Central America',
        'Caribbean Islands',
        'South America',
        'Oceania',

        # Oceans, seas, and major water bodies.
        'North Sea',
        'South China Sea',
        'East China Sea',
        'North Atlantic',
        'South Atlantic',
        'North Pacific',
        'South Pacific',
        'Southern Ocean',

        # US states and territories.
        'North Carolina',
        'South Carolina',
        'North Dakota',
        'South Dakota',
        'West Virginia',

        # Australian states and territories.
        'New South Wales',
        'Northern Territory',
        'South Australia',
        'Western Australia',

        # Canadian provinces and territories.
        'Northwest Territories',

        # UK and Ireland regions.
        'East Anglia',
        'North Yorkshire',
        'South Yorkshire',
        'West Yorkshire',
        'East Yorkshire',
        'West Midlands',
        'East Midlands',
        'Western Isles',

        # Other notable named regions and territories.
        'North Caucasus',
        'South Georgia',
        'South Sandwich Islands',
        'Northern Mariana Islands',
        'Southwest Pacific',
        'Eastern Cape',
        'Eastern Cape (South Africa)',
        'Western Cape',
        'Western Cape (South Africa)',
        'Northern Cape',
        'Northern Cape (South Africa)',
        'North Island',
        'North Island (New Zealand)',
        'South Island',
        'South Island (New Zealand)',
        'South Downs',
        'North Downs',
        'West Country',
        'North Andaman',
        'South Andaman',
    }

    def __init__(self):
        """
        Initialise the geography checker.

        Args:
            None.

        Returns:
            None (mutates the recognised-name cache in place).
        """
        super().__init__()
        self._proper_country_or_region_names = self.ISO_3166_COUNTRIES | self.RECOGNISED_REGIONS

    def check_text(self, section_name: str, text: str) -> List[Violation]:
        """
        Check for geographic naming violations.

        Args:
            section_name (str): Parsed section key supplied by the caller.
            text (str): Parsed section text supplied by the caller.

        Returns:
            List[Violation]: Violations produced by this method.
        """
        violations = []
        violations.extend(self.check_country_names(section_name, text))
        violations.extend(self.check_directional_capitalization(section_name, text))
        return violations

    def check_country_names(self, section_name: str, text: str) -> List[Violation]:
        """
        Check country names against an explicit correction map.

        This method first strips simple inline style markers such as italics,
        bold, superscript, and subscript tags, then applies the explicit
        `COUNTRY_CORRECTIONS` map to the cleaned text and maps matches back to
        the original rich-text span.

        This method is intentionally simple: it only applies the hard-coded
        `COUNTRY_CORRECTIONS` list case-insensitively.

        The list includes:
        - non-preferred country forms such as `Vietnam`, `Laos`, `Burma`,
          `Holland`, `Czech Republic`
        - common misspellings such as `Afganistan`, `Argetina`,
          `Phillipines`, `Portugual`, `Venezeula`

        Examples flagged:
        `found in Vietnam`
        `occurs in Laos`
        `recorded from Burma`
        `distributed in Holland`
        `found in Afganistan`
        `recorded from Argetina`
        `distributed in Phillipines`

        Examples not flagged:
        exact ISO-style names such as `Viet Nam`, `Myanmar`, `Philippines`
        unknown typos that are not yet present in `COUNTRY_CORRECTIONS`
        arbitrary capitalized phrases that are not in the correction map

        Args:
            section_name (str): Parsed section key supplied by the caller.
            text (str): Parsed section text supplied by the caller.

        Returns:
            List[Violation]: Violations produced by this method.
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )

        for incorrect, correct in self.COUNTRY_CORRECTIONS.items():
            pattern = re.compile(rf'\b{re.escape(incorrect)}\b', re.IGNORECASE)
            for match in pattern.finditer(cleaned_text):
                if match.group(0).lower() == correct.lower():
                    continue
                if self.is_within_known_country_or_region(cleaned_text, match.span()):
                    continue
                original_start = index_map[match.start()]
                original_end = index_map[match.end() - 1] + 1

                violations.append(self.create_violation(
                    section_name=section_name,
                    text=text,
                    span=(original_start, original_end),
                    message=f"Use ISO 3166 country name: '{correct}' instead of '{match.group(0)}'",
                    suggested_fix=correct,
                ))

        return violations

    def is_within_known_country_or_region(self, text: str, span: tuple[int, int]) -> bool:
        """
        Return True when a match sits inside a known ISO country or region name.

        Args:
            text (str): Parsed section text supplied by the caller.
            span (tuple[int, int]): Span tuple supplied by the caller.

        Returns:
            bool: Boolean result described by the summary line above.
        """
        start, end = span
        for proper_name in self._proper_country_or_region_names:
            pattern = re.compile(rf'\b{re.escape(proper_name)}\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                if match.start() <= start and end <= match.end():
                    if match.start() == start and match.end() == end:
                        continue
                    return True
        return False

    def check_directional_capitalization(self, section_name: str, text: str) -> List[Violation]:
        """
        Check capitalization of direction-led geographic phrases.

        This method first strips simple inline style markers such as italics,
        bold, superscript, and subscript tags, then applies the directional
        phrase check to the cleaned text and maps matches back to the original
        rich-text span.

        This method looks for phrases beginning with a capitalized cardinal
        direction or directional adjective, such as:
        `North`, `South`, `East`, `West`,
        `Northern`, `Southern`, `Eastern`, `Western`.

        It then checks whether the full matched phrase is one of the allowed
        proper names:
        - an ISO country name such as `North Korea` or `South Africa`
        - a recognized region such as `North Africa`, `East Asia`,
          `South & Southeast Asia`, `West & Central Asia`, `North America`
        - an explicit country-correction key such as `east timor`

        If the phrase is not one of those allowed names, the method assumes
        the direction should be lowercase and suggests that form.

        There are two explicit sentence-position exceptions:
        - if the matched phrase appears at the start of the paragraph
        - if the cleaned text immediately before it ends with ``. ``

        Examples flagged:
        `Eastern Ecuador` -> suggests `eastern Ecuador`
        `Northern Peru` -> suggests `northern Peru`
        `Western Colombia` -> suggests `western Colombia`

        Examples not flagged:
        paragraph-start `Eastern Ecuador contains...`
        `This changed. Eastern Ecuador contains...`
        `North Korea`
        `South Africa`
        `North America`
        `East Asia`
        `South & Southeast Asia`
        `West & Central Asia`
        direction-led subphrases inside a larger proper name such as
        `South Wales` within `New South Wales`

        Examples not checked:
        already-lowercase forms such as `eastern Ecuador`
        direction words that are not followed by a capitalized geographic phrase
        proper region names not present in `ISO_3166_COUNTRIES`,
        `RECOGNISED_REGIONS`, or the correction-key list

        Args:
            section_name (str): Parsed section key supplied by the caller.
            text (str): Parsed section text supplied by the caller.

        Returns:
            List[Violation]: Violations produced by this method.
        """
        violations = []
        cleaned_text, index_map = self.strip_style_markers(
            text,
            italics=True,
            bold=True,
            superscript=True,
            subscript=True,
        )
        directions = ['North', 'South', 'East', 'West', 'Northern', 'Southern', 'Eastern', 'Western']
        directions_pattern = '|'.join(re.escape(direction) for direction in directions)
        pattern = re.compile(
            rf"\b(?P<direction>{directions_pattern})(?P<rest>(?:\s*&\s*[A-Z][A-Za-z-]+)?(?:\s+[A-Z][A-Za-z-]+){{1,3}})\b"
        )

        allowed_phrases = {
            name.lower() for name in (self.ISO_3166_COUNTRIES | self.RECOGNISED_REGIONS)
        }
        allowed_phrases.update(self.COUNTRY_CORRECTIONS.keys())

        for match in pattern.finditer(cleaned_text):
            matched_phrase = match.group(0)
            if match.start() == 0 or cleaned_text[:match.start()].endswith(". "):
                continue
            if matched_phrase.lower() in allowed_phrases:
                continue
            if self.is_within_known_country_or_region(cleaned_text, match.span()):
                continue

            corrected = f"{match.group('direction').lower()}{match.group('rest')}"
            original_start = index_map[match.start()]
            original_end = index_map[match.end() - 1] + 1
            violations.append(self.create_violation(
                section_name=section_name,
                text=text,
                span=(original_start, original_end),
                message=f"'{matched_phrase}' should be lower case (unless it is a proper region name)",
                suggested_fix=corrected,
            ))

        return violations
