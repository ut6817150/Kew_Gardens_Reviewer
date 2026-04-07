# Checkers Reference

This folder contains the rule-specific checker classes used by the IUCN reviewer.
Each checker receives one `(section_name, section_text)` pair from the parsed
full report and returns a list of `Violation` objects.

This README is intentionally practical. For each live checker module, it explains:
- how each method works
- what each method catches
- what each method misses

## Shared Base Class

### `base.py` - `BaseChecker`

Method reference:
- `begin_sweep(...)`
  Hook called before a full parsed report is reviewed.
  The base implementation does nothing.
- `end_sweep(...)`
  Hook called after a full parsed report is reviewed.
  The base implementation does nothing.
- `check(...)`
  Unpacks one `(section_name, text)` tuple and delegates to `check_text(...)`.
- `strip_style_markers(...)`
  Removes selected inline HTML-style markers and returns both cleaned text and
  an `index_map` back into the original text.
  Supported marker groups:
  italics: `<i>`, `</i>`, `<em>`, `</em>`
  bold: `<b>`, `</b>`, `<strong>`, `</strong>`
  superscript: `<sup>`, `</sup>`
  subscript: `<sub>`, `</sub>`
  Defaults:
  italics and bold are stripped by default; superscript and subscript are opt-in.
- `create_violation(...)`
  Builds a `Violation` with:
  `rule_class`, `rule_method`, `matched_text`, `matched_snippet`, `message`,
  `suggested_fix`, and a normalized `section_name`.
- `normalize_section_name(...)`
  Removes a trailing `[paragraph N]` suffix from a section name but leaves table
  row suffixes intact.
- `get_rule_method_name(...)`
  Uses the call stack to record which checker method directly created the violation.

What it does not do:
- it does not contain any domain-specific IUCN rules
- it does not decide which sections a checker should or should not run on
- it does not deduplicate overlapping violations from different checkers
- `strip_style_markers(...)` only handles those exact simple tags, not arbitrary
  HTML, Markdown, or richer markup

## Individual Checkers

General note on `check_text(...)`:
- unless a section says otherwise, each checker's `check_text(...)` method is
  just the dispatcher that calls the `check_...(...)` methods documented in
  that section and concatenates their `Violation` lists

### `abbreviations.py` - `AbbreviationChecker`

Category: `Abbreviations`

This checker is made of five narrow, regex-based methods. Together they cover a
small set of repeatable abbreviation and Latin-term formatting rules.

Aggregated method list:
- `check_eg_and_ie(...)`
  Flags `e.g.` and `i.e.`-style abbreviations in body text and suggests plain-English alternatives.
- `check_abbreviation_formats(...)`
  Flags a fixed set of abbreviations that are missing a final period or using a non-preferred short form.
- `check_latin_terms_without_period(...)`
  Checks a fixed list of Latin expressions that must be italicized and should not contain periods.
- `check_et_al(...)`
  Checks that `et al.` uses the canonical wording and italicizes the letters `et al`.
- `check_title_abbreviations(...)`
  Flags selected courtesy titles that use a trailing period when the checker expects the UK-style form without one.

Detailed coverage:

- `check_eg_and_ie(...)`
  How it works:
  strips all simple style markers, then matches standalone `e.g.` / `i.e.` variants with a case-insensitive regex.
  Hard-coded scope:
  only `e.g.` / `eg`-style variants and `i.e.` / `ie`-style variants are covered by this method.

  - Catches:
    - `E.g.`, `Eg.`, `e.g.`, `E.G.`
    - `ie`, `IE`, `i.e.`, `I.e.`
    - `(e.g. in gallery forest)`
    - `<i>E</i><i>.g.</i>`
    - `<b>I</b><b>.e.</b>`
    - `<sup>eg</sup>`

  - Misses:
    - `segment`, `diet`, `species`, `siege`
    - special-context decisions such as quotations, copied headings, or bibliography-only exceptions

- `check_abbreviation_formats(...)`
  How it works:
  strips all simple style markers, then applies separate regexes for a fixed shortlist of abbreviations.
  Hard-coded list dependency:
  this method only checks the abbreviations hardcoded in the source, currently including `etc`, `in lit`, `pers comm`, `pers obs`, and `Prof`.

  - Catches:
    - `etc`, `ETC`
    - `in lit`, `IN LIT`
    - `pers comm`, `pers comm.`, `pers. comm`
    - `pers obs`, `pers obs.`, `pers. obs`
    - `Prof`, `PROF`
    - `<i>etc</i>`, `<b>in lit</b>`, `<i>Pers</i> <b>Comm</b>`, `<sub>Prof</sub>`

  - Misses:
    - already-correct `etc.`, `in lit.`, `pers. comm.`, `pers. obs.`
    - `Professor Smith`
    - `etcetera`
    - `p e r s comm`
    - `Assoc Prof`
    - abbreviations outside the hard-coded shortlist such as `cf.` or `ca.`

- `check_latin_terms_without_period(...)`
  How it works:
  strips non-italic style markers but keeps `<i>` / `<em>`, checks whether the term is inside italics, then checks whether the italicized fragment contains periods.
  Hard-coded list dependency:
  this method only checks the Latin expressions hardcoded in the source, such as `in situ`, `ex situ`, `de facto`, `in vitro`, `in vivo`, `sensu lato`, `sensu stricto`, and `vice versa`.

  - Catches:
    - plain `in situ`, `de facto`, `vice versa`
    - plain or dotted `sensu. lato`
    - `<i>in. situ</i>`
    - `<i>in.situ</i>`
    - `<i>in situ</i>.`
    - `<b><i>in. vitro</i></b>`

  - Misses:
    - correct `<i>in situ</i>` and `<em>de facto</em>`
    - unlisted terms such as `status quo`
    - Markdown forms such as `*in situ*`
    - malformed variants such as `in .. situ` or `in . situ`
    - hyphenated forms such as `in-situ`
    - Latin expressions not present in the hard-coded term list

- `check_et_al(...)`
  How it works:
  strips all simple style markers for matching, then checks whether the letters `et al` are italicized in the original text. It requires the wording `et al.` but does not require the trailing period itself to sit inside italics.
  Hard-coded scope:
  this method is only about the literal citation term `et al.`.

  - Catches:
    - `et al`
    - `ET AL`
    - plain-text `et al.`
    - `et. al.`
    - `<i>et al</i>`
    - `<i>et.al.</i>`

  - Misses:
    - correct `<i>et al.</i>`
    - correct `<em>et al.</em>`
    - correct `<i>et al</i>.`
    - `et-al`
    - `etal`
    - `ibid.`

- `check_title_abbreviations(...)`
  How it works:
  strips all simple style markers, then flags a small fixed list of courtesy titles that still carry a trailing period.
  Hard-coded list dependency:
  this method only checks the courtesy titles hardcoded in the source, currently `Dr.`, `Mr.`, `Mrs.`, and `Ms.` variants.

  - Catches:
    - `Dr. Smith`
    - `Mr. Brown,`
    - `Mrs. Jones)`
    - `ms. Patel`
    - `MRS. WHITE`
    - `<sup>Mrs.</sup>` and `<sub>Ms.</sub>`

  - Misses:
    - already-correct `Dr Smith`, `Mr Brown`, `Mrs White`, `Ms Black`
    - `Prof.` because that is handled by `check_abbreviation_formats(...)`
    - titles outside the list such as `Rev.`, `Sir`, `St.`
    - any other title not in the hard-coded set

Aggregated misses for `AbbreviationChecker` as a whole:
- it only covers the abbreviations and Latin expressions hardcoded into the source file
- it does not parse references, quotations, or copied text as special contexts
- it only recognizes HTML italics (`<i>` / `<em>`), not Markdown or richer document markup
- it does not perform deeper citation parsing or semantic reasoning

### `dates.py` - `DateChecker`

Category: `Dates`

This checker is made of three narrower methods. It focuses on:
- ordinal day-month date style
- written-out century style
- a narrow decade-formatting rule

It is not a full date parser or calendar engine.

Aggregated method list:
- `check_ordinal_dates(...)`
  Checks ordinal day-month dates such as `11th January`, including a basic validity check for the day-month combination.
- `check_century_format(...)`
  Checks written-out centuries such as `nineteenth century` and rewrites them to numeric form such as `19th century`.
- `check_decade_format(...)`
  Checks decade forms such as `1980's` and `1980 s` and rewrites them to `1980s`.

Detailed coverage:

- `check_ordinal_dates(...)`
  How it works:
  strips all simple style markers, then matches day-month forms with an ordinal suffix and a month name or covered month abbreviation; after matching, it also checks whether the day-month pair is calendar-plausible.
  Hard-coded list dependency:
  month names and covered month abbreviations come from a fixed hard-coded month list rather than a general date parser.

  - Catches:
    - style cases such as `11th January`, `3rd of May`, `21st Jan`, `4th Sept.`
    - case-insensitive variants such as `11TH january`
    - invalid dates such as `31st April`, `30th Feb.`, `0th January`, `32nd December`
    - `29th February` with a leap-year warning
    - styled inputs such as `<b>11th</b> <i>January</i>`

  - Misses:
    - `11 January`
    - `January 11th` and `Sept. 4th`
    - `11/01/2024` and `2024-05-15`
    - misspellings such as `11th Jnauary`
    - year-aware leap-year validation for `29th February 2024` / `29th February 2023`
    - weekday validation such as `Monday, 11th January`
    - month spellings or abbreviations outside the hard-coded set

- `check_century_format(...)`
  How it works:
  strips all simple style markers, then matches written century phrases from the hardcoded map and rewrites them to numeric forms, preferring longer phrases first to avoid nested overlaps.
  Hard-coded list dependency:
  this method only knows the written century phrases present in the `CENTURY_WORDS` map.

  - Catches:
    - `first century`
    - `ninth century`
    - `nineteenth century`
    - `twentieth century`
    - `twenty-first century`
    - `twenty first century`
    - styled forms such as `<i>nineteenth</i> <b>century</b>`

  - Misses:
    - `19th century`
    - `XIX century`
    - `twenty-second century`
    - `nineteenth-century art`
    - `nineteenth centuries`
    - `21st-century taxonomy`
    - any century wording outside the hard-coded `CENTURY_WORDS` map

- `check_decade_format(...)`
  How it works:
  strips all simple style markers, then matches four-digit decades followed by an apostrophe or whitespace before `s`, and rewrites them to the compact no-apostrophe form.
  Hard-coded scope:
  this is a regex rule for four-digit decade forms only; it is not a general period-style checker for all decade wording.

  - Catches:
    - `1980's`
    - `1980 s`
    - `1990   s`
    - `2000' s`
    - styled forms such as `<i>1980</i><b>'s</b>`

  - Misses:
    - `1980s`
    - `'80s`
    - `80's`
    - `the eighties`
    - `1980s-1990s`
    - `1980’s` with a curly apostrophe
    - `2000s and 2010s`

Aggregated misses for `DateChecker` as a whole:
- it is not a general-purpose date parser or calendar validator
- it does not validate numeric dates, ISO dates, date ranges, or time-of-day formats
- it does not check misspelled month names
- it does not use the year when evaluating leap years
- century handling is limited to the hardcoded written forms present in `CENTURY_WORDS`

### `formatting.py` - `FormattingChecker`

Category: `Formatting`

This checker is sweep-aware rather than purely section-local. During a full
review it can harvest names from a taxonomy ladder in one section and use those
harvested names while checking later sections in the same assessment.

Aggregated method list:
- `begin_sweep(...)`
  Clears any harvested taxonomy names from a previous assessment review.
- `check_genus_and_species(...)`
  Uses a harvested genus and species to check later occurrences for italics and case.
- `check_higher_order_taxonomy_formatting(...)`
  Uses harvested higher-taxonomy names from ladder entries to enforce capitalization and no italics.
- `check_eoo_aoo_capitalization(...)`
  Checks capitalization of the spelled-out phrases `extent of occurrence` and `area of occupancy`.
- `end_sweep(...)`
  Clears the temporary harvested taxonomy names after the review finishes.

Detailed coverage:

- `begin_sweep(...)`
  How it works:
  clears the in-memory stores for harvested higher taxonomy names, genus, and species before a new full-report sweep begins.

  - Catches:
    - no violations; this is state reset only

  - Misses:
    - all text checking, because it does not inspect text directly

- `check_genus_and_species(...)`
  How it works:
  strips bold, superscript, and subscript tags but preserves italics, harvests genus/species names from ladder entries such as `PLANTAE - ... - Acrocarpus - fraxinifolius`, and then checks later occurrences of those harvested names for expected italics and case.
  Hard-coded/structural dependency:
  this method depends on seeing a ladder-shaped taxonomy entry first; without a harvested genus/species pair, it has nothing to check.

  - Catches:
    - `Acrocarpus` after harvest when it is plain text instead of italicized
    - `acrocarpus`
    - `fraxinifolius` when it is plain text
    - `Fraxinifolius`
    - `<i>Fraxinifolius</i>`

  - Misses:
    - the taxonomy ladder entry itself
    - names before any ladder has been harvested in the current sweep
    - unrelated genus/species names that do not match the harvested pair
    - Markdown italics such as `*Acrocarpus*`
    - assessments where the taxonomy ladder is missing or written in a different structure

- `check_higher_order_taxonomy_formatting(...)`
  How it works:
  strips non-italic style tags, harvests higher taxonomy names from ladder entries, then checks later occurrences of those harvested names for two conditions: correct capitalization and no italics.
  Hard-coded/structural dependency:
  this method depends on seeing a ladder-shaped taxonomy entry first; it no longer infers higher taxa from suffixes alone.

  - Catches:
    - harvested names reused later such as `plantae` or `<i>Magnoliopsida</i>`
    - harvested names reused later such as `<i>Fabaceae</i>`

  - Misses:
    - literal rank labels such as `family`, `order`, `class`
    - non-harvested family-like names such as `orchidaceae` or `Felidae`
    - Markdown italics
    - taxonomic truth; it checks style, not biological correctness
    - higher taxa that were never harvested from a ladder entry

- `check_eoo_aoo_capitalization(...)`
  How it works:
  strips all simple style markers, then searches case-insensitively for `extent of occurrence` and `area of occupancy`. The fully lowercase form is preferred, except for sentence-start forms where only the first word is capitalized after paragraph start, `. `, `? `, or `: `.
  Hard-coded list dependency:
  this method only checks the two literal phrases `extent of occurrence` and `area of occupancy`.

  - Catches:
    - `The Extent of Occurrence was revised`
    - `The Extent of occurrence was revised`
    - `the area of Occupancy was recalculated`
    - `Summary, Area of occupancy was recalculated`
    - `(Extent of occurrence) was revised`

  - Misses:
    - lowercase `extent of occurrence` and `area of occupancy`
    - `Extent of occurrence ...` at paragraph start
    - `This was revised. Extent of occurrence ...`
    - `Summary: Area of occupancy ...`
    - abbreviations `EOO` / `AOO`
    - misspellings such as `extent of occurence`

- `end_sweep(...)`
  How it works:
  clears the same harvested taxonomy stores after the full-report review finishes.

  - Catches:
    - no violations; this is state reset only

  - Misses:
    - all text checking, because it does not inspect text directly

Aggregated misses for `FormattingChecker` as a whole:
- it depends on simple HTML-style italics tags such as `<i>` and `<em>`
- it does not parse Markdown italics or richer markup
- genus/species and harvested higher-taxonomy checks only work after a taxonomy ladder has already appeared in the same sweep
- it only checks taxonomy names harvested from ladder-shaped entries; it does not infer higher taxa from suffixes alone

### `geography.py` - `GeographyChecker`

Category: `Geography`

This checker has two geography-specific rules:
- `check_country_names(...)`
  Enforces ISO-style country naming using an explicit correction list only.
- `check_directional_capitalization(...)`
  Lowercases directional adjectives such as `Eastern` or `Northern` when they describe part of a country rather than a recognized proper region name.

Detailed coverage:

- `check_country_names(...)`
  How it works:
  strips all simple style markers, then applies the explicit `COUNTRY_CORRECTIONS` map case-insensitively and maps matches back to the original rich-text span. It also skips a correction match if it sits inside a larger known ISO country or recognized region name.
  Hard-coded list dependency:
  this method relies on three hard-coded sets: `COUNTRY_CORRECTIONS`, `ISO_3166_COUNTRIES`, and `RECOGNISED_REGIONS`.

  - Catches:
    - `Vietnam` -> `Viet Nam`
    - `Laos` -> `Lao PDR`
    - `Burma` -> `Myanmar`
    - `Holland` -> `Netherlands`
    - `Afganistan`
    - `Argetina`
    - `Phillipines`

  - Misses:
    - exact preferred forms such as `Viet Nam`, `Myanmar`, `Philippines`
    - typos that are not present in `COUNTRY_CORRECTIONS`
    - arbitrary place-name errors outside the explicit map
    - coordinate or polygon validation
    - any non-preferred country form not present in the hard-coded correction map

- `check_directional_capitalization(...)`
  How it works:
  strips all simple style markers, then looks for capitalized direction-led phrases such as `Eastern Ecuador` and compares the full phrase against an allowlist built from ISO country names, recognized regions, and correction-key phrases. If the phrase is not allowed, it suggests lowercasing the direction word.
  Hard-coded list dependency:
  allowed phrases come from the hard-coded `ISO_3166_COUNTRIES`, `RECOGNISED_REGIONS`, and correction-key lists.

  - Catches:
    - `Eastern Ecuador` -> `eastern Ecuador`
    - `Northern Peru` -> `northern Peru`
    - `Western Colombia` -> `western Colombia`

  - Misses:
    - proper names such as `North Korea`, `South Africa`, `North America`, `East Asia`, `West & Central Asia`
    - already-lowercase forms such as `eastern Ecuador`
    - proper region names that are not in the checker's allowlists
    - deeper geopolitical reasoning
    - special-case region names not present in the hard-coded allowlists

Aggregated misses for `GeographyChecker` as a whole:
- it does not validate the biological truth of a distribution statement
- country-name typo detection depends entirely on the explicit hard-coded correction list
- directional-capitalization handling depends on a fixed allowlist of proper-region names

### `iucn_terms.py` - `IUCNTermsChecker`

Category: `IUCN Terms`

Aggregated method list:
- `check_the_iucn(...)`
  Flags `the IUCN` and suggests `IUCN`.
- `check_CE_abbreviation(...)`
  Flags the incorrect Red List abbreviation `CE` and suggests `CR`.
- `check_category_full_name_capitalization(...)`
  Enforces canonical case for full category names such as `Critically Endangered`.
- `check_category_abbreviation_capitalization(...)`
  Enforces canonical case for category abbreviations such as `CR`.
- `check_threatened_case(...)`
  Lowercases mid-sentence `Threatened` when it is being used collectively for CR/EN/VU species.

Detailed coverage:

- `check_the_iucn(...)`
  How it works:
  strips all simple style markers, then matches `the IUCN` case-insensitively and suggests `IUCN`.
  Hard-coded scope:
  this method only checks the literal phrase `the IUCN`.

  - Catches:
    - `the IUCN`
    - `The IUCN`
    - `THE IUCN`

  - Misses:
    - `IUCN` without the article
    - surrounding grammar adjustments after the article is removed

- `check_CE_abbreviation(...)`
  How it works:
  strips all simple style markers, then matches standalone `CE` case-insensitively with whole-word logic and suggests `CR`.
  Hard-coded scope:
  this method only checks the literal abbreviation `CE`.

  - Catches:
    - `CE`
    - `ce`
    - `(Ce)`

  - Misses:
    - correct `CR`
    - larger words that merely contain `ce`
    - wrong abbreviations other than `CE`

- `check_category_full_name_capitalization(...)`
  How it works:
  strips all simple style markers, then loops through the hardcoded full category names in `CATEGORIES` and checks each full phrase for exact canonical case.
  Hard-coded list dependency:
  this method only knows the full category names present in the `CATEGORIES` list.

  - Catches:
    - `critically endangered`
    - `Near threatened`
    - `extinct in the wild`

  - Misses:
    - correctly cased `Critically Endangered`, `Near Threatened`, `Extinct in the Wild`
    - misspellings that no longer match the category phrase
    - substantive correctness of the chosen Red List category
    - category names not present in the hard-coded `CATEGORIES` list

- `check_category_abbreviation_capitalization(...)`
  How it works:
  strips all simple style markers, then loops through the hardcoded category abbreviations in `CATEGORIES` and checks each abbreviation for exact canonical case. Matching rejects letters, digits, and hyphens on either side, and it also skips the Latin phrase `ex situ`.
  Hard-coded list dependency:
  this method only knows the category abbreviations present in the `CATEGORIES` list.

  - Catches:
    - `cr`
    - `Vu`
    - `ew`

  - Misses:
    - correctly cased `CR`, `VU`, `EW`
    - the Latin phrase `ex situ`
    - hyphenated compounds such as `Ex-situ`
    - abbreviations embedded inside larger words
    - category abbreviations not present in the hard-coded `CATEGORIES` list

- `check_threatened_case(...)`
  How it works:
  strips all simple style markers, then looks for capitalized `Threatened` when it appears after a lowercase letter and a space, suggesting lowercase `threatened`. It skips the category phrase `Near Threatened`.
  Hard-coded scope:
  this method only checks the literal word `Threatened` in that narrow regex context, with a specific exception for `Near Threatened`.

  - Catches:
    - `many Threatened species`
    - `the Threatened categories include CR, EN and VU`

  - Misses:
    - sentence-start `Threatened`
    - already-lowercase `threatened`
    - the category phrase `Near Threatened`
    - context-aware exceptions beyond the regex

Aggregated misses for `IUCNTermsChecker`:
- it does not validate whether the chosen Red List category is substantively correct
- it does not verify criteria strings such as `B2ab(iii)`
- it focuses on wording and capitalization, not assessment logic

### `numbers.py` - `NumberChecker`

Category: `Numbers`

Aggregated method list:
- `check_small_numbers(...)`
  Flags standalone prose numerals `1` to `9` and suggests writing them out.
- `check_large_numbers(...)`
  Enforces standard comma placement in larger numbers.
- `check_sentence_start(...)`
  Flags sentences that begin with a numeral.
- `check_very_large_numbers(...)`
  Rewrites rounded large counts into `million` / `billion` wording.

Detailed coverage:

- `check_small_numbers(...)`
  How it works:
  strips all simple style markers, then matches standalone numerals `1` to `9`. Before creating a violation it applies deterministic exclusions for dates, units, percentages, degrees, decimals, ranges, and any digit adjacent to another digit.
  Hard-coded list dependency:
  this method relies on the hard-coded `NUMBER_WORDS`, `MONTHS`, and `SMALL_NUMBER_UNITS` lists.

  - Catches:
    - `There were 3 sites`
    - `The species survives in 2 valleys`

  - Misses:
    - `3 May`
    - `5 km`, `7 ha`, `3 sq km`, `4 m asl`, `3 m3`, `4 km2`, `2 t`
    - `6%`, `8°`
    - `1.5`
    - `4-5`, `2-3`, `2–3`, `2—3`
    - small-number contexts outside the hard-coded month and unit lists

- `check_large_numbers(...)`
  How it works:
  strips all simple style markers, then scans number-like tokens for missing commas, incorrect comma grouping, or decimal forms whose integer part should be grouped. It intentionally skips likely years from `1800` to `2100`, plain rounded million/billion candidates that are handed off to `check_very_large_numbers(...)`, numbers that fall inside DOI or URL spans, and numbers immediately preceded by `#`.
  Hard-coded threshold dependency:
  the likely-year exclusion is hard-coded to the range `1800` through `2100`, DOI/URL exclusion depends on fixed DOI/URL-style regex patterns, and hash-prefixed exclusion is a literal `#` check.

  - Catches:
    - `1000` -> `1,000`
    - `12,34` -> `1,234`
    - `1234.56` -> `1,234.56`
    - `12,34.56` -> `1,234.56`

  - Misses:
    - likely years such as `2024`
    - already-correct `1,000` and `1,234.56`
    - plain rounded million/billion values such as `1500000`
    - non-year numbers that still fall inside the hard-coded year range
    - DOI and URL numbers such as `https://doi.org/10.1038/s41598-020-64668-z`
    - hash-prefixed identifiers such as `#2916`

- `check_sentence_start(...)`
  How it works:
  strips all simple style markers, then looks for digit strings at the start of the text or immediately after `. `, `! `, or `? `. It also skips numeral starts immediately preceded by `et al. `, by `c.` plus spaces, or by `comm.` plus spaces.
  Hard-coded scope:
  this method only treats `. `, `! `, and `? ` as sentence-start triggers, and only uses the literal exclusions `et al. `, `c.` plus spaces, and `comm.` plus spaces.

  - Catches:
    - `3 sites were surveyed.`
    - `The assessment was updated. 12 records were added.`
    - `Was it revised? 4 locations remained.`
    - `<b>3</b> sites were surveyed.`

  - Misses:
    - mid-sentence numbers
    - written-out starts such as `Three sites were surveyed.`
    - bibliography-style `et al. 2006`
    - approximate forms such as `c. 800` and `c.  800`
    - communication-style references such as `pers. comm. 2020`
    - starts after punctuation not covered by the regex

- `check_very_large_numbers(...)`
  How it works:
  strips all simple style markers, then matches plain digit strings with seven or more digits but only flags values rounded enough to write cleanly as millions or billions. Million-scale values must be divisible by `10,000`; billion-scale values must be divisible by `10,000,000`.
  Hard-coded threshold dependency:
  the million/billion thresholds and divisibility rules are fixed in the code rather than configurable.

  - Catches:
    - `1000000` -> `1 million`
    - `1500000` -> `1.5 million`
    - `1570000` -> `1.57 million`
    - `25000000` -> `25 million`
    - `1500000000` -> `1.5 billion`
    - `1570000000` -> `1.57 billion`
    - `3000000000` -> `3 billion`

  - Misses:
    - more precise values such as `1234567` and `9876543210`
    - comma-separated values, because the regex only targets plain digit strings
    - large-number styles outside the hard-coded million/billion rounding rules

Aggregated misses for `NumberChecker`:
- it does not validate arithmetic or numeric truth
- it does not deeply parse every identifier, date, or range context
- the exclusions are deterministic and intentionally narrow
- the million/billion rule is stylistic and may not be desired in every context

### `punctuation.py` - `PunctuationChecker`

Category: `Punctuation`

Aggregated method list:
- `check_range_dashes(...)`
  Ensures numeric ranges use an unspaced en dash.
- `check_for_example_commas(...)`
  Ensures `for example` is enclosed by commas when used mid-sentence.
- `check_colon_spacing(...)`
  Removes spaces before a colon.
- `check_semicolon_spacing(...)`
  Removes spaces before a semicolon.

Detailed coverage:

- `check_range_dashes(...)`
  How it works:
  strips italic and bold markers but preserves superscript/subscript markup, then looks for numeric pairs such as `10-20`, `10 - 20`, or `10 – 20`. It rewrites the separator to an unspaced en dash, still allows shared trailing units such as `10-20 km`, and skips date-like three-part numeric chains.
  Hard-coded list dependency:
  the rule uses the hard-coded `RANGE_UNITS` list to recognize repeated-unit range structures that should be skipped.

  - Catches:
    - `10-20`
    - `10 - 20`
    - `10 – 20`
    - `10-20 km`
    - `600-1200 m`
    - `14-26 °C`

  - Misses:
    - already-correct unspaced en-dash ranges
    - repeated-unit expressions such as `10 km - 20 km`, `5% - 7%`, `10 km<sup>2</sup> - 20 km<sup>2</sup>`
    - date-like chains such as `2022-08-01`, `08-01-2022`, `08-2022-01`
    - `10 to 20`
    - endpoints longer than four digits
    - repeated-unit ranges whose unit is not present in the hard-coded `RANGE_UNITS` skip list

- `check_for_example_commas(...)`
  How it works:
  strips all simple style markers, then finds `for example` case-insensitively and checks whether a comma appears immediately before and after the phrase. Sentence-start uses are skipped at paragraph start or immediately after `.`, `!`, or `?`.
  Hard-coded scope:
  this method only checks the literal phrase `for example`.

  - Catches:
    - `The species for example, occurs...`
    - `The species, for example occurs...`
    - `The species for example occurs...`

  - Misses:
    - correct `The species, for example, occurs...`
    - sentence-start `For example, ...`
    - broader punctuation reasoning beyond this narrow comma rule

- `check_colon_spacing(...)`
  How it works:
  strips all simple style markers, then flags whitespace immediately before `:`.
  Hard-coded scope:
  this method only checks spaces immediately before the literal character `:`.

  - Catches:
    - `Altitude : 200 m`
    - `Countries  : Peru, Ecuador`

  - Misses:
    - spacing after the colon
    - whether the colon itself is the right punctuation mark

- `check_semicolon_spacing(...)`
  How it works:
  strips all simple style markers, then flags whitespace immediately before `;`.
  Hard-coded scope:
  this method only checks spaces immediately before the literal character `;`.

  - Catches:
    - `Peru ; Ecuador`
    - `The range is broad ; however, records are sparse.`

  - Misses:
    - spacing after the semicolon
    - whether the semicolon itself is the right punctuation mark

Aggregated misses for `PunctuationChecker`:
- it does not fully parse sentence punctuation
- it does not validate quotation punctuation, parenthesis balancing, or apostrophes
- the range-dash rule is still regex-based and intentionally narrow; it skips repeated-unit structures and date-like numeric chains rather than performing full syntax parsing

### `bibliography.py` - `BibliographyChecker`

Category: `Bibliography`

Aggregated method list:
- `begin_sweep(...)`
  Starts the embedded helper checkers used by bibliography review.
- `check_ampersand_usage(...)`
  Replaces `&` with `and` in bibliography sections.
- `check_text(...)`
  Runs the bibliography-local ampersand rule plus selected imported rules from other checkers.
- `end_sweep(...)`
  Ends the embedded helper checkers used by bibliography review.

Detailed coverage:

- `begin_sweep(...)`
  How it works:
  forwards `begin_sweep()` to the embedded `AbbreviationChecker`, `NumberChecker`, and `PunctuationChecker` helpers.

  - Catches:
    - no violations; this is lifecycle setup only

  - Misses:
    - all text checking, because it does not inspect text directly

- `check_text(...)`
  How it works:
  returns no violations unless `section_name` contains `Bibliography`. In bibliography sections, it combines:
  - `check_ampersand_usage(...)`
  - `AbbreviationChecker.check_et_al(...)`
  - `PunctuationChecker.check_range_dashes(...)`
  - `NumberChecker.check_large_numbers(...)`

  The imported helper methods keep their original `rule_class` and `rule_method` values, so bibliography output can still contain `AbbreviationChecker`, `PunctuationChecker`, and `NumberChecker` violations.
  Hard-coded/structural dependency:
  this dispatcher depends on the literal section-name substring `Bibliography` and on the behavior of the embedded helper checkers.

  - Catches:
    - bibliography ampersands such as `Smith & Jones 2020`
    - bibliography `et al.` issues such as `Mishra et al. 2015`
    - bibliography numeric ranges such as `Journal 10-20`
    - bibliography large numbers such as `Flora 5000 species`

  - Misses:
    - all non-bibliography sections
    - bibliography issues outside the four checks listed above
    - any deeper reference parsing beyond what the embedded helper methods already do

- `check_ampersand_usage(...)`
  How it works:
  only runs when `section_name` contains `Bibliography`, strips all simple style markers, then flags every literal `&` and suggests `and`.
  Hard-coded scope:
  this method depends on the literal section-name substring `Bibliography` and the literal character `&`.

  - Catches:
    - `Smith & Jones 2020`
    - `Smith&Jones 2020`
    - `<i>Smith</i> <b>&</b> <i>Jones</i> 2020`

  - Misses:
    - the same text outside bibliography sections
    - cases where no literal `&` is present
    - because it is intentionally broad, it can also catch non-author ampersands inside bibliography text

- `end_sweep(...)`
  How it works:
  forwards `end_sweep()` to the embedded `AbbreviationChecker`, `NumberChecker`, and `PunctuationChecker` helpers.

  - Catches:
    - no violations; this is lifecycle cleanup only

  - Misses:
    - all text checking, because it does not inspect text directly

Aggregated misses for `BibliographyChecker`:
- it does not validate full bibliography structure
- it only checks ampersands plus the embedded `et al.`, range-dash, and large-number rules
- it does not check DOI, URL, title, journal, page, or author-order formatting

### `references.py` - `ReferenceChecker`

Category: `References`

Aggregated method list:
- `check_citation_comma(...)`
  Removes a comma between citation author text and a four-digit year in parentheses.

Detailed coverage:

- `check_citation_comma(...)`
  How it works:
  strips all simple style markers, then looks for a parenthetical shell of `(` + non-nested body + comma + four-digit year + `)` and removes the comma before the year.
  Hard-coded scope:
  this method only checks parenthetical forms ending with a comma plus a four-digit year.

  - Catches:
    - `(Smith, 2020)` -> `(Smith 2020)`
    - `(Mishra et al., 2015)` -> `(Mishra et al. 2015)`
    - other similar parenthetical forms ending `, 2020)`

  - Misses:
    - already-correct `(Smith 2020)`
    - square-bracketed forms such as `[GBIF.org, 2021]`
    - unbracketed `Smith, 2020`
    - citations with nested parentheses in the citation body
    - it can also match non-citation parenthetical text that happens to end with `, 2020)`

Aggregated misses for `ReferenceChecker`:
- it does not parse all author-name formats
- it does not check DOI, URL, title, journal, or page formatting

### `scientific.py` - `ScientificNameChecker`

Category: `Scientific Names`

Aggregated method list:
- `check_species_abbreviations(...)`
  Ensures `sp` and `spp` are written with a trailing period.

Detailed coverage:

- `check_species_abbreviations(...)`
  How it works:
  strips all simple style markers, then runs two narrow regex checks: `spp` must end with a period and `sp` must end with a period. Whole-word matching prevents the rules from firing inside larger words.
  Hard-coded scope:
  this method only checks the literal abbreviations `sp` and `spp`.

  - Catches:
    - `sp` -> `sp.`
    - `spp` -> `spp.`
    - `<i>sp</i>`
    - `<b>spp</b>`
    - `<i>s</i><i>p</i>`
    - `<sup>s</sup><sub>p</sub>`

  - Misses:
    - correct `sp.` and `spp.`
    - larger words such as `species` and `sppx`
    - broader scientific-name formatting issues beyond this missing-period rule

Aggregated misses for `ScientificNameChecker`:
- it only checks `sp` and `spp`
- it does not validate taxonomy, authorship, or italicization beyond those abbreviations

### `spelling.py` - `SpellingChecker`

Category: `Language`

Aggregated method list:
- `check_text(...)`
  Applies a hardcoded general spelling map plus a separate `-ize` house-style map.

Detailed coverage:

- `check_text(...)`
  How it works:
  strips all simple style markers, then runs two whole-word passes: `UK_SPELLING_MAP` for general replacements and `IZE_WORDS` for house-style `-ize` preferences. The suggested fix preserves simple capitalization patterns, and no-op replacements are skipped.
  Hard-coded list dependency:
  this method only knows the spellings present in the hard-coded `UK_SPELLING_MAP` and `IZE_WORDS` dictionaries.

  - Catches:
    - `color` -> `colour`
    - `center` -> `centre`
    - `liter` -> `litre`
    - `analyze` -> `analyse`
    - `organisation` -> `organization`
    - `organised` -> `organized`
    - `realise` -> `realize`
    - `characterise` -> `characterize`
    - split or wrapped simple-tag forms built from those words

  - Misses:
    - unknown typos outside the hardcoded maps
    - context-sensitive spelling decisions
    - words broken by unsupported markup
    - no-op entries such as `recognition` -> `recognition`, which are intentionally ignored
    - any spelling variant not present in the hard-coded dictionaries

Aggregated misses for `SpellingChecker`:
- it is not a general spellchecker
- it only knows the words present in the two static maps
- some entries reflect house style rather than universal UK spelling

### `symbols.py` - `SymbolChecker`

Category: `Symbols`

Current default reviewer:
included in the standard non-bibliography `IUCNAssessmentReviewer` flow.
Bibliography sections still do not use `SymbolChecker`, because those sections
are routed to `BibliographyChecker` only.

Aggregated method list:
- `check_area_units(...)`
  Normalizes a small set of text-based area units into squared-symbol forms.
- `check_degree_text(...)`
  Converts written-out degree forms to `°` notation.
- `check_degree_symbol_spacing(...)`
  Removes spacing around an existing degree symbol.
- `check_percentage(...)`
  Converts `percent` / `per cent` wording into `%`.
- `check_percentage_symbol_spacing(...)`
  Removes spaces before `%` and adds spaces between integers and a short list of units.

Detailed coverage:

- `check_area_units(...)`
  How it works:
  strips simple bold/italic tags, then checks a small fixed set of non-preferred area forms and suggests the corresponding squared-symbol form.
  Hard-coded list dependency:
  this method only normalizes the hard-coded forms `sq km`, `sqkm`, `km2`, `m2`, `cm2`, and `mm2`.

  - Catches:
    - `sq km` or `sqkm` -> `km²`
    - `km2` -> `km²`
    - `m2` -> `m²`
    - `cm2` -> `cm²`
    - `mm2` -> `mm²`
    - simple-tag wrapped or split forms built from those patterns

  - Misses:
    - already-correct `km²`, `m²`, `cm²`, `mm²`
    - other area-unit variants outside the hardcoded list
    - arbitrary markup beyond the supported simple tags
    - any non-preferred area form not present in the hard-coded set

- `check_degree_text(...)`
  How it works:
  strips all simple style markers, then checks written-out direction forms such as `12 degrees N` and written-out Celsius forms such as `20 degrees C` / `20 degrees Celsius`.
  Hard-coded list dependency:
  this method only knows the direction letters `N`, `S`, `E`, `W` and Celsius forms `C` / `Celsius`.

  - Catches:
    - `12 degrees N`
    - `12.5 degrees n`
    - `20 degrees C`
    - `20.75 degrees Celsius`

  - Misses:
    - correct `12°N` and `20°C`
    - Fahrenheit
    - more complex coordinate notation
    - temperature or coordinate notations outside the hard-coded direction/Celsius cases

- `check_degree_symbol_spacing(...)`
  How it works:
  strips all simple style markers, then checks existing degree-symbol forms with incorrect spacing such as `12 °C`, `12° C`, or `12.5 ° N`.
  Hard-coded list dependency:
  this method only knows the direction letters `N`, `S`, `E`, `W` and Celsius form `C`.
  Range behavior:
  shared-unit ranges such as `14-26 °C`, `14–26 °C`, and `14—26 °C` are skipped.

  - Catches:
    - `12 °C`
    - `12° C`
    - `12.5 ° N`

  - Misses:
    - correct `12°N` and `20°C`
    - shared-unit ranges such as `14-26 °C`, `14–26 °C`, and `14—26 °C`
    - Fahrenheit
    - more complex coordinate notation
    - temperature or coordinate notations outside the hard-coded direction/Celsius spacing cases

- `check_percentage(...)`
  How it works:
  strips all simple style markers, then looks for `number + percent` and `number + per cent`, supporting both integers and decimals.
  Hard-coded scope:
  this method only checks the literal wordings `percent` and `per cent`.

  - Catches:
    - `12 percent` -> `12%`
    - `12.5 percent` -> `12.5%`
    - `7 per cent` -> `7%`
    - `7.25 per cent` -> `7.25%`

  - Misses:
    - already-symbolized forms such as `12%`
    - spacing around an existing `%` symbol
    - other paraphrases such as `percentage`
    - percentage phrasings outside the two hard-coded forms

- `check_percentage_symbol_spacing(...)`
  How it works:
  strips all simple style markers, then applies two spacing checks: remove spaces before `%` and add spaces between integers and the short fixed unit list `km`, `ha`, `kg`, `m`, `cm`, `mm`, `ml`, `l`.
  Hard-coded list dependency:
  the number-unit spacing part of this rule only uses the hard-coded short unit list `km`, `ha`, `kg`, `m`, `cm`, `mm`, `ml`, `l`.
  Range behavior:
  shared-unit ranges such as `12-15 %`, `12–15 %`, `12—15 %`, and `600-1200m` are skipped.

  - Catches:
    - `12 %` -> `12%`
    - `12km` -> `12 km`
    - `12ha` -> `12 ha`
    - `12kg` -> `12 kg`
    - `12m` -> `12 m`
    - `12cm` -> `12 cm`

  - Misses:
    - decimal+unit spacing in this rule
    - shared-unit ranges such as `12-15 %`, `12–15 %`, `12—15 %`, and `600-1200m`
    - units outside that short list
    - richer unit forms such as `sq km`, `m3`, `km2`
    - number-unit spacing for any unit outside the hard-coded unit list

Aggregated misses for `SymbolChecker`:
- it does not normalize every unit in the style guide
- it is formatting-oriented, not measurement-aware
- most rules use short fixed unit lists rather than general SI parsing

## Notes On Overlap

- `FormattingChecker` and `ScientificNameChecker` both touch taxonomic text.
  `FormattingChecker` focuses on harvested genus/species formatting, higher-taxon formatting, and EOO/AOO capitalization.
  `ScientificNameChecker` focuses on `sp.` / `spp.` formatting.

- `GeographyChecker` and `IUCNTermsChecker` can both touch assessment wording, but neither validates the biological truth of the statement.

- `NumberChecker`, `PunctuationChecker`, and `SymbolChecker` overlap around numeric style, but they do different jobs.
  `NumberChecker` handles word-vs-digit style, comma placement, sentence starts, and million/billion wording.
  `PunctuationChecker` handles bare numeric range separators and a few punctuation-spacing rules.
  `SymbolChecker` handles `%`, degree expressions, and a small set of unit symbol normalizations.

## Scope Reminder

These checkers are still rule-based utilities, not full language-understanding systems.
They are strongest on repeatable formatting and house-style conventions.
They are weaker on:
- nuanced context
- ambiguous prose
- assessment logic
- scientific correctness
- layout or markup beyond the simple patterns the code currently recognizes
