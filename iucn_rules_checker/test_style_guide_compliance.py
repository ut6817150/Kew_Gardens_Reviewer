"""
Comprehensive test for IUCN Style Guide Rules (Sections 3.4-3.5)
Tests against both text and JSON input.
"""

import json
from engine import IUCNRuleChecker

print("=" * 80)
print("IUCN STYLE GUIDE COMPLIANCE TEST")
print("Sections 3.4 (Numbers & Dates) and 3.5 (Punctuation)")
print("=" * 80)

checker = IUCNRuleChecker()

# ==============================================================================
# SECTION 3.4.1: NUMBERS
# ==============================================================================
print("\n" + "=" * 80)
print("3.4.1 NUMBERS TESTING")
print("=" * 80)

numbers_tests = [
    # Rule 1: Numbers 1-9 in full
    ("Species found at 3 sites", True, "Should flag: 3 → three"),
    ("Species found at three sites", False, "Correct: written out"),
    ("Species found at 10 sites", False, "Correct: >9 as numeral"),
    
    # Rule 2: Numbers >9 as numerals
    ("After thirty two sightings", True, "Should flag: thirty two → 32"),
    ("After 32 sightings", False, "Correct: numeral"),
    
    # Rule 3: Sentence start written out
    ("15 grouse were spotted", True, "Should flag: 15 → Fifteen"),
    ("Fifteen grouse were spotted", False, "Correct: written out"),
    
    # Rule 4: Commas in numbers ≥1000
    ("Depth of 2000 m", True, "Should flag: 2000 → 2,000"),
    ("Depth of 2,000 m", False, "Correct: has comma"),
    ("Depth of 2.000 m", True, "Should flag: period → comma"),
    
    # Rule 5: Million/billion format
    ("Population of 2400000", True, "Should flag: 2400000 → 2.4 million"),
    ("Population of 2.4 million", False, "Correct: million format"),
]

print("\nRule 1: Numbers 1-9 written out")
print("Rule 2: Numbers >9 as numerals")
print("Rule 3: Sentence start written out")
print("Rule 4: Commas for ≥1,000")
print("Rule 5: Million/billion format")
print("-" * 80)

for text, should_violate, description in numbers_tests:
    report = checker.check(text)
    number_violations = [v for v in report.violations if v.category == 'Numbers']
    has_violation = len(number_violations) > 0
    
    status = "✓" if has_violation == should_violate else "✗"
    result = "PASS" if has_violation == should_violate else "FAIL"
    
    print(f"{status} [{result}] {description}")
    print(f"   Text: '{text}'")
    if has_violation and should_violate:
        print(f"   Found: {number_violations[0].message}")
    elif not has_violation and not should_violate:
        print(f"   Correctly accepted")
    print()

# ==============================================================================
# SECTION 3.4.2: DATES
# ==============================================================================
print("\n" + "=" * 80)
print("3.4.2 DATES TESTING")
print("=" * 80)

dates_tests = [
    # Rule 1: dd/month/yyyy format
    ("11th January 2005", True, "Should flag: 11th → 11"),
    ("11 January 2005", False, "Correct: no ordinal"),
    ("January 11, 2005", True, "Should flag: US format"),
    
    # Rule 2: Century format
    ("nineteenth century", True, "Should flag: nineteenth → 19th"),
    ("19th century", False, "Correct: numeric"),
    ("19th Century", True, "Should flag: Century → century"),
    
    # Rule 2: Decade format
    ("1980's", True, "Should flag: 1980's → 1980s"),
    ("1980 s", True, "Should flag: 1980 s → 1980s"),
    ("1980s", False, "Correct: no apostrophe"),
]

print("\nRule 1: dd/month/yyyy (no ordinals, no US format)")
print("Rule 2: Centuries as 19th century")
print("Rule 2: Decades as 1980s (no apostrophe)")
print("-" * 80)

for text, should_violate, description in dates_tests:
    report = checker.check(text)
    date_violations = [v for v in report.violations if v.category == 'Dates']
    has_violation = len(date_violations) > 0
    
    status = "✓" if has_violation == should_violate else "✗"
    result = "PASS" if has_violation == should_violate else "FAIL"
    
    print(f"{status} [{result}] {description}")
    print(f"   Text: '{text}'")
    if has_violation and should_violate:
        print(f"   Found: {date_violations[0].message}")
    elif not has_violation and not should_violate:
        print(f"   Correctly accepted")
    print()

# ==============================================================================
# SECTION 3.5.2: DASHES
# ==============================================================================
print("\n" + "=" * 80)
print("3.5.2 DASHES TESTING")
print("=" * 80)

dash_tests = [
    # Rule 1: En dash for ranges
    ("112-600 m", True, "Should flag: hyphen → en dash"),
    ("112–600 m", False, "Correct: en dash"),
    ("15-31 March", True, "Should flag: hyphen → en dash"),
    ("15–31 March", False, "Correct: en dash"),
    ("9:00-5:00", True, "Should flag: hyphen → en dash"),
    ("9:00–5:00", False, "Correct: en dash"),
]

print("\nRule 1: En dashes (–) for ranges")
print("-" * 80)

for text, should_violate, description in dash_tests:
    report = checker.check(text)
    punct_violations = [v for v in report.violations if v.category == 'Punctuation' and 'dash' in v.message.lower()]
    has_violation = len(punct_violations) > 0
    
    status = "✓" if has_violation == should_violate else "✗"
    result = "PASS" if has_violation == should_violate else "FAIL"
    
    print(f"{status} [{result}] {description}")
    print(f"   Text: '{text}'")
    if has_violation and should_violate:
        print(f"   Found: {punct_violations[0].message}")
    elif not has_violation and not should_violate:
        print(f"   Correctly accepted")
    print()

# ==============================================================================
# SECTION 3.5.6: COMMAS
# ==============================================================================
print("\n" + "=" * 80)
print("3.5.6 COMMAS TESTING")
print("=" * 80)

comma_tests = [
    # Rule 5: Commas around "for example"
    ("Species include for example sharks", True, "Should flag: missing comma before"),
    ("Species include, for example sharks", True, "Should flag: missing comma after"),
    ("Species include, for example, sharks", False, "Correct: commas both sides"),
]

print("\nRule 5: Commas around 'for example'")
print("-" * 80)

for text, should_violate, description in comma_tests:
    report = checker.check(text)
    comma_violations = [v for v in report.violations if 'for example' in v.message.lower()]
    has_violation = len(comma_violations) > 0
    
    status = "✓" if has_violation == should_violate else "✗"
    result = "PASS" if has_violation == should_violate else "FAIL"
    
    print(f"{status} [{result}] {description}")
    print(f"   Text: '{text}'")
    if has_violation and should_violate:
        print(f"   Found: {comma_violations[0].message}")
    elif not has_violation and not should_violate:
        print(f"   Correctly accepted")
    print()

# ==============================================================================
# JSON INPUT TEST
# ==============================================================================
print("\n" + "=" * 80)
print("JSON INPUT TESTING")
print("=" * 80)

# Create test JSON with style guide violations
test_json = {
    'title': 'Test Assessment',
    'blocks': [
        {'type': 'table', 'rows': [['Red List Status'], ['VU - Vulnerable, B2ab(iii)']]}
    ],
    'children': [
        {
            'title': 'Red List Assessment',
            'children': [
                {
                    'title': 'Assessment Rationale',
                    'blocks': [
                        {
                            'type': 'paragraph',
                            'text': '''This species occurs at 3 locations with an extent of occurrence 
                            of 50000 km2. It was first recorded in the nineteenth century. 
                            Populations declined in the 1980's. The range is 1000-2000 m elevation.
                            There are for example several threats affecting the species.'''
                        }
                    ],
                    'children': []
                }
            ]
        }
    ]
}

print("\nTest JSON with multiple style guide violations:")
report = checker.check_json(test_json)

print(f"\nTotal violations: {len(report.violations)}")
print(f"\nBy Category:")
by_category = {}
for v in report.violations:
    if v.category not in by_category:
        by_category[v.category] = 0
    by_category[v.category] += 1
for cat, count in sorted(by_category.items()):
    print(f"  {cat}: {count}")

print(f"\nDetailed violations:")
for i, v in enumerate(report.violations[:15], 1):
    print(f"\n{i}. [{v.severity.value.upper()}] {v.category}")
    print(f"   {v.message}")
    if v.suggested_fix:
        print(f"   Fix: {v.suggested_fix}")

# ==============================================================================
# REAL FILE TEST
# ==============================================================================
print("\n" + "=" * 80)
print("REAL ASSESSMENT FILE TEST")
print("=" * 80)

try:
    with open('Acianthera_odontotepala_draft_status_Jun2025__1_.json') as f:
        real_assessment = json.load(f)
    
    print("\nTesting real assessment file...")
    report = checker.check_json(real_assessment)
    
    print(f"\nTotal violations: {len(report.violations)}")
    
    # Show style guide specific violations
    style_violations = {
        'Numbers': [],
        'Dates': [],
        'Punctuation': []
    }
    
    for v in report.violations:
        if v.category in style_violations:
            style_violations[v.category].append(v)
    
    print(f"\nStyle Guide Violations:")
    for cat, viols in style_violations.items():
        if viols:
            print(f"\n  {cat}: {len(viols)} violations")
            for v in viols[:3]:
                print(f"    - {v.message}")
            if len(viols) > 3:
                print(f"    ... and {len(viols)-3} more")
    
    print("\n✅ Real file test complete!")
    
except FileNotFoundError:
    print("\n⚠️  Real assessment file not found, skipping")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

total_tests = len(numbers_tests) + len(dates_tests) + len(dash_tests) + len(comma_tests)
print(f"\nTotal test cases: {total_tests}")
print(f"Sections tested:")
print(f"  ✓ 3.4.1 Numbers (5 rules)")
print(f"  ✓ 3.4.2 Dates (3 rules)")
print(f"  ✓ 3.5.2 Dashes (1 rule)")
print(f"  ✓ 3.5.6 Commas (1 rule)")
print(f"\n✅ IUCN Style Guide compliance testing complete!")
print("=" * 80)