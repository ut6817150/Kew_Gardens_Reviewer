"""Test the newly added validation rules."""

import json
from streamlit_json_validator import StreamlitAssessmentValidator

print("=" * 60)
print("Testing New Validation Rules")
print("=" * 60)

# Test 1: EOO/AOO Discrepancy
print("\n1. Testing EOO/AOO Discrepancy Check...")
test1 = {
    'title': 'Test',
    'blocks': [
        {'type': 'table', 'rows': [['Red List Status'], ['EN - Endangered, B2ab(iii)']]}
    ],
    'children': [
        {
            'title': 'Distribution',
            'children': [
                {
                    'title': 'Extent of Occurrence (EOO)',
                    'blocks': [{'type': 'table', 'rows': [['EOO'], ['50000']]}],
                    'children': []
                },
                {
                    'title': 'Area of Occupancy (AOO)',
                    'blocks': [{'type': 'table', 'rows': [['AOO'], ['40']]}],
                    'children': []
                }
            ]
        },
        {
            'title': 'Red Li            'title': 'Red Li chil            'title': 'Red Li            'tient   tionale',            'title': 'Red Li            'title': 'Red Li chil s spe           restr            'title': 'Red Li                      'ildr            'title': 'Red                 'title': 'Red Li            'title':ato        rt            'titvalid       t1)
discrepancy_violations = [v for v in report.violations if 'discrepancy' in v.rule_id.lower()]
if discrepancy_violations:
    for     for     for     for     for          for     for     for essage}")
else:
    print("  ✗ FAIL: Should have flagged EOO/AOO discrepancy"    print("  ✗ FAIL: Should have flagged EOO/AOO discrepancy"    print("  ✗d Check...")
test2 = {
    'title': '    'title': '   s': [{'type': 'table    'title': '    'title': '   s': [{'type': 'table   ab(iii)']]}],
    'children': [{
        'title': 'Red List Assessment',
        'children': [{
            'title'            'tiati                      loc     [{'ty            'title'            'tiati                      loc     [{'ty            'title'            'tiati                      loc     [{'ty            'title'            'tiati                      loc     [{'ty            'title'            'tiati                      loc     [{'ty            'title'            'tiati                      loc     [{'ty            'title(f            'title'            'tiati                      loc     [{'ty            'title' tions")

# Test 3: Criteria T# Test 3: Criteria T# Test 3: Criteria T# Test 3: Criteria T# Test 3: Crit")
test3 = test3 = test3 = test3 = test3 = test3 = test3 = test3 = test3 = test3 = test3 = test3 = test3 = test3 = test3 = test3 = test3 = test3 = test3 = test3 = te': 'Distribtest3 = test3 = test3 = test3 = test3 = test3 = test3 = test3 = test3 = test3 = test3 = test3 = test3 = test3 = teste',test3 = test3 = ], ['600']]}],
            'children': []
        }]
    }]
}

report3, _ = validator.validate(test3)
threshold_violations = [v for v in report3.violations if 'vu' in v.rule_id.lower() or 'threshold' in v.message.lower()]
if threshold_violations:
    for v in threshold_violations:
        print(f"  ✓ PASS: {v.message}")
else:
    print("  ✗ FAIL: Should have flagged EN with VU threshold")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
