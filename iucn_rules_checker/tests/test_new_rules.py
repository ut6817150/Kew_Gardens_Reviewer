"""Test the newly added validation rules."""

import json
from streamlit_json_validator import StreamlitAssessmentValidator

print("=" * 60)
print("Testing New Validation Rules")
print("=" * 60)


# Test 1: EOO/AOO Discrepancy
print("\n1. Testing EOO/AOO Discrepancy Check...")

test1 = {
    "title": "Assessment",
    "blocks": [
        {
            "type": "table",
            "rows": [
                ["Category"],
                ["EN"],
                ["Criteria"],
                ["B2ab(iii)"]
            ]
        }
    ],
    "children": [
        {
            "title": "Geographic Range",
            "children": [
                {
                    "title": "Extent of Occurrence (EOO)",
                    "blocks": [
                        {
                            "type": "table",
                            "rows": [
                                ["EOO (km²)"],
                                ["50000"]
                            ]
                        }
                    ],
                    "children": []
                },
                {
                    "title": "Area of Occupancy (AOO)",
                    "blocks": [
                        {
                            "type": "table",
                            "rows": [
                                ["AOO (km²)"],
                                ["40"]
                            ]
                        }
                    ],
                    "children": []
                }
            ]
        },
        {
            "title": "Red List Assessment",
            "children": [
                {
                    "title": "Assessment Rationale",
                    "blocks": [
                        {
                            "type": "paragraph",
                            "text": "This species has a restricted range with small AOO."
                        }
                    ],
                    "children": []
                }
            ]
        }
    ]
}

validator1 = StreamlitAssessmentValidator()
report1, _ = validator1.validate(test1)

print("\nViolations found:", len(report1.violations))
for v in report1.violations:
    print("✓", v.rule_id, "|", v.message)



# Test 2: Locations Not Explained
print("\n2. Testing Locations Not Explained Check...")

test2 = {
    "title": "Assessment",
    "blocks": [
        {
            "type": "table",
            "rows": [
                ["Category"],
                ["VU"],
                ["Criteria"],
                ["B2ab(iii)"]
            ]
        }
    ],
    "children": [
        {
            "title": "Red List Assessment",
            "children": [
                {
                    "title": "Assessment Rationale",
                    "blocks": [
                        {
                            "type": "paragraph",
                            "text": "This species is known from 5 locations with an AOO of 40 km²."
                        }
                    ],
                    "children": []
                }
            ]
        }
    ]
}

validator2 = StreamlitAssessmentValidator()
report2, _ = validator2.validate(test2)

print("\nViolations found:", len(report2.violations))
for v in report2.violations:
    print("✓", v.rule_id, "|", v.message)



# Test 3: Threshold mismatch
print("\n3. Testing Threshold Mismatch Check...")

test3 = {
    "title": "Assessment",
    "blocks": [
        {
            "type": "table",
            "rows": [
                ["Category"],
                ["EN"],
                ["Criteria"],
                ["B2ab(iii)"]
            ]
        }
    ],
    "children": [
        {
            "title": "Geographic Range",
            "children": [
                {
                    "title": "Area of Occupancy (AOO)",
                    "blocks": [
                        {
                            "type": "table",
                            "rows": [
                                ["AOO (km²)"],
                                ["600"]
                            ]
                        }
                    ],
                    "children": []
                }
            ]
        }
    ]
}

validator3 = StreamlitAssessmentValidator()
report3, _ = validator3.validate(test3)

print("\nViolations found:", len(report3.violations))
for v in report3.violations:
    print("✓", v.rule_id, "|", v.message)


print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)