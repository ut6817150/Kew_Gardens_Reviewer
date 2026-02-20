"""Quick demo of the IUCN Rule Checker."""

import json
from engine import IUCNRuleChecker

print("=" * 60)
print("IUCN RULE CHECKER DEMO")
print("=" * 60)

# Example 1: Text checking
print("\n1. Text-based checking:")
checker = IUCNRuleChecker()
text = """
The species occurs in 3 locations in Vietnam.
It has an Extent of Occurrence of 50000 km2.
Populations are maintained ex situ.
"""
report = checker.check(text)
print(f"   Found {len(report.violations)} violations")

# Example 2: JSON checking  
print("\n2. JSON-based checking:")
with open('Acianthera_odontotepala_draft_status_Jun2025__1_.json') as f:
    assessment = json.load(f)
report = checker.check_json(assessment)
print(f"   Found {len(report.violations)} violations")

by_cat = {}
for v in report.violations:
    by_cat[v.category] = by_cat.get(v.category, 0) + 1

print(f"\n   Categories:")
for cat, count in sorted(by_cat.items()):
    print(f"     {cat}: {count}")

print("\n✅ Demo complete!")
