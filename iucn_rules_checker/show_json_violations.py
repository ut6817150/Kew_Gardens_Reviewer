#!/usr/bin/env python3
"""Show JSON violations with detailed section information."""

import json
import sys
from engine import IUCNRuleChecker

if len(sys.argv) < 2:
    print("Usage: python show_json_violations.py <json_file>")
    sys.exit(1)

filename = sys.argv[1]

# Load JSON
with open(filename) as f:
    assessment = json.load(f)

# Check
checker = IUCNRuleChecker()
report = checker.check_json(assessment)

print("="*80)
print("JSON ASSESSMENT VALIDATION REPORT")
print("="*80)
print(f"\nFile: {filename}")
print(f"Total Violations: {len(report.violations)}")
print()

# Group by category
by_category = {}
for v in report.violations:
    if v.category not in by_category:
        by_category[v.category] = []
    by_category[v.category].append(v)

# Show violations by category with section info
for category in sorted(by_category.keys()):
    violations = by_category[category]
    print(f"\n{'='*80}")
    print(f"📋 {category.upper()} ({len(violations)} violations)")
    print("="*80)
    
    for i, v in enumerate(violations, 1):
        print(f"\n[{i}] {v.severity.value.upper()}")
        
        # Show section if available
        section = v.assessment_section
        if section and section != "Whole Document":
            print(f" Section: {section}")
        elif v.metadata and 'section' in v.metadata:
            print(f" Section: {v.metadata['section']}")
        
        # Message
        print(f"  {v.message}")
        
        # Show what was found
        if v.matched_text:
            preview = v.matched_text[:150]
            if len(v.matched_text) > 150:
                preview += "..."
            print(f"    🔍 Found: \"{preview}\"")
        
        # Show fix
        if v.suggested_fix:
            fix_preview = v.suggested_fix[:150]
            if len(v.suggested_fix) > 150:
                fix_preview += "..."
            print(f"    ✅ Fix: {fix_preview}")
        
        # Show context if available
        if v.context and v.context != "":
            context_preview = v.context[:200]
            if len(v.context) > 200:
                context_preview += "..."
            print(f"    📝 Context: {context_preview}")

print(f"\n{'='*80}")
print(f" COMPLETE: {len(report.violations)} total violations")
print("="*80)

# Summary by severity
by_severity = {}
for v in report.violations:
    sev = v.severity.value
    by_severity[sev] = by_severity.get(sev, 0) + 1

print(f"\n Summary:")
print(f"   Errors: {by_severity.get('error', 0)}")
print(f"   Warnings: {by_severity.get('warning', 0)}")
print(f"   Info: {by_severity.get('info', 0)}")