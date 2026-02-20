"""
Example: How to use the IUCN Rule Checker with Streamlit JSON

This shows you exactly how to integrate the checker into your workflow.
"""

import json
from engine import IUCNRuleChecker

# ============================================================================
# EXAMPLE 1: Check Streamlit JSON
# ============================================================================

print("=" * 60)
print("EXAMPLE 1: Checking Streamlit JSON")
print("=" * 60)

# Load JSON from Streamlit
with open('sample.json', 'r') as f:
    assessment_json = json.load(f)

# Create checker
checker = IUCNRuleChecker()

# Check the JSON
report = checker.check_json(assessment_json)

# Display results
print(f"\n Results:")
print(f"Total violations: {report.total_violations}")
print(f"Text length: {report.text_length} characters")

print(f"\nBy Severity:")
for severity, count in report.violations_by_severity.items():
    print(f"  {severity.value}: {count}")

print(f"\nBy Category:")
for category, count in report.violations_by_category.items():
    print(f"  {category}: {count}")

# Show first 5 violations
if report.violations:
    print(f"\nFirst 5 Violations:")
    for i, v in enumerate(report.violations[:5], 1):
        print(f"\n  [{i}] {v.severity.value.upper()} - {v.category}")
        print(f"      {v.message}")
        if v.matched_text:
            print(f"      Found: {v.matched_text[:60]}...")
        if v.suggested_fix:
            print(f"      Fix: {v.suggested_fix}")


# ============================================================================
# EXAMPLE 2: Check Plain Text
# ============================================================================

print("\n" + "=" * 60)
print("EXAMPLE 2: Checking Plain Text")
print("=" * 60)

text = """
The species has an EOO of 73697 km2 and AOO of 40 km2. 
The color of the bird is gray and it lives in 3 locations.
It was assessed by Smith et al 2020.
"""

report2 = checker.check(text)

print(f"\n📊 Results:")
print(f"Total violations: {report2.total_violations}")

for i, v in enumerate(report2.violations, 1):
    print(f"\n  [{i}] {v.severity.value.upper()} - {v.category}")
    print(f"      {v.message}")
    if v.suggested_fix:
        print(f"      Fix: {v.suggested_fix}")


# ============================================================================
# EXAMPLE 3: Filter by Category
# ============================================================================

print("\n" + "=" * 60)
print("EXAMPLE 3: Check Only Specific Categories")
print("=" * 60)

# Only check spelling and numbers
checker_filtered = IUCNRuleChecker(
    enabled_categories={'Language', 'Numbers'}
)

report3 = checker_filtered.check(text)

print(f"\n Results (Language + Numbers only):")
print(f"Total violations: {report3.total_violations}")
print(f"Checked rules: {len(report3.checked_rules)}")
print(f"Skipped rules: {len(report3.skipped_rules)}")


# ============================================================================
# EXAMPLE 4: Export to JSON
# ============================================================================

print("\n" + "=" * 60)
print("EXAMPLE 4: Export Results to JSON")
print("=" * 60)

# Get JSON output
json_output = report.to_json()

# Save to file
with open('sample.json', 'w') as f:
    f.write(json_output)

print("\n✅ Report saved to: /mnt/user-data/outputs/report_example.json")
print(f"Preview:\n{json_output[:200]}...")


# ============================================================================
# EXAMPLE 5: Integrate with Your Streamlit App
# ============================================================================

print("\n" + "=" * 60)
print("EXAMPLE 5: How to Use in Streamlit")
print("=" * 60)

print("""
# In your Streamlit app:

import streamlit as st
from engine import IUCNRuleChecker

# When user uploads a document and you convert it to JSON:
if uploaded_file:
    # Your code to convert DOCX -> JSON
    assessment_json = convert_docx_to_json(uploaded_file)
    
    # Run the checker
    checker = IUCNRuleChecker()
    report = checker.check_json(assessment_json)
    
    # Display results in Streamlit
    st.write(f"Total Violations: {report.total_violations}")
    
    for violation in report.violations:
        if violation.severity.value == 'error':
            st.error(f"{violation.category}: {violation.message}")
        elif violation.severity.value == 'warning':
            st.warning(f"{violation.category}: {violation.message}")
        else:
            st.info(f"{violation.category}: {violation.message}")
""")

print("\n" + "=" * 60)
print("✅ All examples complete!")
print("=" * 60)
