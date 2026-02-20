# IUCN Checker

A Python-based linting tool that validates text against IUCN Red List assessment formatting, style, and terminology rules.

## Overview

IUCN Checker enforces standardized conventions for scientific assessments, including UK spelling, number formatting, date conventions, IUCN-specific terminology, and more. It can be used as a command-line tool or imported as a Python library.

## Features

- **11 specialized checkers** covering 50+ distinct rules
- **Accurate position tracking** with line/column information for each violation
- **Suggested fixes** for most violations
- **Flexible filtering** by category, severity, or specific rules
- **Multiple output formats** including JSON and pretty-printed summaries
- **No external dependencies** - uses only Python standard library
- **Dual interface** - CLI tool and importable Python library
- **Comprehensive test suite** - 22+ tests covering all major checkers

## Installation

No installation required beyond Python 3.7+. Clone the repository and run directly:

```bash
git clone <repository-url>
cd code
```

### Development Setup (Recommended)

For development and testing, set up a virtual environment:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows

# Install in editable mode with dependencies
pip install -e .

# Install testing dependencies
pip install pytest pytest-cov
```


## Usage

### Command Line

```bash
# Check a file
python -m iucn_checker input.txt

# Check from stdin
echo "The color of the species is grey." | python -m iucn_checker

# Pretty-print violations
python -m iucn_checker input.txt --pretty

# Show summary only
python -m iucn_checker input.txt --summary

# Output to JSON file
python -m iucn_checker input.txt -o report.json

# Filter by categories
python -m iucn_checker input.txt --categories Language Numbers

# Filter by minimum severity (error, warning, info)
python -m iucn_checker input.txt --severity warning

# Check plain text (skip formatting checks that require HTML tags)
python -m iucn_checker input.txt --plain-text

# List available categories
python -m iucn_checker --list-categories
```

### As a Python Library

```python
from iucn_checker import IUCNRuleChecker, check_text, Severity

# Quick check
report = check_text("Your assessment text here...")
print(report.to_json())

# Check with filtering options
checker = IUCNRuleChecker(
    enabled_categories={'Language', 'Numbers'},
    min_severity=Severity.WARNING
)
report = checker.check("Your assessment text here...")

# Iterate through violations
for violation in report.violations:
    print(f"Line {violation.position.line}: {violation.message}")
    if violation.suggested_fix:
        print(f"  Suggestion: {violation.suggested_fix}")
```

## Testing

The project includes a comprehensive test suite to ensure accuracy and reliability.

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_formatting.py -v

# Run tests with coverage report
python -m pytest tests/ --cov=checkers --cov-report=html

# Run tests with detailed output
python -m pytest tests/ -v -s
```

### Test Coverage

The test suite includes:
- **Unit tests** for individual checkers (formatting, numbers, punctuation, spelling)
- **Integration tests** for the complete checking pipeline
- **Edge case tests** for unusual inputs and performance
- **CLI tests** for command-line interface functionality

Current test metrics:
- **22+ comprehensive tests**
- **Coverage of all major rule categories**
- **Validation of suggested fixes**
- **False positive detection and prevention**

### Test Structure
```
tests/
├── conftest.py              # Shared fixtures
├── test_formatting.py       # Scientific name and italics tests
├── test_numbers.py          # Number formatting tests
├── test_punctuation.py      # En-dash and punctuation tests
├── test_spelling.py         # UK/US spelling tests
├── test_integration.py      # End-to-end tests
├── test_edge_cases.py       # Edge cases and performance
└── test_cli.py              # Command-line interface tests
```

## Categories

The checker includes the following rule categories:

| Category | Description |
|----------|-------------|
| **Language** | UK spelling enforcement (colour, centre, grey, -ise endings) |
| **Numbers** | Number formatting (1-9 as words, commas for thousands) |
| **Dates** | Date conventions (no ordinals, numeric centuries) |
| **Abbreviations** | Abbreviation rules (et al., etc., Latin terms) |
| **Symbols** | Units and symbols (km², °, %) |
| **Punctuation** | Punctuation standards (en-dashes for ranges) |
| **IUCN Terms** | IUCN-specific terminology (no "the IUCN", category capitalization) |
| **Geography** | Geographic naming (ISO 3166 country names) |
| **Scientific Names** | Species name formatting (spp., sp., italics) |
| **References** | Citation formatting (author separators, et al.) |
| **Formatting** | Text formatting rules (italics usage — requires HTML-tagged input) |

> **Note:** The Formatting category checks for correct use of italics via HTML tags (`<i>`, `<em>`). If your input is plain text without HTML markup, use the `--plain-text` flag to skip these checks and avoid false positives.

## Output Format

Reports are output in JSON format with the following structure:

```json
{
  "summary": {
    "text_length": 1234,
    "total_violations": 5,
    "by_severity": {
      "error": 1,
      "warning": 3,
      "info": 1
    },
    "by_category": {
      "Language": 2,
      "Numbers": 3
    }
  },
  "violations": [
    {
      "rule_id": "spelling_uk",
      "rule_name": "UK English spelling required",
      "category": "Language",
      "matched_text": "color",
      "position": {
        "start": 42,
        "end": 47,
        "line": 2,
        "column": 15
      },
      "severity": "warning",
      "message": "Use UK spelling 'colour' instead of 'color'",
      "suggested_fix": "colour",
      "context": "...the color of the..."
    }
  ]
}
```

## Exit Codes

When used as a CLI tool:

| Code | Meaning |
|------|---------|
| 0 | No violations found |
| 1 | Warnings or info-level violations found |
| 2 | Errors found |

## Project Structure

```
code/
├── checkers/
│   ├── __init__.py          # Package exports
│   ├── base.py              # Abstract base classes
│   ├── spelling.py          # UK spelling rules
│   ├── numbers.py           # Number formatting
│   ├── dates.py             # Date formatting
│   ├── abbreviations.py     # Abbreviation rules
│   ├── symbols.py           # Symbols and units
│   ├── punctuation.py       # Punctuation rules
│   ├── iucn_terms.py        # IUCN terminology
│   ├── geography.py         # Geographic naming
│   ├── scientific.py        # Scientific names
│   ├── references.py        # Citation formatting
│   └── formatting.py        # Text formatting
├── tests/                   # Comprehensive test suite
│   ├── conftest.py
│   ├── test_formatting.py
│   ├── test_numbers.py
│   ├── test_punctuation.py
│   ├── test_spelling.py
│   ├── test_integration.py
│   ├── test_edge_cases.py
│   └── test_cli.py
├── engine.py                # Core checking orchestration
├── models.py                # Data models (Violation, Report)
├── main.py                  # CLI entry point
├── setup.py                 # Package configuration
├── IUCN_Assessment_Rules.json
└── IUCN_Assessment_Rules.xlsx
```

## Examples

### Checking an Assessment

```bash
python -m iucn_checker assessment.txt --pretty
```

Sample output:

```
IUCN Rule Checker Report
========================

Violations Found: 3

[WARNING] Line 5, Col 12 (Language)
  Rule: UK English spelling required
  Found: "color"
  Message: Use UK spelling 'colour' instead of 'color'
  Suggestion: colour

[WARNING] Line 8, Col 1 (Numbers)
  Rule: Numbers at sentence start
  Found: "5 species"
  Message: Spell out numbers at the start of a sentence
  Suggestion: Five species

[INFO] Line 12, Col 23 (IUCN Terms)
  Rule: IUCN without article
  Found: "the IUCN"
  Message: Use 'IUCN' without 'the'
  Suggestion: IUCN
```

### CI/CD Integration

```bash
# Exit with non-zero status if errors found
python -m iucn_checker assessment.txt --severity error
if [ $? -eq 2 ]; then
    echo "Assessment contains errors"
    exit 1
fi
```

## Requirements

- Python 3.7 or higher
- No external dependencies
- pytest 7.0+ (for running tests)

## License

This project was developed as part of an academic group project at Imperial College London.
