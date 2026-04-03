# IUCN Rules Checker

Utilities for parsing an IUCN assessment JSON tree and running rule-based text checks on the parsed content.
This package also powers the repo-level Streamlit interface in `app.py`.

This README documents the code that currently exists in `iucn_rules_checker/`. For rule-by-rule checker behavior, see `checkers/README.md`.

## What This Folder Contains

The package currently has three main responsibilities:

- `assessment_parser.py`
  Converts a hierarchical assessment tree into a flat `section_name -> text` mapping.
- `assessment_reviewer.py`
  Runs the configured checker classes over an already parsed report and returns a list of violations.
- `violation.py`
  Defines the `Violation` dataclass used as the shared output format.

The checker implementations live in `checkers/`, and their regression tests live in `unittests/`.

## Requirements

### Runtime Dependencies

The current Python code in `iucn_rules_checker/` uses only the Python standard library.

External runtime dependencies:

- none

Standard-library modules used in the current codebase include:

- `abc`
- `dataclasses`
- `inspect`
- `json`
- `pathlib`
- `re`
- `typing`
- `unittest`

### Optional Tooling

- Jupyter
  Only needed if you want to open or run `test_json_file/test.ipynb`.

## AssessmentParser

`AssessmentParser` expects a Python `dict` representing the assessment tree used in this project.

Expected structure:

- each node has a `title`
- each node has a `blocks` list
- each node has a `children` list

Current parser behavior:

- walks the tree recursively
- builds keys like:
  - `Title [paragraph 1]`
  - `Title > Section > Subsection [paragraph 2]`
  - `Title > Section [table 1] [row 3]`
- prefers `text_rich` over `text` for paragraph blocks
- prefers `rows_rich` over `rows` for table blocks
- emits one output entry per table row, not one per table
- preserves Unicode characters as-is
- applies style blocks to entries already collected in the same node

Parser input:

- hierarchical assessment `dict`

Parser output:

- `dict[str, str]`
  - keys are parsed section/block paths
  - values are the corresponding paragraph or table-row strings

Example:

```python
import json
from pathlib import Path

from iucn_rules_checker.assessment_parser import AssessmentParser

json_path = Path("iucn_rules_checker/test_json_file/Acianthera odontotepala_draft_status_Jun2025 (1).json")

with json_path.open(encoding="utf-8") as handle:
    assessment = json.load(handle)

full_report = AssessmentParser().parse(assessment)
print(len(full_report))
print(next(iter(full_report.items())))
```

## IUCNAssessmentReviewer

`IUCNAssessmentReviewer` reviews an already parsed report.

The intended workflow is now explicitly two-step:

1. run `AssessmentParser.parse(assessment)` to build a flat `full_report`
2. pass that `full_report` into `IUCNAssessmentReviewer.review_full_report(...)`

Current reviewer behavior:

- skips empty sections
- skips parsed table sections entirely
- routes bibliography sections to a dedicated `BibliographyChecker`
- runs all other non-table sections through the normal checker list
- calls `begin_sweep()` on every checker before a review pass
- calls `end_sweep()` on every checker after the pass finishes

### Section Routing

Normal non-bibliography sections are checked by `self.checkers`:

- `AbbreviationChecker`
- `DateChecker`
- `FormattingChecker`
- `GeographyChecker`
- `IUCNTermsChecker`
- `NumberChecker`
- `PunctuationChecker`
- `ReferenceChecker`
- `ScientificNameChecker`
- `SpellingChecker`
- `SymbolChecker`

Bibliography sections are routed to `self.bibliography_checker` only:

- `BibliographyChecker`

Current `BibliographyChecker` behavior combines:

- `check_ampersand_usage(...)`
- `AbbreviationChecker.check_et_al(...)`
- `PunctuationChecker.check_range_dashes(...)`
- `NumberChecker.check_large_numbers(...)`

Not included in the normal reviewer flow:

- `LanguageChecker`

Example:

```python
import json
from pathlib import Path

from iucn_rules_checker.assessment_parser import AssessmentParser
from iucn_rules_checker.assessment_reviewer import IUCNAssessmentReviewer

json_path = Path("iucn_rules_checker/test_json_file/Acianthera odontotepala_draft_status_Jun2025 (1).json")

with json_path.open(encoding="utf-8") as handle:
    assessment = json.load(handle)

full_report = AssessmentParser().parse(assessment)
reviewer = IUCNAssessmentReviewer()
violations = reviewer.review_full_report(full_report)

print(f"Violations: {len(violations)}")
print(violations[0].to_dict())
```

You can also review an already-flat report:

```python
from iucn_rules_checker.assessment_reviewer import IUCNAssessmentReviewer

full_report = {
    "Assessment > Notes [paragraph 1]": "Examples occur e.g. in text.",
    "Assessment > Bibliography [paragraph 1]": "Cheng, W.J. 1985. Tree flora of china. Vol II.",
}

reviewer = IUCNAssessmentReviewer()
violations = reviewer.review_full_report(full_report)
```

## Violation Output

Each rule hit is returned as a `Violation` object.

Current fields:

- `rule_class`
- `rule_method`
- `matched_text`
- `matched_snippet`
- `message`
- `suggested_fix`
- `section_name`

Meaning of the text-related fields:

- `matched_text`
  The exact span that triggered the rule.
- `matched_snippet`
  A short nearby context snippet.
- `section_name`
  The display section name. Paragraph suffixes such as `[paragraph 2]` are normalized away when violations are created.

Example `to_dict()` output:

```json
{
  "rule_class": "SpellingChecker",
  "rule_method": "SpellingChecker.check_text",
  "section_name": "Assessment > Geographic Range",
  "matched_text": "color",
  "matched_snippet": "the color of the species",
  "message": "Use UK spelling 'colour' instead of 'color'",
  "suggested_fix": "colour"
}
```

## BaseChecker Helpers

All checker classes inherit from `checkers/base.py`.

Shared methods include:

- `check((section_name, text))`
- `begin_sweep()`
- `end_sweep()`
- `strip_style_markers(...)`
- `create_violation(...)`
- `normalize_section_name(...)`
- `get_rule_method_name()`

`strip_style_markers(...)` can selectively remove:

- italic tags: `<i>`, `<em>`
- bold tags: `<b>`, `<strong>`
- superscript tags: `<sup>`
- subscript tags: `<sub>`

It returns:

- cleaned text
- an index map from cleaned positions back to original-text positions

That lets a checker match against normalized text but still create violations against the original rich-text source.

## Checkers

Checker modules currently present in `checkers/`:

- `abbreviations.py`
- `bibliography.py`
- `dates.py`
- `formatting.py`
- `geography.py`
- `iucn_terms.py`
- `numbers.py`
- `punctuation.py`
- `references.py`
- `scientific.py`
- `spelling.py`
- `symbols.py`

See `checkers/README.md` for the method-by-method rule documentation.

## Unit Tests

The unit test suite uses the standard library `unittest` runner.

Tests are designed to be run from the repository root, because they import modules using package paths such as `iucn_rules_checker.assessment_reviewer`.

Run the full suite:

```bash
python -m unittest discover -s iucn_rules_checker/unittests -p "test_*.py"
```

Run one test module:

```bash
python -m unittest iucn_rules_checker.unittests.test_assessment_parser
python -m unittest iucn_rules_checker.unittests.test_assessment_reviewer
```

## Test JSON File

Sample files used for testing and notebook-based inspection now live in `test_json_file/`:

- `test_json_file/Acianthera odontotepala_draft_status_Jun2025 (1).json`
- `test_json_file/test.ipynb`

The notebook is used for:

- fresh imports from disk
- parser checks
- reviewer checks
- JSON-style printing of parsed output and violations

### Running Your Own JSON File

You can run the parser and reviewer on any assessment JSON file that follows the
same tree structure expected by `AssessmentParser`.

#### From A Python Script

```python
import json
from pathlib import Path

from iucn_rules_checker.assessment_parser import AssessmentParser
from iucn_rules_checker.assessment_reviewer import IUCNAssessmentReviewer

json_path = Path(r"PATH\\TO\\YOUR\\ASSESSMENT.json")

with json_path.open(encoding="utf-8") as handle:
    assessment = json.load(handle)

parser = AssessmentParser()
full_report = parser.parse(assessment)

reviewer = IUCNAssessmentReviewer()
violations = reviewer.review_full_report(full_report)

for violation in violations:
    print(violation.to_dict())
```

#### From The Test Notebook

Open `test_json_file/test.ipynb` and edit the JSON path cell:

- leave `CUSTOM_JSON_PATH = None` to use the bundled sample JSON
- set `CUSTOM_JSON_PATH` to your own file path to run a different assessment

Example:

```python
CUSTOM_JSON_PATH = r"G:\\path\\to\\your_assessment.json"
```

The rest of the notebook will then load that file, parse it, and run the
reviewer against the resulting `full_report`.

## Project Structure

```text
iucn_rules_checker/
|- checkers/
|  |- README.md
|  |- abbreviations.py
|  |- base.py
|  |- bibliography.py
|  |- dates.py
|  |- formatting.py
|  |- geography.py
|  |- iucn_terms.py
|  |- numbers.py
|  |- punctuation.py
|  |- references.py
|  |- scientific.py
|  |- spelling.py
|  `- symbols.py
|- test_json_file/
|  |- Acianthera odontotepala_draft_status_Jun2025 (1).json
|  `- test.ipynb
|- unittests/
|  |- test_abbreviations.py
|  |- test_bibliography.py
|  |- test_assessment_parser.py
|  |- test_assessment_reviewer.py
|  |- test_base.py
|  |- test_dates.py
|  |- test_formatting.py
|  |- test_geography.py
|  |- test_iucn_terms.py
|  |- test_numbers.py
|  |- test_punctuation.py
|  |- test_references.py
|  |- test_scientific.py
|  |- test_spelling.py
|  `- test_symbols.py
|- .gitignore
|- README.md
|- assessment_parser.py
|- assessment_reviewer.py
`- violation.py
```

## Adding Or Updating Rules

### Add A Rule To An Existing Checker

1. Open the relevant checker module under `checkers/`.
2. Add the logic inside `check_text(...)` or a helper method it calls.
3. Create violations with `self.create_violation(...)`.
4. Add regression coverage in the matching file under `unittests/`.

Example:

```python
violations.append(
    self.create_violation(
        section_name=section_name,
        text=text,
        span=match.span(),
        message="Describe the problem here",
        suggested_fix="corrected text",
    )
)
```

### Add A New Checker

1. Create a new checker under `checkers/` as a `BaseChecker` subclass.
2. Implement `check_text(...)`.
3. Wire it into `IUCNAssessmentReviewer.__init__`:
   add it to `self.checkers` for normal sections, or compose it into `self.bibliography_checker` if it should only run on bibliography content.
4. Add tests under `unittests/`.

Minimal skeleton:

```python
from typing import List

from .base import BaseChecker
from ..violation import Violation


class MyChecker(BaseChecker):
    def check_text(self, section_name: str, text: str) -> List[Violation]:
        violations: List[Violation] = []
        return violations
```

## Related Documentation

- `checkers/README.md`
  Detailed checker and rule documentation.

## License

Developed as part of an academic group project at Imperial College London.
