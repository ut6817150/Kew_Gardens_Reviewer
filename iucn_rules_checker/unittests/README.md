# Unit Tests

This folder contains the `unittest` regression suite for the
`iucn_rules_checker` package.

## Files In This Folder

Core files in `iucn_rules_checker/unittests/`:

- `test_base.py`
  Tests the shared `BaseChecker` helpers.
- `test_assessment_parser.py`
  Tests `AssessmentParser`.
- `test_assessment_reviewer.py`
  Tests `IUCNAssessmentReviewer`.
- `test_checker_helpers.py`
  Tests checker helper utilities and smaller shared behaviors.
- `test_abbreviations.py`
  Tests `AbbreviationChecker`.
- `test_bibliography.py`
  Tests `BibliographyChecker`.
- `test_dates.py`
  Tests `DateChecker`.
- `test_formatting.py`
  Tests `FormattingChecker`.
- `test_geography.py`
  Tests `GeographyChecker`.
- `test_iucn_terms.py`
  Tests `IUCNTermsChecker`.
- `test_numbers.py`
  Tests `NumberChecker`.
- `test_punctuation.py`
  Tests `PunctuationChecker`.
- `test_references.py`
  Tests `ReferenceChecker`.
- `test_scientific.py`
  Tests `ScientificNameChecker`.
- `test_spelling.py`
  Tests `SpellingChecker`.
- `test_symbols.py`
  Tests `SymbolChecker`.
- `test_tables.py`
  Tests `TableChecker`.
- `README.md`
  This document.

## Where To Run The Tests From

The unit tests are intended to be run from the repository root:

- `IUCN_Reviewer/`

That is the folder that contains:

- `app.py`
- `iucn_rules_checker/`

The tests use package imports such as:

- `iucn_rules_checker.assessment_reviewer`
- `iucn_rules_checker.checkers.abbreviations`

so running them from the repository root ensures those imports resolve
consistently.

## Run The Full Suite

```bash
python -m unittest discover -s iucn_rules_checker/unittests -p "test_*.py"
```

## Run One Test Module

```bash
python -m unittest iucn_rules_checker.unittests.test_assessment_parser
python -m unittest iucn_rules_checker.unittests.test_assessment_reviewer
```
