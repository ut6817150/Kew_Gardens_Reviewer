# Unit Tests

This folder contains the `unittest` regression suite for the
`iucn_rules_checker` package.

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
