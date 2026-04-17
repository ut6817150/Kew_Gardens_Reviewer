# Evaluation

This folder records how the rules-based system was evaluated, audited, and
refined over time.

It contains:

- the results of an initial large-scale sweep across Kew-prepared assessments
- a custom Word document used for targeted rule testing
- a notebook for running checker-by-checker evaluations on that custom document

## Initial Sweep

The rules-based system was first swept through `109` assessments prepared by
Kew. This first evaluation pass was carried out with the help of an LLM.

The results of that sweep are stored in:

- `IUCN_submissions_evaluation.xlsx`

That workbook was used to identify false positives, false negatives, and rule
areas that needed refinement. The rules-based system was then updated based on
those findings.

## Targeted Refinement

After the initial sweep, the rules-based system was refined further using a
custom Word document:

- `test_doc_rules_based.docx`

That document is paired with:

- `evaluation.ipynb`

The notebook is intended for focused evaluation of specific checker behavior.
It is especially useful when we want to confirm:

- which parsed sections a checker is actually running on
- which violations are produced for a known test document
- whether a rule catches intended errors but ignores intended exceptions

## Notebook Workflow

`evaluation.ipynb` runs the following workflow:

1. It sets the evaluation folder as the working context and derives the repo
   root from there.
2. It loads `parse_to_dict` from:
   `repo root > preprocessing > assessment_processor.py`
3. It converts the Word document into a Python assessment dictionary.
4. It runs `AssessmentParser.parse(...)` to create the `full_report`
   section-to-text mapping used by the rules-based system.
5. It reloads the checker modules fresh so notebook runs reflect recent code
   changes.
6. It runs each checker only on parsed sections whose section name contains the
   script name associated with that checker.
7. It prints:
   - the matching parsed sections
   - the violation count
   - the cleaned violation payload for that checker

This means the notebook is not just running the full reviewer end to end. It is
also acting as a targeted checker-inspection tool.

## Checkers Covered In The Notebook

The notebook currently includes dedicated cells for:

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
- `TableChecker`
- `BibliographyChecker`

The `TableChecker` cell is important because table-derived content is handled
separately from the main checker pipeline in `IUCNAssessmentReviewer`.

## Using The Bundled Evaluation Document

By default, the notebook uses:

- `test_doc_rules_based.docx`

This document contains deliberately chosen examples for the checker scripts so
that rule behavior can be inspected section by section.

To use the bundled document:

1. Open `evaluation.ipynb`.
2. Run the setup cells first.
3. Leave `CUSTOM_DOCX_PATH = None`.
4. Run the remaining cells in order.

## Using Your Own Word Document

You can also point the notebook at your own `.docx` file.

In the configuration cell near the top of the notebook, set:

- `CUSTOM_DOCX_PATH`

to either:

- a string path to your `.docx` file
- or a `Path(...)` object pointing to your `.docx` file

Then rerun the notebook from the top so that:

- the Word document is reconverted with `parse_to_dict(...)`
- `full_report` is rebuilt from the new input
- the checker outputs refresh against the new parsed sections

## Notes On Running The Notebook

- The notebook assumes `preprocessing/assessment_processor.py` exists at the
  repo root.
- It expects to load a Word document, not a prebuilt JSON file.
- The checker cells depend on parsed section names containing script labels
  such as `abbreviations.py`, `tables.py`, or `bibliography.py`.
- The notebook reloads `iucn_rules_checker` modules before each checker run so
  recent code changes are picked up more reliably inside Jupyter.

## Files In This Folder

- `evaluation.ipynb`
  Targeted evaluation notebook for running checker-specific inspections against
  a Word document.
- `test_doc_rules_based.docx`
  Custom Word document used for targeted rules-based refinement.
- `IUCN_submissions_evaluation.xlsx`
  Workbook containing the results of the initial `109`-assessment sweep.
- `README.md`
  This document.
