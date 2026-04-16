# Test Word Document

This folder contains a simple notebook-based workflow for testing the
rules-based system on a Word document.

## What This Folder Contains

- `Acrocarpus_fraxinifolius_JP.docx`
  Bundled sample Word document used for the default notebook run.
- `test_word_document.ipynb`
  Notebook that loads a Word document, converts it to a Python dict, parses the resulting dictionary with
  `AssessmentParser`, and then generates violations with
  `IUCNAssessmentReviewer`.

## How It Works

The notebook uses the same two-stage rules-based flow as the rest of the
package:

1. The `.docx` file is first converted to a Python dict using the
   `parse_to_dict` function from `repo root > preprocessing > assessment_processor.py`.
2. `AssessmentParser.parse(...)` converts that assessment dictionary into the
   flat `full_report` mapping used by the reviewer.
3. `IUCNAssessmentReviewer.review_full_report(...)` generates the rule
   violations.
4. `clean_up_violations(...)` is applied before the results are displayed so
   the printed output is easier to read.

## Run The Bundled Sample

Open:

- `test_word_document/test_word_document.ipynb`

Leave:

```python
CUSTOM_DOCX_PATH = None
```

and run the notebook cells in order. The notebook will use:

- `test_word_document/Acrocarpus_fraxinifolius_JP.docx`

## Test Your Own Word Document

Open the same notebook and change the path cell so that `CUSTOM_DOCX_PATH`
points to your own `.docx` file.

Example:

```python
CUSTOM_DOCX_PATH = r"G:\\path\\to\\your_assessment.docx"
```

Then run the notebook cells again. The notebook will:

- load your Word document
- convert it to a Python dict
- build the parsed `full_report`
- generate the rules-based violations
- print the results

## Notes

- The notebook expects the `parse_to_dict` function from
  `repo root > preprocessing > assessment_processor.py` to be available for
  the Word-document conversion step.
- The notebook is written to locate the repository root automatically, so it
  can still work if it is launched from the notebook folder or from the repo
  root.
