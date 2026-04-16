# Test Preprocessing

This folder contains notebooks and fixture files for testing the preprocessing parser in `../assessment_processor.py`.

## Contents

### `test_output_formate.ipynb`

Manual output inspection notebook.

- Purpose: parse one `.docx`, `.html`, or `.htm` file using `parse_to_dict`
- Input: set `DOCUMENT_PATH` inside the notebook
- Output: prints the resulting parsed dictionary in standard formatted JSON
- Use this when you want to quickly inspect what the parser produces for one document

### `unit_tests.ipynb`

Function-level and behavior-level test notebook.

- Purpose: test key `AssessmentParser` helpers and parser behavior in small controlled examples
- Output: each test prints `PASS` or `FAIL`
- Fixture style: creates temporary DOCX and HTML examples inside the notebook
- Coverage includes:
  - rich-text wrapping
  - paragraph rich-text extraction
  - style bucket merging
  - heading detection
  - table extraction
  - full DOCX parsing
  - DOCX files without comments
  - HTML parsing
  - HTML style extraction
  - `.htm` extension support
  - `parse_to_dict`
  - raw XML run rendering
  - unsupported file extensions

### `evaluation_tests.ipynb`

Parser-output evaluation notebook.

- Purpose: evaluate the final parsed JSON dictionary using output-level metrics
- Output: each metric prints `PASS` or `FAIL` plus its score
- Fixture in this folder: `Myrcia neosmithii_draft_status_Apr2022_v2.docx`
- Note: evaluation testing has been run across 110 documents in total.

The evaluation metrics include:

- text bigram similarity between the source DOCX text and parsed JSON text
- schema completeness
- heading tree recall
- top-level heading order accuracy
- block type distribution accuracy
- plain-text exact-match recall
- rich-text formatting recall
- rich-text tag coverage
- table cell exact-match accuracy
- style feature recall
- comment output match
- overall parser evaluation score

### `Myrcia neosmithii_draft_status_Apr2022_v2.docx`

Sample DOCX fixture used by the notebooks.

## Testing Setup

Run the notebooks from this folder so relative paths resolve correctly:

```bash
cd preprocessing/test_preprocessing
```

The notebooks import `assessment_processor.py` by walking up the parent directories until the parser file is found. This means they should still work when opened from Jupyter, as long as the repository structure remains the same.

Required Python packages:

- `python-docx`
- `beautifulsoup4`
- `IPython`
- Jupyter or a compatible notebook environment

## How To Run

### Manual Output Inspection

1. Open `test_output_formate.ipynb`
2. Set `DOCUMENT_PATH` to the document you want to parse
3. Run all cells
4. Inspect the printed JSON dictionary

### Unit Tests

1. Open `unit_tests.ipynb`
2. Run all cells
3. Confirm every test prints `PASS`

### Evaluation Tests

1. Open `evaluation_tests.ipynb`
2. Confirm the sample document path 
3. Run all cells
4. Review each metric score and the overall parser evaluation score

## Interpreting Results

- `PASS` means the test or metric matched the expected parser output for the current fixture.
- `FAIL` means the parser output changed or the expected value in the notebook needs to be reviewed.
- The evaluation metrics are intended as regression checks for the final JSON dictionary, not as replacements for manual review of difficult documents.
- The bigram metric is useful for detecting text-loss regressions because it compares the text content in the source Word document with the text content preserved in the parsed JSON.
