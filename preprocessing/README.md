# Preprocessing

This folder contains the preprocessing parser used to convert assessment files into structured JSON.

## Contents

### `assessment_processor.py`

This is the main preprocessing script.

- Purpose: parse assessment documents into a structured Python dictionary / JSON output
- Supported input file types: `.docx`, `.html`, `.htm`
- Output:
  - batch mode: one `.json` file per input document
  - single-document mode: JSON printed to stdout
- Default input folder variable: `DEFAULT_INPUT_FOLDER`
- Default output folder variable: `DEFAULT_OUTPUT_FOLDER`

What it extracts:

- heading structure
- paragraphs
- list blocks
- tables
- comments and comment anchor context for DOCX files
- rich-text fields where available

## How To Run

Run from inside `preprocessing/`:

```bash
python3.12 assessment_processor.py
```

### Batch Mode

Batch mode runs when you call the script with no arguments.

```bash
python3.12 assessment_processor.py
```

Behavior:

- reads every `.docx`, `.html`, and `.htm` file in the folder pointed to by `DEFAULT_INPUT_FOLDER`
- creates the folder pointed to by `DEFAULT_OUTPUT_FOLDER` if it does not already exist
- saves one JSON file per input document into the folder pointed to by `DEFAULT_OUTPUT_FOLDER`
- writes `_errors.json` into the folder pointed to by `DEFAULT_OUTPUT_FOLDER`

Use batch mode when you want to process a whole folder at once.

Batch workflow:

1. Put your `.docx`, `.html`, or `.htm` files in the folder pointed to by `DEFAULT_INPUT_FOLDER`
2. Run `python3.12 assessment_processor.py`
3. Open the generated JSON files in the folder pointed to by `DEFAULT_OUTPUT_FOLDER`
4. If anything fails, inspect `_errors.json` in the folder pointed to by `DEFAULT_OUTPUT_FOLDER`

### Single-Document Mode

Single-document mode runs when you pass one supported file path to the script.

```bash
python3.12 assessment_processor.py "<path-to-file>.docx"
```

You can also pass HTML:

```bash
python3.12 assessment_processor.py "<path-to-file>.html"
```

Behavior:

- parses only the one file you pass in
- prints the JSON to stdout
- does not automatically save a `.json` file

If you want to save the result manually:

```bash
python3.12 assessment_processor.py "<path-to-file>.docx" > "output.json"
```

## Test Notebooks

The folder `test_preprocessing/` contains notebooks for manual testing, unit testing, and evaluation testing.

### `test_output_formate.ipynb`

Use this notebook to quickly inspect parser output for a single document.

- purpose: choose a document path, run `parse_to_dict`, and print the parsed dictionary in standard JSON format
- input: set `DOCUMENT_PATH` inside the notebook
- output: formatted JSON printed in the notebook

### `unit_tests.ipynb`

Use this notebook to test individual parser functions and expected parser behavior.

- purpose: run focused tests for rich-text rendering, table extraction, heading detection, DOCX parsing, HTML parsing, and error handling
- output: each test section prints `PASS` or `FAIL`
- note: the tests use small temporary DOCX and HTML fixtures created inside the notebook

### `evaluation_tests.ipynb`

Use this notebook to evaluate the final parsed JSON dictionary using parser-output metrics.

- purpose: parse the sample document and score the output dictionary using evaluation metrics
- sample file: `Myrcia neosmithii_draft_status_Apr2022_v2.docx`
- metrics include schema completeness, heading tree recall, block type distribution accuracy, plain text exact-match recall, rich-text formatting recall, table cell exact-match accuracy, style feature recall, comment output match, and overall parser evaluation score
- output: each metric prints `PASS` or `FAIL` plus the metric score

## Recommended Process

### If your source files are already `.docx` or `HTML`

1. Put the files in the folder pointed to by `DEFAULT_INPUT_FOLDER`
2. Run `python3.12 assessment_processor.py` for batch parsing

### If you want to inspect just one file

1. Keep the document wherever you want
2. Run `python3.12 assessment_processor.py "<path-to-file>"`
3. Review the printed JSON directly in the terminal

### If you want to test interactively

1. Open `test_preprocessing/test_output_formate.ipynb`
2. Set `DOCUMENT_PATH`
3. Run the notebook cells

### If you want to run tests

1. Open `test_preprocessing/unit_tests.ipynb`
2. Run all cells
3. Check that every test prints `PASS`

### If you want to run parser-output evaluation metrics

1. Open `test_preprocessing/evaluation_tests.ipynb`
2. Run all cells
3. Review each printed score and the overall parser evaluation score

