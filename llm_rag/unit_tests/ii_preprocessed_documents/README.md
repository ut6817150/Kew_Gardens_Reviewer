# unit_tests/ii_preprocessed_documents

## Purpose

This folder contains unit tests for
`llm_rag/ii_preprocessed_documents/preprocess_pdfs.py`.

The tests focus on the preprocessing helpers that convert raw PDF content into
retrieval-ready JSONL records.

## Latest Test Result

The latest full `llm_rag` unit-test run passed:

```text
127 passed in 9.60s
```

This folder contributes 25 test functions in one test module.

## Test Module

### `test_preprocess_pdfs_unit.py`

This module tests:

- retrieval-record construction
- text cleanup
- repeated header and footer normalization
- heading detection and heading-level inference
- section-path updates
- contextualized retrieval text formatting
- layout block extraction with mocked PDF objects
- repeated edge-text detection
- text-block merging
- table matrix normalization
- table header construction
- row serialization into `header: value` pairs
- nearest-section lookup for tables
- table-title inference
- page-text extraction for fallback parsing
- supporting-information table detection
- flattened table-schema detection
- enumerated-entry splitting
- synthetic supporting-information fallback rows
- standard table extraction
- duplicate text-record removal when table records already cover the content
- JSONL writing
- single-PDF processing
- corpus-level preprocessing entry point

## Why These Tests Matter

Preprocessing mistakes propagate into every downstream RAG stage. If a table row
is dropped, a heading is misread, or contextualized text is malformed, the
vector DB can still build successfully but retrieval quality may degrade.

These tests protect the small helper functions that make preprocessing
repeatable and debuggable.

## Running These Tests

From the repo root:

```bash
python -m pytest -q llm_rag/unit_tests/ii_preprocessed_documents
```

Run the module directly:

```bash
python -m pytest -q llm_rag/unit_tests/ii_preprocessed_documents/test_preprocess_pdfs_unit.py
```

## Related Code

- `llm_rag/ii_preprocessed_documents/preprocess_pdfs.py`
  Code under test.
- `llm_rag/ii_preprocessed_documents/README.md`
  Documentation for the preprocessing stage.

## Related Documentation
- `llm_rag/ii_preprocessed_documents/README.md`
- `llm_rag/i_raw_documents/README.md`
- `llm_rag/iii_vector_db/README.md`
