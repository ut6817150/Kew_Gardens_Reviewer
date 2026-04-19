# ii_preprocessed_documents

## Purpose

`llm_rag/ii_preprocessed_documents/` contains the structured intermediate
outputs built from the raw reference PDFs in `llm_rag/i_raw_documents/`.

This stage turns PDFs into retrieval-ready records. It is the bridge between
raw documents and the vector-db build step. The outputs are intentionally stored
as plain JSONL, JSON, and CSV files so they can be inspected, tested, and
rebuilt without running the Streamlit app.

## Main Script

`preprocess_pdfs.py` is the preprocessing entry point.

Run it from the repo root:

```bash
python llm_rag/ii_preprocessed_documents/preprocess_pdfs.py
```

The script reads every PDF in `llm_rag/i_raw_documents/` and writes one
processed subfolder per PDF.

## What Preprocessing Does

The preprocessing pipeline:

1. extracts layout-aware text blocks with page numbers, bounding boxes, font
   size cues, and simple bold detection
2. normalizes whitespace and PDF punctuation artifacts
3. detects repeated running headers or footers
4. identifies likely headings and maintains section paths
5. merges narrative text into retrieval-sized text records
6. extracts standard PDF tables with `pdfplumber`
7. serializes extracted tables as parent table records and row-level records
8. reconstructs difficult supporting-information tables with fallback logic
9. removes obvious duplicated text when table-derived records already cover the
   same content
10. writes per-document manifests and a corpus-level `summary.json`

The output is designed for retrieval, not for recreating the original PDF
layout exactly.

## Output Files

Each processed document folder usually contains:

| File | Purpose |
|---|---|
| `raw_page_blocks.jsonl` | Raw layout-aware text blocks extracted from the PDF before retrieval-specific merging. |
| `retrieval_blocks.jsonl` | Contextualized retrieval records consumed by the vector-db build step. |
| `manifest.json` | Per-document counts and preprocessing metadata. |
| `tables/*.csv` | CSV exports of extracted tables when standard table extraction succeeds. |

The corpus-level file is:

| File | Purpose |
|---|---|
| `summary.json` | List of all per-document manifest summaries. Useful for checking the whole preprocessing output at a glance. |

## Retrieval Record Shape

`retrieval_blocks.jsonl` records include provenance and retrieval fields such as:

- `doc_id`
- `source_file`
- `page`
- `block_type`
- `section_path`
- `section_title`
- `table_id`
- `table_title`
- `row_id`
- `text`
- `metadata`

The `text` field is contextualized. It includes source, page, block type, table
or section context, and the actual content. This gives both sparse and dense
retrieval more useful cues than raw extracted text alone.

## Current Corpus Summary

The current preprocessed corpus contains 1,170 retrieval records across seven
source PDFs.

| Document Folder | Raw Blocks | Retrieval Blocks | Text Blocks | Table Parents | Table Rows | Fallback Rows | Synthetic Parents |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Guidelines_for_Reporting_Proportion_Threatened_ver_1_2/` | 38 | 14 | 14 | 0 | 0 | 0 | 0 |
| `Mapping_Standards_Version_1.20_Jan2024/` | 380 | 200 | 118 | 10 | 72 | 0 | 0 |
| `RedListGuidelines/` | 1,033 | 428 | 348 | 10 | 70 | 0 | 0 |
| `Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments/` | 186 | 68 | 27 | 3 | 38 | 38 | 3 |
| `RL_categories_and_criteria/` | 402 | 120 | 97 | 6 | 17 | 0 | 0 |
| `RL_criteria_summary_sheet/` | 65 | 37 | 8 | 5 | 24 | 0 | 0 |
| `RL_Standards_Consistency/` | 1,259 | 303 | 253 | 6 | 44 | 0 | 0 |

The supporting-information document is the only current source that uses
synthetic fallback table recovery. Its visible tables are important for RAG
answers, but they are not reliably captured as conventional PDF tables.

## Relationship To The Vector DB

The next stage is:

```bash
python llm_rag/iii_vector_db/build_reference_db.py --reset
```

That script reads `retrieval_blocks.jsonl` files from this folder and produces:

- `llm_rag/iii_vector_db/reference_corpus.jsonl`
- `llm_rag/iii_vector_db/parent_contexts.jsonl`
- `llm_rag/iii_vector_db/build_summary.json`
- `llm_rag/iii_vector_db/chroma_db/reference_docs/`

If preprocessing changes, rebuild the vector-db assets because chunk contents,
metadata, and row-level records may all change.

## Quality Checks

After preprocessing, check:

- `summary.json` for unexpected count changes
- each `manifest.json` for missing table rows or sudden block-count drops
- `tables/*.csv` files for malformed extraction
- `llm_rag/evaluation/smoke_and_inspection/inspect_preprocessed_docs.ipynb`
  for notebook-based inspection

The unit tests for this stage are in:

```text
llm_rag/unit_tests/ii_preprocessed_documents/
```

The latest full unit-test run passed:

```text
127 passed in 9.60s
```

## Files In This Folder

- `preprocess_pdfs.py`
  Preprocessing script for all raw reference PDFs.
- `summary.json`
  Corpus-level preprocessing summary.
- one subfolder per source PDF
  Per-document raw blocks, retrieval blocks, manifest, and optional tables.
- `README.md`
  This document.

## Related Documentation
- `llm_rag/README.md`
- `llm_rag/i_raw_documents/README.md`
- `llm_rag/iii_vector_db/README.md`
- `llm_rag/evaluation/smoke_and_inspection/README.md`
- `llm_rag/unit_tests/ii_preprocessed_documents/README.md`
