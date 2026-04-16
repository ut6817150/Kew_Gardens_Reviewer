# ii_preprocessed_documents

## Purpose

This folder holds the intermediate assets produced from the raw IUCN PDFs before the reference index is built.

The goal is retrieval quality, not perfect PDF reconstruction. The preprocessing step tries to preserve:

- page provenance
- section structure
- narrative blocks
- table parents
- table rows

That structure is what makes later supporting-information retrieval much stronger than plain text chunking.

## Input and output locations

Input PDFs live in:

```text
llm_rag/i_raw_documents/
```

This folder receives the outputs:

```text
llm_rag/ii_preprocessed_documents/
```

Each source document gets its own subfolder with:

- `raw_page_blocks.jsonl`
- `retrieval_blocks.jsonl`
- `manifest.json`
- `tables/*.csv` when standard extraction succeeds

The folder also contains a corpus-wide `summary.json`.

## Main script

`preprocess_pdfs.py` is the entry point for this stage.

It now resolves paths relative to the `llm_rag` folder, so it reads from `i_raw_documents/` and writes back into this directory consistently.

Run it from the repo root with:

```bash
python llm_rag/ii_preprocessed_documents/preprocess_pdfs.py
```

## What the preprocessing script does

### 1. Extract page blocks

The script uses PyMuPDF to pull out block-level text and layout hints such as page number, bounding box, font size, and boldness.

### 2. Remove repeated noise

Repeated headers, footers, and boilerplate are suppressed when they look like cross-page edge text rather than document content.

### 3. Recover section structure

Heuristics are used to detect headings and build a best-effort section path that can be carried forward into retrieval metadata.

### 4. Build text retrieval blocks

Narrative content is merged into retrieval-ready `text` blocks with source, page, block type, and section metadata.

### 5. Extract tables

The script uses `pdfplumber` to extract standard tables when possible. For each extracted table it creates:

- a parent `table` block
- child `table_row` blocks
- a CSV export in `tables/`

### 6. Recover difficult requirement tables

The supporting-information PDF includes important visible tables that are often not encoded as real tables in the PDF structure.

For that case the script creates synthetic fallback table records for Table 1, Table 2, and Table 3 by reading flattened page text and reconstructing:

- parent `table` blocks
- child `table_row` blocks

### 7. Reduce overlap

Because narrative extraction and table extraction can both capture similar text, the script reduces duplication by preferring:

- `table_row` over `table`
- `table` over `text`

This keeps the downstream corpus cleaner.

## Retrieval block types

- `text`: narrative prose
- `table`: parent table context
- `table_row`: row-level table evidence

The `table_row` records are especially important for questions about required supporting information.

## What to inspect after running

Useful files to spot-check are:

1. per-document `manifest.json`
2. per-document `retrieval_blocks.jsonl`
3. the corpus-level `summary.json`

For the supporting-information PDF, the main sanity check is that synthetic table parents and row-level entries were created successfully.

## Limitations

- heading detection is heuristic
- some row text still contains PDF extraction artifacts
- the fallback table logic is tailored to the current IUCN PDFs, not to arbitrary documents
