# RL_categories_and_criteria

## Purpose

This folder contains the preprocessed retrieval assets for:

```text
llm_rag/i_raw_documents/RL_categories_and_criteria.pdf
```

This document provides formal Red List category and criteria language. It is a
core reference for questions about category definitions, criteria structure,
and official threshold wording.

## Current Manifest Summary

| Metric | Count |
|---|---:|
| Raw text blocks | 402 |
| Retrieval blocks | 120 |
| Text retrieval blocks | 97 |
| Table parent blocks | 6 |
| Table row blocks | 17 |
| Fallback table rows | 0 |
| Synthetic table parents | 0 |

## Tables

The current preprocessing run extracted six CSV tables into `tables/`.

Some table text from this PDF is visually complex and can be extracted in
unusual order. The deterministic threshold lookup in
`llm_rag/iii_vector_db/thresholds.json` is therefore also used for stable
numeric threshold answers.

See `llm_rag/ii_preprocessed_documents/RL_categories_and_criteria/tables/README.md`
for the table file inventory.

## Retrieval Role

This source is useful for questions such as:

- what the official Red List categories mean
- how criteria A-E are structured
- what formal criteria language applies to a category
- where a threshold statement originates

For exact numeric threshold answers, the runtime may combine this source with
deterministic lookup rather than relying only on free-text retrieval.

## Files In This Folder

- `raw_page_blocks.jsonl`
  Raw layout-aware text blocks extracted from the PDF.
- `retrieval_blocks.jsonl`
  Text, table-parent, and table-row records used by the vector-db build step.
- `manifest.json`
  Counts and preprocessing metadata for this document.
- `tables/`
  CSV exports of extracted tables.
- `README.md`
  This document.

## Related Documentation
- `llm_rag/ii_preprocessed_documents/README.md`
- `llm_rag/i_raw_documents/README.md`
- `llm_rag/iii_vector_db/README.md`
- `llm_rag/ii_preprocessed_documents/RL_categories_and_criteria/tables/README.md`
