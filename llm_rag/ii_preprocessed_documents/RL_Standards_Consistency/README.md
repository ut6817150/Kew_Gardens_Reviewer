# RL_Standards_Consistency

## Purpose

This folder contains the preprocessed retrieval assets for:

```text
llm_rag/i_raw_documents/RL_Standards_Consistency.pdf
```

This source supports review-style questions about consistency, common errors,
required fields, rationale quality, mapping fields, and assessment checks. It is
especially useful when the RAG system is asked what should be checked in an
uploaded assessment rather than what a threshold value is.

## Current Manifest Summary

| Metric | Count |
|---|---:|
| Raw text blocks | 1,259 |
| Retrieval blocks | 303 |
| Text retrieval blocks | 253 |
| Table parent blocks | 6 |
| Table row blocks | 44 |
| Fallback table rows | 0 |
| Synthetic table parents | 0 |

## Tables

The current preprocessing run extracted six CSV tables into `tables/`.

The extracted tables include field and coding tables that can support spatial
and consistency-review questions.

See `llm_rag/ii_preprocessed_documents/RL_Standards_Consistency/tables/README.md`
for the table file inventory.

## Retrieval Role

This source is useful for questions such as:

- what common assessment errors should be checked
- whether rationale text justifies the assigned category and criteria
- what supporting information should be clear for Criterion B
- what consistency checks apply to mapping and geographic range fields
- how standards guidance frames required fields and review expectations

It often complements `Required_and_Recommended_Supporting_Information...` and
`Mapping_Standards_Version_1.20_Jan2024`.

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
- `llm_rag/ii_preprocessed_documents/RL_Standards_Consistency/tables/README.md`
