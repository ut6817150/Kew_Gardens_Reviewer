# Mapping_Standards_Version_1.20_Jan2024

## Purpose

This folder contains the preprocessed retrieval assets for:

```text
llm_rag/i_raw_documents/Mapping_Standards_Version_1.20_Jan2024.pdf
```

This document is the main reference source for Red List mapping and spatial-data
questions. It is especially important for review prompts about map status,
georeferenced distribution data, polygon and point attributes, PRESENCE,
ORIGIN, SEASONALITY, EOO, AOO, and spatial-data consistency.

## Current Manifest Summary

| Metric | Count |
|---|---:|
| Raw text blocks | 380 |
| Retrieval blocks | 200 |
| Text retrieval blocks | 118 |
| Table parent blocks | 10 |
| Table row blocks | 72 |
| Fallback table rows | 0 |
| Synthetic table parents | 0 |

## Tables

The current preprocessing run extracted 10 CSV tables into `tables/`.

The extracted tables include mapping attribute tables and coding tables used by
the retriever for spatial-review questions. Some headers are imperfect because
they come from PDF table extraction, but the records still preserve useful
row-level evidence for retrieval.

See `llm_rag/ii_preprocessed_documents/Mapping_Standards_Version_1.20_Jan2024/tables/README.md`
for the table file inventory.

## Retrieval Role

This source is useful for questions such as:

- what spatial data are required for assessments using Criteria B or D2
- whether a map status is sufficient
- what attributes should exist in point or polygon spatial data
- how PRESENCE, ORIGIN, and SEASONALITY codes affect mapped range
- how EOO and AOO should be calculated or interpreted

It is not the only source for spatial review. The RAG system may also retrieve
`RL_Standards_Consistency` and `RedListGuidelines` when a question asks about
assessment consistency or Criterion B interpretation.

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
- `llm_rag/ii_preprocessed_documents/Mapping_Standards_Version_1.20_Jan2024/tables/README.md`
