# RedListGuidelines

## Purpose

This folder contains the preprocessed retrieval assets for:

```text
llm_rag/i_raw_documents/RedListGuidelines.pdf
```

This is one of the most important reference sources in the RAG corpus. It
contains detailed guidance for applying the IUCN Red List Criteria, including
definitions, examples, interpretation guidance, uncertainty, continuing
decline, locations, generation length, population reduction, EOO, AOO, and
criterion-specific edge cases.

## Current Manifest Summary

| Metric | Count |
|---|---:|
| Raw text blocks | 1,033 |
| Retrieval blocks | 428 |
| Text retrieval blocks | 348 |
| Table parent blocks | 10 |
| Table row blocks | 70 |
| Fallback table rows | 0 |
| Synthetic table parents | 0 |

## Tables

The current preprocessing run extracted 10 CSV tables into `tables/`.

These tables are mostly example or interpretation tables rather than the compact
criteria summary tables. They help retrieval when questions refer to population
reduction examples, subpopulation calculations, biological characteristics, or
other guideline examples.

See `llm_rag/ii_preprocessed_documents/RedListGuidelines/tables/README.md` for
the table file inventory.

## Retrieval Role

This source is useful for questions such as:

- how to apply Criterion B
- how severe fragmentation and number of locations should be interpreted
- how continuing decline should be assessed
- how AOO and EOO are defined and calculated
- how population reduction and generation length should be handled
- what uncertainty or inference can be used in an assessment rationale

Because this document is broad, the retriever may return it for many questions.
The reranker and query-decomposition logic try to keep those broad hits focused
by also retrieving more compact criteria and supporting-information sources when
needed.

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
- `llm_rag/ii_preprocessed_documents/RedListGuidelines/tables/README.md`
