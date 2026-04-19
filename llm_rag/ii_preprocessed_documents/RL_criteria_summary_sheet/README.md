# RL_criteria_summary_sheet

## Purpose

This folder contains the preprocessed retrieval assets for:

```text
llm_rag/i_raw_documents/RL_criteria_summary_sheet.pdf
```

This compact source summarizes the Red List criteria and thresholds. It is
valuable because it provides concise table-like evidence for Criteria A-E, which
can be easier to retrieve than longer guideline prose.

## Current Manifest Summary

| Metric | Count |
|---|---:|
| Raw text blocks | 65 |
| Retrieval blocks | 37 |
| Text retrieval blocks | 8 |
| Table parent blocks | 5 |
| Table row blocks | 24 |
| Fallback table rows | 0 |
| Synthetic table parents | 0 |

## Tables

The current preprocessing run extracted five CSV tables into `tables/`, one for
each broad criteria group A-E.

See `llm_rag/ii_preprocessed_documents/RL_criteria_summary_sheet/tables/README.md`
for the table file inventory.

## Retrieval Role

This source is useful for concise answers about:

- population reduction thresholds
- geographic range thresholds
- small population size and decline
- very small or restricted populations
- quantitative analysis thresholds

The deterministic threshold lookup still handles many direct threshold prompts,
but these rows provide supporting reference context for hybrid RAG answers.

## Files In This Folder

- `raw_page_blocks.jsonl`
  Raw layout-aware text blocks extracted from the PDF.
- `retrieval_blocks.jsonl`
  Text, table-parent, and table-row records used by the vector-db build step.
- `manifest.json`
  Counts and preprocessing metadata for this document.
- `tables/`
  CSV exports of extracted criteria summary tables.
- `README.md`
  This document.

## Related Documentation
- `llm_rag/ii_preprocessed_documents/README.md`
- `llm_rag/i_raw_documents/README.md`
- `llm_rag/iii_vector_db/README.md`
- `llm_rag/ii_preprocessed_documents/RL_criteria_summary_sheet/tables/README.md`
