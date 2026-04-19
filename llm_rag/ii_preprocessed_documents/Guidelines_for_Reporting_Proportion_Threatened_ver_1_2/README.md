# Guidelines_for_Reporting_Proportion_Threatened_ver_1_2

## Purpose

This folder contains the preprocessed retrieval assets for:

```text
llm_rag/i_raw_documents/Guidelines_for_Reporting_Proportion_Threatened_ver_1_2.pdf
```

The document supports questions about reporting proportions of threatened taxa
and appropriate use of Red List data. It is a smaller source document than most
of the other reference PDFs, and its current preprocessing output is entirely
text-based.

## Current Manifest Summary

| Metric | Count |
|---|---:|
| Raw text blocks | 38 |
| Retrieval blocks | 14 |
| Text retrieval blocks | 14 |
| Table parent blocks | 0 |
| Table row blocks | 0 |
| Fallback table rows | 0 |
| Synthetic table parents | 0 |

## How To Interpret This Output

No tables were extracted from this document in the current preprocessing run.
That is expected for the current asset set. The retrieval contribution is made
through contextualized text blocks with page and section provenance.

This source is most useful when a RAG question asks about:

- reporting threatened proportions
- appropriate use of Red List data
- high-level Red List reporting language

It is not the main source for criteria thresholds, supporting-information table
requirements, or spatial mapping standards.

## Files In This Folder

- `raw_page_blocks.jsonl`
  Raw layout-aware text blocks extracted from the PDF.
- `retrieval_blocks.jsonl`
  Contextualized text records used by the vector-db build step.
- `manifest.json`
  Counts and preprocessing metadata for this document.
- `README.md`
  This document.

## Related Documentation
- `llm_rag/ii_preprocessed_documents/README.md`
- `llm_rag/i_raw_documents/README.md`
- `llm_rag/iii_vector_db/README.md`
