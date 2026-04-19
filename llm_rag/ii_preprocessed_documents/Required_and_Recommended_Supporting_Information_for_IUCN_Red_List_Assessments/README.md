# Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments

## Purpose

This folder contains the preprocessed retrieval assets for:

```text
llm_rag/i_raw_documents/Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf
```

This document is the main source for questions about required and recommended
supporting information in Red List assessments. It is especially important for
checking whether an uploaded assessment contains the fields, narratives, and
evidence expected for a threatened assessment.

## Current Manifest Summary

| Metric | Count |
|---|---:|
| Raw text blocks | 186 |
| Retrieval blocks | 68 |
| Text retrieval blocks | 27 |
| Table parent blocks | 3 |
| Table row blocks | 38 |
| Fallback table rows | 38 |
| Synthetic table parents | 3 |

## Synthetic Table Recovery

This document uses custom fallback table recovery.

The visible supporting-information tables are important for retrieval, but they
are not reliably extracted as conventional PDF tables. The preprocessing script
therefore reconstructs Table 1, Table 2, and Table 3 from flattened page text.

The synthetic output contains:

- three parent table records
- 38 fallback row records
- no CSV exports

This behavior is intentional. The fallback records are the retrieval assets; the
absence of CSV files does not mean the tables were ignored.

## Retrieval Role

This source is useful for questions such as:

- what information is required for all Red List assessments
- what additional information is required under specific criteria or conditions
- what narrative fields should be present for threatened taxa
- whether an assessment contains enough support for population trend, threats,
  habitats, rationale, or criterion-specific parameters

The retrieval engine has special query expansion for supporting-information
questions. It tries to retrieve both baseline all-assessment requirements and
condition-specific requirements.

## Files In This Folder

- `raw_page_blocks.jsonl`
  Raw layout-aware text blocks extracted from the PDF.
- `retrieval_blocks.jsonl`
  Text records plus synthetic parent table and table-row records.
- `manifest.json`
  Counts and preprocessing metadata for this document.
- `README.md`
  This document.

## Related Documentation
- `llm_rag/ii_preprocessed_documents/README.md`
- `llm_rag/i_raw_documents/README.md`
- `llm_rag/iii_vector_db/README.md`
