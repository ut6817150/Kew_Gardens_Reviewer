# i_raw_documents

## Purpose

`llm_rag/i_raw_documents/` stores the source reference PDFs used to build the
RAG reference corpus.

These files are the upstream source of truth for the reference side of the RAG
pipeline. They are not queried directly by the app. Instead, they are processed
by `llm_rag/ii_preprocessed_documents/preprocess_pdfs.py`, converted into
retrieval records, and then indexed by `llm_rag/iii_vector_db/build_reference_db.py`.

## Current Source Documents

| File | Role In Retrieval |
|---|---|
| `Guidelines_for_Reporting_Proportion_Threatened_ver_1_2.pdf` | Guidance for reporting proportions of threatened taxa. Useful for questions about threatened proportions and Red List data-use reporting. |
| `Mapping_Standards_Version_1.20_Jan2024.pdf` | Mapping standards for Red List spatial data. Useful for questions about map status, spatial attributes, presence/origin/seasonality coding, EOO, AOO, and mapped distribution data. |
| `RedListGuidelines.pdf` | Main Red List guidance document. Useful for detailed interpretation of criteria, subcriteria, continuing decline, locations, population concepts, uncertainty, and application guidance. |
| `Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf` | Supporting-information guidance. This is the key source for required and recommended SIS fields, including conditional requirements for threatened taxa. |
| `RL_categories_and_criteria.pdf` | Red List Categories and Criteria document. Useful for category definitions, formal criteria, and official threshold language. |
| `RL_criteria_summary_sheet.pdf` | Compact criteria summary sheet. Useful for concise threshold and criteria lookup, especially when the retriever needs table-style criterion summaries. |
| `RL_Standards_Consistency.pdf` | Standards and consistency checks. Useful for review-style questions about rationale quality, supporting information, mapping consistency, and common errors. |

## How These Files Are Used

The pipeline uses these PDFs in this order:

1. `preprocess_pdfs.py` reads each PDF.
2. Layout-aware text blocks are extracted from each page.
3. Repeated headers and footers are removed when detected.
4. Headings are used to maintain section-path context.
5. Tables are extracted where possible.
6. Difficult supporting-information tables are reconstructed with custom
   fallback logic.
7. JSONL retrieval records are written under
   `llm_rag/ii_preprocessed_documents/`.
8. The vector-db build step reads those records and creates the reference
   retrieval assets in `llm_rag/iii_vector_db/`.

## Updating The Raw Corpus

If a source PDF is added, removed, or replaced, rebuild the downstream assets
from the repo root:

```bash
python llm_rag/ii_preprocessed_documents/preprocess_pdfs.py
python llm_rag/iii_vector_db/build_reference_db.py --reset
python -m pytest -q llm_rag/unit_tests
```

After rebuilding, inspect:

- `llm_rag/ii_preprocessed_documents/summary.json`
- the per-document `manifest.json` files
- `llm_rag/iii_vector_db/build_summary.json`
- the evaluation notebooks in `llm_rag/evaluation/smoke_and_inspection/`

## Versioning Notes

These PDFs are project data, not generated cache files. They should remain in
the repository unless the team decides to source them from an external storage
location.

When replacing a PDF with a newer version, keep the filename meaningful and
expect retrieval output to change. A source-document change can affect:

- extracted text block counts
- table recovery
- chunk counts
- Chroma index contents
- deterministic and hybrid retrieval behavior
- evaluation notebook outputs

## Files In This Folder

- `README.md`
  This document.
- `*.pdf`
  Raw IUCN reference documents used by the preprocessing and retrieval build
  stages.

## Related Documentation
- `llm_rag/README.md`
- `llm_rag/ii_preprocessed_documents/README.md`
- `llm_rag/iii_vector_db/README.md`
