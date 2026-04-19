# RL_categories_and_criteria/tables

## Purpose

This folder contains CSV exports of tables extracted from
`RL_categories_and_criteria.pdf`.

The tables are used as inspection artifacts and as the source of row-level
retrieval records.

## Current Table Inventory

| File | Rows | Columns | Notes |
|---|---:|---:|---|
| `table_001.csv` | 1 | 2 | Title/front-matter style table extraction. |
| `table_002.csv` | 3 | 5 | Criterion A threshold-style extracted table. |
| `table_003.csv` | 3 | 8 | Criterion B extracted table. |
| `table_004.csv` | 4 | 9 | Criteria B and C extracted table content. |
| `table_005.csv` | 3 | 4 | Criterion D extracted table. |
| `table_006.csv` | 3 | 3 | Criterion E quantitative-analysis table. |

## Notes

The PDF layout makes some extracted rows appear in unusual order. These CSVs
are useful for tracing what preprocessing recovered, but exact threshold facts
are also encoded in `llm_rag/iii_vector_db/thresholds.json` for deterministic
lookup.

## Related Documentation
- `llm_rag/ii_preprocessed_documents/RL_categories_and_criteria/README.md`
- `llm_rag/ii_preprocessed_documents/README.md`
- `llm_rag/iii_vector_db/README.md`
