# RL_criteria_summary_sheet/tables

## Purpose

This folder contains CSV exports of the compact Red List criteria summary
tables.

These rows are especially useful for retrieval because they keep formal
criterion names and threshold language close together.

## Current Table Inventory

| File | Rows | Columns | Notes |
|---|---:|---:|---|
| `table_001.csv` | 4 | 4 | Criterion A population-size reduction summary. |
| `table_002.csv` | 7 | 5 | Criterion B geographic-range summary. |
| `table_003.csv` | 8 | 6 | Criterion C small-population summary. |
| `table_004.csv` | 3 | 4 | Criterion D very-small or restricted-population summary. |
| `table_005.csv` | 2 | 4 | Criterion E quantitative-analysis summary. |

## How These Tables Are Used

The vector-db build keeps each row atomic, which helps exact criterion and
threshold prompts retrieve compact evidence. The broader parent table context is
also exported so row hits can be shown with surrounding criteria context.

## Related Documentation
- `llm_rag/ii_preprocessed_documents/RL_criteria_summary_sheet/README.md`
- `llm_rag/ii_preprocessed_documents/README.md`
- `llm_rag/iii_vector_db/README.md`
