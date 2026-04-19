# smoke_and_inspection

## Purpose

This folder contains notebooks for inspecting the RAG pipeline and smoke-testing
retrieval behavior.

These notebooks are not formal benchmarks. They are practical debugging and QA
workspaces for understanding where retrieval behavior is coming from. They are
especially useful after changing preprocessing, rebuilding the vector database,
editing deterministic routing, or changing the retrieval engine.

## Recommended Flow

Run the notebooks in this order when investigating retrieval behavior:

1. `inspect_preprocessed_docs.ipynb`
   Check whether the preprocessed PDF outputs look sensible.
2. `inspect_vector_db_assets.ipynb`
   Check whether the built vector-db assets look sensible.
3. `eval_threshold_lookup.ipynb`
   Check whether deterministic threshold lookup handles threshold-style
   questions correctly.
4. `eval_retrieval_smoke.ipynb`
   Run broad natural-language prompts through `answer_query(...)` and inspect
   routes, scaffolds, and retrieved candidates.

This order follows the pipeline from source assets to retrieval behavior. If a
smoke prompt looks wrong, the earlier notebooks help narrow down whether the
issue is in preprocessing, vector-db assets, deterministic lookup, or hybrid
retrieval.

## Current Saved Notebook Outputs

The notebooks in this folder have been run and currently include saved cell
outputs.

### `inspect_preprocessed_docs.ipynb`

This notebook evaluates the preprocessing outputs.

It checks:

- which source PDFs have been preprocessed
- manifest-level counts for each processed document
- example `retrieval_blocks.jsonl` rows
- example `raw_page_blocks.jsonl` rows

Current saved output:

- Seven preprocessed source-document folders are visible in the manifest table.
- The inspected target document is
  `Guidelines_for_Reporting_Proportion_Threatened_ver_1_2`.
- That target shows 38 raw page blocks and 14 retrieval blocks.
- The target retrieval blocks are text blocks rather than extracted table rows.

This matters because weak retrieval can start with missing or poorly structured
preprocessing output. The current output gives a quick sanity check that the
preprocessed files exist and that at least one target document can be inspected
from raw page blocks through retrieval blocks.

### `inspect_vector_db_assets.ipynb`

This notebook evaluates the vector-db build outputs.

It checks:

- `build_summary.json`
- `reference_corpus.jsonl`
- `parent_contexts.jsonl`
- `thresholds.json`
- source and block-type distributions
- example parent contexts used to give table rows and chunks broader context

Current saved output:

- The vector build reports 1,170 raw records and 1,743 chunks.
- Chunking is configured with a chunk size of 800 and overlap of 100.
- The Chroma build flag is `True`.
- The build includes 265 extracted table rows and 38 fallback table rows.
- Three synthetic parent contexts are present.
- The embedding model recorded in the build summary is `BAAI/bge-m3`.
- The Chroma collection name is `iucn_reference_docs`.
- Threshold entries are present for Criteria A, B, C, D, and E.

This matters because the retrieval engine depends on both vector-search assets
and deterministic lookup assets. The current output confirms that dense
retrieval, table-row support, parent context support, and threshold lookup data
are all represented in the built assets.

### `eval_threshold_lookup.ipynb`

This notebook evaluates deterministic threshold lookup in isolation.

It checks:

- whether prompts are recognized as threshold-style questions
- which criterion is inferred
- whether an authoritative deterministic answer is returned
- whether general non-threshold prompts are left out of the deterministic route

Current saved output:

- Eight prompts were tested.
- Seven prompts were recognized as threshold queries.
- One general prompt about assessment documentation was correctly not treated as
  a threshold query.
- Criterion B was inferred for EOO, AOO, and locations prompts.
- Criterion D was inferred for mature-individual and D2 prompts.
- Criterion E was inferred for extinction-probability prompts.

This matters because thresholds are stable numeric facts and should not rely
only on generative retrieval behavior. The current output suggests the
deterministic path is routing the intended threshold-style questions correctly
while avoiding at least one broad non-threshold prompt.

### `eval_retrieval_smoke.ipynb`

This notebook evaluates broad reference retrieval behavior.

It runs natural-language prompts through:

```python
llm_rag.iii_vector_db.retrieval_engine.answer_query(...)
```

It displays:

- selected route
- deterministic threshold answer, when applicable
- internal retrieval subqueries
- answer scaffold
- retrieved candidates
- score and metadata summaries

Current saved output:

- Eight smoke prompts were tested.
- Four prompts used the `hybrid_rag` route.
- Four prompts used the `deterministic_threshold_lookup` route.
- Deterministic-threshold prompts returned direct threshold answers and no
  vector-search result list.
- Hybrid prompts returned retrieved reference candidates.
- Several hybrid prompts expanded into multiple internal subqueries, which makes
  the retrieval package easier to inspect question-by-question.

This matters because it gives a fast human-readable signal after rebuilding the
index or changing retrieval logic. The current output confirms that both main
retrieval routes are being exercised and that the notebook exposes the internal
objects needed for debugging.

## Interpreting The Smoke Outputs

The smoke notebooks are diagnostic rather than pass/fail tests.

Useful signs include:

- threshold-style prompts route to deterministic lookup
- broad guidance prompts route to hybrid retrieval
- retrieved reference items come from expected source documents
- table rows appear when a question needs tabular guidance
- answer scaffolds reflect the shape of the question
- non-threshold prompts are not forced into deterministic lookup

Warning signs include:

- a threshold prompt falling through to broad hybrid retrieval
- a broad prompt receiving only generic or unrelated reference hits
- important table rows missing from questions that need criteria or supporting
  information tables
- noisy draft or reference hits crowding out directly relevant evidence

These warning signs should usually be investigated with the inspection notebooks
before editing inference behavior.

## Files In This Folder

- `eval_retrieval_smoke.ipynb`
  Broad natural-language reference retrieval smoke test.
- `eval_threshold_lookup.ipynb`
  Deterministic threshold lookup inspection notebook.
- `inspect_preprocessed_docs.ipynb`
  Preprocessing output inspection notebook.
- `inspect_vector_db_assets.ipynb`
  Vector-db build asset inspection notebook.
- `README.md`
  This document.

## Related Documentation
- `llm_rag/evaluation/README.md`
- `llm_rag/ii_preprocessed_documents/README.md`
- `llm_rag/iii_vector_db/README.md`
- `llm_rag/unit_tests/README.md`
