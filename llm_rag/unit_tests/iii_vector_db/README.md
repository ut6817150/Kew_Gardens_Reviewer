# unit_tests/iii_vector_db

## Purpose

This folder contains unit tests for the reference retrieval and vector-db stage
in `llm_rag/iii_vector_db/`.

The tests cover build-time helpers, query-time retrieval helpers, deterministic
threshold lookup, embedding loading, and smoke-test output formatting.

## Latest Test Result

The latest full `llm_rag` unit-test run passed:

```text
127 passed in 9.60s
```

This folder contributes 47 test functions across five test modules.

## Test Modules

### `test_build_reference_db_unit.py`

Tests the reference asset build helpers:

- extracting display text from contextualized records
- building enriched search text
- parsing JSONL records and defaults
- loading retrieval records from processed document folders
- preserving expected metadata
- keeping table rows atomic during splitting
- splitting text records
- flattening document chunks
- writing `reference_corpus.jsonl`
- writing unique parent table contexts
- requiring embeddings during strict build
- resetting and persisting Chroma collections
- writing build summaries
- running the build pipeline with `--skip-chroma`

### `test_embedding_loader_unit.py`

Tests embedding-model loading behavior:

- cached snapshot resolution
- network fallback when cache lookup misses
- raising after both resolution attempts fail
- constructing the Hugging Face embedding wrapper
- strict failure behavior
- non-strict sparse-only fallback behavior

### `test_reference_retrieval_smoke_test_unit.py`

Tests the command-line smoke-test script:

- default argument parsing
- custom argument parsing
- deterministic threshold output formatting
- hybrid retrieval output formatting
- parent-context display

### `test_retrieval_engine_unit.py`

Tests query-time retrieval helpers:

- tokenization and phrase matching
- query-mode detection
- supporting-information query expansion
- candidate construction
- table labeling
- best-table candidate selection
- sparse-corpus backfill
- dense and sparse hit merging
- supporting-information coverage forcing
- parent-context attachment
- answer scaffold construction
- deterministic threshold route
- hybrid route fallback when threshold lookup does not answer

### `test_threshold_lookup_unit.py`

Tests deterministic threshold lookup:

- threshold-query detection
- criterion inference
- field request detection
- answer-line joining
- Criterion B field-specific answers
- Criterion B summary answers
- Criterion D field-specific answers
- Criterion E field-specific answers
- field-only queries without explicit criterion labels
- non-threshold prompts returning no deterministic answer

## Why These Tests Matter

This stage controls what official reference evidence reaches the inference
prompt. Failures here can cause missing thresholds, noisy evidence, poor table
coverage, or broken dense retrieval.

The unit tests keep the build and retrieval helpers stable without requiring a
full external LLM call.

## Running These Tests

From the repo root:

```bash
python -m pytest -q llm_rag/unit_tests/iii_vector_db
```

Run a specific module:

```bash
python -m pytest -q llm_rag/unit_tests/iii_vector_db/test_retrieval_engine_unit.py
```

## Related Code

- `llm_rag/iii_vector_db/build_reference_db.py`
- `llm_rag/iii_vector_db/embedding_loader.py`
- `llm_rag/iii_vector_db/reference_retrieval_smoke_test.py`
- `llm_rag/iii_vector_db/retrieval_engine.py`
- `llm_rag/iii_vector_db/threshold_lookup.py`

## Related Documentation
- `llm_rag/unit_tests/README.md`
- `llm_rag/iii_vector_db/README.md`
- `llm_rag/ii_preprocessed_documents/README.md`
