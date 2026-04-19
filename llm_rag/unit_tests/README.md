# unit_tests

## Purpose

`llm_rag/unit_tests/` contains pytest-based unit tests for the RAG pipeline.

The tests are grouped by pipeline stage so failures are easier to localize:

- `ii_preprocessed_documents/`
  PDF preprocessing helpers and output-writing logic.
- `iii_vector_db/`
  reference asset build helpers, embedding loading, retrieval, threshold lookup,
  and smoke-test formatting.
- `iv_inference/`
  uploaded-draft parsing, draft retrieval, RAG runtime orchestration, external
  LLM response normalization, and UI helper behavior.

## Latest Test Result

The latest full unit-test run was:

```bash
python -m pytest -q llm_rag/unit_tests
```

Result:

```text
127 passed in 9.60s
```

## Test Coverage Summary

| Folder | Test Files | Test Functions | Focus |
|---|---:|---:|---|
| `ii_preprocessed_documents/` | 1 | 25 | PDF preprocessing helpers and output generation. |
| `iii_vector_db/` | 5 | 47 | Build pipeline helpers, embedding loading, retrieval engine, threshold lookup, and smoke-test output. |
| `iv_inference/` | 4 | 55 | Draft parsing, draft retrieval, prompt assembly, model-call normalization, debug payloads, and UI helpers. |
| Total | 10 | 127 | Stage-specific RAG unit tests. |

## Running Tests

Run the full `llm_rag` unit suite from the repo root:

```bash
python -m pytest -q llm_rag/unit_tests
```

Run a stage-specific suite:

```bash
python -m pytest -q llm_rag/unit_tests/ii_preprocessed_documents
python -m pytest -q llm_rag/unit_tests/iii_vector_db
python -m pytest -q llm_rag/unit_tests/iv_inference
```

Run one test module:

```bash
python -m pytest -q llm_rag/unit_tests/iii_vector_db/test_retrieval_engine_unit.py
```

Run one test function:

```bash
python -m pytest -q llm_rag/unit_tests/iv_inference/test_rag_runtime_unit.py::test_answer_rag_question_combines_threshold_reference_draft_and_model_output
```

## What These Tests Are And Are Not

These are unit tests. They use small fixtures, monkeypatching, and lightweight
test doubles to check specific behavior.

They do check:

- helper logic
- metadata handling
- scoring and selection behavior
- prompt assembly structure
- deterministic threshold routing
- error handling and response normalization

They do not fully benchmark:

- end-to-end retrieval quality over the complete corpus
- external LLM answer quality
- Chroma embedding quality
- notebook-based evaluation outputs

For retrieval-quality inspection, use the notebooks in
`llm_rag/evaluation/`.

## Adding New Tests

Add tests close to the pipeline stage they cover:

- preprocessing tests in `ii_preprocessed_documents/`
- vector-db and retrieval tests in `iii_vector_db/`
- inference-time tests in `iv_inference/`

Prefer small, deterministic fixtures. Avoid tests that require network access,
live external LLM calls, or a full vector-db rebuild unless they are explicitly
marked and documented as integration tests.

## Files In This Folder

- `README.md`
  This document.
- `ii_preprocessed_documents/`
  Preprocessing unit tests.
- `iii_vector_db/`
  Vector-db and reference-retrieval unit tests.
- `iv_inference/`
  Inference-time unit tests.

## Related Documentation
- `llm_rag/README.md`
- `llm_rag/unit_tests/ii_preprocessed_documents/README.md`
- `llm_rag/unit_tests/iii_vector_db/README.md`
- `llm_rag/unit_tests/iv_inference/README.md`
