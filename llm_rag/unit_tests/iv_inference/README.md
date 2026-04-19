# unit_tests/iv_inference

## Purpose

This folder contains unit tests for the inference-time RAG layer in
`llm_rag/iv_inference/`.

The tests cover uploaded-draft parsing, draft retrieval, prompt assembly,
external LLM response normalization, session-state behavior, debug payload
formatting, and UI-facing helpers.

## Latest Test Result

The latest full `llm_rag` unit-test run passed:

```text
127 passed in 9.60s
```

This folder contributes 55 test functions across four test modules.

## Test Modules

### `test_draft_retrieval_unit.py`

Tests draft-side retrieval helpers:

- whitespace normalization
- simple HTML stripping
- token normalization and stopword filtering
- section-path marker cleanup
- phrase matching
- supporting-information query detection
- draft-store construction
- ignoring malformed or empty report entries
- deduplicating identical source/text chunks
- draft-hit formatting
- empty-store behavior
- fallback behavior for tokenless queries
- population-query boosts
- status-query boosts
- supporting-information section boosts

### `test_inference_assessment_parser_unit.py`

Tests `InferenceAssessmentParser`:

- rejecting non-dictionary input
- empty-document behavior
- nested section-path construction
- assessment-title fallback behavior
- paragraph rendering
- avoiding double-wrapping existing paragraph HTML
- table rendering
- malformed table-row handling
- nested cell flattening
- scalar row handling
- ignoring unsupported block types
- merging repeated section content
- rich-cell stringification

### `test_rag_runtime_unit.py`

Tests RAG runtime orchestration:

- OpenAI-compatible base URL normalization
- LLM request payload construction
- SDK completion object normalization
- unsupported completion response handling
- nested response text stringification
- reasoning-text extraction
- output and reasoning parsing
- normalized LLM error payloads
- Streamlit session-state initialization
- upload-signature synchronization
- report building from parsed assessments
- draft-store caching
- reference-result formatting
- generation prompt assembly
- external LLM configuration checks
- incomplete-config behavior
- missing-client error behavior
- successful external LLM call normalization
- end-to-end `answer_rag_question(...)` orchestration with mocked helpers
- debug payload serialization

### `test_ui_helpers_unit.py`

Tests UI-facing section-name cleanup:

- paragraph marker removal
- table marker removal
- row marker removal
- case-insensitive marker cleanup
- preserving plain section names
- leaving unsupported suffix patterns untouched

## Why These Tests Matter

The inference stage is the point where reference evidence, uploaded-draft
evidence, prompt construction, and model-call handling meet. Unit tests here
help ensure that UI-facing behavior stays stable while retrieval and model
configuration continue to evolve.

## Running These Tests

From the repo root:

```bash
python -m pytest -q llm_rag/unit_tests/iv_inference
```

Run a specific module:

```bash
python -m pytest -q llm_rag/unit_tests/iv_inference/test_rag_runtime_unit.py
```

## Related Code

- `llm_rag/iv_inference/draft_retrieval.py`
- `llm_rag/iv_inference/inference_assessment_parser.py`
- `llm_rag/iv_inference/rag_runtime.py`
- `llm_rag/iv_inference/ui_helpers.py`

## Related Documentation
- `llm_rag/unit_tests/README.md`
- `llm_rag/iv_inference/README.md`
- `llm_rag/iii_vector_db/README.md`
