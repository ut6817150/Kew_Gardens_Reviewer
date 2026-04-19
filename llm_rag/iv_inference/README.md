# iv_inference

## Purpose

`llm_rag/iv_inference/` contains the runtime RAG logic used after an assessment
draft has been uploaded and parsed.

This stage combines:

- deterministic threshold facts
- retrieved IUCN reference evidence
- retrieved uploaded-draft evidence
- grounded prompt assembly
- optional external LLM calls
- debug payload formatting for the UI and evaluation notebooks

The key design principle is separation of evidence. Reference guidance and
uploaded-draft content are retrieved separately and shown separately in the
final prompt.

## Runtime Flow

The main entry point is `answer_rag_question(...)` in `rag_runtime.py`.

For each user question, the runtime:

1. checks whether deterministic threshold facts are relevant
2. calls the reference retriever in `llm_rag/iii_vector_db/retrieval_engine.py`
3. retrieves relevant sections from the uploaded draft
4. assembles a grounded generation prompt
5. optionally sends the prompt to an OpenAI-compatible external LLM endpoint
6. normalizes the answer or request error
7. builds a debug payload containing route, evidence, prompt, model output, and
   errors

During full RAG inference, deterministic threshold facts do not prevent hybrid
retrieval. The runtime calls the reference retriever with threshold
short-circuiting disabled so the final prompt can include both exact threshold
facts and supporting reference context.

## Files In This Folder

| File | Purpose |
|---|---|
| `inference_assessment_parser.py` | Converts the parsed assessment tree into a flatter section-level HTML report suitable for draft retrieval. |
| `draft_retrieval.py` | Builds a draft-side retrieval store from the report and scores draft sections against user questions. |
| `rag_runtime.py` | Orchestrates threshold lookup, reference retrieval, draft retrieval, prompt assembly, external LLM calls, session-state helpers, and debug payloads. |
| `ui_helpers.py` | Small UI-facing helper for cleaning parser-added section markers before display. |
| `README.md` | This document. |

## `inference_assessment_parser.py`

`InferenceAssessmentParser` turns an assessment dictionary into a mapping of:

```text
section_path -> {"content": html}
```

It preserves nested section paths and supports common block types such as:

- paragraphs
- tables
- rows and cells
- rich text values

The inference layer does not need the full original tree at prompt time. It
needs section-level content that can be indexed and shown back to the user.

## `draft_retrieval.py`

`draft_retrieval.py` creates a lightweight retrieval store for the uploaded
assessment draft.

The store keeps:

- normalized section paths
- original parser source keys
- original HTML content
- stripped plain text
- retrieval tokens built from both section paths and content

`retrieve_from_draft(...)` scores draft chunks with:

- token overlap
- section-path overlap
- phrase matches
- intent-specific boosts for common assessment-review prompts

Intent boosts currently cover themes such as:

- Red List status and category
- assessment rationale
- data deficiency
- EOO, AOO, locations, and D2
- population trend
- threats
- habitats
- conservation and research needs
- supporting-information questions

Draft retrieval is deliberately lightweight and local. It does not currently
use embeddings.

## `rag_runtime.py`

`rag_runtime.py` coordinates the full inference path.

Important helpers include:

| Helper | Role |
|---|---|
| `init_rag_session_state(...)` | Adds missing Streamlit session-state keys for RAG chat, cached draft state, and parsed report data. |
| `sync_rag_state_with_upload(...)` | Resets cached draft state when a new uploaded file is detected. |
| `build_report_from_assessment(...)` | Converts the parsed assessment dictionary into the section-level report. |
| `ensure_draft_store_from_report(...)` | Builds or reuses the cached draft retrieval store. |
| `build_reference_result_lines(...)` | Formats retrieved reference chunks and parent contexts for the prompt. |
| `build_generation_prompt(...)` | Creates the final grounded prompt with separate reference and draft evidence sections. |
| `external_llm_is_configured(...)` | Checks whether a model configuration has enough information to call an external LLM. |
| `maybe_call_external_llm(...)` | Calls the external model and normalizes success or error output. |
| `answer_rag_question(...)` | Runs the complete RAG workflow for one user question. |
| `build_rag_debug_payload(...)` | Creates the structured debug view shown in the UI and used by evaluation notebooks. |
| `format_rag_debug_payload(...)` | Pretty-prints the debug payload as JSON. |

## External LLM Calls

The runtime uses the OpenAI client against an OpenAI-compatible endpoint. In the
current app flow, this is intended for OpenRouter-style model configs.

The LLM call receives:

- one system message
- one user message containing the full grounded RAG prompt
- low temperature
- optional provider-specific reasoning settings when configured

The runtime normalizes several response shapes, including dictionary responses,
Pydantic-style SDK objects, and response content made of structured parts.

If the request fails, the runtime returns a normalized error payload instead of
raising directly into the UI. This lets the app show a user-facing error while
still preserving debug details.

## Prompt Structure

`build_generation_prompt(...)` keeps evidence types separate:

- deterministic threshold facts
- reference-evidence scaffold
- retrieved IUCN reference excerpts
- uploaded-draft excerpts
- answer-formatting instructions

This separation is important because a reviewer needs to know whether a claim
comes from official guidance or from the uploaded assessment draft.

## Debug Payload

The debug payload can include:

- route
- deterministic threshold answer
- internal retrieval subqueries
- reference answer scaffold
- retrieved reference evidence
- parent table context
- retrieved draft hits
- full prompt text
- model output
- provider reasoning details when available
- request errors

This debug-first design makes it easier to diagnose retrieval failures, noisy
evidence, missing draft sections, or external LLM errors.

## Unit Tests

The inference unit tests live in:

```text
llm_rag/unit_tests/iv_inference/
```

They cover:

- parser behavior for nested sections, paragraphs, tables, and rich cells
- draft-store creation and draft retrieval ranking
- prompt assembly
- external LLM response normalization
- session-state synchronization
- debug payload formatting
- UI section-name cleanup

The latest full unit-test run passed:

```text
127 passed in 9.60s
```

## Relationship To The App

The Streamlit app imports these helpers to support the RAG tab. The app handles
file upload, sidebar configuration, and UI rendering. This folder handles the
RAG-specific runtime behavior.

The inference code should remain usable from notebooks and tests without
depending on Streamlit rendering. Session-state helpers are intentionally small
wrappers around the core parsing and retrieval functions.

## Related Documentation
- `llm_rag/README.md`
- `llm_rag/iii_vector_db/README.md`
- `llm_rag/evaluation/README.md`
- `llm_rag/unit_tests/iv_inference/README.md`
