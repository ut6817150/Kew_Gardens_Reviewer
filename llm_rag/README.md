# llm_rag

## Overview

`llm_rag` contains the retrieval-augmented generation pipeline used by `app.py` in the repo root.

The pipeline has four stages:

1. Raw IUCN reference PDFs are stored in `i_raw_documents/`.
2. `ii_preprocessed_documents/preprocess_pdfs.py` converts those PDFs into retrieval-ready blocks, including table rows and synthetic fallback tables for difficult PDFs.
3. `iii_vector_db/build_reference_db.py` turns those blocks into a hybrid retrieval store made up of:
   - a Chroma dense index
   - a sparse JSONL corpus
   - parent table contexts
   - deterministic threshold lookup data
4. `iv_inference/rag_runtime.py` combines reference retrieval with uploaded-draft retrieval and sends a grounded prompt to the external LLM configured in `app.py` in the repo root.

## Folder structure

```text
llm_rag/
|- i_raw_documents/
|- ii_preprocessed_documents/
|- iii_vector_db/
|- iv_inference/
|- README.md
`- RAG_updates.md
```

## What each folder does

### `i_raw_documents/`

Stores the source PDFs used to build the reference corpus.

### `ii_preprocessed_documents/`

Stores intermediate retrieval assets produced from the PDFs:

- `raw_page_blocks.jsonl`
- `retrieval_blocks.jsonl`
- per-document `manifest.json`
- extracted CSV tables where available
- `summary.json` across the whole corpus

### `iii_vector_db/`

Stores the reference retrieval layer:

- `reference_corpus.jsonl`
- `parent_contexts.jsonl`
- `thresholds.json`
- `build_summary.json`
- `chroma_db/reference_docs/`
- the runtime retrieval code

### `iv_inference/`

Stores the inference-time logic used by the app:

- `inference_assessment_parser.py` flattens the uploaded draft into section-level HTML
- `draft_retrieval.py` scores uploaded-draft sections against the user query
- `rag_runtime.py` orchestrates threshold lookup, reference retrieval, draft retrieval, prompt assembly, external LLM calls, and debug payloads
- `ui_helpers.py` contains small UI-facing helpers

## Importing the inference helpers into `app.py`

The app can import the inference helpers directly from the repo root.

The key pattern is:

```python
from llm_rag.iv_inference.rag_runtime import answer_rag_question
from llm_rag.iv_inference.rag_runtime import build_rag_debug_payload
from llm_rag.iv_inference.rag_runtime import build_report_from_assessment
from llm_rag.iv_inference.rag_runtime import ensure_draft_store_from_report
from llm_rag.iv_inference.rag_runtime import init_rag_session_state
from llm_rag.iv_inference.rag_runtime import sync_rag_state_with_upload
from llm_rag.iv_inference.ui_helpers import normalize_display_section_name
```

These are the main inference-side functions used by the app:

- `init_rag_session_state(...)`: creates the RAG session-state keys
- `sync_rag_state_with_upload(...)`: resets cached RAG state when the uploaded file changes
- `build_report_from_assessment(...)`: converts the parsed assessment tree into the section-level report dictionary
- `ensure_draft_store_from_report(...)`: builds or reuses the cached draft retrieval store
- `answer_rag_question(...)`: runs the end-to-end RAG pipeline for one prompt
- `build_rag_debug_payload(...)`: formats the debug information shown in the UI

## Current runtime flow

When a user uploads a draft in the RAG app from the repo root:

1. The uploaded assessment is parsed with `parse_dict(...)`.
2. `InferenceAssessmentParser` converts the parsed tree into a `section_path -> html` report dictionary.
3. A draft retrieval store is built from those sections and cached in Streamlit session state.
4. For each prompt:
   - threshold facts are looked up deterministically when relevant
   - reference evidence is retrieved with hybrid dense + sparse search
   - top draft sections are retrieved from the uploaded assessment
   - a grounded prompt is assembled
   - the external LLM is called through the OpenAI client against OpenRouter
   - debug output is prepared, including prompt, reference excerpts, draft hits, and request errors

## Key design decisions

- Table rows are preserved because many IUCN requirements live in tables rather than prose.
- Parent table contexts are stored separately so row-level hits can be expanded with larger context.
- Threshold questions are answered deterministically where possible instead of relying fully on free-form generation.
- Uploaded drafts are retrieved separately from the reference corpus so the system can distinguish:
  - what the IUCN reference documents require
  - what the uploaded draft appears to contain

## Typical commands

From the repo root:

```bash
python llm_rag/ii_preprocessed_documents/preprocess_pdfs.py
python llm_rag/iii_vector_db/build_reference_db.py --reset
streamlit run app.py
```

## When to rebuild

Re-run preprocessing or rebuild the vector DB when:

- the PDFs in `i_raw_documents/` change
- preprocessing logic changes
- chunking, embeddings, or retrieval-build logic changes

You do not need to rebuild the vector DB for UI-only changes or most inference-layer changes.
