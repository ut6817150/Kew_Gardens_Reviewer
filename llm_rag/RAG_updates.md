# RAG updates

## Overview

This note records how the `llm_rag` solution has evolved from a reference-only retrieval experiment into the current app-integrated RAG workflow.

Today the system supports:

- preprocessing IUCN PDFs into retrieval-friendly blocks
- building a hybrid reference retriever
- deterministic threshold lookup
- parsing an uploaded assessment draft into section-level evidence
- retrieving draft and reference evidence side by side
- assembling a grounded prompt for the external LLM
- exposing detailed debug output inside the Streamlit RAG chat

## Evolution of the solution

### Stage 1: basic reference RAG

The first version focused on the reference PDFs only:

- chunk the documents
- embed the chunks
- store them in a vector database

This gave broad topical matches, but it often missed exact IUCN requirements and struggled with tables.

### Stage 2: structure-aware preprocessing

The pipeline was then upgraded to preserve more document structure:

- section paths
- page provenance
- text blocks
- extracted tables

This improved traceability, but many high-value requirement tables were still not being captured well enough.

### Stage 3: synthetic table recovery

The supporting-information PDF turned out to be the key pain point. Important tables were visible to humans but not reliably encoded as machine-readable PDF tables.

To handle that, the preprocessing stage learned to create synthetic table records for Table 1, Table 2, and Table 3 by reconstructing them from flattened page text.

That change made row-level requirement retrieval much more reliable.

### Stage 4: hybrid retrieval and parent contexts

Pure dense retrieval was not enough for this corpus because the questions often depend on:

- exact criterion names
- exact phrasing
- short symbolic labels
- table rows that need bigger surrounding context

The retrieval layer therefore evolved into a hybrid system:

- dense retrieval with BGE-M3 + Chroma
- sparse retrieval over a JSONL corpus
- parent table-context attachment
- heuristic reranking

This was the point where the reference side became strong enough to answer many policy-style questions more faithfully.

### Stage 5: deterministic thresholds

Some answers, especially official thresholds, were too risky to leave entirely to retrieval plus generation.

A deterministic threshold lane was added for:

- AOO
- EOO
- number of locations
- mature individuals
- extinction probability

That gave the system an authoritative route for stable numeric facts.

### Stage 6: supporting-information query decomposition

Questions about required supporting information were still difficult because they often need both:

- baseline requirements for all assessments
- extra requirements under specific conditions

The retrieval engine was then extended to:

- expand one user question into internal subqueries
- target Table 1 and Table 2 differently
- enforce coverage when key evidence would otherwise be missed
- build an answer scaffold that separates the two requirement groups

### Stage 7: uploaded-draft retrieval

The next major step was moving from reference-only retrieval to actual draft review.

Two inference-side components were added:

- `InferenceAssessmentParser`, which converts an uploaded parsed draft into section-level HTML
- `draft_retrieval.py`, which retrieves the most relevant draft sections for a user question

This made it possible to compare:

- what the IUCN references require
- what the uploaded draft appears to contain

### Stage 8: app integration

The RAG workflow was then wired into `app.py` in the repo root:

- the uploaded draft is parsed once
- RAG context is prepared automatically when a new file is uploaded
- the chat stays in a fixed-height pane
- debug mode can show scaffold, retrieved evidence, prompt text, model steps, and errors

This moved the project from backend experiments into a usable review interface.

### Stage 9: robustness and simplification

Recent work has focused on making the system easier to maintain and less fragile:

- path handling was cleaned up
- duplicate and stale UI elements were removed
- debug payload generation was centralized
- the external LLM call now uses the OpenAI client against OpenRouter
- request failures are shown in debug mode
- embedding loading was hardened with cached snapshot resolution and sparse-only fallback
- the `llm_rag` documentation has been refreshed to match the current code

## Current shape of the system

The solution now has three clear layers:

1. `ii_preprocessed_documents/`
   Turns PDFs into retrieval-ready records.
2. `iii_vector_db/`
   Builds and serves the reference retrieval system.
3. `iv_inference/`
   Adds uploaded-draft retrieval, prompt assembly, external LLM calls, and debug output for the app.

## What changed most conceptually

The biggest shift is that the solution is no longer just "RAG over reference PDFs."

It is now a comparison-oriented review pipeline:

- deterministic facts where possible
- reference retrieval for official requirements
- draft retrieval for uploaded content
- explicit prompt construction that keeps those evidence sources separate

That separation is what makes the app useful for review rather than just for generic question answering.

## Current status

The pipeline is in a good place for iterative app work:

- reference retrieval is already strong on table-heavy requirement questions
- draft retrieval is integrated
- request failures are inspectable in debug mode
- the documentation and folder structure now line up with the current implementation
