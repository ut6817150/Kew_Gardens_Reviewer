# RAG Development History

## Purpose

This note records how the `llm_rag` solution evolved from a reference-only
retrieval experiment into the current app-integrated assessment-review RAG
workflow.

It is historical context, not an implementation guide. For current commands,
folder contents, and future work, see `llm_rag/README.md`.

## Current Capabilities

The current system supports:

- preprocessing IUCN PDFs into retrieval-friendly records
- preserving section context, page provenance, table parents, and table rows
- synthetic recovery of difficult supporting-information tables
- building a hybrid reference retriever
- deterministic lookup for stable Red List threshold facts
- parsing an uploaded assessment draft into section-level evidence
- retrieving draft and reference evidence side by side
- assembling a grounded prompt for an external LLM
- exposing detailed debug output for the Streamlit RAG chat
- generating retrieval-judging prompts for external LLM evaluation
- running stage-specific unit tests for preprocessing, vector retrieval, and
  inference

## Evolution Of The Solution

### Stage 1: Basic Reference RAG

The first version focused on the reference PDFs only:

- chunk the documents
- embed the chunks
- store them in a vector database
- retrieve broad topical matches for a user question

This gave useful first-pass retrieval, but it often missed exact IUCN
requirements and struggled with tables.

### Stage 2: Structure-Aware Preprocessing

The pipeline was then upgraded to preserve more document structure:

- section paths
- page provenance
- text blocks
- extracted tables
- block types
- contextualized retrieval text

This improved traceability, but many high-value requirement tables were still
not being captured well enough.

### Stage 3: Synthetic Table Recovery

The supporting-information PDF became a key pain point. Important tables were
visible to humans but not reliably encoded as machine-readable PDF tables.

To handle that, the preprocessing stage learned to create synthetic table
records for Table 1, Table 2, and Table 3 by reconstructing them from flattened
page text.

That change made row-level requirement retrieval much more reliable for
questions about required and recommended supporting information.

### Stage 4: Hybrid Retrieval And Parent Contexts

Pure dense retrieval was not enough for this corpus because the questions often
depend on:

- exact criterion names
- exact wording
- short symbolic labels
- table rows that need larger surrounding context

The retrieval layer therefore evolved into a hybrid system:

- dense retrieval with BGE-M3 and Chroma
- sparse BM25-style retrieval over a JSONL corpus
- reciprocal-rank fusion
- parent table-context attachment
- heuristic reranking

This was the point where the reference side became strong enough to answer many
policy-style questions more faithfully.

### Stage 5: Deterministic Thresholds

Some answers, especially official threshold facts, were too risky to leave
entirely to retrieval plus generation.

A deterministic threshold lane was added for:

- AOO
- EOO
- number of locations
- mature individuals
- extinction probability

That gave the system an authoritative route for stable numeric facts.

### Stage 6: Supporting-Information Query Decomposition

Questions about required supporting information were still difficult because
they often need both:

- baseline requirements for all assessments
- extra requirements under specific criteria or conditions

The retrieval engine was extended to:

- expand one user question into internal subqueries
- target all-assessment requirements and condition-specific requirements
  separately
- backfill key supporting-information tables when normal ranking missed them
- build an answer scaffold that separates the requirement groups

### Stage 7: Uploaded-Draft Retrieval

The next major step was moving from reference-only retrieval to actual draft
review.

Two inference-side components were added:

- `InferenceAssessmentParser`, which converts an uploaded parsed draft into
  section-level HTML
- `draft_retrieval.py`, which retrieves the most relevant draft sections for a
  user question

This made it possible to compare:

- what the IUCN references require
- what the uploaded draft appears to contain

### Stage 8: App Integration

The RAG workflow was then wired into the Streamlit app:

- the uploaded draft is parsed once
- RAG context is prepared when a new file is uploaded
- the sidebar model and API-key configuration can be passed into RAG
- debug mode can show scaffold, retrieved evidence, prompt text, model steps,
  and errors
- external LLM calls use the OpenAI client against OpenRouter-style model
  configs

This moved the project from backend experiments into a usable review interface.

### Stage 9: Evaluation Workflows

The evaluation workflow was then split into clearer notebook groups:

- `llm_rag/evaluation/smoke_and_inspection/`
  Inspects preprocessing outputs, vector-db assets, deterministic threshold
  lookup, and broad retrieval behavior.
- `llm_rag/evaluation/retrieval_judging/`
  Generates retrieval-only Markdown prompts that can be manually passed to an
  external LLM judge.

The retrieval-judging flow deliberately avoids testing the final answer from an
external LLM. It evaluates the retrieved evidence package directly.

Recent saved external-LLM judgments show that retrieval is often useful but
still uneven when a question needs specific threshold rows, cross-referenced
sections, or nearby draft fields. That finding now informs the future-work
discussion around better preprocessing, graph-style evidence links, and
connected chat context.

### Stage 10: Unit-Test Reorganization

Unit tests were moved into `llm_rag/unit_tests/` and grouped by pipeline stage:

- preprocessing tests
- vector-db and retrieval tests
- inference tests

The suite currently covers helper logic without requiring live external LLM
calls. The latest full run passed:

```text
127 passed in 9.60s
```

### Stage 11: Documentation Refresh

Documentation was expanded across the RAG folders so each stage now explains:

- what the folder contains
- how the files are generated or used
- what scripts are responsible for
- what tests exist
- what current generated counts and evaluation outputs mean

The repo-root README is intentionally separate from this RAG-specific
documentation.

## Current Shape Of The System

The solution now has five clear RAG areas:

1. `i_raw_documents/`
   Stores the raw IUCN reference PDFs.
2. `ii_preprocessed_documents/`
   Turns PDFs into retrieval-ready records.
3. `iii_vector_db/`
   Builds and serves the reference retrieval system.
4. `iv_inference/`
   Adds uploaded-draft retrieval, prompt assembly, external LLM calls, and
   debug output.
5. `evaluation/` and `unit_tests/`
   Provide notebook-based inspection, retrieval judging, and automated unit
   tests.

## What Changed Most Conceptually

The biggest shift is that the solution is no longer just "RAG over reference
PDFs."

It is now a comparison-oriented review pipeline:

- deterministic facts where possible
- reference retrieval for official requirements
- draft retrieval for uploaded content
- prompt construction that keeps those evidence sources separate
- debug output that makes retrieval behavior inspectable
- evaluation workflows that judge retrieval quality directly

That separation is what makes the app useful for review rather than just for
generic question answering.

## Current Status

The pipeline is ready for iterative app and retrieval work:

- reference preprocessing and vector-db assets are built
- deterministic threshold lookup is covered by unit tests
- uploaded-draft retrieval is integrated into inference
- request failures are normalized for debug output
- evaluation notebooks and external-LLM judging prompts are in place
- all current `llm_rag` unit tests pass

Known improvement areas are now RAG-specific and documented in
`llm_rag/README.md`, especially better preprocessing, graph-style links between
chunks, connected chat context, expanded retrieval evaluation, and stronger
draft retrieval.
