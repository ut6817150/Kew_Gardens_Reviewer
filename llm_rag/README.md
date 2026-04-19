# llm_rag

## Purpose

`llm_rag/` contains the retrieval-augmented generation pipeline used to review
IUCN Red List assessment drafts against a curated reference library of official
IUCN documents.

The pipeline is not designed as generic chat over PDFs. It is designed as a
review-support workflow:

- official reference evidence is retrieved from the IUCN corpus
- uploaded assessment evidence is retrieved from the current draft
- deterministic Red List threshold facts are supplied where possible
- the final prompt keeps reference evidence and draft evidence separate
- debug output exposes the retrieval route, evidence package, prompt, and model
  response details

## How To Use This README

This README is the RAG architecture hub. Use it when you need the overall
workflow, design decisions, rebuild commands, test commands, and future-work
direction.

For stage-specific detail, open:

- `llm_rag/i_raw_documents/README.md`
  Raw IUCN PDFs and when to replace or rebuild them.
- `llm_rag/ii_preprocessed_documents/README.md`
  PDF preprocessing, retrieval-record shape, generated manifests, and
  per-document output counts.
- `llm_rag/iii_vector_db/README.md`
  Reference retrieval assets, Chroma, sparse retrieval, deterministic
  thresholds, and retrieval routes.
- `llm_rag/iv_inference/README.md`
  Uploaded-draft retrieval, prompt assembly, external LLM calls, and debug
  payloads.
- `llm_rag/evaluation/README.md`
  Notebook-based smoke tests, inspection notebooks, and external LLM retrieval
  judging.
- `llm_rag/unit_tests/README.md`
  Unit-test structure, latest result, and how to run tests.
- `llm_rag/DEVELOPMENT_HISTORY.md`
  Project evolution narrative.

## Folder Structure

```text
llm_rag/
|- DEVELOPMENT_HISTORY.md
|- README.md
|- evaluation/
|- i_raw_documents/
|- ii_preprocessed_documents/
|- iii_vector_db/
|- iv_inference/
`- unit_tests/
```

## Pipeline At A Glance

The RAG pipeline has five main stages:

1. Store raw reference PDFs in `llm_rag/i_raw_documents/`.
2. Preprocess PDFs into structured retrieval blocks in
   `llm_rag/ii_preprocessed_documents/`.
3. Build reference retrieval assets in `llm_rag/iii_vector_db/`.
4. Combine reference retrieval with uploaded-draft retrieval in
   `llm_rag/iv_inference/`.
5. Evaluate retrieval behavior through `llm_rag/evaluation/` and protect helper
   behavior with `llm_rag/unit_tests/`.

## Stage 1: Raw Reference Documents

The source PDFs live in:

```text
llm_rag/i_raw_documents/
```

Current sources include:

- Red List Guidelines
- Red List Categories and Criteria
- Red List criteria summary sheet
- mapping standards
- required and recommended supporting-information guidance
- standards and consistency guidance
- guidance for reporting proportion threatened

These PDFs are the upstream source of truth for reference-side retrieval. They
are not queried directly at runtime. They must be preprocessed before they can
be indexed.

More detail:

```text
llm_rag/i_raw_documents/README.md
```

## Stage 2: PDF Preprocessing

The preprocessing script is:

```text
llm_rag/ii_preprocessed_documents/preprocess_pdfs.py
```

Run from the repo root:

```bash
python llm_rag/ii_preprocessed_documents/preprocess_pdfs.py
```

Preprocessing converts raw PDFs into:

- raw page blocks
- contextualized retrieval blocks
- table parent records
- table row records
- synthetic fallback table rows where needed
- per-document manifests
- a corpus-level summary

The current preprocessed corpus contains 1,170 retrieval records across seven
source PDFs.

Important generated files:

- `llm_rag/ii_preprocessed_documents/summary.json`
- `llm_rag/ii_preprocessed_documents/*/manifest.json`
- `llm_rag/ii_preprocessed_documents/*/retrieval_blocks.jsonl`
- `llm_rag/ii_preprocessed_documents/*/raw_page_blocks.jsonl`
- `llm_rag/ii_preprocessed_documents/*/tables/*.csv`

More detail:

```text
llm_rag/ii_preprocessed_documents/README.md
```

Each processed document folder also has its own README explaining document
role, counts, and table output.

## Stage 3: Reference Retrieval Build

The reference build script is:

```text
llm_rag/iii_vector_db/build_reference_db.py
```

Run from the repo root:

```bash
python llm_rag/iii_vector_db/build_reference_db.py --reset
```

The build stage reads all preprocessed `retrieval_blocks.jsonl` files and
creates:

- `llm_rag/iii_vector_db/reference_corpus.jsonl`
- `llm_rag/iii_vector_db/parent_contexts.jsonl`
- `llm_rag/iii_vector_db/build_summary.json`
- `llm_rag/iii_vector_db/chroma_db/reference_docs/`

The current build contains:

| Metric | Value |
|---|---:|
| Raw retrieval records | 1,170 |
| Built chunks | 1,743 |
| Table rows | 265 |
| Fallback table rows | 38 |
| Synthetic parent contexts | 3 |
| Dense embedding model | `BAAI/bge-m3` |
| Chroma collection | `iucn_reference_docs` |

More detail:

```text
llm_rag/iii_vector_db/README.md
```

## Stage 4: Reference Retrieval At Query Time

The main reference-side query layer is:

```text
llm_rag/iii_vector_db/retrieval_engine.py
```

It supports two retrieval routes.

### Deterministic threshold lookup

Stable Red List numeric facts are encoded in:

```text
llm_rag/iii_vector_db/thresholds.json
```

The threshold route can answer questions about:

- AOO thresholds
- EOO thresholds
- number-of-locations thresholds
- mature-individual thresholds
- extinction-probability thresholds

This avoids asking a generative model to reconstruct official threshold values
from memory.

### Hybrid retrieval

Broader reference questions use a hybrid retriever:

- dense Chroma retrieval
- local BM25-style sparse retrieval
- reciprocal-rank fusion
- heuristic reranking
- table-row boosts
- source and query-intent boosts
- parent table-context attachment

This is needed because IUCN questions often combine semantic language with
exact symbols such as `AOO`, `EOO`, `D2`, `Criterion B`, table labels, and field
names.

## Stage 5: Uploaded-Draft Inference

The inference runtime lives in:

```text
llm_rag/iv_inference/
```

The main orchestration function is:

```python
answer_rag_question(...)
```

The runtime flow is:

1. receive the user question and current uploaded-draft retrieval store
2. look up deterministic threshold facts when relevant
3. retrieve official reference evidence
4. retrieve relevant draft sections from the uploaded assessment
5. build a prompt that keeps reference evidence and draft evidence separate
6. call an external OpenAI-compatible LLM when configured
7. normalize the answer, model metadata, and any request errors
8. build a debug payload for the UI or evaluation notebooks

More detail:

```text
llm_rag/iv_inference/README.md
```

## Runtime Call Chain

The app-facing flow is:

```text
app.py
  -> ui/app_rag_tab.py
    -> llm_rag/iv_inference/rag_runtime.py
      -> llm_rag/iii_vector_db/retrieval_engine.py
      -> llm_rag/iii_vector_db/threshold_lookup.py
      -> llm_rag/iv_inference/draft_retrieval.py
      -> external OpenAI-compatible LLM, when configured
```

The reference asset build flow is:

```text
llm_rag/i_raw_documents/*.pdf
  -> llm_rag/ii_preprocessed_documents/preprocess_pdfs.py
  -> llm_rag/ii_preprocessed_documents/*/retrieval_blocks.jsonl
  -> llm_rag/iii_vector_db/build_reference_db.py
  -> llm_rag/iii_vector_db/reference_corpus.jsonl
  -> llm_rag/iii_vector_db/chroma_db/reference_docs/
```

## Key Design Decisions

### Structure-aware preprocessing

Plain text chunking was not enough for this corpus. Many important Red List
requirements are table rows, short criterion-specific fragments, or field-like
statements. Preprocessing therefore preserves block type, page provenance, table
metadata, and section paths.

### Row-level table retrieval

Tables are represented both as parent table records and as row-level records.
Row-level records improve recall for precise questions. Parent table contexts
provide larger surrounding evidence when a row is selected.

### Synthetic supporting-information tables

The supporting-information PDF contains important visible tables that are not
always extractable as conventional PDF tables. The preprocessing stage
reconstructs synthetic Table 1, Table 2, and Table 3 records from flattened page
text so those requirements remain retrievable.

### Hybrid dense and sparse retrieval

Dense retrieval handles semantic matches. Sparse retrieval handles exact labels,
symbols, and table language. The final result uses both, because Red List review
questions often need both capabilities at once.

### Deterministic thresholds

Official numeric thresholds are stable enough to encode deterministically. This
reduces the chance of generated threshold errors and makes threshold answers
easier to test.

### Separate draft and reference evidence

The uploaded assessment draft is not mixed into the reference corpus. It is
retrieved separately so the prompt can distinguish:

- what official IUCN guidance says
- what the uploaded draft says

This separation is central to review work.

### Debug-first workflow

Retrieval failures are often more important than generation failures. The debug
payload therefore includes route, threshold facts, subqueries, scaffold,
reference evidence, draft hits, full prompt, model output, and errors.

## Evaluation

Evaluation material lives in:

```text
llm_rag/evaluation/
```

It is split into:

- `retrieval_judging/`
  Generates retrieval-only Markdown prompts for manual external LLM judging.
- `smoke_and_inspection/`
  Contains notebooks for inspecting preprocessing outputs, vector-db assets,
  deterministic threshold lookup, and broad retrieval behavior.

Current external LLM retrieval-judging results show:

| Metric | Average |
|---|---:|
| Relevance | 3.50 / 5 |
| Coverage | 3.00 / 5 |
| Focus | 2.67 / 5 |

Those results suggest that retrieval is often directionally useful, but coverage
needs improvement when a question depends on exact threshold rows, cross-linked
reference sections, or nearby draft fields.

More detail:

```text
llm_rag/evaluation/README.md
llm_rag/evaluation/retrieval_judging/README.md
llm_rag/evaluation/smoke_and_inspection/README.md
```

## Testing

Unit tests live in:

```text
llm_rag/unit_tests/
```

Run the full suite from the repo root:

```bash
python -m pytest -q llm_rag/unit_tests
```

Latest full run:

```text
127 passed in 9.60s
```

Run stage-specific suites:

```bash
python -m pytest -q llm_rag/unit_tests/ii_preprocessed_documents
python -m pytest -q llm_rag/unit_tests/iii_vector_db
python -m pytest -q llm_rag/unit_tests/iv_inference
```

More detail:

```text
llm_rag/unit_tests/README.md
```

## Rebuild Guide

Use this when deciding whether to rebuild assets.

| Change | Re-run Preprocessing | Rebuild Vector DB | Re-run Unit Tests | Re-run Evaluation Notebooks |
|---|---:|---:|---:|---:|
| Raw PDFs changed | yes | yes | yes | recommended |
| Preprocessing logic changed | yes | yes | yes | recommended |
| Chunking or build logic changed | no | yes | yes | recommended |
| Threshold data changed | no | no, unless summary assets should refresh | yes | recommended |
| Retrieval reranking changed | no | no | yes | recommended |
| Inference prompt logic changed | no | no | yes | recommended |
| UI-only app changes | no | no | relevant UI tests only | optional |

Common rebuild commands:

```bash
python llm_rag/ii_preprocessed_documents/preprocess_pdfs.py
python llm_rag/iii_vector_db/build_reference_db.py --reset
python -m pytest -q llm_rag/unit_tests
```

## Manual Retrieval Smoke Test

Run a quick reference-side smoke test:

```bash
python llm_rag/iii_vector_db/reference_retrieval_smoke_test.py "What supporting information is required for a threatened species assessment?"
```

This prints route, deterministic output if present, subqueries, scaffold, and
retrieved reference candidates.

## Deployment Notes

Generated Chroma assets in `llm_rag/iii_vector_db/chroma_db/reference_docs/`
are not the same as disposable Python cache. They may be needed if a deployed
environment should start without rebuilding the dense reference index.

Disposable local artifacts include:

- `__pycache__/`
- `.pytest_cache/`
- `.ipynb_checkpoints/`
- `.DS_Store`
- `Thumbs.db`
- `*.pyc`

Those can be removed safely.

## Files In This Folder

- `DEVELOPMENT_HISTORY.md`
  Project evolution notes for the RAG pipeline.
- `README.md`
  This architecture and workflow guide.
- `evaluation/`
  Notebook-based evaluation, retrieval judging, and inspection workflows.
- `i_raw_documents/`
  Raw IUCN reference PDFs.
- `ii_preprocessed_documents/`
  Preprocessed JSONL, manifest, and table assets built from the raw PDFs.
- `iii_vector_db/`
  Reference retrieval build artifacts and query-time retrieval code.
- `iv_inference/`
  Uploaded-draft retrieval, prompt assembly, external LLM calls, and debug
  payload generation.
- `unit_tests/`
  Unit tests grouped by pipeline stage.

## Development History

See:

```text
llm_rag/DEVELOPMENT_HISTORY.md
```

In short, the project moved through these stages:

1. basic reference-only vector search
2. structure-aware preprocessing
3. synthetic recovery of difficult supporting-information tables
4. hybrid dense plus sparse retrieval with parent table contexts
5. deterministic threshold lookup
6. supporting-information query decomposition
7. uploaded-draft retrieval
8. Streamlit app integration
9. retrieval-judging and smoke/inspection evaluation
10. unit-test reorganization and documentation refresh

## Future Work

This is the only README that should hold future-work items for the RAG system.
Stage-level READMEs describe current behavior only.

### 1. Better preprocessing with MarkItDown

The current preprocessing pipeline is deliberately structure-aware, but it is
still custom code tuned around the current IUCN PDF set. A useful next RAG
improvement would be to evaluate Microsoft's
[MarkItDown](https://github.com/microsoft/markitdown) as either a replacement
preprocessing layer or a companion parser.

MarkItDown converts files, including PDFs and Microsoft Office documents, into
Markdown for LLM and text-analysis workflows. For this project, it could help
with:

- preserving headings, lists, tables, and document order in a format that is
  easier for LLM-oriented retrieval to consume
- simplifying the PDF-to-text and DOCX-to-text conversion path
- producing cleaner Markdown chunks for uploaded assessment drafts
- reducing the amount of bespoke parsing logic needed for future document
  formats
- making preprocessing QA easier, because Markdown output can be inspected by
  humans more naturally than raw extracted blocks

This should be tested carefully rather than swapped in blindly. The current
pipeline already preserves page provenance, table metadata, row-level table
records, and synthetic fallback tables. A MarkItDown experiment should compare:

- whether important IUCN table rows are preserved cleanly
- whether page and section provenance can still be attached to each chunk
- whether supporting-information tables are easier or harder to recover
- whether generated Markdown improves retrieval for questions about criteria,
  thresholds, mapping standards, and supporting information
- whether the output is stable enough for deterministic tests and repeatable
  vector-db rebuilds

The best likely direction is a hybrid parser: use MarkItDown for cleaner
document structure where it performs well, while keeping custom recovery logic
for Red List tables or other domain-specific structures that need exact
row-level retrieval.

### 2. GraphRAG-style links between chunks

The current retriever stores chunks, table rows, parent table contexts, and
draft sections, but most relationships between them are implicit. Once a chunk
is retrieved, the system does not have a rich graph of connected evidence that
can reliably pull in adjacent, referenced, or dependent sections.

This shows up in the retrieval-judging results in
`llm_rag/evaluation/retrieval_judging/external_llm_evaluation/llm_output.md`.
Several judged outputs say that retrieval found a broadly relevant document or
section, but missed the specific row, threshold, or nearby draft section needed
for full coverage. For example, a retrieved chunk may introduce the relevant
criterion or topic, but the answer may actually depend on another table row,
annex, subsection, or draft field that the chunk points toward but does not
include.

GraphRAG-style retrieval could make those connections explicit. Potential graph
edges include:

- consecutive chunks within the same section
- child rows linked to parent tables
- parent tables linked back to their row records
- cross-references such as "see section", "see table", "see annex", or
  criterion-specific references
- threshold facts linked to their source criteria
- draft rationale sections linked to related draft fields such as AOO, EOO,
  locations, threats, population trend, and continuing decline
- reference guidance linked to the draft fields it is meant to check

With those links in place, retrieval could start from the highest-scoring
chunks, then expand to graph neighbors that are structurally important even if
their lexical or embedding score is lower. That should help reduce cases where
the retriever finds the "doorway" chunk but misses the room behind it.

### 3. Connected chat context across prompts

The current RAG setup treats prompts as mostly disconnected turns. That is a
reasonable starting point for independent assessment-review questions, but it
means the system does not naturally carry forward retrieval context from one
question to the next.

A connected chat interface could make multi-step review smoother. For example,
a reviewer might first ask about the assessment rationale, then ask whether the
AOO value supports the category, then ask what evidence is missing. Those turns
are related, and the system could use conversation context to:

- reuse relevant retrieved evidence from previous turns
- avoid re-retrieving the same broad background every time
- preserve user intent across follow-up questions
- let the reviewer drill into a specific criterion, document section, or draft
  field without restating all context
- display an accumulated evidence trail for the review session

This should still keep reference evidence and uploaded-draft evidence separate.
The chat history should guide retrieval, not blur the distinction between what
the guidance says and what the uploaded assessment says.

### 4. Expanded retrieval evaluation

The current retrieval-judging workflow creates external-LLM judging prompts for
a small set of questions against one uploaded assessment document.

Useful next steps:

- add more assessment documents with different categories, criteria, and
  document-quality problems
- add more domain-specific questions for thresholds, supporting information,
  mapping standards, threats, population trend, and rationale consistency
- compare retrieval quality across vector-db rebuilds
- track regressions when chunking, preprocessing, reranking, or graph expansion
  changes
- turn the manually summarized external-LLM judgments into a structured
  retrieval-evaluation dataset

This would make it easier to tell whether preprocessing changes, MarkItDown
experiments, GraphRAG links, or chat-context retrieval actually improve evidence
quality.

### 5. Stronger draft retrieval

Draft retrieval currently uses lightweight lexical and heuristic scoring over
the uploaded assessment's parsed section tree.

Future work could include:

- embedded draft retrieval using the same or a lighter embedding model than the
  reference retriever
- graph links between draft sections, such as rationale to AOO, EOO, locations,
  threats, population trend, habitat decline, and bibliography
- curated draft-section relevance labels for representative assessments
- better retrieval over comments, tables, field labels, and section aliases
- comparison between current heuristic retrieval, embedded retrieval, and
  graph-expanded retrieval
