# iii_vector_db

## Purpose

This folder contains the reference-side retrieval system used by the RAG app in the repo root.

It turns the preprocessed retrieval blocks into:

- a persistent Chroma dense index
- a sparse JSONL corpus for lexical retrieval
- a parent-context store for table expansion
- a deterministic threshold lookup layer

## Main files

- `build_reference_db.py`: builds the retrieval assets from `ii_preprocessed_documents/`
- `retrieval_engine.py`: runs deterministic threshold lookup plus hybrid retrieval at query time
- `embedding_loader.py`: resolves and loads the embedding model more robustly, with sparse-only fallback support
- `threshold_lookup.py`: deterministic answers for official thresholds
- `test_reference_db.py`: lightweight local retrieval smoke test

## Main outputs

- `reference_corpus.jsonl`
- `parent_contexts.jsonl`
- `build_summary.json`
- `thresholds.json`
- `chroma_db/reference_docs/`

## Build flow

### 1. Read retrieval blocks

`build_reference_db.py` reads the `retrieval_blocks.jsonl` files created in `llm_rag/ii_preprocessed_documents/` relative to the repo root.

### 2. Chunk by block type

The build step treats block types differently:

- `table_row` stays atomic
- `table` uses larger chunks
- long narrative `text` uses standard chunking

### 3. Embed and index

The dense retrieval model is `BAAI/bge-m3`.

The embeddings are stored in Chroma under:

```text
llm_rag/iii_vector_db/chroma_db/reference_docs/
```

### 4. Export sparse and parent-context data

The build also writes:

- `reference_corpus.jsonl` for sparse retrieval
- `parent_contexts.jsonl` for table expansion
- `build_summary.json` for a quick build snapshot

## Query-time flow

At inference time the retrieval engine follows two lanes.

### Lane 1: deterministic threshold lookup

Questions about official thresholds such as AOO, EOO, number of locations, mature individuals, or extinction probability are handled by `threshold_lookup.py` when possible.

This reduces the risk of the LLM inventing numeric facts.

### Lane 2: hybrid reference retrieval

When the query is not answered deterministically, `retrieval_engine.py` performs:

1. dense retrieval over Chroma
2. sparse retrieval over the JSONL corpus
3. fusion and heuristic reranking
4. parent table-context attachment when relevant
5. final candidate selection
6. answer-scaffold construction for the prompt/debug layer

## Supporting-information handling

Supporting-information questions get extra treatment because the important evidence often lives across Table 1 and Table 2.

The retrieval engine can:

- expand one user query into multiple internal subqueries
- boost table-specific evidence differently for baseline and conditional requirements
- enforce coverage when a key table would otherwise be missing
- build a scaffold that separates:
  - requirements for all assessments
  - additional requirements under specific conditions

## Robustness notes

`embedding_loader.py` was added to make embedding loading less fragile:

- it prefers cached local Hugging Face snapshots first
- it can fall back to a network download when needed
- inference can continue in sparse-only mode if embeddings cannot be loaded

## Rebuild commands

From the repo root:

```bash
python llm_rag/iii_vector_db/build_reference_db.py --reset
```

Optional smoke tests:

```bash
python llm_rag/iii_vector_db/test_reference_db.py "What supporting information is required for a threatened species assessment?"
python llm_rag/iii_vector_db/test_reference_db.py "What are the Criterion B thresholds for EOO and AOO?"
```

## When you need to rebuild

Rebuild the reference DB if:

- the PDFs change
- preprocessing output changes
- chunking or embedding settings change
- retrieval-build logic changes

You do not need to rebuild just because the app UI or debug presentation changed.
