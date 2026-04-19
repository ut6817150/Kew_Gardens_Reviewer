# evaluation

## Purpose

This folder contains notebook-based evaluation and inspection workflows for the
`llm_rag` retrieval pipeline.

The evaluation material is split into two groups:

- `retrieval_judging/`
  Builds external-LLM grading prompts for retrieval quality, using a real test
  assessment document and the same retrieval helpers used during inference.
- `smoke_and_inspection/`
  Contains notebooks for inspecting preprocessing outputs, vector-db assets,
  deterministic threshold lookup, and broad retrieval behavior.

These notebooks complement the automated unit tests in `llm_rag/unit_tests/`.
Unit tests check whether code paths behave as expected. The evaluation notebooks
make retrieval quality, asset health, routing behavior, and retrieved evidence
visible for human review.

## Folder Structure

```text
llm_rag/evaluation/
|- README.md
|- retrieval_judging/
|  |- external_llm_evaluation/
|  |- generate_external_llm_evaluation_prompts.ipynb
|  |- prompt_skeleton.md
|  `- test_document.docx
`- smoke_and_inspection/
   |- eval_retrieval_smoke.ipynb
   |- eval_threshold_lookup.ipynb
   |- inspect_preprocessed_docs.ipynb
   `- inspect_vector_db_assets.ipynb
```

## Current Evaluation Outputs

The current evaluation folder contains two kinds of results:

- Retrieval-judging outputs saved in
  `llm_rag/evaluation/retrieval_judging/external_llm_evaluation/llm_output.md`.
- Saved notebook outputs inside the notebooks in
  `llm_rag/evaluation/smoke_and_inspection/`.

The retrieval-judging run evaluated six assessment-review questions. The
external LLM judged retrieval relevance, coverage, and focus on a 1-5 scale.
The current average scores are:

| Metric | Average |
|---|---:|
| Relevance | 3.50 / 5 |
| Coverage | 3.00 / 5 |
| Focus | 2.67 / 5 |

The broad pattern is that retrieval often finds partially useful evidence, but
coverage is uneven when the question needs very specific criterion thresholds,
table rows, or nearby draft sections. The strongest judged result is the
locations and severe-fragmentation question, where the retrieval package found
the locations table, the assessment rationale, threat context, and relevant
Criterion B guidance. The weakest judged result is the EOO/AOO-only category
question, where draft values were retrieved but the decisive Criterion B
threshold evidence was not retrieved clearly enough.

The smoke and inspection notebooks have also been run and now contain saved
cell outputs. Those outputs show that preprocessing and vector-db assets are
present, deterministic threshold routing is working for threshold-style prompts,
and broad smoke prompts exercise both deterministic and hybrid retrieval routes.

## Retrieval Judging

`retrieval_judging/` is for judging whether the retrieval system returned good
evidence for a question. It does not evaluate the final response from an answer
generation LLM.

The current flow is:

1. Load `test_document.docx`.
2. Parse it with `preprocessing.assessment_processor.parse_to_dict(...)`.
3. Build the draft retrieval store with `llm_rag.iv_inference` helpers.
4. Run retrieval for several questions without calling an answer-generation LLM.
5. Build a retrieval-only payload for each question.
6. Insert that payload into `prompt_skeleton.md`.
7. Write one Markdown prompt per question into `external_llm_evaluation/`.
8. Manually pass each prompt to an external LLM judge.
9. Save the external LLM's raw JSON outputs in `llm_output.md`.

`llm_output.md` now contains a human-readable summary above the raw outputs. The
raw outputs are still preserved in per-question JSON code blocks so the original
judge responses remain inspectable.

## Smoke And Inspection

`smoke_and_inspection/` is for understanding the pipeline before or after
changes. The notebooks are useful when retrieval behavior changes and we need to
work out whether the cause is preprocessing, vector-db build output,
deterministic routing, or hybrid retrieval.

The recommended order is:

1. `inspect_preprocessed_docs.ipynb`
   Check that preprocessing outputs look sensible.
2. `inspect_vector_db_assets.ipynb`
   Check that the built retrieval assets look sensible.
3. `eval_threshold_lookup.ipynb`
   Check deterministic threshold lookup in isolation.
4. `eval_retrieval_smoke.ipynb`
   Run natural-language retrieval smoke prompts.

The current saved notebook outputs show:

- `inspect_preprocessed_docs.ipynb`
  Found seven preprocessed source-document folders. The inspected target,
  `Guidelines_for_Reporting_Proportion_Threatened_ver_1_2`, contains 38 raw
  page blocks and 14 retrieval blocks.
- `inspect_vector_db_assets.ipynb`
  Found a built Chroma asset set with 1,170 raw records, 1,743 chunks, 265 table
  rows, 38 fallback table rows, three synthetic parent contexts, and threshold
  entries for Criteria A-E.
- `eval_threshold_lookup.ipynb`
  Tested eight prompts. Seven were recognized as deterministic threshold
  queries and one general documentation prompt was correctly left outside the
  deterministic threshold path.
- `eval_retrieval_smoke.ipynb`
  Tested eight natural-language prompts. The saved outputs show both
  `hybrid_rag` and `deterministic_threshold_lookup` routes, including internal
  subqueries, answer scaffolds, and retrieved candidates for the hybrid runs.

## How To Read The Evaluation Results

The retrieval-judging results and smoke notebooks answer slightly different
questions:

- Retrieval judging asks whether a retrieved evidence package is good enough
  for an external judge to answer a specific assessment-review question.
- Smoke testing asks whether routing, retrieval, deterministic lookup, and
  asset loading are behaving sensibly across representative prompts.
- Inspection notebooks ask whether the underlying preprocessed and vector-db
  assets look healthy.

Read the retrieval-judging scores as a quality signal, not as a formal
benchmark. The current results are especially useful for identifying where
retrieval needs better targeted table-row retrieval, stronger draft-section
coverage, or cleaner filtering of broad but low-value reference hits.

## Files In This Folder

- `README.md`
  This document.
- `retrieval_judging/`
  Retrieval-evaluation prompt generation and manually saved external LLM
  judgments.
- `smoke_and_inspection/`
  Inspection and smoke-test notebooks for the pipeline.

## Related Documentation
- `llm_rag/README.md`
- `llm_rag/evaluation/retrieval_judging/README.md`
- `llm_rag/evaluation/smoke_and_inspection/README.md`
- `llm_rag/unit_tests/README.md`
