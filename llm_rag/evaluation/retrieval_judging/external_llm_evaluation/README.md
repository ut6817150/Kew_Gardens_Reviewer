# external_llm_evaluation

## Purpose

This folder stores the generated Markdown prompts and manually saved outputs for
external LLM retrieval judging.

The generated prompts are produced by:

```text
llm_rag/evaluation/retrieval_judging/generate_external_llm_evaluation_prompts.ipynb
```

The prompts are designed to evaluate retrieval quality only. They ask an
external LLM judge to inspect the retrieval package returned by the RAG system
and score whether the retrieved evidence is relevant, complete, and focused.

## How The Prompt Files Are Created

The parent notebook loads `test_document.docx`, parses it into an assessment
dictionary, builds the draft retrieval store, and runs each evaluation question
through the RAG retrieval path with `llm_config=None`. No answer-generation LLM
is called during this step.

Each generated prompt contains:

- the retrieval-evaluation rubric from `prompt_skeleton.md`
- one user question
- the retrieval route used
- deterministic threshold evidence, if any
- retrieved reference items
- retrieved draft hits from `test_document.docx`
- the reference documents available to the retriever

The prompt also tells the external judge that the test document may be uploaded
alongside the prompt. If the document is provided, it should only be used to
check whether the retrieved draft hits covered the relevant assessment content.
The retrieval system should not receive credit for evidence that exists in the
document but was not retrieved.

## How To Use The Generated Prompts

1. Open one generated `question_*.md` file.
2. Paste the full contents into an external LLM.
3. Optionally upload `test_document.docx` to the same LLM conversation.
4. Ask the LLM to return the JSON judgment requested by the prompt.
5. Save the returned JSON in `llm_output.md` if it should become part of the
   tracked evaluation record.

The generated prompts are intended for manual evaluation. They are not runtime
inputs for the app or inference pipeline.

## Current Saved Output

`llm_output.md` contains the current manually saved external-LLM judgments for
the six generated prompts.

The file is structured as:

- a short description of what the file contains
- a summary section derived from the raw JSON outputs
- a score table for the six judged questions
- per-question notes
- the raw LLM outputs in per-question JSON code blocks

The raw outputs are intentionally preserved underneath the summary. This makes
the file useful both as a quick status report and as an audit trail of the
external judge's original response.

## Current Score Summary

| Question | Evaluation Focus | Relevance | Coverage | Focus |
|---:|---|---:|---:|---:|
| 1 | EOO/AOO category support | 3 | 2 | 3 |
| 2 | Locations and severe fragmentation | 4 | 4 | 3 |
| 3 | Supporting information requirements | 4 | 3 | 2 |
| 4 | Mapping and spatial-data checks | 3 | 3 | 3 |
| 5 | Population trend and continuing decline | 3 | 3 | 2 |
| 6 | Assessment rationale consistency | 4 | 3 | 3 |

Average scores:

| Metric | Average |
|---|---:|
| Relevance | 3.50 / 5 |
| Coverage | 3.00 / 5 |
| Focus | 2.67 / 5 |

The current results suggest that retrieval is often directionally useful but
still incomplete for more exacting review questions. The most frequent weakness
is not that retrieval finds nothing; it is that it finds broad relevant
documents while missing the specific threshold rows, criterion details, or draft
sections needed for full coverage.

## Reading The Results

Use the scores as a guide to where retrieval needs attention:

- Higher relevance means the returned evidence is broadly connected to the
  question.
- Higher coverage means the returned evidence contains enough of the necessary
  facts to answer the question.
- Higher focus means the returned package is not padded with off-topic or weakly
  related hits.

The current run shows the best performance on the locations question because it
retrieved both the draft locations evidence and relevant Criterion B guidance.
It shows the clearest weakness on the EOO/AOO category question because the
retrieval package included EOO/AOO values but did not retrieve the official
threshold evidence needed to classify those values confidently.

## Files In This Folder

- `question_*.md`
  Generated retrieval-judging prompts, one per tested question.
- `llm_output.md`
  Manually saved external LLM judgments, including a summary and raw JSON
  outputs.
- `README.md`
  This document.

## Related Documentation
- `llm_rag/evaluation/retrieval_judging/README.md`
- `llm_rag/evaluation/README.md`
- `llm_rag/README.md`
