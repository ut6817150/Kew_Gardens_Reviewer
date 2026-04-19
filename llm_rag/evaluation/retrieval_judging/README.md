# retrieval_judging

## Purpose

This folder contains the retrieval-judging prompt-generation workflow.

The goal is to produce self-contained Markdown prompts that can be passed to an
external LLM judge. Each prompt asks the judge to evaluate whether the retrieval
system returned useful evidence for one assessment-review question.

This workflow evaluates retrieval quality only. It does not evaluate the final
LLM answer quality, because the notebook deliberately runs retrieval with
`llm_config=None`.

## Current Flow

The notebook flow is:

1. `generate_external_llm_evaluation_prompts.ipynb` loads `test_document.docx`.
2. The document is parsed with
   `preprocessing.assessment_processor.parse_to_dict(...)`.
3. The parsed assessment is converted into the RAG draft report with
   `build_report_from_assessment(...)`.
4. A draft retrieval store is built with `ensure_draft_store_from_report(...)`.
5. Each configured question is sent through `answer_rag_question(...)` with
   `llm_config=None`.
6. Because `llm_config=None`, no external answer-generation LLM is called.
7. The notebook extracts the retrieval evidence package:
   - selected retrieval route
   - deterministic threshold answer, if present
   - reference documents available to the retriever
   - answer scaffold
   - internal subqueries
   - retrieved reference items
   - retrieved draft hits
8. The retrieval evidence package is inserted into `prompt_skeleton.md`.
9. One Markdown judge prompt is written per question into
   `external_llm_evaluation/`.
10. The generated prompts are manually passed to an external LLM judge.
11. The raw external LLM outputs are saved in
    `external_llm_evaluation/llm_output.md`.

## Why This Flow Is Retrieval-Only

The notebook is designed to evaluate the retrieval system without mixing in the
behavior of a separate answer-generation model. That keeps the evaluation
focused on whether the system found the right reference guidance and the right
sections of the uploaded assessment.

This matters because an answer-generation LLM can sometimes hide weak retrieval
by filling gaps from prior knowledge, or make strong retrieval look worse by
writing a poor answer. Retrieval judging avoids that confusion by judging the
evidence package directly.

## Test Document

`test_document.docx` is the uploaded-assessment fixture used by this workflow.
The notebook parses this document into the same dictionary-style assessment
structure used by the app flow. The generated external-LLM prompts also tell the
judge that the test document may be uploaded alongside the prompt.

If the test document is provided to the judge, it should be used only to check
whether the retrieved draft hits covered the relevant parts of the assessment.
The judge should not give the retrieval system credit for evidence that was
present in the document but absent from the retrieval package.

## What Is Being Evaluated

The external judge prompt asks the LLM to score:

- relevance
  Whether the returned evidence is useful for the question.
- coverage
  Whether the evidence covers the important facts needed to answer the question.
- focus
  Whether the evidence package is clean or noisy.

The judge is told not to grade final answer wording because no final answer is
being tested in this workflow.

## Current External LLM Results

The latest manually saved judge output is in:

```text
llm_rag/evaluation/retrieval_judging/external_llm_evaluation/llm_output.md
```

That file now contains a summary section followed by the raw per-question JSON
outputs. The raw outputs are preserved in code blocks so the original judge
responses remain available for inspection.

Current average scores:

| Metric | Average |
|---|---:|
| Relevance | 3.50 / 5 |
| Coverage | 3.00 / 5 |
| Focus | 2.67 / 5 |

Current question-level results:

| Question | Main Focus | Relevance | Coverage | Focus | Interpretation |
|---:|---|---:|---:|---:|---|
| 1 | EOO/AOO category support | 3 | 2 | 3 | Draft EOO/AOO evidence was found, but the decisive Criterion B threshold evidence was not retrieved clearly enough. |
| 2 | Locations and severe fragmentation | 4 | 4 | 3 | Retrieval was strong for number of locations and Criterion B guidance, but severe-fragmentation evidence was less explicit. |
| 3 | Required or recommended supporting information | 4 | 3 | 2 | The main supporting-information reference was found, but recommended-information coverage was incomplete and some draft hits were noisy. |
| 4 | Mapping and spatial-data checks | 3 | 3 | 3 | Several useful mapping references and the draft map-status section were found, but key draft spatial sections were missed. |
| 5 | Population trend and continuing decline | 3 | 3 | 2 | Some relevant population-trend and continuing-decline evidence was found, but important population and threats narrative was missed. |
| 6 | Assessment-rationale consistency | 4 | 3 | 3 | Rationale guidance was strong, but draft-side support for AOO, locations, and continuing decline was incomplete. |

The overall pattern is that retrieval is often useful but not yet consistently
complete. The most important improvement areas are targeted threshold retrieval,
better draft-section coverage around nearby supporting fields, and reducing
off-topic reference hits when a question needs specific table rows.

## Running The Notebook

Open and run:

```text
llm_rag/evaluation/retrieval_judging/generate_external_llm_evaluation_prompts.ipynb
```

The notebook is designed to be run from Jupyter. It locates the repo root,
imports the existing parser and inference helpers, then writes prompt files into
`external_llm_evaluation/`.

The notebook does not call an external LLM. It only prepares prompt files for
manual judging.

## Output Files

The output files are Markdown prompts named by question number, for example:

```text
question_01_based_on_extent_of_occurrence_eoo_and_area_of_occupancy_aoo_.md
```

Each generated question file can be pasted into an external LLM to obtain a
structured JSON judgment of retrieval quality.

`llm_output.md` is the manually assembled result file. It is not generated by
the notebook. It records the judge outputs from the current prompt set and adds
a short summary so the results can be read quickly.

## Files In This Folder

- `generate_external_llm_evaluation_prompts.ipynb`
  Runs the retrieval-only evaluation setup and writes Markdown judge prompts.
- `prompt_skeleton.md`
  Rubric and prompt template used to create each external-LLM grading prompt.
- `test_document.docx`
  Sample uploaded assessment draft used by the notebook.
- `external_llm_evaluation/`
  Output folder containing generated Markdown prompts and the manually saved
  external LLM output file.
- `README.md`
  This document.

## Related Documentation
- `llm_rag/evaluation/README.md`
- `llm_rag/evaluation/retrieval_judging/external_llm_evaluation/README.md`
- `llm_rag/iv_inference/README.md`
- `llm_rag/iii_vector_db/README.md`
