You are evaluating a retrieval system for an IUCN Red List assessment review assistant.

The retrieval system has two evidence sources:

1. Official IUCN reference material
   - `Guidelines_for_Reporting_Proportion_Threatened_ver_1_2.pdf`
   - `Mapping_Standards_Version_1.20_Jan2024.pdf`
   - `RL_Standards_Consistency.pdf`
   - `RL_categories_and_criteria.pdf`
   - `RL_criteria_summary_sheet.pdf`
   - `RedListGuidelines.pdf`
   - `Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf`

2. The uploaded draft assessment document
   - this is parsed into section-level draft chunks before retrieval
   - it may also be attached to the external LLM conversation as `test_document.docx`

Your job is to judge only the retrieval output. Do not judge a final model answer,
writing style, or wording quality. The question is whether the retrieval system
brought back the right evidence for the user question.

If `test_document.docx` is attached, use it only as a reference for judging
draft-side retrieval coverage. In other words:

- you may check whether the retrieved `draft_hits` include the relevant parts of
  the uploaded assessment
- if the document contains important draft evidence that was not retrieved, count
  that as missing retrieval evidence
- do not give the retrieval system credit for evidence that exists in the
  attached document but is absent from `draft_hits`
- do not use the attached draft document to judge official IUCN reference
  requirements; judge reference-side coverage from `deterministic_answer`,
  `retrieved_reference_items`, and `answer_scaffold`

The input JSON contains:

- `question`: the user question being tested
- `route`: the retrieval route used by the system
- `deterministic_answer`: any deterministic threshold answer returned before retrieval
- `reference_documents`: the reference PDFs available to the system
- `retrieved_reference_items`: reference-corpus chunks returned by retrieval
- `draft_hits`: uploaded-draft chunks returned by retrieval
- `answer_scaffold`: any retrieval-generated scaffold from the reference retriever

Use integer scores from 1 to 5 only, where higher is better.

Scoring rubric:

1. `relevance_score`
How useful are the returned reference items and draft hits for answering the
question?

- `5`: Almost all retrieved evidence is directly useful.
- `4`: Most retrieved evidence is useful, with minor weak items.
- `3`: Mixed; some useful items, some weak or indirect items.
- `2`: Only a small portion of the evidence is useful.
- `1`: The retrieved evidence is mostly not useful.

2. `coverage_score`
How completely does the retrieved evidence cover what is needed to answer the
question?

- `5`: Covers all or nearly all key evidence needed.
- `4`: Covers most key evidence, with minor gaps.
- `3`: Covers some important evidence, but major gaps remain.
- `2`: Covers only a small part of the needed evidence.
- `1`: Fails to cover the necessary evidence.

3. `focus_score`
How clean and non-distracting is the retrieved evidence package?

- `5`: Very focused; almost no irrelevant or distracting content.
- `4`: Mostly focused, with a small amount of noise.
- `3`: Moderate noise or unnecessary material.
- `2`: Significant irrelevant or distracting content.
- `1`: Dominated by irrelevant or distracting content.

Be conservative:

- Do not infer evidence that was not retrieved.
- Penalize missing evidence even if the retrieved items are individually relevant.
- Penalize noisy retrieval even if one or two items are useful.
- If the question requires both reference rules and draft-specific values, judge
  whether both sides were retrieved.
- Treat deterministic threshold answers as part of the evidence package when
  present.
- If `test_document.docx` is attached, use it to identify missed draft-side
  evidence, not to fill gaps in the retrieved evidence package.

Return JSON only.
Do not include Markdown fences.
Do not include text before or after the JSON.

Return JSON with exactly this structure:

{
  "question": "<repeat the user question>",
  "route": "<repeat the route>",
  "relevance_score": 1,
  "coverage_score": 1,
  "focus_score": 1,
  "covered_evidence": [
    "<short description of evidence that was successfully retrieved>"
  ],
  "missing_evidence": [
    "<short description of important evidence that was not retrieved>"
  ],
  "reference_item_labels": [
    {
      "rank": 1,
      "source": "<pdf or file name>",
      "page": 1,
      "section": "<section title>",
      "label": "useful | partly_useful | not_useful",
      "reason": "<short evidence-focused reason>"
    }
  ],
  "draft_hit_labels": [
    {
      "rank": 1,
      "section_path": "<draft section path>",
      "score": 1.0,
      "label": "useful | partly_useful | not_useful",
      "reason": "<short evidence-focused reason>"
    }
  ],
  "summary": "<1-3 sentence summary of retrieval quality>"
}

Input:
{...JSON loaded...}
