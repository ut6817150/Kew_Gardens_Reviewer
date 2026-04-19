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
{
  "question": "What evidence does the uploaded assessment provide for number of locations and severe fragmentation under Criterion B?",
  "route": "deterministic_plus_hybrid_rag",
  "deterministic_answer": "Typical number of locations thresholds under Criterion B: CR = 1, EN ≤ 5, VU ≤ 10.",
  "reference_documents": [
    "Guidelines_for_Reporting_Proportion_Threatened_ver_1_2.pdf",
    "Mapping_Standards_Version_1.20_Jan2024.pdf",
    "RL_Standards_Consistency.pdf",
    "RL_categories_and_criteria.pdf",
    "RL_criteria_summary_sheet.pdf",
    "RedListGuidelines.pdf",
    "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf"
  ],
  "answer_scaffold": "Answer scaffold\n---------------\nMost relevant retrieved reference evidence:\n- Section: 6. Guidelines for Applying Criterion B | Source: RedListGuidelines.pdf | Page: 72 | Evidence: . Therefore, if a taxon has met the distributional requirement for the Endangered category and option (c) extreme fluctuation, but none of the other options, it would not qualify as Endangered (or Vulnerable) under criterion B. To qualify, it would also have t\n- Section: 12. Guidelines for Threatening Processes | Source: RedListGuidelines.pdf | Page: 98 | Evidence: If a taxon is not currently severely fragmented (see section 4.8 ), this cannot be used to meet the severe fragmentation subcriteria (e.g., criterion B1a) even if there is evidence to infer that it may become so under future climates. However, projected future\n- Section: Common Errors | Source: RL_Standards_Consistency.pdf | Page: 65 | Evidence: 62 o If criteria B1a or B2a are used, check that the assessment is clear about whether severe fragmentation or number of locations have been used for the assessment. Also check that the number of locations has been estimated appropriately (based on the most se\n- Section: 6. Guidelines for Applying Criterion B | Source: RedListGuidelines.pdf | Page: 73 | Evidence: assessors make this distinction by explicitly specifying in their documentation: (1) whether the taxon is severely fragmented, and (2) the number of locations. Some of the problems encountered when applying criterion B are dealt with elsewhere in this document",
  "subqueries": [
    "What evidence does the uploaded assessment provide for number of locations and severe fragmentation under Criterion B?"
  ],
  "retrieved_reference_items": [
    {
      "rank": 1,
      "block_type": "text",
      "source": "RedListGuidelines.pdf",
      "page": 72,
      "section": "6. Guidelines for Applying Criterion B",
      "text": ". Therefore, if a taxon has met the distributional requirement for the Endangered category and option (c) extreme fluctuation, but none of the other options, it would not qualify as Endangered (or Vulnerable) under criterion B. To qualify, it would also have to meet either (a) or (b). An example of the proper use of criterion B is Endangered: B1ab(v). This means that the taxon is judged to have an extent of occurrence of less than 5,000 km 2 , the population is severely fragmented or known to exist at no more than five locations, and there is a continuing decline in the number of mature individuals. Subcriterion (a) requires severe fragmentation and/or limited number of locations. The numbering in the criteria does not allow distinguishing between these two conditions. We recommend that"
    },
    {
      "rank": 2,
      "block_type": "text",
      "source": "RedListGuidelines.pdf",
      "page": 98,
      "section": "12. Guidelines for Threatening Processes",
      "text": "If a taxon is not currently severely fragmented (see section 4.8 ), this cannot be used to meet the severe fragmentation subcriteria (e.g., criterion B1a) even if there is evidence to infer that it may become so under future climates. However, projected future fragmentation can be used to infer continuing decline, if certain conditions are met. Continuing decline is recent, current or projected future decline (see section 4.6 ). Severe fragmentation can for some species lead to local extinctions of subpopulations inhabiting the smallest habitat fragments. If the population density and the projected distribution of fragments justify a prediction of increasing rate of local extinctions in the near future, this may be used to infer continuing future decline in population size"
    },
    {
      "rank": 3,
      "block_type": "text",
      "source": "RL_Standards_Consistency.pdf",
      "page": 65,
      "section": "Common Errors",
      "text": "62 o If criteria B1a or B2a are used, check that the assessment is clear about whether severe fragmentation or number of locations have been used for the assessment. Also check that the number of locations has been estimated appropriately (based on the most serious threat rather than simply on collection sites). o If VU D2 is used, check that there is a plausible threat to the species rather than having a restricted range and no threats at all. o If criterion E is used, ensure the quantitative model (with the assumptions used in this) is available for inspection. • Check for contradictions between information in the summary documentation and the data fields (e.g., text says population has declined by 32% but data field records decline of at least 50%)"
    },
    {
      "rank": 4,
      "block_type": "text",
      "source": "RedListGuidelines.pdf",
      "page": 73,
      "section": "6. Guidelines for Applying Criterion B",
      "text": "assessors make this distinction by explicitly specifying in their documentation: (1) whether the taxon is severely fragmented, and (2) the number of locations. Some of the problems encountered when applying criterion B are dealt with elsewhere in this document, i.e. definitions of \"subpopulations\" ( section 4.2 ), \"location\" ( section 4.11 ), \"continuing decline\" ( section 4.6) , \"extreme fluctuations\" ( section 4.7 ), \"severely fragmented\" ( section 4.8 ), \"extent of occurrence\" ( section 4.9 ) and \"area of occupancy\" ( section 4.10 ). The different types of information used in criterion B need not be based on the same area at the same time of the year"
    },
    {
      "rank": 5,
      "block_type": "text",
      "source": "RedListGuidelines.pdf",
      "page": 51,
      "section": "Indian Ocean",
      "text": ". 2000). The definition of severe fragmentation is based on the distribution of subpopulations. This is often confused with the concept of \"location\" (see section 4.11 ), but is independent of it. A taxon may be severely fragmented, yet all the isolated subpopulations may be threatened by the same major factor (single location), or each subpopulation may be threatened by a different factor (many locations). Also, severe fragmentation does not require an ongoing threat; small and isolated subpopulations of a severely fragmented taxon can go extinct due to natural, stochastic (demographic and environmental) processes. 4.9 Extent of occurrence (criteria A and B)"
    },
    {
      "rank": 6,
      "block_type": "text",
      "source": "RL_categories_and_criteria.pdf",
      "page": 33,
      "section": "≤ 10 (a) Severely fragmented OR Number of locations",
      "text": "< 5,000 km2 < 500 km2 B. Geographic range in the form of either B1 (extent of occurrence) AND/OR B2 (area of occupancy) ≥ 70% ≥ 50% ≤ 5 based on any of the following: < 100 km2 < 10 km2 A1 Population reduction observed, estimated, inferred, or suspected in the past where the causes of the reduction are clearly reversible AND understood AND have ceased. A2 Population reduction observed, estimated, inferred, or suspected in the past where the causes of reduction may not have ceased OR may not be understood OR may not be reversible. A3 Population reduction projected, inferred or suspected to be met in the future (up to a maximum of 100 years) [(a) cannot be used for A3]"
    },
    {
      "rank": 7,
      "block_type": "text",
      "source": "RL_Standards_Consistency.pdf",
      "page": 19,
      "section": "2.2.2. Geographic Range",
      "text": "1) Estimated extent of occurrence (EOO) in km², with an indication of how this was estimated. 2) Estimated area of occupancy (AOO) in km², with an indication of how this was estimated. 3) Estimated number of locations (if number of locations is the basis of using criteria B1a+2a) with reference to major threats and how these affect the taxon to justify this estimate. 3) Justification of why the taxon is severely fragmented (if severe fragmentation is the basis of using criteria B1a+2a)."
    },
    {
      "rank": 8,
      "block_type": "text",
      "source": "RL_categories_and_criteria.pdf",
      "page": 16,
      "section": "8. Severely fragmented (Criterion B)",
      "text": "The phrase ‘severely fragmented’ refers to the situation in which increased extinction risk to the taxon results from the fact that most of its individuals are found in small and relatively isolated subpopulations (in certain circumstances this may be inferred from habitat information). These small subpopulations may go extinct, with a reduced probability of recolonization."
    }
  ],
  "draft_hits": [
    {
      "rank": 1,
      "section_path": "test_document > Distribution > Locations Information",
      "source_key": "test_document > Distribution > Locations Information",
      "score": 8.25,
      "text": "<table> <tr><td><b>Number of Locations</b></td><td><b>Justification</b></td></tr> <tr><td>11</td><td>It occupies total 11 locations in DRC and Zambia. The specimens have been collected from 11 locations and each of them face different threats ranging from mining, agricultural land-use to dam development.</td></tr> </table>"
    },
    {
      "rank": 2,
      "section_path": "test_document > Distribution > Very restricted AOO or number of locations (triggers VU D2)",
      "source_key": "test_document > Distribution > Very restricted AOO or number of locations (triggers VU D2)",
      "score": 8.2,
      "text": "<table> <tr><td><b>Very restricted in area of occupancy (AOO) and/or # of locations</b></td><td><b>Justification</b></td></tr> <tr><td>No</td><td>The species is found at multiple locations in Zambia and DRC.</td></tr> </table>"
    },
    {
      "rank": 3,
      "section_path": "test_document > Red List Assessment > Assessment Rationale",
      "source_key": "test_document > Red List Assessment > Assessment Rationale",
      "score": 8.15,
      "text": "<p>Bulbostylis atracuminata occurs in a restricted area ranging from southern DRC to northern Zambia. Its extent of occurrence of about 114,000 km2, exceeds the threshold value for threatened categories. However, its actual suitable habitat within this range is considered to be smaller because its habitat is restricted to riverbanks and wetlands. This is reflected in the area of occupancy (44 km2), which falls within the threshold values for the Endangered category under criterion B2. There are an estimated 11 locations for this species in Zambia and DRC, with some locations found in natural environments with low human disturbance levels. However, the majority of its habitat is threatened by mining activity and agricultural land conversion, resulting in an overall continuing deterioration of habitat quality.. Though this species has a restricted AOO and is experiencing a continuing decline in habitat quality, the total number of locations marginally exceeds the threshold value for threatened categories under criterion B, and the existence of some high-quality habitat may enable the species to persist without being globally threatened in the near future. Therefore, this species is assessed as Endangered B2ab(iii). Despite these pressures, the species is unlikely to face major threats in the near future.</p>"
    },
    {
      "rank": 4,
      "section_path": "test_document > Threats",
      "source_key": "test_document > Threats",
      "score": 3.0,
      "text": "<p>The habitats of copper bog sedge are predominantly threatened by mining activity and agricultural land conversion (Global Forest Watch 2023, Google Earth Pro 2023). Five out of the total eleven locations are situated on mining sites managed by different mining companies in both Zambia and the DRC (Global Forest Watch 2023). Four locations in the DRC are currently experiencing the threat of agricultural land conversion (Google Earth Pro 2023), with three of these locations affected by both mining and agricultural land use simultaneously. Agricultural land conversion is likely to continue affecting these local subpopulations in the future. Additionally, a dam has been identified near one location in the DRC (Google Earth Pro 2023). This could affect the soil moisture content, subsequently impacting the habitat suitability for this species. Three locations in the DRC are situated within two national parks (IUCN and UNEP-WCMC 2023). While national parks can provide some protection from human disturbance, it's important to note that these parks are not strictly managed and remain open to activities such as bird watching, hiking, walking, and camping (Virunga National Park 2023). Currently, the habitats within the national parks are not threatened. Several sites are also increasingly affected by road construction and urban expansion. Unregulated fire is also a serious threat in some areas.</p>"
    },
    {
      "rank": 5,
      "section_path": "test_document > Red List Assessment > Assessment Information",
      "source_key": "test_document > Red List Assessment > Assessment Information",
      "score": 1.75,
      "text": "<p><b>Assessor(s): </b>Lemboye, B. & Lyu, E</p> <p><b>Institution(s): </b>Royal Botanic Gardens, Kew</p> <p><b>Regions: </b>Global</p>"
    },
    {
      "rank": 6,
      "section_path": "test_document > Distribution > Map Status",
      "source_key": "test_document > Distribution > Map Status",
      "score": 1.75,
      "text": "<table> <tr><td><b>Map Status</b></td><td><b>Use map from previous assessment</b></td><td><b>How the map was created, including data sources/methods used:</b></td><td><b>Please state reason for map not available:</b></td><td><b>Data Sensitive?</b></td><td><b>Justification</b></td><td><b>Geographic range this applies to:</b></td><td><b>Date restriction imposed:</b></td></tr> <tr><td>Done</td><td>-</td><td>Prepared in GeoCat (Moat et al. 2023) based on 11 georeferenced herbarium specimens (Global Biodiversity Information Facility 2023)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr> </table>"
    }
  ]
}
