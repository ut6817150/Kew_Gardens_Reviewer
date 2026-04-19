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
  "question": "Based on Extent of occurrence (EOO) and Area of occupancy (AOO) only, what IUCN category would the uploaded assessment support?",
  "route": "hybrid_rag",
  "deterministic_answer": null,
  "reference_documents": [
    "Guidelines_for_Reporting_Proportion_Threatened_ver_1_2.pdf",
    "Mapping_Standards_Version_1.20_Jan2024.pdf",
    "RL_Standards_Consistency.pdf",
    "RL_categories_and_criteria.pdf",
    "RL_criteria_summary_sheet.pdf",
    "RedListGuidelines.pdf",
    "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf"
  ],
  "answer_scaffold": "Answer scaffold\n---------------\nMost relevant retrieved reference evidence:\n- Section: 7.1.1 How should EOO be calculated for polygon maps and point maps? | Source: Mapping_Standards_Version_1.20_Jan2024.pdf | Page: 21 | Evidence: Figure 1 : Schematic representation of polygon presence coding and calculation of EOO. Figure 2: Schematic representation of point presence coding and calculation of EOO. 7.2 Area of occupancy (AOO) Area of occupancy (AOO) is a scaled metric that represents th\n- Section: 7.1 Extent of occurrence (EOO) | Source: Mapping_Standards_Version_1.20_Jan2024.pdf | Page: 19 | Evidence: Extent of occurrence (EOO) is defined as \"the area contained within the shortest continuous imaginary boundary which can be drawn to encompass all the known, inferred or projected sites of present occurrence of a taxon, excluding cases of vagrancy\". EOO is a p\n- Section: Annex 4: Summary of the IUCN Red List Criteria | Source: RL_categories_and_criteria.pdf | Page: 33 | Evidence: . 1 (b) Continuing decline observed, estimated, inferred or projected in any of: (i) extent of occurrence; (ii) area of occupancy; (iii) area, extent and/or quality of habitat; (iv) number of locations or subpopulations; (v) number of mature individuals (c) Ex\n- Section: 4.10 Area of occupancy (criteria A, B and D) | Source: RedListGuidelines.pdf | Page: 54 | Evidence: . 2018). Habitat maps with higher resolutions can be used for other aspects of a Red List assessment, such as calculating reduction in habitat quality as a basis of population reduction for criterion A2(c) or estimating continuing decline in habitat area for B",
  "subqueries": [
    "Based on Extent of occurrence (EOO) and Area of occupancy (AOO) only, what IUCN category would the uploaded assessment support?"
  ],
  "retrieved_reference_items": [
    {
      "rank": 1,
      "block_type": "text",
      "source": "Mapping_Standards_Version_1.20_Jan2024.pdf",
      "page": 21,
      "section": "7.1.1 How should EOO be calculated for polygon maps and point maps?",
      "text": "Figure 1 : Schematic representation of polygon presence coding and calculation of EOO. Figure 2: Schematic representation of point presence coding and calculation of EOO. 7.2 Area of occupancy (AOO) Area of occupancy (AOO) is a scaled metric that represents the area of suitable habitat currently occupied by the taxon. Area of occupancy is included in the IUCN Red List Criteria for two main reasons: 21"
    },
    {
      "rank": 2,
      "block_type": "text",
      "source": "Mapping_Standards_Version_1.20_Jan2024.pdf",
      "page": 19,
      "section": "7.1 Extent of occurrence (EOO)",
      "text": "Extent of occurrence (EOO) is defined as \"the area contained within the shortest continuous imaginary boundary which can be drawn to encompass all the known, inferred or projected sites of present occurrence of a taxon, excluding cases of vagrancy\". EOO is a parameter that measures the spatial spread of the areas currently occupied by the taxon. It is included in the IUCN Red List Criteria as a measure of the degree to which risks from threats are spread spatially across the taxon’s geographical distribution (see section 4.9 of the Guidelines for Using the IUCN Red List Categories and Criteria )."
    },
    {
      "rank": 3,
      "block_type": "text",
      "source": "RL_categories_and_criteria.pdf",
      "page": 33,
      "section": "Annex 4: Summary of the IUCN Red List Criteria",
      "text": ". 1 (b) Continuing decline observed, estimated, inferred or projected in any of: (i) extent of occurrence; (ii) area of occupancy; (iii) area, extent and/or quality of habitat; (iv) number of locations or subpopulations; (v) number of mature individuals (c) Extreme ﬂuctuations in any of: (i) extent of occurrence; (ii) area of occupancy; (iii) number of locations or subpopulations; (iv) number of mature individuals (a) direct observation [except A3] (b) an index of abundance appropriate to the taxon (c) a decline in area of occupancy (AOO), extent of occurrence (EOO) and/or habitat quality (d) actual or potential levels of exploitation (e) eﬀects of introduced taxa, hybridization, pathogens, pollutants, competitors or parasites. A. Population size reduction"
    },
    {
      "rank": 4,
      "block_type": "text",
      "source": "RedListGuidelines.pdf",
      "page": 54,
      "section": "4.10 Area of occupancy (criteria A, B and D)",
      "text": ". 2018). Habitat maps with higher resolutions can be used for other aspects of a Red List assessment, such as calculating reduction in habitat quality as a basis of population reduction for criterion A2(c) or estimating continuing decline in habitat area for B2(b), as well as for conservation planning. Recognizing the role of AOO and the importance of valid scaling, IUCN (2001, 2012b) includes the following text: “Area of occupancy is defined as the area within its 'extent of occurrence' (see 4.9 above), which is occupied by a taxon, excluding cases of vagrancy. The measure reflects the fact that a taxon will not usually occur throughout the area of its extent of occurrence, which may"
    },
    {
      "rank": 5,
      "block_type": "text",
      "source": "Mapping_Standards_Version_1.20_Jan2024.pdf",
      "page": 20,
      "section": "7.1.1 How should EOO be calculated for polygon maps and point maps?",
      "text": "Note that MCPs are in some cases considered less suitable as a method for comparing two or more temporal estimates of EOO for assessing reductions or continuing declines. Therefore, a method such as the α-hull (a generalization of a convex hull) is recommended for assessing reductions in EOO. For further information, refer to Section 4.9 in the Guidelines for Using the IUCN Red List Categories and Criteria . In the case of migratory species, EOO should be based on the minimum of the breeding or nonbreeding (wintering) areas, but not both, because such species are dependent on both areas, and the bulk of the population is found in only one of these areas at any time"
    },
    {
      "rank": 6,
      "block_type": "text",
      "source": "RedListGuidelines.pdf",
      "page": 60,
      "section": "4.10 Area of occupancy (criteria A, B and D)",
      "text": "AOO should be used for taxa in all types of habitat distribution, including taxa with linear ranges living in rivers or along coastlines. 4.10.7 AOO and EOO based on habitat maps and models Both AOO and EOO may be estimated based on “…known, inferred or projected sites of present occurrences…” (IUCN 2001). In this case, ‘known’ refers to confirmed extant records of the taxon; ‘inferred’ refers to the use of information about habitat characteristics, dispersal capability, rates and effects of habitat destruction and other relevant factors, based on known sites, to deduce a very high likelihood of presence at other sites; and ‘projected’ refers to spatially predicted sites on the basis of habitat maps or models, subject to the three conditions outlined below"
    },
    {
      "rank": 7,
      "block_type": "text",
      "source": "RedListGuidelines.pdf",
      "page": 62,
      "section": "4.10 Area of occupancy (criteria A, B and D)",
      "text": "Underestimation of AOO will affect the outcome of Red List assessments under criterion B2, e.g. if the estimated AOO is less than, or close to, 2,000 km 2 , the lower threshold of the VU category. In such cases, assessors may not be able to justify the assumption that AOO is estimated accurately from a simple intersection of current records with a standard 2 × 2 km grid, and an alternative assumption must be made in support of a more accurate estimate. Assessors should follow section 3.2 to deal with uncertainty in estimates of AOO for potentially threatened taxa that have poorly sampled distributions. A plausible lower bound of AOO would be no smaller than that based on an intersection of current records with a 2 × 2 km grid, but could be larger"
    },
    {
      "rank": 8,
      "block_type": "text",
      "source": "RedListGuidelines.pdf",
      "page": 56,
      "section": "4.10 Area of occupancy (criteria A, B and D)",
      "text": ". Scales finer (smaller) than 2 × 2 km grid size tend to list more taxa at higher threat categories than the definitions of these categories imply. Assessors should avoid using estimates of AOO at other scales. The scale for AOO should not be based on EOO (or other measures of range area), because AOO and EOO measure different factors affecting extinction risk (see below). If AOO can be calculated directly at the reference scale of 4 km 2 (2 × 2 km) cells, you can skip sections 4.10.4 and 4.10.5. If AOO cannot be calculated at the reference scale (e.g., because it has already been calculated at another scale and original maps are not available), then the methods described in the following two sections may be helpful. 4.10.4 Scale-area relationships"
    }
  ],
  "draft_hits": [
    {
      "rank": 1,
      "section_path": "test_document > Red List Assessment > Assessment Rationale",
      "source_key": "test_document > Red List Assessment > Assessment Rationale",
      "score": 8.05,
      "text": "<p>Bulbostylis atracuminata occurs in a restricted area ranging from southern DRC to northern Zambia. Its extent of occurrence of about 114,000 km2, exceeds the threshold value for threatened categories. However, its actual suitable habitat within this range is considered to be smaller because its habitat is restricted to riverbanks and wetlands. This is reflected in the area of occupancy (44 km2), which falls within the threshold values for the Endangered category under criterion B2. There are an estimated 11 locations for this species in Zambia and DRC, with some locations found in natural environments with low human disturbance levels. However, the majority of its habitat is threatened by mining activity and agricultural land conversion, resulting in an overall continuing deterioration of habitat quality.. Though this species has a restricted AOO and is experiencing a continuing decline in habitat quality, the total number of locations marginally exceeds the threshold value for threatened categories under criterion B, and the existence of some high-quality habitat may enable the species to persist without being globally threatened in the near future. Therefore, this species is assessed as Endangered B2ab(iii). Despite these pressures, the species is unlikely to face major threats in the near future.</p>"
    },
    {
      "rank": 2,
      "section_path": "test_document > Distribution > Area of Occupancy (AOO)",
      "source_key": "test_document > Distribution > Area of Occupancy (AOO)",
      "score": 6.25,
      "text": "<table> <tr><td><b>Estimated area of occupancy (AOO) - in km2</b></td><td><b>Justification</b></td></tr> <tr><td>44</td><td>Calculated in GeoCat (Moat et al. 2023) based on 11 georeferenced herbarium specimens including all available specimen data in online herbarium catalogues.</td></tr> </table>"
    },
    {
      "rank": 3,
      "section_path": "test_document > Red List Assessment > Assessment Information",
      "source_key": "test_document > Red List Assessment > Assessment Information",
      "score": 5.15,
      "text": "<p><b>Assessor(s): </b>Lemboye, B. & Lyu, E</p> <p><b>Institution(s): </b>Royal Botanic Gardens, Kew</p> <p><b>Regions: </b>Global</p>"
    },
    {
      "rank": 4,
      "section_path": "test_document > Occurrence > Countries of Occurrence",
      "source_key": "test_document > Occurrence > Countries of Occurrence",
      "score": 4.95,
      "text": "<table> <tr><td><b>Country</b></td><td><b>Presence</b></td><td><b>Origin</b></td><td><b>Formerly Bred</b></td><td><b>Seasonality</b></td></tr> <tr><td>Congo, The Democratic Republic of the</td><td>Extant</td><td>Native</td><td>Unknown</td><td>Resident</td></tr> <tr><td>Zambia</td><td>Possibly Extinct</td><td>Native</td><td>Unknown</td><td>Resident</td></tr> </table>"
    },
    {
      "rank": 5,
      "section_path": "test_document > Distribution > Very restricted AOO or number of locations (triggers VU D2)",
      "source_key": "test_document > Distribution > Very restricted AOO or number of locations (triggers VU D2)",
      "score": 3.0,
      "text": "<table> <tr><td><b>Very restricted in area of occupancy (AOO) and/or # of locations</b></td><td><b>Justification</b></td></tr> <tr><td>No</td><td>The species is found at multiple locations in Zambia and DRC.</td></tr> </table>"
    },
    {
      "rank": 6,
      "section_path": "test_document",
      "source_key": "test_document",
      "score": 2.0,
      "text": "<p><b>Draft</b></p> <p><b><i>Bulbostylis atracuminata</i></b><b> - (Larridon, Reynders & Goetgh.) Larridon & Roalson</b></p> <p>PLANTAE - TRACHEOPHYTA - LILIOPSIDA - POALES - CYPERACEAE - Bulbostylis - atracuminata</p> <p><b>Common Names: </b>No Common Names <b>Synonyms: </b>Nemum atracuminatum Larridon, Reynders & Goetgh.</p> <table> <tr><td><b>Red List Status</b></td></tr> <tr><td>EN, B2ab(iii) (IUCN version 3.1)</td></tr> </table>"
    }
  ]
}
