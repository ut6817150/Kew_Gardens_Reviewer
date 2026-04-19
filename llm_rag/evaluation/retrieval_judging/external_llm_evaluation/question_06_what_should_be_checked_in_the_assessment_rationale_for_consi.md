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
  "question": "What should be checked in the assessment rationale for consistency with IUCN Red List guidance?",
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
  "answer_scaffold": "Answer scaffold\n---------------\nMost relevant retrieved reference evidence:\n- Section: Required Information Purpose Guidance Notes | Source: Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf | Page: 3 | Evidence: Entry 5: 5. A rationale for the Red List • To justify the Red List Include any inferences or uncertainty assessment Category and Criteria that relate to the interpretation of the selected data and information in relation to the criteria and their thresholds. 1\n- Section: Required Information Purpose Guidance Notes | Source: Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf | Page: 3 | Evidence: Page 3 | Entry 5: 5. A rationale for the Red List • To justify the Red List Include any inferences or uncertainty assessment Category and Criteria that relate to the interpretation of the selected data and information in relation to the criteria and their thre\n- Section: 2.2.8. Assessment Rationale | Source: RL_Standards_Consistency.pdf | Page: 23 | Evidence: All assessments published on the IUCN Red List re quire a rationale― sometimes also referred to as the “justification” ( Table 1 ). The rationale justifies the IUCN Red List Category and Criteria selected for the taxon. In SIS, this is recorded in the Rational\n- Section: Required Information Purpose Guidance Notes | Source: Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf | Page: 3 | Evidence: The Red List Category and Criteria represent the most fundamental elements of a Red List assessment. Application of the categories and criteria must be in accordance with the IUCN Red List Categories and Criteria. Version 3.1 and the current version of the Gui",
  "subqueries": [
    "What should be checked in the assessment rationale for consistency with IUCN Red List guidance?"
  ],
  "retrieved_reference_items": [
    {
      "rank": 1,
      "block_type": "table_row",
      "source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
      "page": 3,
      "section": "Required Information Purpose Guidance Notes",
      "text": "Entry 5: 5. A rationale for the Red List • To justify the Red List Include any inferences or uncertainty assessment Category and Criteria that relate to the interpretation of the selected data and information in relation to the criteria and their thresholds. 1 Note that all taxa assessed must be validly published in accordance with the appropriate international nomenclatural codes and should be currently accepted names. Standard taxonomic checklists should be used wherever possible for names. The standard lists adopted by IUCN are periodically reviewed and listed on the Red List website: http://www.iucnredlist.org/info/info_sources_quality.html. For many groups no standards are available, or there may be a valid reason for adopting another treatment. In such cases, the taxonomic treatment followed should be indicated and if not one of the standards followed by IUCN, the reference should be cited in full and a reason for the deviation given This should include the date of publication, except in the case of plant names. The abbreviations used for author names of plants should follow Brummitt and Powell (1992) and subsequent updates on the International Plant Names Index website (http://www.ipni.org/index.html) 3",
      "parent_text": "Table 1: Required supporting information for all assessments\nSchema: Required Information | Purpose | Guidance | Notes\nPage 3 | Entry 1: 1. Scientific name1 • To identify which taxon is If the taxon is already in SIS, this being assessed information requires no additional effort from the Assessors. If the taxon is not • To support Red List website yet recorded in SIS, Assessors must functionality provide this information to the Red List Unit.\nPage 3 | Entry 2: 2. Higher taxonomy details • To identify which taxon is If the taxon is already in SIS, this (Kingdom, Phylum, Class, being assessed requires no additional effort from the Order, Family) Assessors. If the taxon is not yet • To support Red List website recorded in SIS, Assessors must functionality provide this information to the Red List Unit.\nPage 3 | Entry 3: 3. Taxonomic authorities for all • To identify which taxon is If the taxon is already in SIS, this specific and infra-specific being assessed information requires no additional effort names used, following the from the Assessors. If the taxon is not appropriate nomenclatural yet recorded in SIS, Assessors must rules2 provide this information to the Red List Unit.\nPage 3 | Entry 4: 4. IUCN Red List Category and • To identify the current status The Red List Category and Criteria Criteria (including sub- of the taxon represent the most fundamental criteria) met at the highest elements of a Red List assessment. • To support Red List website category of threat functionality Application of the categories and criteria must be in accordance with the • To allow basic analyses IUCN Red List Categories and Criteria. Version 3.1 and the current version of the Guidelines for Using the IUCN Red List Categories and Criteria.\nPage 3 | Entry 5: 5. A rationale for the Red List • To justify the Red List Include any inferences or uncertainty assessment Category and Criteria that relate to the interpretation of the selected data and information in relation to the criteria and their thresholds. 1 Note that all taxa assessed must be validly published in accordance with the appropriate international nomenclatural codes and should be currently accepted names. Standard taxonomic checklists should be used wherever possible for names. The standard lists adopted by IUCN are periodically reviewed and listed on the Red List website: http://www.iucnredlist.org/info/info_sources_quality.html. For many groups no standards are available, or there may be a valid reason for adopting another treatment. In such cases, the taxonomic treatment followed should be indicated and if not one of the standards followed by IUCN, the reference should be cited in full and a reason for the deviation given This should include the date of publication, except in the case of plant names. The abbreviations used for author names of plants should follow Brummitt and Powell (1992) and subsequent updates on the International Plant Names Index website (http://www.ipni.org/index.html) 3\nPage 4 | Entry 6: 6. Data for parameters • To underpin and justify the Enter these data either into the triggering the Red List Red List Category and Criteria relevant coded/numerical fields or in Criteria met at the highest used the relevant narrative (text) fields in Category level SIS. If data are entered into the data fields, this allows the Red List Criteria calculator to be used in SIS, which automatically checks for errors, omissions and inconsistencies, reducing the burden of manual checking by Assessors, RLA Coordinators and project coordinators. If data are included within the narrative (text) fields, the text must clearly indicate all of the relevant subcriteria parameters and qualifiers (observed, estimated, inferred, projected or suspected) used.\nPage 4 | Entry 7: 7. Countries of occurrence (for • To support Red List website SIS automatically records Presence = native and reintroduced functionality (especially Extant and Origin = Native by default taxa), including Presence country searches) as countries are selected. and Origin coding A tool will be made available to • To allow basic analyses determine countries of occurrence automatically from GIS maps. Countries of occurrence are not strictly required for vagrant and introduced ranges. 4\nPage 5 | Entry 8: 8. Geo-referenced distribution • To support Red List website Spatial distribution data are not data for all taxa with a known functionality required for taxa of unknown distribution provenance (e.g. taxa assessed as • To allow basic analyses Data Deficient because their range is • Spatial distribution data are not known). essential for supporting Spatial data may be geo-referenced assessments under criteria B polygons or point localities, and may and D2 (and arguably also for be provided in any format, including as demonstrating that these a paper map, text file of coordinates, thresholds are not met) pdf, graphics file or GIS shapefile. A GIS shapefile is preferred (but is not strictly required), given their value for conducting spatial analyses, visual displays on the Red List website, and future functionality on the Red List website that will allow spatial searches. Although additional distributional documentation is desirable for taxa qualifying under criterion B (e.g., 2x2 km grids showing occupancy), this is not Required. Note that any distributional data can be coded as sensitive to avoid this being distributed or displayed on the Red List website (see Annex 5).\nPage 5 | Entry 9: 9. Direction of current • To support Red List website population trend (stable, functionality increasing, decreasing, • To allow basic analyses unknown)\nPage 5 | Entry 10: 10. Coding for occurrence in • To support Red List website freshwater (= inland waters), functionality terrestrial, and marine ecosystems (i.e., “System” in • To allow basic analyses SIS)\nPage 5 | Entry 11: 11. Suitable habitats utilized • To support the assessment To speed up entering such coding in (coded to lowest level in SIS, habitat importance is set to • To support Red List website Habitats Classification 'suitable' by default for any habitat functionality Scheme) selected. • To allow basic analyses\nPage 5 | Entry 12: 12. Bibliography (cited in full; • To underpin the assessment In SIS, references are recorded in the including unpublished data and provide all sources of Reference Manager. sources but not personal data and information used to communications) support the Red List assessment 5"
    },
    {
      "rank": 2,
      "block_type": "table",
      "source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
      "page": 3,
      "section": "Required Information Purpose Guidance Notes",
      "text": "Page 3 | Entry 5: 5. A rationale for the Red List • To justify the Red List Include any inferences or uncertainty assessment Category and Criteria that relate to the interpretation of the selected data and information in relation to the criteria and their thresholds. 1 Note that all taxa assessed must be validly published in accordance with the appropriate international nomenclatural codes and should be currently accepted names. Standard taxonomic checklists should be used wherever possible for names. The standard lists adopted by IUCN are periodically reviewed and listed on the Red List website: http://www.iucnredlist.org/info/info_sources_quality.html. For many groups no standards are available, or there may be a valid reason for adopting another treatment. In such cases, the taxonomic treatment followed should be indicated and if not one of the standards followed by IUCN, the reference should be cited in full and a reason for the deviation given This should include the date of publication, except in the case of plant names. The abbreviations used for author names of plants should follow Brummitt and Powell (1992) and subsequent updates on the International Plant Names Index website (http://www.ipni.org/index.html) 3\nPage 4 | Entry 6: 6. Data for parameters • To underpin and justify the Enter these data either into the triggering the Red List Red List Category and Criteria relevant coded/numerical fields or in Criteria met at the highest used the relevant narrative (text) fields in Category level SIS. If data are entered into the data fields, this allows the Red List Criteria calculator to be used in SIS, which automatically checks for errors, omissions and inconsistencies, reducing the burden of manual checking by Assessors, RLA Coordinators and project coordinators. If data are included within the narrative (text) fields, the text must clearly indicate all of the relevant subcriteria parameters and qualifiers (observed, estimated, inferred, projected or suspected) used.",
      "parent_text": "Table 1: Required supporting information for all assessments\nSchema: Required Information | Purpose | Guidance | Notes\nPage 3 | Entry 1: 1. Scientific name1 • To identify which taxon is If the taxon is already in SIS, this being assessed information requires no additional effort from the Assessors. If the taxon is not • To support Red List website yet recorded in SIS, Assessors must functionality provide this information to the Red List Unit.\nPage 3 | Entry 2: 2. Higher taxonomy details • To identify which taxon is If the taxon is already in SIS, this (Kingdom, Phylum, Class, being assessed requires no additional effort from the Order, Family) Assessors. If the taxon is not yet • To support Red List website recorded in SIS, Assessors must functionality provide this information to the Red List Unit.\nPage 3 | Entry 3: 3. Taxonomic authorities for all • To identify which taxon is If the taxon is already in SIS, this specific and infra-specific being assessed information requires no additional effort names used, following the from the Assessors. If the taxon is not appropriate nomenclatural yet recorded in SIS, Assessors must rules2 provide this information to the Red List Unit.\nPage 3 | Entry 4: 4. IUCN Red List Category and • To identify the current status The Red List Category and Criteria Criteria (including sub- of the taxon represent the most fundamental criteria) met at the highest elements of a Red List assessment. • To support Red List website category of threat functionality Application of the categories and criteria must be in accordance with the • To allow basic analyses IUCN Red List Categories and Criteria. Version 3.1 and the current version of the Guidelines for Using the IUCN Red List Categories and Criteria.\nPage 3 | Entry 5: 5. A rationale for the Red List • To justify the Red List Include any inferences or uncertainty assessment Category and Criteria that relate to the interpretation of the selected data and information in relation to the criteria and their thresholds. 1 Note that all taxa assessed must be validly published in accordance with the appropriate international nomenclatural codes and should be currently accepted names. Standard taxonomic checklists should be used wherever possible for names. The standard lists adopted by IUCN are periodically reviewed and listed on the Red List website: http://www.iucnredlist.org/info/info_sources_quality.html. For many groups no standards are available, or there may be a valid reason for adopting another treatment. In such cases, the taxonomic treatment followed should be indicated and if not one of the standards followed by IUCN, the reference should be cited in full and a reason for the deviation given This should include the date of publication, except in the case of plant names. The abbreviations used for author names of plants should follow Brummitt and Powell (1992) and subsequent updates on the International Plant Names Index website (http://www.ipni.org/index.html) 3\nPage 4 | Entry 6: 6. Data for parameters • To underpin and justify the Enter these data either into the triggering the Red List Red List Category and Criteria relevant coded/numerical fields or in Criteria met at the highest used the relevant narrative (text) fields in Category level SIS. If data are entered into the data fields, this allows the Red List Criteria calculator to be used in SIS, which automatically checks for errors, omissions and inconsistencies, reducing the burden of manual checking by Assessors, RLA Coordinators and project coordinators. If data are included within the narrative (text) fields, the text must clearly indicate all of the relevant subcriteria parameters and qualifiers (observed, estimated, inferred, projected or suspected) used.\nPage 4 | Entry 7: 7. Countries of occurrence (for • To support Red List website SIS automatically records Presence = native and reintroduced functionality (especially Extant and Origin = Native by default taxa), including Presence country searches) as countries are selected. and Origin coding A tool will be made available to • To allow basic analyses determine countries of occurrence automatically from GIS maps. Countries of occurrence are not strictly required for vagrant and introduced ranges. 4\nPage 5 | Entry 8: 8. Geo-referenced distribution • To support Red List website Spatial distribution data are not data for all taxa with a known functionality required for taxa of unknown distribution provenance (e.g. taxa assessed as • To allow basic analyses Data Deficient because their range is • Spatial distribution data are not known). essential for supporting Spatial data may be geo-referenced assessments under criteria B polygons or point localities, and may and D2 (and arguably also for be provided in any format, including as demonstrating that these a paper map, text file of coordinates, thresholds are not met) pdf, graphics file or GIS shapefile. A GIS shapefile is preferred (but is not strictly required), given their value for conducting spatial analyses, visual displays on the Red List website, and future functionality on the Red List website that will allow spatial searches. Although additional distributional documentation is desirable for taxa qualifying under criterion B (e.g., 2x2 km grids showing occupancy), this is not Required. Note that any distributional data can be coded as sensitive to avoid this being distributed or displayed on the Red List website (see Annex 5).\nPage 5 | Entry 9: 9. Direction of current • To support Red List website population trend (stable, functionality increasing, decreasing, • To allow basic analyses unknown)\nPage 5 | Entry 10: 10. Coding for occurrence in • To support Red List website freshwater (= inland waters), functionality terrestrial, and marine ecosystems (i.e., “System” in • To allow basic analyses SIS)\nPage 5 | Entry 11: 11. Suitable habitats utilized • To support the assessment To speed up entering such coding in (coded to lowest level in SIS, habitat importance is set to • To support Red List website Habitats Classification 'suitable' by default for any habitat functionality Scheme) selected. • To allow basic analyses\nPage 5 | Entry 12: 12. Bibliography (cited in full; • To underpin the assessment In SIS, references are recorded in the including unpublished data and provide all sources of Reference Manager. sources but not personal data and information used to communications) support the Red List assessment 5"
    },
    {
      "rank": 3,
      "block_type": "text",
      "source": "RL_Standards_Consistency.pdf",
      "page": 23,
      "section": "2.2.8. Assessment Rationale",
      "text": "All assessments published on the IUCN Red List re quire a rationale― sometimes also referred to as the “justification” ( Table 1 ). The rationale justifies the IUCN Red List Category and Criteria selected for the taxon. In SIS, this is recorded in the Rationale for the Red List Assessment field. The rationale should not simply quote the Red List Criteria thresholds that are met (the criteria code already indicates these); instead it should use the key issues highlighted in the other documentation sections to summarize the reasons why the taxon qualifies for the assigned category. Include in the rationale any inferences or uncertainty that relate to the interpretation of the available data and information in relation to the criteria and their thresholds"
    },
    {
      "rank": 4,
      "block_type": "text",
      "source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
      "page": 3,
      "section": "Required Information Purpose Guidance Notes",
      "text": "The Red List Category and Criteria represent the most fundamental elements of a Red List assessment. Application of the categories and criteria must be in accordance with the IUCN Red List Categories and Criteria. Version 3.1 and the current version of the Guidelines for Using the IUCN Red List Categories and Criteria . • To identify the current status of the taxon • To support Red List website functionality • To allow basic analyses 5. A rationale for the Red List assessment Include any inferences or uncertainty that relate to the interpretation of the data and information in relation to the criteria and their thresholds. • To justify the Red List Category and Criteria selected"
    },
    {
      "rank": 5,
      "block_type": "text",
      "source": "RL_Standards_Consistency.pdf",
      "page": 65,
      "section": "5.2. Consistency Checks",
      "text": "Before a large number of assessments are submitted for publication on the IUCN Red List, it is also important to check the assessments for consistency in how the IUCN Red List Categories and Criteria have been applied to different taxa, particularly taxa occurring in the same area and facing the same threats. Different Assessors may apply the IUCN Red List Criteria slightly differently because of differences in attitudes. When faced with uncertain data, some Assessors will be more precautionary in their interpretation of the data, tending to list taxa in higher threat categories, while others are more evidentiary and tend to seek out further evidence before listing a taxon in a higher threat category"
    },
    {
      "rank": 6,
      "block_type": "text",
      "source": "RL_Standards_Consistency.pdf",
      "page": 33,
      "section": "2.5.5. Rationale ( Assessment Rational section in SIS)",
      "text": "A rationale is required information for all IUCN Red List assessments ( Table 1 and section 2.2.8. )"
    },
    {
      "rank": 7,
      "block_type": "text",
      "source": "RL_Standards_Consistency.pdf",
      "page": 6,
      "section": "Required Information Purpose Guidance Notes",
      "text": "4. IUCN Red List Category and Criteria (including subcriteria) met at the highest category of threat. • To identify the current status of the taxon The Red List Category and Criteria represent the most fundamental elements of a Red List assessment. • To support Red List website functionality Use of the categories and criteria must be in accordance with the IUCN Red List Categories and Criteria. Version 3.1 and the current version of the Guidelines for Using the IUCN Red List Categories and Criteria . Both documents are available to download from the IUCN Red List website ( http://www.iucnredlist.org/technical- documents/red-list-documents ). See section 2.5.1 . • To allow basic analyses 5. A rationale for the Red List assessment. • To justify the Red List Category and Criteria selected"
    },
    {
      "rank": 8,
      "block_type": "text",
      "source": "RL_categories_and_criteria.pdf",
      "page": 31,
      "section": "Annex 3: Required and Recommended Supporting Information for IUCN Red List Assessments",
      "text": "All assessments published on the IUCN Red List are freely available for public use. To ensure assessments are fully justified and to allow Red List assessment data to be analysed, thus making the IUCN Red List a powerful tool for conservation and policy decisions, a set of supporting information is required to accompany every assessment submitted for publication on the IUCN Red List of Threatened Species TM . The reference document Documentation Standards and Consistency Checks for IUCN Red List Assessments and Species Accounts is available to download from the Red List website (www.iucnredlist.org) and provides guidance on the following: Required supporting information for all IUCN Red List assessments.   Required supporting information under specific conditions (e.g"
    }
  ],
  "draft_hits": [
    {
      "rank": 1,
      "section_path": "test_document > Red List Assessment > Assessment Rationale",
      "source_key": "test_document > Red List Assessment > Assessment Rationale",
      "score": 10.8,
      "text": "<p>Bulbostylis atracuminata occurs in a restricted area ranging from southern DRC to northern Zambia. Its extent of occurrence of about 114,000 km2, exceeds the threshold value for threatened categories. However, its actual suitable habitat within this range is considered to be smaller because its habitat is restricted to riverbanks and wetlands. This is reflected in the area of occupancy (44 km2), which falls within the threshold values for the Endangered category under criterion B2. There are an estimated 11 locations for this species in Zambia and DRC, with some locations found in natural environments with low human disturbance levels. However, the majority of its habitat is threatened by mining activity and agricultural land conversion, resulting in an overall continuing deterioration of habitat quality.. Though this species has a restricted AOO and is experiencing a continuing decline in habitat quality, the total number of locations marginally exceeds the threshold value for threatened categories under criterion B, and the existence of some high-quality habitat may enable the species to persist without being globally threatened in the near future. Therefore, this species is assessed as Endangered B2ab(iii). Despite these pressures, the species is unlikely to face major threats in the near future.</p>"
    },
    {
      "rank": 2,
      "section_path": "test_document > Red List Assessment > Assessment Information",
      "source_key": "test_document > Red List Assessment > Assessment Information",
      "score": 5.25,
      "text": "<p><b>Assessor(s): </b>Lemboye, B. & Lyu, E</p> <p><b>Institution(s): </b>Royal Botanic Gardens, Kew</p> <p><b>Regions: </b>Global</p>"
    },
    {
      "rank": 3,
      "section_path": "test_document",
      "source_key": "test_document",
      "score": 3.0,
      "text": "<p><b>Draft</b></p> <p><b><i>Bulbostylis atracuminata</i></b><b> - (Larridon, Reynders & Goetgh.) Larridon & Roalson</b></p> <p>PLANTAE - TRACHEOPHYTA - LILIOPSIDA - POALES - CYPERACEAE - Bulbostylis - atracuminata</p> <p><b>Common Names: </b>No Common Names <b>Synonyms: </b>Nemum atracuminatum Larridon, Reynders & Goetgh.</p> <table> <tr><td><b>Red List Status</b></td></tr> <tr><td>EN, B2ab(iii) (IUCN version 3.1)</td></tr> </table>"
    },
    {
      "rank": 4,
      "section_path": "test_document > Threats",
      "source_key": "test_document > Threats",
      "score": 2.0,
      "text": "<p>The habitats of copper bog sedge are predominantly threatened by mining activity and agricultural land conversion (Global Forest Watch 2023, Google Earth Pro 2023). Five out of the total eleven locations are situated on mining sites managed by different mining companies in both Zambia and the DRC (Global Forest Watch 2023). Four locations in the DRC are currently experiencing the threat of agricultural land conversion (Google Earth Pro 2023), with three of these locations affected by both mining and agricultural land use simultaneously. Agricultural land conversion is likely to continue affecting these local subpopulations in the future. Additionally, a dam has been identified near one location in the DRC (Google Earth Pro 2023). This could affect the soil moisture content, subsequently impacting the habitat suitability for this species. Three locations in the DRC are situated within two national parks (IUCN and UNEP-WCMC 2023). While national parks can provide some protection from human disturbance, it's important to note that these parks are not strictly managed and remain open to activities such as bird watching, hiking, walking, and camping (Virunga National Park 2023). Currently, the habitats within the national parks are not threatened. Several sites are also increasingly affected by road construction and urban expansion. Unregulated fire is also a serious threat in some areas.</p>"
    },
    {
      "rank": 5,
      "section_path": "test_document > Habitats and Ecology > IUCN Habitats Classification Scheme",
      "source_key": "test_document > Habitats and Ecology > IUCN Habitats Classification Scheme",
      "score": 1.75,
      "text": "<table> <tr><td><b>Habitat</b></td><td><b>Season</b></td><td><b>Suitability</b></td><td><b>Major Importance?</b></td></tr> <tr><td>5.2. Wetlands (inland) -> Wetlands (inland) - Seasonal/Intermittent/Irregular Rivers/Streams/Creeks</td><td>Resident</td><td>Suitable</td><td>-</td></tr> <tr><td>5.8. Wetlands (inland) -> Wetlands (inland) - Seasonal/Intermittent Freshwater Marshes/Pools (under 8ha)</td><td>Resident</td><td>Suitable</td><td>-</td></tr> <tr><td>5.2. woodland (inland) -> woodland (inland) –</td><td>Resident</td><td>Suitable</td><td>-</td></tr> </table>"
    },
    {
      "rank": 6,
      "section_path": "test_document > Distribution > Map Status",
      "source_key": "test_document > Distribution > Map Status",
      "score": 1.0,
      "text": "<table> <tr><td><b>Map Status</b></td><td><b>Use map from previous assessment</b></td><td><b>How the map was created, including data sources/methods used:</b></td><td><b>Please state reason for map not available:</b></td><td><b>Data Sensitive?</b></td><td><b>Justification</b></td><td><b>Geographic range this applies to:</b></td><td><b>Date restriction imposed:</b></td></tr> <tr><td>Done</td><td>-</td><td>Prepared in GeoCat (Moat et al. 2023) based on 11 georeferenced herbarium specimens (Global Biodiversity Information Facility 2023)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr> </table>"
    }
  ]
}
