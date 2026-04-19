# External LLM Retrieval Evaluation Output

This file contains the manually saved external-LLM judgments for the retrieval-judging prompts generated in `llm_rag/evaluation/retrieval_judging/external_llm_evaluation/`.

The summary below is derived from the raw JSON outputs. The raw outputs are preserved underneath in per-question code blocks without reformatting the JSON bodies.

## Summary

- Questions evaluated: `6`
- Average relevance score: `3.50 / 5`
- Average coverage score: `3.00 / 5`
- Average focus score: `2.67 / 5`
- Overall pattern: retrieval is often partially useful, but coverage gaps remain when the question needs very specific criterion thresholds or draft sections.
- Strongest result: Question 2, where locations evidence and relevant Criterion B guidance were retrieved well.
- Most important weakness: Question 1, where draft EOO/AOO values were retrieved but the reference-side CR/EN/VU threshold evidence was not retrieved clearly enough.
- Repeated issue: some runs retrieve the right broad reference document but miss specific rows or sections needed for complete coverage.
- Draft-side issue: several questions retrieve the main draft rationale but miss nearby supporting sections such as AOO, locations, geographic range, threats, or population narrative.

### Score Table

| Question | Evaluation Focus | Relevance | Coverage | Focus | Main Takeaway |
|---:|---|---:|---:|---:|---|
| 1 | EOO/AOO category support | 3 | 2 | 3 | Draft-side retrieval is reasonably good because it brings back the key EOO and AOO evidence, especially in the assessment rationale and AOO section. |
| 2 | Locations and severe fragmentation | 4 | 4 | 3 | The retrieval is strong on number-of-locations evidence: it brings back the key draft locations table, the rationale, and supporting threat context, along with relevant Criterion B guidance. |
| 3 | Supporting information requirements | 4 | 3 | 2 | The retrieval package found the right core reference document and several genuinely useful required-information rows, especially for all-assessment requirements and some threatened-taxon conditional items. |
| 4 | Mapping and spatial-data checks | 3 | 3 | 3 | The retrieval package gets several important mapping references and one directly relevant draft section, so it is not a miss. |
| 5 | Population trend and continuing decline | 3 | 3 | 2 | The retrieval package finds some of the right evidence for continuing decline and the coded population-trend field, but it is incomplete and noisy. |
| 6 | Assessment rationale consistency | 4 | 3 | 3 | The retrieval package is strong on reference-side guidance for what a rationale should contain, especially that it should justify the category, not merely restate thresholds, and should include uncertainty or inference. |

### Per-Question Notes

- Question 1: Draft-side retrieval found useful EOO/AOO evidence, but reference retrieval missed the decisive Criterion B threshold evidence needed to classify from those values.
- Question 2: Retrieval performed well for number of locations and relevant Criterion B guidance, but did not retrieve an explicit severe-fragmentation statement from the draft.
- Question 3: Retrieval found the core supporting-information reference document and several required-information rows, but missed recommended-information evidence and included noisy draft hits.
- Question 4: Retrieval found several useful mapping references and the draft map-status section, but missed important draft spatial sections such as geographic range, countries of occurrence, AOO, and locations.
- Question 5: Retrieval found some continuing-decline and population-trend evidence, but missed stronger nearby population and threats narrative, and reference hits included several irrelevant special cases.
- Question 6: Reference-side rationale guidance was strong, but draft-side retrieval missed key supporting sections needed to verify consistency with AOO, locations, and continuing decline claims.

## Raw LLM Outputs

### Question 1

Question: Based on Extent of occurrence (EOO) and Area of occupancy (AOO) only, what IUCN category would the uploaded assessment support?

```json
{"question":"Based on Extent of occurrence (EOO) and Area of occupancy (AOO) only, what IUCN category would the uploaded assessment support?","route":"hybrid_rag","relevance_score":3,"coverage_score":2,"focus_score":3,"covered_evidence":["Draft retrieval captured the key AOO value (44 km2) in a dedicated distribution section.","Draft retrieval also captured the assessment rationale stating EOO is about 114,000 km2 and AOO is 44 km2.","Reference retrieval included background material explaining what EOO and AOO are and how they are calculated/used under criterion B.","Reference retrieval included some criterion-B-related material, including an Annex 4 summary passage mentioning EOO/AOO within subcriteria."],"missing_evidence":["The key official threshold table or text mapping EOO and AOO values to CR/EN/VU categories was not retrieved.","Reference evidence did not clearly retrieve the criterion B1/B2 cutoffs needed to classify 114,000 km2 EOO and 44 km2 AOO.","Reference retrieval focused on definitions and calculation guidance rather than the specific category-determination rules the question requires.","No retrieved reference item clearly supports the final category decision from EOO/AOO alone.","Draft retrieval did not bring a dedicated EOO section or tabular source for the EOO value; the EOO evidence appears only inside the rationale text."],"reference_item_labels":[{"rank":1,"source":"Mapping_Standards_Version_1.20_Jan2024.pdf","page":21,"section":"7.1.1 How should EOO be calculated for polygon maps and point maps?","label":"partly_useful","reason":"Useful for understanding AOO concept, but it does not provide the category thresholds needed to decide CR/EN/VU from the reported values."},{"rank":2,"source":"Mapping_Standards_Version_1.20_Jan2024.pdf","page":19,"section":"7.1 Extent of occurrence (EOO)","label":"partly_useful","reason":"Defines EOO, but does not provide the threshold cutoffs required to classify the species."},{"rank":3,"source":"RL_categories_and_criteria.pdf","page":33,"section":"Annex 4: Summary of the IUCN Red List Criteria","label":"partly_useful","reason":"This is the closest item to criterion B decision logic, but the retrieved passage is about subcriteria and does not expose the EOO/AOO category thresholds needed for the question."},{"rank":4,"source":"RedListGuidelines.pdf","page":54,"section":"4.10 Area of occupancy (criteria A, B and D)","label":"partly_useful","reason":"Relevant to how AOO is interpreted and calculated, but not to mapping the draft values to an IUCN category."},{"rank":5,"source":"Mapping_Standards_Version_1.20_Jan2024.pdf","page":20,"section":"7.1.1 How should EOO be calculated for polygon maps and point maps?","label":"not_useful","reason":"Methodological detail about MCP and migratory species is not needed for deciding category from already-given EOO/AOO values."},{"rank":6,"source":"RedListGuidelines.pdf","page":60,"section":"4.10 Area of occupancy (criteria A, B and D)","label":"partly_useful","reason":"General guidance on known/inferred/projected occurrences is related to AOO/EOO estimation, but it does not answer the category-threshold question."},{"rank":7,"source":"RedListGuidelines.pdf","page":62,"section":"4.10 Area of occupancy (criteria A, B and D)","label":"partly_useful","reason":"Mentions the VU AOO threshold contextually, but it is incomplete for full classification and does not provide the full threshold set needed for EN/CR comparison."},{"rank":8,"source":"RedListGuidelines.pdf","page":56,"section":"4.10 Area of occupancy (criteria A, B and D)","label":"partly_useful","reason":"Useful for confirming 2x2 km grid scale expectations for AOO, but not for determining the resulting category from the retrieved draft values."}],"draft_hit_labels":[{"rank":1,"section_path":"test_document > Red List Assessment > Assessment Rationale","score":8.05,"label":"useful","reason":"Contains both critical draft-side values: EOO about 114,000 km2 and AOO 44 km2, which are central to the question."},{"rank":2,"section_path":"test_document > Distribution > Area of Occupancy (AOO)","score":6.25,"label":"useful","reason":"Directly provides the AOO value and its justification, which is essential evidence for the question."},{"rank":3,"section_path":"test_document > Red List Assessment > Assessment Information","score":5.15,"label":"not_useful","reason":"Assessor names and institution do not help determine category from EOO/AOO."},{"rank":4,"section_path":"test_document > Occurrence > Countries of Occurrence","score":4.95,"label":"not_useful","reason":"Country presence information is not needed for deciding category from EOO/AOO only."},{"rank":5,"section_path":"test_document > Distribution > Very restricted AOO or number of locations (triggers VU D2)","score":3.0,"label":"partly_useful","reason":"Touches on restricted range context, but it is about D2 and locations rather than the direct EOO/AOO-only categorization requested."},{"rank":6,"section_path":"test_document","score":2.0,"label":"partly_useful","reason":"The document root includes the existing Red List status, but this is less useful than the underlying EOO/AOO evidence and does not by itself justify the category from the asked basis."}],"summary":"Draft-side retrieval is reasonably good because it brings back the key EOO and AOO evidence, especially in the assessment rationale and AOO section. Reference-side retrieval is the main weakness: it mostly returns definitions and calculation guidance, but misses the core criterion B threshold evidence needed to map those values to an IUCN category. Overall, the package is partially relevant but incomplete for this question."}
```

### Question 2

Question: What evidence does the uploaded assessment provide for number of locations and severe fragmentation under Criterion B?

```json
{
"question": "What evidence does the uploaded assessment provide for number of locations and severe fragmentation under Criterion B?",
"route": "deterministic_plus_hybrid_rag",
"relevance_score": 4,
"coverage_score": 4,
"focus_score": 3,
"covered_evidence": [
"Draft Locations Information chunk directly gives the estimate of 11 locations and a justification based on different threats across sites in DRC and Zambia.",
"Draft Assessment Rationale states there are 11 locations, notes this marginally exceeds the Criterion B threatened threshold, and ties the assessment to B2ab(iii).",
"Draft Threats chunk provides site-level threat context (mining, agriculture, dam development) that supports the location justification.",
"Reference guidance explains that Criterion B subcriterion (a) depends on severe fragmentation and/or number of locations and that assessors should explicitly state which is being used.",
"Reference standards state that assessments using B1a/B2a should clearly document number of locations and, if relevant, justify severe fragmentation."
],
"missing_evidence": [
"No retrieved draft chunk explicitly states whether the taxon is severely fragmented or provides a severe-fragmentation justification.",
"No retrieved draft chunk explicitly says that severe fragmentation was not used under Criterion B; this must be inferred indirectly from the retrieved rationale and locations material.",
"The retrieval includes a few weak draft items unrelated to the question, rather than adding more direct evidence about Criterion B documentation."
],
"reference_item_labels": [
{
"rank": 1,
"source": "RedListGuidelines.pdf",
"page": 72,
"section": "6. Guidelines for Applying Criterion B",
"label": "useful",
"reason": "Directly explains that Criterion B requires severe fragmentation and/or limited number of locations and gives an EN example with no more than five locations."
},
{
"rank": 2,
"source": "RedListGuidelines.pdf",
"page": 98,
"section": "12. Guidelines for Threatening Processes",
"label": "partly_useful",
"reason": "Relevant to how severe fragmentation may or may not be used, but more about future threats than the draft's actual evidence."
},
{
"rank": 3,
"source": "RL_Standards_Consistency.pdf",
"page": 65,
"section": "Common Errors",
"label": "useful",
"reason": "Directly says assessments using B1a/B2a should be clear about whether severe fragmentation or number of locations was used."
},
{
"rank": 4,
"source": "RedListGuidelines.pdf",
"page": 73,
"section": "6. Guidelines for Applying Criterion B",
"label": "useful",
"reason": "Explicitly recommends documenting both whether the taxon is severely fragmented and the number of locations."
},
{
"rank": 5,
"source": "RedListGuidelines.pdf",
"page": 51,
"section": "Indian Ocean",
"label": "useful",
"reason": "Clarifies that severe fragmentation is distinct from number of locations, which is central to the question."
},
{
"rank": 6,
"source": "RL_categories_and_criteria.pdf",
"page": 33,
"section": "≤ 10 (a) Severely fragmented OR Number of locations",
"label": "partly_useful",
"reason": "Provides general threshold context for Criterion B, but the chunk is noisy and less targeted than the stronger guidance items."
},
{
"rank": 7,
"source": "RL_Standards_Consistency.pdf",
"page": 19,
"section": "2.2.2. Geographic Range",
"label": "useful",
"reason": "Directly states the supporting information expected for number of locations and severe fragmentation under B1a/B2a."
},
{
"rank": 8,
"source": "RL_categories_and_criteria.pdf",
"page": 16,
"section": "8. Severely fragmented (Criterion B)",
"label": "useful",
"reason": "Provides the formal meaning of severe fragmentation, which helps judge whether the draft offers such evidence."
}
],
"draft_hit_labels": [
{
"rank": 1,
"section_path": "test_document > Distribution > Locations Information",
"score": 8.25,
"label": "useful",
"reason": "This is the strongest draft evidence: it gives the 11-location estimate and a direct justification tied to different threats."
},
{
"rank": 2,
"section_path": "test_document > Distribution > Very restricted AOO or number of locations (triggers VU D2)",
"score": 8.2,
"label": "partly_useful",
"reason": "It is relevant to location count, but it is framed around VU D2 rather than Criterion B and adds limited new evidence."
},
{
"rank": 3,
"section_path": "test_document > Red List Assessment > Assessment Rationale",
"score": 8.15,
"label": "useful",
"reason": "Directly discusses the 11 locations, says this exceeds the threatened threshold under Criterion B, and shows how the assessment rationale uses that fact."
},
{
"rank": 4,
"section_path": "test_document > Threats",
"score": 3.0,
"label": "partly_useful",
"reason": "Supports the location justification by describing different threats across sites, but it is indirect for the specific severe-fragmentation question."
},
{
"rank": 5,
"section_path": "test_document > Red List Assessment > Assessment Information",
"score": 1.75,
"label": "not_useful",
"reason": "Administrative metadata only; it does not bear on locations or severe fragmentation."
},
{
"rank": 6,
"section_path": "test_document > Distribution > Map Status",
"score": 1.75,
"label": "not_useful",
"reason": "Map preparation details are not materially useful for answering the question about number of locations and severe fragmentation."
}
],
"summary": "The retrieval is strong on number-of-locations evidence: it brings back the key draft locations table, the rationale, and supporting threat context, along with relevant Criterion B guidance. The main gap is that no draft hit explicitly addresses severe fragmentation status or justification, so the package supports an answer about locations much better than one about severe fragmentation."
}
```

### Question 3

Question: What supporting information is required or recommended for a threatened Red List assessment?

```json
{
"question": "What supporting information is required or recommended for a threatened Red List assessment?",
"route": "hybrid_rag",
"relevance_score": 4,
"coverage_score": 3,
"focus_score": 2,
"covered_evidence": [
"Retrieved the main reference source for this question: the Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments PDF.",
"Captured required supporting information for all assessments, including core items such as scientific name, taxonomy, Red List category/criteria, rationale, parameter data, countries of occurrence, distribution data, population trend, habitats, and bibliography.",
"Captured several threatened-taxon conditional requirements from Table 2, including generation length, 3-generation period for A4, criterion E model/data, major threats, and required narrative text on range/population/habitat/threats.",
"The answer scaffold correctly points toward required-for-all material and additional conditional requirements relevant to threatened taxa."
],
"missing_evidence": [
"No retrieved evidence for the recommended supporting information portion of the question; the package is almost entirely limited to required information tables.",
"Threatened-taxon conditional coverage is incomplete: an important criterion B item, coding as severely fragmented or number of locations, was present in the source table but not retrieved as its own hit.",
"Other conditional threatened-assessment items that may matter depending on the category/criterion were not retrieved, such as some additional Table 2 rows beyond the selected subset.",
"Draft-side hits are mostly not needed for this question and do not help recover missing reference-side coverage."
],
"reference_item_labels": [
{
"rank": 1,
"source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
"page": 3,
"section": "Required Information Purpose Guidance Notes",
"label": "useful",
"reason": "Directly relevant core table for required supporting information for all assessments; this is central evidence for the question."
},
{
"rank": 2,
"source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
"page": 8,
"section": "Specific Condition Purpose Guidance Notes",
"label": "partly_useful",
"reason": "Relevant only for criterion A4 threatened taxa; useful but too narrow to answer the full question."
},
{
"rank": 3,
"source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
"page": 8,
"section": "Specific Condition Purpose Guidance Notes",
"label": "not_useful",
"reason": "Near Threatened-specific entry does not answer a question about threatened Red List assessments."
},
{
"rank": 4,
"source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
"page": 8,
"section": "Specific Condition Purpose Guidance Notes",
"label": "useful",
"reason": "Generation length is an important conditional requirement for threatened taxa under criteria A and C1."
},
{
"rank": 5,
"source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
"page": 10,
"section": "Specific Condition Purpose Guidance Notes",
"label": "useful",
"reason": "Narrative text on range, population, habitat/ecology, and threats is directly relevant and broadly applicable to threatened taxa."
},
{
"rank": 6,
"source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
"page": 8,
"section": "Specific Condition Purpose Guidance Notes",
"label": "partly_useful",
"reason": "Relevant only when criterion E is used; useful conditional evidence but narrow."
},
{
"rank": 7,
"source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
"page": 9,
"section": "Specific Condition Purpose Guidance Notes",
"label": "useful",
"reason": "Major threats are directly relevant because they are required for threatened and Near Threatened taxa."
},
{
"rank": 8,
"source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
"page": 7,
"section": "Specific Condition Purpose Guidance Notes",
"label": "not_useful",
"reason": "Possibly Extinct/Possibly Extinct in the Wild tagging is specific to certain Critically Endangered cases and is not generally responsive to this question."
}
],
"draft_hit_labels": [
{
"rank": 1,
"section_path": "test_document > Red List Assessment > Assessment Information",
"score": 9.2,
"label": "not_useful",
"reason": "Contains assessor metadata from the draft, not the reference rules about required or recommended supporting information."
},
{
"rank": 2,
"section_path": "test_document > Red List Assessment > Assessment Rationale",
"score": 8.65,
"label": "partly_useful",
"reason": "Shows an example of assessment rationale/narrative that corresponds to a required support element, but does not answer the general reference-side question."
},
{
"rank": 3,
"section_path": "test_document > Threats > Threats Classification Scheme",
"score": 5.2,
"label": "partly_useful",
"reason": "Shows coded threats from the draft, which aligns with one threatened-taxon requirement, but is still secondary to the missing reference guidance."
},
{
"rank": 4,
"section_path": "test_document > Distribution > Locations Information",
"score": 4.55,
"label": "partly_useful",
"reason": "Provides number of locations, which is relevant to one threatened criterion-B conditional requirement, but the question is primarily about official required/recommended information categories."
},
{
"rank": 5,
"section_path": "test_document > Population > Population Information",
"score": 3.75,
"label": "partly_useful",
"reason": "Contains current population trend, one of the required support items, but only as draft content rather than official reference guidance."
},
{
"rank": 6,
"section_path": "test_document > Threats",
"score": 2.8,
"label": "partly_useful",
"reason": "Provides narrative threats text that matches a required threatened-assessment narrative element, but it is draft evidence rather than the governing reference list."
}
],
"summary": "The retrieval package found the right core reference document and several genuinely useful required-information rows, especially for all-assessment requirements and some threatened-taxon conditional items. Coverage is incomplete because the question explicitly asks for required or recommended information, but recommended-information evidence was not retrieved, and the package is noisy because multiple draft hits and a few narrowly relevant or off-target conditional rows were included."
}
```

### Question 4

Question: What mapping or spatial-data checks should be considered when reviewing this assessment?

```json
{
"question": "What mapping or spatial-data checks should be considered when reviewing this assessment?",
"route": "hybrid_rag",
"relevance_score": 3,
"coverage_score": 3,
"focus_score": 3,
"covered_evidence": [
"Reference evidence retrieved the core rule that geo-referenced distribution data are expected, especially for Criteria B and D2, and that shapefiles are preferred though not strictly required.",
"Reference evidence retrieved useful mapping-standard content on map status, sensitivity handling, recommended spatial attributes, and PRESENCE coding implications for whether areas count toward EOO/AOO.",
"Draft retrieval included the Map Status section showing the map is marked Done, built in GeoCAT from 11 georeferenced herbarium specimens, which is directly relevant to mapping review.",
"The assessment rationale draft hit provides some spatially relevant values (EOO, AOO, number of locations, range description), though only indirectly."
],
"missing_evidence": [
"The draft Geographic Range section was not retrieved, so key range-review evidence was missed, including the narrative range description and a possible country-range inconsistency involving Angola.",
"The draft Countries of Occurrence section was not retrieved, so spatial coding checks on country presence/origin were missed, including the notable 'Zambia = Possibly Extinct' entry.",
"The draft AOO justification and Locations Information sections were not retrieved, even though they contain core spatial inputs (44 km2, 11 georeferenced specimens, 11 locations) that should be checked against the map.",
"No retrieved draft chunk exposed whether map status, country coding, and rationale are internally consistent with one another.",
"Reference retrieval missed some broader mapping-review checks such as verifying completeness/justification of an incomplete map, alignment between mapped range and occurrence coding, and more comprehensive attribute requirements beyond the isolated sensitivity/longitude row."
],
"reference_item_labels": [
{
"rank": 1,
"source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
"page": 5,
"section": "Required Information Purpose Guidance Notes",
"label": "useful",
"reason": "Directly states that geo-referenced distribution data are essential for Criteria B and D2, describes acceptable formats, and notes shapefiles are preferred."
},
{
"rank": 2,
"source": "Mapping_Standards_Version_1.20_Jan2024.pdf",
"page": 6,
"section": "1.3.2 Mapping of Subspecies, Varieties or Subpopulations versus Green Status Spatial Units",
"label": "not_useful",
"reason": "This is mainly about Green Status spatial units and adds little to reviewing Red List mapping for this assessment."
},
{
"rank": 3,
"source": "Mapping_Standards_Version_1.20_Jan2024.pdf",
"page": 12,
"section": "4.1.2 Recommended polygon and point attributes",
"label": "useful",
"reason": "Useful for checking whether sensitive spatial data are flagged correctly and whether required supporting attributes are present."
},
{
"rank": 4,
"source": "Mapping_Standards_Version_1.20_Jan2024.pdf",
"page": 15,
"section": "Notes:",
"label": "useful",
"reason": "Useful because PRESENCE coding determines whether mapped areas should count toward EOO/AOO, which is central to spatial-data review."
},
{
"rank": 5,
"source": "Mapping_Standards_Version_1.20_Jan2024.pdf",
"page": 4,
"section": "1.1 Why spatial data are required",
"label": "useful",
"reason": "Provides direct justification that each assessment should provide spatial data and that these are essential for Criteria B and D2."
},
{
"rank": 6,
"source": "Mapping_Standards_Version_1.20_Jan2024.pdf",
"page": 7,
"section": "2.1 Species distribution text and map status in SIS",
"label": "useful",
"reason": "Directly relevant for checking whether the assessment's map status is acceptable and whether submission/publication constraints apply."
},
{
"rank": 7,
"source": "RedListGuidelines.pdf",
"page": 77,
"section": "9. Guidelines for Applying Criterion E",
"label": "not_useful",
"reason": "Discusses spatial data in the context of Criterion E models, which is largely off-target for a mapping review of this B2 assessment."
},
{
"rank": 8,
"source": "Mapping_Standards_Version_1.20_Jan2024.pdf",
"page": 5,
"section": "1.3 IUCN Red List versus IUCN Green Status mapping",
"label": "partly_useful",
"reason": "Contains some mapping context, but it is more about Green Status alignment than the main spatial checks needed here."
}
],
"draft_hit_labels": [
{
"rank": 1,
"section_path": "test_document > Red List Assessment > Assessment Rationale",
"score": 2.75,
"label": "partly_useful",
"reason": "Contains EOO, AOO, locations, and broad range information, but it is not the main draft section for map/spatial-data checks."
},
{
"rank": 2,
"section_path": "test_document > Red List Assessment > Assessment Information",
"score": 1.75,
"label": "not_useful",
"reason": "Assessor metadata does not help review mapping or spatial-data quality."
},
{
"rank": 3,
"section_path": "test_document > Distribution > Map Status",
"score": 1.0,
"label": "useful",
"reason": "Directly relevant because it gives map status, how the map was created, and the georeferenced specimen basis."
},
{
"rank": 4,
"section_path": "test_document > Ecosystem Services > Ecosystem Services Provided by the Species",
"score": 1.0,
"label": "not_useful",
"reason": "Unrelated to mapping or spatial-data review."
},
{
"rank": 5,
"section_path": "test_document > Bibliography",
"score": 1.0,
"label": "partly_useful",
"reason": "Shows potential spatial data sources and tools such as GBIF and GeoCAT, but only indirectly supports the mapping review."
},
{
"rank": 6,
"section_path": "test_document",
"score": 0.0,
"label": "not_useful",
"reason": "The document root only gives title and status and does not add substantive spatial-review evidence."
}
],
"summary": "The retrieval package gets several important mapping references and one directly relevant draft section, so it is not a miss. However, coverage is incomplete because key draft-side spatial evidence was not retrieved, especially the Geographic Range, Countries of Occurrence, and AOO/locations sections that would be needed to actually review mapping consistency in this assessment."
}
```

### Question 5

Question: Does the uploaded assessment include the information needed to justify population trend and continuing decline?

```json
{
"question": "Does the uploaded assessment include the information needed to justify population trend and continuing decline?",
"route": "hybrid_rag",
"relevance_score": 3,
"coverage_score": 3,
"focus_score": 2,
"covered_evidence": [
"Reference retrieval includes the supporting-information table that contains the requirement to record direction of current population trend and broader rationale/data requirements.",
"Reference retrieval includes the requirement for narrative text about population, habitat/ecology, and threats, which is relevant to justifying continuing decline.",
"Draft retrieval includes the assessment rationale stating continuing deterioration of habitat quality and the continuing-decline-in-habitat justification table.",
"Draft retrieval includes the coded population trend field showing the assessment currently records trend as Unknown."
],
"missing_evidence": [
"The retrieved draft hits miss the nearby population narrative stating that there is little direct population information but the population is suspected to be decreasing, which is central for evaluating population-trend justification in the uploaded draft.",
"The retrieved draft hits miss the detailed Threats section describing mining, agricultural conversion, dams, road construction, urban expansion, and fire, which provides stronger draft-side support for continuing decline than the shorter retrieved snippets.",
"Reference retrieval does not bring back a more targeted rule chunk focused specifically on how continuing decline should be documented under the applied criterion context; most reference evidence is generic supporting-information guidance.",
"Several retrieved reference rows are about unrelated special cases such as A4 time period, generation length, Criterion E, Near Threatened justification, and Possibly Extinct tagging."
],
"reference_item_labels": [
{
"rank": 1,
"source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
"page": 3,
"section": "Required Information Purpose Guidance Notes",
"label": "useful",
"reason": "Broad but relevant table; includes the requirement to record current population trend and general rationale/data requirements needed to judge whether the draft contains required support."
},
{
"rank": 2,
"source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
"page": 8,
"section": "Specific Condition Purpose Guidance Notes",
"label": "not_useful",
"reason": "A4 3-generation decline timing is not the key evidence for this question and does not help much with the uploaded draft's population trend or continuing-decline support."
},
{
"rank": 3,
"source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
"page": 8,
"section": "Specific Condition Purpose Guidance Notes",
"label": "not_useful",
"reason": "Near Threatened nearly-met criteria are unrelated to the question."
},
{
"rank": 4,
"source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
"page": 8,
"section": "Specific Condition Purpose Guidance Notes",
"label": "not_useful",
"reason": "Generation length for criteria A and C1 is not relevant to the main evidence needed here."
},
{
"rank": 5,
"source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
"page": 10,
"section": "Specific Condition Purpose Guidance Notes",
"label": "useful",
"reason": "Narrative text about population, habitat/ecology, and threats is directly relevant to checking whether the uploaded assessment contains justification for continuing decline."
},
{
"rank": 6,
"source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
"page": 8,
"section": "Specific Condition Purpose Guidance Notes",
"label": "not_useful",
"reason": "Criterion E / PVA documentation is unrelated to this question."
},
{
"rank": 7,
"source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
"page": 9,
"section": "Specific Condition Purpose Guidance Notes",
"label": "partly_useful",
"reason": "Major threats can help support continuing decline, but this is less directly targeted than the population-trend requirement or the narrative-support requirement."
},
{
"rank": 8,
"source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
"page": 7,
"section": "Specific Condition Purpose Guidance Notes",
"label": "not_useful",
"reason": "Possibly Extinct tagging is irrelevant to the question."
}
],
"draft_hit_labels": [
{
"rank": 1,
"section_path": "test_document > Population > Population Information",
"score": 8.5,
"label": "partly_useful",
"reason": "Directly on topic because it gives the coded trend field, but it only says Current Population Trend: Unknown and does not retrieve the stronger nearby narrative about suspected decrease."
},
{
"rank": 2,
"section_path": "test_document > Conservation > Research Needed",
"score": 5.75,
"label": "partly_useful",
"reason": "Indirectly useful because it signals unresolved needs around population trends and habitat monitoring, but it is not actual justification evidence."
},
{
"rank": 3,
"section_path": "test_document > Red List Assessment > Assessment Information",
"score": 5.7,
"label": "not_useful",
"reason": "Assessor names, institution, and region do not help answer the question."
},
{
"rank": 4,
"section_path": "test_document > Red List Assessment > Assessment Rationale",
"score": 5.55,
"label": "useful",
"reason": "Contains direct draft-side justification that habitat quality is continuing to deteriorate because of mining and agricultural conversion."
},
{
"rank": 5,
"section_path": "test_document > Habitats and Ecology > Continuing Decline in Habitat",
"score": 4.5,
"label": "useful",
"reason": "Highly relevant: explicitly states continuing decline in habitat, qualifier, and justification."
},
{
"rank": 6,
"section_path": "test_document > Conservation > Important Conservation Actions Needed",
"score": 4.05,
"label": "not_useful",
"reason": "Conservation actions needed are not the core evidence for whether the assessment already includes justification for population trend and continuing decline."
}
],
"summary": "The retrieval package finds some of the right evidence for continuing decline and the coded population-trend field, but it is incomplete and noisy. The biggest draft-side miss is the population narrative stating that the population is suspected to be decreasing, and the detailed Threats section that more fully supports continuing decline in the uploaded assessment ."
}
```

### Question 6

Question: What should be checked in the assessment rationale for consistency with IUCN Red List guidance?

```json
{
"question": "What should be checked in the assessment rationale for consistency with IUCN Red List guidance?",
"route": "hybrid_rag",
"relevance_score": 4,
"coverage_score": 3,
"focus_score": 3,
"covered_evidence": [
"Reference guidance that every assessment needs a rationale and that it should justify the selected category and criteria.",
"Reference guidance that the rationale should summarize the reasons the taxon qualifies, not merely repeat the criteria thresholds.",
"Reference guidance that the rationale should include inferences and uncertainty tied to interpretation of data against criteria thresholds.",
"Reference guidance that relevant subcriteria parameters and qualifiers should be clearly indicated in supporting narrative or data fields.",
"Draft retrieval included the Assessment Rationale section itself and a Threats section that could be compared against the rationale."
],
"missing_evidence": [
"Draft-side retrieval missed the key supporting sections for checking whether the rationale is internally consistent with the cited criterion B evidence, especially Area of Occupancy (44 km2).",
"Draft-side retrieval missed the Locations Information section needed to verify the rationale's statement about 11 locations and whether that supports EN under B2ab(iii).",
"Draft-side retrieval missed the Continuing Decline in Habitat section needed to verify the rationale's claim about ongoing habitat-quality decline.",
"Draft-side retrieval missed a direct draft section for Geographic Range / EOO support, even though the rationale cites an EOO of about 114,000 km2.",
"The package contains duplicate reference evidence and several weak draft hits instead of retrieving the core supporting draft fields needed for a consistency check."
],
"reference_item_labels": [
{
"rank": 1,
"source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
"page": 3,
"section": "Required Information Purpose Guidance Notes",
"label": "useful",
"reason": "Directly states what the rationale must do and that it should include inferences or uncertainty related to the criteria and thresholds."
},
{
"rank": 2,
"source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
"page": 3,
"section": "Required Information Purpose Guidance Notes",
"label": "useful",
"reason": "Covers both the rationale requirement and the need for clear parameter and qualifier reporting that supports consistency checking."
},
{
"rank": 3,
"source": "RL_Standards_Consistency.pdf",
"page": 23,
"section": "2.2.8. Assessment Rationale",
"label": "useful",
"reason": "Highly on-point guidance: the rationale should not simply quote thresholds and should summarize why the taxon qualifies."
},
{
"rank": 4,
"source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
"page": 3,
"section": "Required Information Purpose Guidance Notes",
"label": "useful",
"reason": "Concise restatement of the core rationale requirement and its link to the selected category and criteria."
},
{
"rank": 5,
"source": "RL_Standards_Consistency.pdf",
"page": 65,
"section": "5.2. Consistency Checks",
"label": "partly_useful",
"reason": "Relevant to broader assessment consistency, but it is not specifically about what to check inside the rationale text."
},
{
"rank": 6,
"source": "RL_Standards_Consistency.pdf",
"page": 33,
"section": "2.5.5. Rationale ( Assessment Rational section in SIS)",
"label": "partly_useful",
"reason": "Confirms that a rationale is required, but adds little detail about the actual consistency checks needed."
},
{
"rank": 7,
"source": "RL_Standards_Consistency.pdf",
"page": 6,
"section": "Required Information Purpose Guidance Notes",
"label": "partly_useful",
"reason": "Provides general requirement context, but is less specific than the stronger rationale-focused items already retrieved."
},
{
"rank": 8,
"source": "RL_categories_and_criteria.pdf",
"page": 31,
"section": "Annex 3: Required and Recommended Supporting Information for IUCN Red List Assessments",
"label": "not_useful",
"reason": "Mostly points to another guidance document rather than providing concrete rationale-checking evidence."
}
],
"draft_hit_labels": [
{
"rank": 1,
"section_path": "test_document > Red List Assessment > Assessment Rationale",
"score": 10.8,
"label": "useful",
"reason": "This is the main draft section that must be checked against guidance and against the supporting evidence elsewhere in the assessment."
},
{
"rank": 2,
"section_path": "test_document > Red List Assessment > Assessment Information",
"score": 5.25,
"label": "not_useful",
"reason": "Assessor names, institution, and region do not help evaluate rationale consistency."
},
{
"rank": 3,
"section_path": "test_document",
"score": 3.0,
"label": "partly_useful",
"reason": "The overall status and criterion code are relevant context for checking whether the rationale matches the assigned category, but this chunk is not sufficient on its own."
},
{
"rank": 4,
"section_path": "test_document > Threats",
"score": 2.0,
"label": "partly_useful",
"reason": "Threat information can support the rationale's statements about habitat decline, but it does not cover the key B2 parameters such as AOO and number of locations."
},
{
"rank": 5,
"section_path": "test_document > Habitats and Ecology > IUCN Habitats Classification Scheme",
"score": 1.75,
"label": "not_useful",
"reason": "Habitat coding is peripheral here and does not directly check whether the rationale is consistent with the main criterion B justification."
},
{
"rank": 6,
"section_path": "test_document > Distribution > Map Status",
"score": 1.0,
"label": "not_useful",
"reason": "Map preparation metadata is not the key draft evidence needed for checking rationale consistency."
}
],
"summary": "The retrieval package is strong on reference-side guidance for what a rationale should contain, especially that it should justify the category, not merely restate thresholds, and should include uncertainty or inference. Coverage is only partial on the draft side because it retrieves the rationale itself but misses the main supporting sections needed to verify consistency with the rationale's claims about AOO, locations, and continuing decline. The package is also somewhat noisy due to duplicate reference evidence and several weak draft hits."
}
```
