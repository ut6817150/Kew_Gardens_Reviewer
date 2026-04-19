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
  "question": "What mapping or spatial-data checks should be considered when reviewing this assessment?",
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
  "answer_scaffold": "Answer scaffold\n---------------\nMost relevant retrieved reference evidence:\n- Section: Required Information Purpose Guidance Notes | Source: Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf | Page: 5 | Evidence: Entry 8: 8. Geo-referenced distribution • To support Red List website Spatial distribution data are not data for all taxa with a known functionality required for taxa of unknown distribution provenance (e.g. taxa assessed as • To allow basic analyses Data Defi\n- Section: 1.3.2 Mapping of Subspecies, Varieties or Subpopulations versus Green Status Spatial Units | Source: Mapping_Standards_Version_1.20_Jan2024.pdf | Page: 6 | Evidence: . Further guidance will be forthcoming to assist assessors who wish to map Spatial Units in a GSS assessment in how to capture this information in the attribute data.\n- Section: 4.1.2 Recommended polygon and point attributes | Source: Mapping_Standards_Version_1.20_Jan2024.pdf | Page: 12 | Evidence: dec_long: data_sens | The geographical longitude in decimal degrees: Flags up whether or not the polygon distribution/data point is sensitive. [Required if data is sensitive] | E.g., -121.25. Positive values are east of the Greenwich Meridian; negative values \n- Section: Notes: | Source: Mapping_Standards_Version_1.20_Jan2024.pdf | Page: 15 | Evidence: column_1: 1 | CODE:  | column_3:  | column_4: Extant | PRESENCE:  | column_6:  | column_7: The species is known or thought very likely to occur currently in the area, which encompasses localities with current or recent (last 20-30 years) records where suitable",
  "subqueries": [
    "What mapping or spatial-data checks should be considered when reviewing this assessment?"
  ],
  "retrieved_reference_items": [
    {
      "rank": 1,
      "block_type": "table_row",
      "source": "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments.pdf",
      "page": 5,
      "section": "Required Information Purpose Guidance Notes",
      "text": "Entry 8: 8. Geo-referenced distribution • To support Red List website Spatial distribution data are not data for all taxa with a known functionality required for taxa of unknown distribution provenance (e.g. taxa assessed as • To allow basic analyses Data Deficient because their range is • Spatial distribution data are not known). essential for supporting Spatial data may be geo-referenced assessments under criteria B polygons or point localities, and may and D2 (and arguably also for be provided in any format, including as demonstrating that these a paper map, text file of coordinates, thresholds are not met) pdf, graphics file or GIS shapefile. A GIS shapefile is preferred (but is not strictly required), given their value for conducting spatial analyses, visual displays on the Red List website, and future functionality on the Red List website that will allow spatial searches. Although additional distributional documentation is desirable for taxa qualifying under criterion B (e.g., 2x2 km grids showing occupancy), this is not Required. Note that any distributional data can be coded as sensitive to avoid this being distributed or displayed on the Red List website (see Annex 5).",
      "parent_text": "Table 1: Required supporting information for all assessments\nSchema: Required Information | Purpose | Guidance | Notes\nPage 3 | Entry 1: 1. Scientific name1 • To identify which taxon is If the taxon is already in SIS, this being assessed information requires no additional effort from the Assessors. If the taxon is not • To support Red List website yet recorded in SIS, Assessors must functionality provide this information to the Red List Unit.\nPage 3 | Entry 2: 2. Higher taxonomy details • To identify which taxon is If the taxon is already in SIS, this (Kingdom, Phylum, Class, being assessed requires no additional effort from the Order, Family) Assessors. If the taxon is not yet • To support Red List website recorded in SIS, Assessors must functionality provide this information to the Red List Unit.\nPage 3 | Entry 3: 3. Taxonomic authorities for all • To identify which taxon is If the taxon is already in SIS, this specific and infra-specific being assessed information requires no additional effort names used, following the from the Assessors. If the taxon is not appropriate nomenclatural yet recorded in SIS, Assessors must rules2 provide this information to the Red List Unit.\nPage 3 | Entry 4: 4. IUCN Red List Category and • To identify the current status The Red List Category and Criteria Criteria (including sub- of the taxon represent the most fundamental criteria) met at the highest elements of a Red List assessment. • To support Red List website category of threat functionality Application of the categories and criteria must be in accordance with the • To allow basic analyses IUCN Red List Categories and Criteria. Version 3.1 and the current version of the Guidelines for Using the IUCN Red List Categories and Criteria.\nPage 3 | Entry 5: 5. A rationale for the Red List • To justify the Red List Include any inferences or uncertainty assessment Category and Criteria that relate to the interpretation of the selected data and information in relation to the criteria and their thresholds. 1 Note that all taxa assessed must be validly published in accordance with the appropriate international nomenclatural codes and should be currently accepted names. Standard taxonomic checklists should be used wherever possible for names. The standard lists adopted by IUCN are periodically reviewed and listed on the Red List website: http://www.iucnredlist.org/info/info_sources_quality.html. For many groups no standards are available, or there may be a valid reason for adopting another treatment. In such cases, the taxonomic treatment followed should be indicated and if not one of the standards followed by IUCN, the reference should be cited in full and a reason for the deviation given This should include the date of publication, except in the case of plant names. The abbreviations used for author names of plants should follow Brummitt and Powell (1992) and subsequent updates on the International Plant Names Index website (http://www.ipni.org/index.html) 3\nPage 4 | Entry 6: 6. Data for parameters • To underpin and justify the Enter these data either into the triggering the Red List Red List Category and Criteria relevant coded/numerical fields or in Criteria met at the highest used the relevant narrative (text) fields in Category level SIS. If data are entered into the data fields, this allows the Red List Criteria calculator to be used in SIS, which automatically checks for errors, omissions and inconsistencies, reducing the burden of manual checking by Assessors, RLA Coordinators and project coordinators. If data are included within the narrative (text) fields, the text must clearly indicate all of the relevant subcriteria parameters and qualifiers (observed, estimated, inferred, projected or suspected) used.\nPage 4 | Entry 7: 7. Countries of occurrence (for • To support Red List website SIS automatically records Presence = native and reintroduced functionality (especially Extant and Origin = Native by default taxa), including Presence country searches) as countries are selected. and Origin coding A tool will be made available to • To allow basic analyses determine countries of occurrence automatically from GIS maps. Countries of occurrence are not strictly required for vagrant and introduced ranges. 4\nPage 5 | Entry 8: 8. Geo-referenced distribution • To support Red List website Spatial distribution data are not data for all taxa with a known functionality required for taxa of unknown distribution provenance (e.g. taxa assessed as • To allow basic analyses Data Deficient because their range is • Spatial distribution data are not known). essential for supporting Spatial data may be geo-referenced assessments under criteria B polygons or point localities, and may and D2 (and arguably also for be provided in any format, including as demonstrating that these a paper map, text file of coordinates, thresholds are not met) pdf, graphics file or GIS shapefile. A GIS shapefile is preferred (but is not strictly required), given their value for conducting spatial analyses, visual displays on the Red List website, and future functionality on the Red List website that will allow spatial searches. Although additional distributional documentation is desirable for taxa qualifying under criterion B (e.g., 2x2 km grids showing occupancy), this is not Required. Note that any distributional data can be coded as sensitive to avoid this being distributed or displayed on the Red List website (see Annex 5).\nPage 5 | Entry 9: 9. Direction of current • To support Red List website population trend (stable, functionality increasing, decreasing, • To allow basic analyses unknown)\nPage 5 | Entry 10: 10. Coding for occurrence in • To support Red List website freshwater (= inland waters), functionality terrestrial, and marine ecosystems (i.e., “System” in • To allow basic analyses SIS)\nPage 5 | Entry 11: 11. Suitable habitats utilized • To support the assessment To speed up entering such coding in (coded to lowest level in SIS, habitat importance is set to • To support Red List website Habitats Classification 'suitable' by default for any habitat functionality Scheme) selected. • To allow basic analyses\nPage 5 | Entry 12: 12. Bibliography (cited in full; • To underpin the assessment In SIS, references are recorded in the including unpublished data and provide all sources of Reference Manager. sources but not personal data and information used to communications) support the Red List assessment 5"
    },
    {
      "rank": 2,
      "block_type": "text",
      "source": "Mapping_Standards_Version_1.20_Jan2024.pdf",
      "page": 6,
      "section": "1.3.2 Mapping of Subspecies, Varieties or Subpopulations versus Green Status Spatial Units",
      "text": ". Further guidance will be forthcoming to assist assessors who wish to map Spatial Units in a GSS assessment in how to capture this information in the attribute data."
    },
    {
      "rank": 3,
      "block_type": "table_row",
      "source": "Mapping_Standards_Version_1.20_Jan2024.pdf",
      "page": 12,
      "section": "4.1.2 Recommended polygon and point attributes",
      "text": "dec_long: data_sens | The geographical longitude in decimal degrees: Flags up whether or not the polygon distribution/data point is sensitive. [Required if data is sensitive] | E.g., -121.25. Positive values are east of the Greenwich Meridian; negative values are west of it. Valid values lie between -180 and 180.: This is most likely to be the case if the polygon or point shows individual localities of a sensitive nature as determined by the Assessor. True or false field: 1 or 0. If 1 for true/yes, the field sens_comm should be completed. Default is 0 for false/no. | column_4: ✔ | ✔: ✔",
      "parent_text": "Table title: 4.1.2 Recommended polygon and point attributes\nHeaders: dec_long | The geographical longitude in decimal degrees | E.g., -121.25. Positive values are east of the Greenwich Meridian; negative values are west of it. Valid values lie between -180 and 180. | column_4 | ✔\ndec_long: spatialref | The geographical longitude in decimal degrees: The ellipsoid, geodetic datum or spatial reference system (SRS) upon which the geographic coordinates (supplied in dec_lat and dec_long) are based | E.g., -121.25. Positive values are east of the Greenwich Meridian; negative values are west of it. Valid values lie between -180 and 180.: Data is preferred in WGS84. If blank, the default of WGS84 will be assumed. E.g.: “WGS84”; “EPSG:4326”; “NAD27”; “Campo Inchauspe”; “European 1950”; “Clarke 1866”. | column_4:  | ✔: ✔\ndec_long: subspecies | The geographical longitude in decimal degrees: Subspecies Name/Epithet [Required if relevant] | E.g., -121.25. Positive values are east of the Greenwich Meridian; negative values are west of it. Valid values lie between -180 and 180.: To indicate that the data relates to a specific subspecies (or variety) of the assessed taxon. This must then match the infra name in SIS, e.g., “persica”, “brevifolia\". | column_4: ✔ | ✔: ✔\ndec_long: subpop | The geographical longitude in decimal degrees: Subpopulation Name/Epithet [Required if relevant] | E.g., -121.25. Positive values are east of the Greenwich Meridian; negative values are west of it. Valid values lie between -180 and 180.: To indicate that the data relates to a specific subpopulation of the assessed taxon, e.g., “Hawaiian subpopulation”. | column_4: ✔ | ✔: ✔\ndec_long: data_sens | The geographical longitude in decimal degrees: Flags up whether or not the polygon distribution/data point is sensitive. [Required if data is sensitive] | E.g., -121.25. Positive values are east of the Greenwich Meridian; negative values are west of it. Valid values lie between -180 and 180.: This is most likely to be the case if the polygon or point shows individual localities of a sensitive nature as determined by the Assessor. True or false field: 1 or 0. If 1 for true/yes, the field sens_comm should be completed. Default is 0 for false/no. | column_4: ✔ | ✔: ✔\ndec_long: sens_comm | The geographical longitude in decimal degrees: Comments on why the data are considered sensitive [Required if data_sens is 1] | E.g., -121.25. Positive values are east of the Greenwich Meridian; negative values are west of it. Valid values lie between -180 and 180.: [Max. 254 characters] | column_4: ✔ | ✔: ✔"
    },
    {
      "rank": 4,
      "block_type": "table_row",
      "source": "Mapping_Standards_Version_1.20_Jan2024.pdf",
      "page": 15,
      "section": "Notes:",
      "text": "column_1: 1 | CODE:  | column_3:  | column_4: Extant | PRESENCE:  | column_6:  | column_7: The species is known or thought very likely to occur currently in the area, which encompasses localities with current or recent (last 20-30 years) records where suitable habitat at appropriate altitudes remains (see note 2). Extant polygons can include inferred or spatially projected sites of present occurrence (see the Guidelines for Using the IUCN Red List Categories and Criteria for further guidance). Extant ranges should be considered in the calculation of EOO or AOO. When mapping an “assisted colonisation” it is important to note that this range should be treated as Extant. | DEFINITION:  | column_9:",
      "parent_text": "Table title: Notes:\nHeaders: column_1 | CODE | column_3 | column_4 | PRESENCE | column_6 | column_7 | DEFINITION | column_9\ncolumn_1: 1 | CODE:  | column_3:  | column_4: Extant | PRESENCE:  | column_6:  | column_7: The species is known or thought very likely to occur currently in the area, which encompasses localities with current or recent (last 20-30 years) records where suitable habitat at appropriate altitudes remains (see note 2). Extant polygons can include inferred or spatially projected sites of present occurrence (see the Guidelines for Using the IUCN Red List Categories and Criteria for further guidance). Extant ranges should be considered in the calculation of EOO or AOO. When mapping an “assisted colonisation” it is important to note that this range should be treated as Extant. | DEFINITION:  | column_9: \ncolumn_1: 2 | CODE:  | column_3:  | column_4: Probably Extant | PRESENCE:  | column_6:  | column_7: This code value has been discontinued for reasons of ambiguity. It may exist in the spatial data but will gradually be phased out. | DEFINITION:  | column_9: \ncolumn_1: 3 | CODE:  | column_3:  | column_4: Possibly Extant | PRESENCE:  | column_6:  | column_7: There is no record of the species in the area, but the species may possibly occur, based on the distribution of potentially suitable habitat at appropriate altitudes, although the area is beyond where the species is Extant (i.e., beyond the limits of known or likely records), and the degree of probability of the species occurring is lower (e.g., because the area is beyond a geographic barrier, or because the area represents a considerable extension beyond areas of known or probable occurrence). Identifying Possibly Extant areas is useful to flag up areas where the taxon should be searched for. Possibly Extant ranges should not be considered in the calculation of EOO or AOO. | DEFINITION:  | column_9: \ncolumn_1: 4 | CODE:  | column_3:  | column_4: Possibly Extinct | PRESENCE:  | column_6:  | column_7: The species was formerly known or thought very likely to occur in the area (post 1500 CE), but it is most likely now extirpated from the area because habitat loss and/or other threats are thought likely to have extirpated the species, and there have been no confirmed recent records despite searches. Possibly Extinct ranges should not be considered in the calculation of EOO or AOO. | DEFINITION:  | column_9: \ncolumn_1: 5 | CODE:  | column_3:  | column_4: Extinct | PRESENCE:  | column_6:  | column_7: The species was formerly known or thought very likely to occur in the area (post 1500 CE), but it has been confirmed that the species no longer occurs because exhaustive searches have failed to produce recent records, and the intensity and timing of threats could plausibly have extirpated the taxon. Extinct ranges should not be considered in the calculation of EOO or AOO. | DEFINITION:  | column_9: \ncolumn_1: 6 | CODE:  | column_3:  | column_4: Presence Uncertain | PRESENCE:  | column_6:  | column_7: A record exists of the species' presence in the area, but this record requires verification or is rendered questionable owing to uncertainty over the identity or authenticity of the record, or the accuracy of the location. Presence Uncertain records should not be considered in the calculation of EOO or AOO. | DEFINITION:  | column_9: \ncolumn_1: 7 | CODE:  | column_3:  | column_4: Expected Additional Range | PRESENCE:  | column_6:  | column_7: The species is not currently known to occur in the area, but this area is expected to (1) become suitable in the next 100 years, taking into account range shifts resulting from climate change, AND (2) to become occupied by the species, with or without human assistance. Expected additional range is only considered in a Green Status of Species assessment and not for point occurrences. | DEFINITION:  | column_9:"
    },
    {
      "rank": 5,
      "block_type": "text",
      "source": "Mapping_Standards_Version_1.20_Jan2024.pdf",
      "page": 4,
      "section": "1.1 Why spatial data are required",
      "text": "Spatial data are some of the most frequently used data on The IUCN Red List. They are crucial information for conservation planning and enable a broad variety of research in support of conservation. They are essential for supporting Red List assessments under criteria B and D2. Each assessment should provide spatial data in some form. Assessors should produce the most accurate depiction of a taxon’s current and historic distribution based on their knowledge and the available data, in a format that is considered most appropriate to inform conservation action for the taxon. These maps are also displayed and can be used for spatial searches on The IUCN Red List website."
    },
    {
      "rank": 6,
      "block_type": "text",
      "source": "Mapping_Standards_Version_1.20_Jan2024.pdf",
      "page": 7,
      "section": "2.1 Species distribution text and map status in SIS",
      "text": "● Done - the map has been completed to the required standards and will be provided as part of the assessment. It will be published unless the data are marked as sensitive (see sections 4 and 8.5 ). ● Missing - the map is missing and needs to be located. An assessment with this status cannot be submitted or published. ● Incomplete - the map will be provided, but it is known not to be complete, for example there was not enough information to map certain parts of the range. Incomplete maps will be published (unless the data are marked as sensitive, see sections 4 and 8.5 ). A reason for the map being incomplete must be provided in the justification box in SIS. ● Not Possible - making a distribution map for the species is not possible"
    },
    {
      "rank": 7,
      "block_type": "text",
      "source": "RedListGuidelines.pdf",
      "page": 77,
      "section": "9. Guidelines for Applying Criterion E",
      "text": "The types of data that can be used in an assessment include spatial distributions of suitable habitat, local populations or individuals, patterns of occupancy and extinction in habitat patches, presence- absence data, habitat relationships, abundance estimates from surveys and censuses, vital rate (fecundity and survival) estimates from censuses and mark-recapture studies, as well as temporal variation and spatial covariation in these parameters. Not all of these types of data are required for any one model. For more information about data needs of particular types of PVA models, see the references mentioned above. When there is not sufficient data, or when the available information is too uncertain, it is risky to make a criterion E assessment with any method, including PVA"
    },
    {
      "rank": 8,
      "block_type": "text",
      "source": "Mapping_Standards_Version_1.20_Jan2024.pdf",
      "page": 5,
      "section": "1.3 IUCN Red List versus IUCN Green Status mapping",
      "text": "Note that for an IUCN Green Status assessment, assessors are asked to map three things: ● the historical (indigenous) part of the range, back to the benchmark date determined in the Green Status assessment; ● the expected additional range; and ● the delineated spatial units (although this is not compulsory). Since both IUCN Red List and IUCN Green Status assessments will be displayed on the IUCN Red List website with an accompanying map, it is important to be mindful of both processes and align the maps as much as possible to ensure their usefulness to the Red List end user. Here are some key points to remember and be mindful of:"
    }
  ],
  "draft_hits": [
    {
      "rank": 1,
      "section_path": "test_document > Red List Assessment > Assessment Rationale",
      "source_key": "test_document > Red List Assessment > Assessment Rationale",
      "score": 2.75,
      "text": "<p>Bulbostylis atracuminata occurs in a restricted area ranging from southern DRC to northern Zambia. Its extent of occurrence of about 114,000 km2, exceeds the threshold value for threatened categories. However, its actual suitable habitat within this range is considered to be smaller because its habitat is restricted to riverbanks and wetlands. This is reflected in the area of occupancy (44 km2), which falls within the threshold values for the Endangered category under criterion B2. There are an estimated 11 locations for this species in Zambia and DRC, with some locations found in natural environments with low human disturbance levels. However, the majority of its habitat is threatened by mining activity and agricultural land conversion, resulting in an overall continuing deterioration of habitat quality.. Though this species has a restricted AOO and is experiencing a continuing decline in habitat quality, the total number of locations marginally exceeds the threshold value for threatened categories under criterion B, and the existence of some high-quality habitat may enable the species to persist without being globally threatened in the near future. Therefore, this species is assessed as Endangered B2ab(iii). Despite these pressures, the species is unlikely to face major threats in the near future.</p>"
    },
    {
      "rank": 2,
      "section_path": "test_document > Red List Assessment > Assessment Information",
      "source_key": "test_document > Red List Assessment > Assessment Information",
      "score": 1.75,
      "text": "<p><b>Assessor(s): </b>Lemboye, B. & Lyu, E</p> <p><b>Institution(s): </b>Royal Botanic Gardens, Kew</p> <p><b>Regions: </b>Global</p>"
    },
    {
      "rank": 3,
      "section_path": "test_document > Distribution > Map Status",
      "source_key": "test_document > Distribution > Map Status",
      "score": 1.0,
      "text": "<table> <tr><td><b>Map Status</b></td><td><b>Use map from previous assessment</b></td><td><b>How the map was created, including data sources/methods used:</b></td><td><b>Please state reason for map not available:</b></td><td><b>Data Sensitive?</b></td><td><b>Justification</b></td><td><b>Geographic range this applies to:</b></td><td><b>Date restriction imposed:</b></td></tr> <tr><td>Done</td><td>-</td><td>Prepared in GeoCat (Moat et al. 2023) based on 11 georeferenced herbarium specimens (Global Biodiversity Information Facility 2023)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr> </table>"
    },
    {
      "rank": 4,
      "section_path": "test_document > Ecosystem Services > Ecosystem Services Provided by the Species",
      "source_key": "test_document > Ecosystem Services > Ecosystem Services Provided by the Species",
      "score": 1.0,
      "text": "<table> <tr><td><b>Insufficient Information Available</b></td><td><b>All coded services should have an importance score of 5 - Not Known.</b></td></tr> <tr><td>true</td><td>-</td></tr> </table>"
    },
    {
      "rank": 5,
      "section_path": "test_document > Bibliography",
      "source_key": "test_document > Bibliography",
      "score": 1.0,
      "text": "<p>2023. Botanic Gardens Conservation Internationa. Available at: . (Accessed: Sept 4, 2023).</p> <p>2023. Google Earth Pro. (Accessed: Sept 4, 2023).</p> <p>2023. Virunga National Park. Available at: . (Accessed: Sept 4, 2023).</p> <p>2023. World Database on Protected Areas. Available at: . (Accessed: Sept 4, 2023).</p> <p>GBIF: the Global Biodiversity Information Facility. 2022. Bulbostylis microelegans. Occurance data for Bulbostylis microelegans. GBIF. Available at: . (Accessed: Sept 4, 2023).</p> <p>Global Forest Watch. Available at: . (Accessed: Sept 4, 2023).</p> <p>Browning, Jane, Lock. M, John Michael, Beentje, Henk Jaap, Vollesen, Kaj, Larridon, Isabel, Xanthos, Martin Cheek, Martin Darbyshire, IainGoyder, D. J. 2020. Flora Zambesiaca. <i>Royal Botanic Gardens, Kew</i> London.</p> <p>Hoenselaar.K, Verdcourt.B & Beentje.H.J. 2010. <i>Flora of Tropical East Africa</i>. Royal Botanic Gardens Kew, London.</p> <p>Larridon, Reynders & Goetghebeur. 2008. Novelties in Nemum (Cyperaceae). <i>Belgian Journal of Botany</i> 141(2): 157-177.</p> <p>Moat, J., Bachman, S., & Walker, B. 2023. ShinyGeoCAT - Geospatial Conservation Assessment Tools (BETA).</p> <p>Roalson et al. 2019. A broader circumscription of Bulbostylis including Nemum (Abildgaardieae: Cyperaceae). <i>Phytotaxa</i> Vol. 395 (3): 119-208.</p> <p>Utsav, Tandon. 2024. SWE project. Vol. 20.</p> <p>[Roalson 2019]</p>"
    },
    {
      "rank": 6,
      "section_path": "test_document",
      "source_key": "test_document",
      "score": 0.0,
      "text": "<p><b>Draft</b></p> <p><b><i>Bulbostylis atracuminata</i></b><b> - (Larridon, Reynders & Goetgh.) Larridon & Roalson</b></p> <p>PLANTAE - TRACHEOPHYTA - LILIOPSIDA - POALES - CYPERACEAE - Bulbostylis - atracuminata</p> <p><b>Common Names: </b>No Common Names <b>Synonyms: </b>Nemum atracuminatum Larridon, Reynders & Goetgh.</p> <table> <tr><td><b>Red List Status</b></td></tr> <tr><td>EN, B2ab(iii) (IUCN version 3.1)</td></tr> </table>"
    }
  ]
}
