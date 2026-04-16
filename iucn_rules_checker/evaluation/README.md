# Evaluation

This folder records how the rules-based system was evaluated and refined.

## Initial Sweep

The rules-based system was first swept through 109 assessments prepared by Kew.
This first evaluation pass was carried out with the help of an LLM.

The results of that sweep are displayed in the Excel workbook saved in this
folder:

- `IUCN_submissions_evaluation.xlsx`

The rules based system was refined based on this output to reduce the number of false positives and false negatives.

## Further Refinement

After the initial sweep, the rules-based system was refined further using a
custom Word document:

- `test_doc_rules_based.docx`

That document is used together with:

- `evaluation.ipynb`

The notebook first converts the Word document to a Python dict using the
`parse_to_dict` function from `repo root > preprocessing > assessment_processor.py`,
then parses that result into the `full_report` structure used by the
rules-based system, and then runs the checker classes so their output can be
reviewed section by section.

## Files In This Folder

- `evaluation.ipynb`
  Notebook for running the custom Word document through the rules-based
  evaluation flow.
- `test_doc_rules_based.docx`
  Custom document used to refine and inspect rule behavior.
- `IUCN_submissions_evaluation.xlsx`
  Excel workbook containing the results of the initial 109-assessment sweep.
