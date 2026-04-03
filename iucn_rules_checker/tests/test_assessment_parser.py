"""Regression tests for the block-level assessment parser."""

import json
import unittest
from pathlib import Path

from iucn_rules_checker.assessment_parser import AssessmentParser


class AssessmentParserTests(unittest.TestCase):
    """Lock in the parser's current block-by-block behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        json_path = (
            Path(__file__).resolve().parent.parent
            / "test_json_file"
            / "Acianthera odontotepala_draft_status_Jun2025 (1).json"
        )
        with json_path.open(encoding="utf-8") as handle:
            cls.assessment = json.load(handle)

    def test_parse_returns_block_level_full_report(self) -> None:
        full_report = AssessmentParser().parse(self.assessment)
        title = self.assessment["title"]
        paragraph_entries = [key for key in full_report if "[paragraph " in key]
        table_row_entries = [key for key in full_report if "[table " in key and "[row " in key]

        root_paragraph_1 = f"{title} [paragraph 1]"
        root_paragraph_2 = f"{title} [paragraph 2]"
        root_table_row_1 = f"{title} [table 1] [row 1]"
        root_table_row_2 = f"{title} [table 1] [row 2]"
        assessment_info_1 = f"{title} > Red List Assessment > Assessment Information [paragraph 1]"
        rationale_1 = f"{title} > Red List Assessment > Assessment Rationale [paragraph 1]"
        bibliography_1 = f"{title} > Bibliography [paragraph 1]"
        self.assertIsInstance(full_report, dict)
        self.assertEqual(len(full_report), 79)
        self.assertEqual(len(paragraph_entries), 33)
        self.assertEqual(len(table_row_entries), 46)
        self.assertTrue(all(isinstance(key, str) for key in full_report))
        self.assertTrue(
            all(isinstance(value, str) and value.strip() for value in full_report.values())
        )
        self.assertFalse(any("[style " in key for key in full_report))

        for entry in [
            root_paragraph_1,
            root_paragraph_2,
            root_table_row_1,
            root_table_row_2,
            assessment_info_1,
            rationale_1,
            bibliography_1,
        ]:
            self.assertIn(entry, full_report)

        self.assertEqual(full_report[root_paragraph_1], "<b>Draft</b>")
        self.assertIn("<b><i>Acrocarpus fraxinifolius</i></b>", full_report[root_paragraph_2])
        self.assertEqual(full_report[root_table_row_1], "Red List Status")
        self.assertIn("LC - Least Concern", full_report[root_table_row_2])
        self.assertIn("<b>Date of Assessment:</b>", full_report[assessment_info_1])
        self.assertIn("<i>Acrocarpus fraxinifolius</i>", full_report[rationale_1])
        self.assertIn("<i>Tree flora of china. Vol II</i>", full_report[bibliography_1])
        self.assertIn("<sup>2</sup>", full_report[rationale_1])
        self.assertFalse(any("&amp;" in value for value in full_report.values()))
        self.assertFalse(any("-&gt;" in value for value in full_report.values()))

    def test_parse_prefers_rich_block_fields_when_present(self) -> None:
        assessment = {
            "title": "Root",
            "blocks": [
                {
                    "type": "paragraph",
                    "text": "Area 10 km2",
                    "text_rich": "Area 10 km<sup>2</sup>",
                },
                {
                    "type": "table",
                    "rows": [["CO2"]],
                    "rows_rich": [["CO<sub>2</sub>"]],
                },
            ],
            "children": [],
        }

        full_report = AssessmentParser().parse(assessment)

        self.assertEqual(full_report["Root [paragraph 1]"], "Area 10 km<sup>2</sup>")
        self.assertEqual(full_report["Root [table 1] [row 1]"], "CO<sub>2</sub>")

    def test_parse_preserves_non_ascii_characters_as_is(self) -> None:
        assessment = {
            "title": "Région – Root",
            "blocks": [
                {
                    "type": "paragraph",
                    "text": "Méndez observed 14–26 °C.",
                    "text_rich": "Méndez observed 14–26 °C.",
                },
                {
                    "type": "table",
                    "rows": [["Área", "14–26 °C"]],
                    "rows_rich": [["Área", "14–26 °C"]],
                },
            ],
            "children": [],
        }

        full_report = AssessmentParser().parse(assessment)

        self.assertIn("Région – Root [paragraph 1]", full_report)
        self.assertEqual(
            full_report["Région – Root [paragraph 1]"],
            "Méndez observed 14–26 °C.",
        )
        self.assertEqual(
            full_report["Région – Root [table 1] [row 1]"],
            "Área | 14–26 °C",
        )


if __name__ == "__main__":
    unittest.main()
