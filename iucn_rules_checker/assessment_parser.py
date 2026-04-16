"""Parse a known assessment tree into a block-level report mapping."""

from typing import Any, Dict, List, Optional


class AssessmentParser:
    """
    Convert an assessment tree dict into a block-level full report mapping.

    Purpose:
        This class converts assessment-tree dictionaries into the block-level report mapping used by the reviewer.
    """

    def parse(self, assessment: Dict[str, Any]) -> Dict[str, str]:
        """
        Return a dict mapping block paths to paragraph and table-row rich text.

        Args:
            assessment (Dict[str, Any]): Input value used by this method.

        Returns:
            Dict[str, str]: Dictionary value produced by this method.
        """
        if not isinstance(assessment, dict):
            raise TypeError("AssessmentParser.parse() expects a Python dictionary.")

        full_report: Dict[str, str] = {}
        self.parse_document(assessment, full_report)
        return full_report

    def parse_document(
        self,
        node: Dict[str, Any],
        full_report: Dict[str, str],
        path: Optional[List[str]] = None,
    ) -> None:
        """
        Walk the known tree structure and collect rich text by block path.

        Args:
            node (Dict[str, Any]): Assessment tree node supplied to the parser.
            full_report (Dict[str, str]): Parsed ``section -> text`` mapping supplied by the caller.
            path (Optional[List[str]]): Input value used by this method.

        Returns:
            None: Value produced by this method.
        """
        if path is None:
            path = []

        title = self.coerce_text(node["title"])
        if title:
            path = path + [title]

        block_counts: Dict[str, int] = {}
        for block in node["blocks"]:
            if block.get("type") == "style":
                continue

            if block.get("type") == "table":
                table_rows = self.extract_table_rows(block)
                if table_rows:
                    table_key = self.build_block_key(path, block, block_counts)
                    for row_index, row_text in enumerate(table_rows, start=1):
                        row_key = f"{table_key} [row {row_index}]"
                        full_report[row_key] = row_text
                continue

            block_text = self.extract_block_text(block)
            if block_text:
                section_key = self.build_block_key(path, block, block_counts)
                full_report[section_key] = block_text

        for child in node["children"]:
            self.parse_document(child, full_report, path)

    def build_block_key(
        self,
        path: List[str],
        block: Dict[str, Any],
        block_counts: Dict[str, int],
    ) -> str:
        """
        Build a unique key for one block under a section path.

        Args:
            path (List[str]): Input value used by this method.
            block (Dict[str, Any]): Single block dictionary extracted from the assessment tree.
            block_counts (Dict[str, int]): Input value used by this method.

        Returns:
            str: String value produced by this method.
        """
        section_key = " > ".join(path)
        block_type = str(block.get("type", "block")).strip().lower() or "block"

        block_counts[block_type] = block_counts.get(block_type, 0) + 1
        return f"{section_key} [{block_type} {block_counts[block_type]}]"

    def extract_block_text(self, block: Dict[str, Any]) -> str:
        """
        Extract one paragraph block from `text_rich` only.

        Args:
            block (Dict[str, Any]): Single block dictionary extracted from the assessment tree.

        Returns:
            str: String value produced by this method.
        """
        block_type = block.get("type")

        if block_type == "paragraph":
            return self.coerce_text(block.get("text_rich", "")).strip()

        return ""

    def coerce_text(self, value: Any) -> str:
        """
        Convert parser input values to text while normalizing non-breaking spaces.

        Args:
            value (Any): Input value consumed by this helper.

        Returns:
            str: String value produced by this method.
        """
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8").replace("\u00A0", " ")
        if isinstance(value, str):
            return value.replace("\u00A0", " ")
        return str(value).replace("\u00A0", " ")

    def extract_table_rows(self, block: Dict[str, Any]) -> List[str]:
        """
        Render one table block from `rows_rich` only as per-row strings.

        Args:
            block (Dict[str, Any]): Single block dictionary extracted from the assessment tree.

        Returns:
            List[str]: List value produced by this method.
        """
        rows = block.get("rows_rich", [])
        formatted_rows: List[str] = []

        for row in rows:
            if not isinstance(row, list):
                row_text = self.coerce_text(row).strip()
                if row_text:
                    formatted_rows.append(row_text)
                continue

            cells = [self.stringify_cell(cell) for cell in row]
            cells = [cell for cell in cells if cell]
            if cells:
                formatted_rows.append(" | ".join(cells))

        return formatted_rows

    def stringify_cell(self, cell: Any) -> str:
        """
        Convert one table cell into text.

        Args:
            cell (Any): Single table cell supplied by the caller.

        Returns:
            str: String value produced by this method.
        """
        if cell is None:
            return ""

        if isinstance(cell, list):
            return " ; ".join(
                cell_text for cell_text in (self.stringify_cell(item) for item in cell) if cell_text
            )

        return self.coerce_text(cell).strip()
