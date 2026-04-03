"""Parse a known assessment tree into a block-level report mapping."""

from typing import Any, Dict, List, Optional


class AssessmentParser:
    """Convert an assessment tree dict into a block-level full report mapping."""

    def parse(self, assessment: Dict[str, Any]) -> Dict[str, str]:
        """Return a dict mapping block paths to paragraph and table-row text."""
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
        """Walk the known tree structure and collect text by block path."""
        if path is None:
            path = []

        title = self.coerce_text(node["title"])
        if title:
            path = path + [title]

        block_counts: Dict[str, int] = {}
        node_entry_keys: List[str] = []
        for block in node["blocks"]:
            if block.get("type") == "style":
                self.apply_style_block(block, node_entry_keys, full_report)
                continue

            if block.get("type") == "table":
                table_rows = self.extract_table_rows(block)
                if table_rows:
                    table_key = self.build_block_key(path, block, block_counts)
                    for row_index, row_text in enumerate(table_rows, start=1):
                        row_key = f"{table_key} [row {row_index}]"
                        full_report[row_key] = row_text
                        node_entry_keys.append(row_key)
                continue

            block_text = self.extract_block_text(block)
            if block_text:
                section_key = self.build_block_key(path, block, block_counts)
                full_report[section_key] = block_text
                node_entry_keys.append(section_key)

        for child in node["children"]:
            self.parse_document(child, full_report, path)

    def build_block_key(
        self,
        path: List[str],
        block: Dict[str, Any],
        block_counts: Dict[str, int],
    ) -> str:
        """Build a unique key for one block under a section path."""
        section_key = " > ".join(path)
        block_type = str(block.get("type", "block")).strip().lower() or "block"

        block_counts[block_type] = block_counts.get(block_type, 0) + 1
        return f"{section_key} [{block_type} {block_counts[block_type]}]"

    def extract_block_text(self, block: Dict[str, Any]) -> str:
        """Convert one known non-table block into rich text for the full report."""
        block_type = block.get("type")

        if block_type == "paragraph":
            return self.coerce_text(block.get("text_rich", block.get("text", ""))).strip()

        return ""

    def coerce_text(self, value: Any) -> str:
        """Convert parser input values to text while preserving Unicode characters."""
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, str):
            return value
        return str(value)

    def apply_style_block(
        self,
        block: Dict[str, Any],
        entry_keys: List[str],
        full_report: Dict[str, str],
    ) -> None:
        """Apply one style block to previously collected entries in the same node."""
        data = block.get("data", {})
        if not isinstance(data, dict):
            return

        bold_terms = self.normalize_style_terms(data.get("bold", []))
        italic_terms = self.normalize_style_terms(data.get("italic", []))
        if not bold_terms and not italic_terms:
            return

        for entry_key in entry_keys:
            full_report[entry_key] = self.apply_inline_styles(
                full_report[entry_key],
                bold_terms=bold_terms,
                italic_terms=italic_terms,
            )

    def normalize_style_terms(self, values: Any) -> List[str]:
        """Normalize one style list into non-empty string terms."""
        if not isinstance(values, list):
            return []

        normalized_terms: List[str] = []
        for value in values:
            term = self.coerce_text(value).strip()
            if term:
                normalized_terms.append(term)
        return normalized_terms

    def apply_inline_styles(
        self,
        text: str,
        bold_terms: List[str],
        italic_terms: List[str],
    ) -> str:
        """Wrap matching substrings in bold/italic tags based on one style block."""
        if not text:
            return text

        bold_mask = [False] * len(text)
        italic_mask = [False] * len(text)

        for term in bold_terms:
            self.mark_term_spans(text, term, bold_mask)
        for term in italic_terms:
            self.mark_term_spans(text, term, italic_mask)

        styled_parts: List[str] = []
        bold_open = False
        italic_open = False

        for index, character in enumerate(text):
            desired_bold = bold_mask[index]
            desired_italic = italic_mask[index]

            if (bold_open, italic_open) != (desired_bold, desired_italic):
                if italic_open:
                    styled_parts.append("</i>")
                if bold_open:
                    styled_parts.append("</b>")
                if desired_bold:
                    styled_parts.append("<b>")
                if desired_italic:
                    styled_parts.append("<i>")
                bold_open = desired_bold
                italic_open = desired_italic

            styled_parts.append(character)

        if italic_open:
            styled_parts.append("</i>")
        if bold_open:
            styled_parts.append("</b>")

        return "".join(styled_parts)

    def mark_term_spans(self, text: str, term: str, mask: List[bool]) -> None:
        """Mark every occurrence of one term in a boolean style mask."""
        if not term:
            return

        start = 0
        while True:
            index = text.find(term, start)
            if index == -1:
                return

            end = index + len(term)
            for position in range(index, end):
                mask[position] = True
            start = end

    def extract_table_rows(self, block: Dict[str, Any]) -> List[str]:
        """Render one table block as a list of per-row strings."""
        rows = block.get("rows_rich", block.get("rows", []))
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
        """Convert one table cell into text."""
        if cell is None:
            return ""

        if isinstance(cell, list):
            return " ; ".join(
                cell_text for cell_text in (self.stringify_cell(item) for item in cell) if cell_text
            )

        return self.coerce_text(cell).strip()
