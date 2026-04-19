"""Inference-side assessment parser for RAG draft retrieval.

Purpose:
    This module converts the uploaded assessment tree dict into a section-level
    HTML report mapping. The flatter report format is easier for draft-side
    retrieval to index while still preserving nested section paths and rendered
    paragraph/table content.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class InferenceAssessmentParser:
    """Convert an assessment tree dict into a section-level HTML report mapping.

    The inference pipeline does not need the full original assessment tree at
    prompt time. Instead it needs a flatter structure that:
    - preserves section paths
    - keeps paragraph and table content
    - is easy to index for draft-side retrieval

    This parser produces exactly that intermediate representation.
    """

    def parse(self, assessment: Dict[str, Any]) -> Dict[str, str]:
        """Return a mapping from section path to combined HTML content.

        The returned dictionary merges all supported block content found under
        each section into one HTML string. It is the main input to
        ``build_draft_store_from_report(...)`` during inference.
        """
        if not isinstance(assessment, dict):
            raise TypeError("InferenceAssessmentParser.parse() expects a Python dictionary.")

        full_report: Dict[str, str] = {}
        self.parse_document(assessment, full_report)
        return full_report

    def parse_document(
        self,
        node: Dict[str, Any],
        full_report: Dict[str, str],
        path: Optional[List[str]] = None,
    ) -> None:
        """Walk the assessment tree recursively and collect HTML by section path.

        This method:
        - extends the current section path with the node title
        - renders supported blocks in the current node
        - appends the rendered HTML into ``full_report``
        - then continues into child nodes
        """
        if path is None:
            path = []

        title = self.coerce_text(node.get("title", "")).strip()
        if title:
            path = path + [title]

        section_key = " > ".join(path) if path else "Assessment"
        html_fragments: List[str] = []

        for block in node.get("blocks", []):
            block_html = self.render_block_html(block)
            if block_html:
                html_fragments.append(block_html)

        if html_fragments:
            combined_html = "\n".join(html_fragments)
            existing_html = full_report.get(section_key, "")
            full_report[section_key] = (
                f"{existing_html}\n{combined_html}".strip() if existing_html else combined_html
            )

        for child in node.get("children", []):
            self.parse_document(child, full_report, path)

    def render_block_html(self, block: Dict[str, Any]) -> str:
        """Render one supported block into HTML using rich-text fields only.

        The inference parser currently supports:
        - paragraph blocks
        - table blocks

        Unsupported block types are ignored rather than raising an error.
        """
        block_type = str(block.get("type", "")).strip().lower()
        if block_type == "paragraph":
            return self.render_paragraph_html(block)
        if block_type == "table":
            return self.render_table_html(block)
        return ""

    def render_paragraph_html(self, block: Dict[str, Any]) -> str:
        """Render one paragraph block using ``text_rich``.

        If the paragraph already looks like one wrapped outer ``<p>`` element,
        the method leaves it unchanged. Otherwise it wraps the content in a
        simple paragraph tag.
        """
        text_rich = self.coerce_text(block.get("text_rich", "")).strip()
        if not text_rich:
            return ""
        if self.looks_like_wrapped_html(text_rich, "p"):
            return text_rich
        return f"<p>{text_rich}</p>"

    def render_table_html(self, block: Dict[str, Any]) -> str:
        """Render one table block using ``rows_rich``.

        The method builds a lightweight HTML table made up of:
        - ``<table>``
        - one ``<tr>`` per row
        - one ``<td>`` per rendered cell

        Empty or malformed row lists are skipped.
        """
        rows = block.get("rows_rich", [])
        if not isinstance(rows, list) or not rows:
            return ""

        rendered_rows: List[str] = []
        for row in rows:
            row_html = self.render_table_row_html(row)
            if row_html:
                rendered_rows.append(f"  <tr>{row_html}</tr>")

        if not rendered_rows:
            return ""

        return "<table>\n" + "\n".join(rendered_rows) + "\n</table>"

    def render_table_row_html(self, row: Any) -> str:
        """Render one rich table row into ``<td>`` cells.

        Rows may arrive either as:
        - a list of cells
        - a single cell-like value

        Empty rows produce an empty string so callers can ignore them.
        """
        if isinstance(row, list):
            cells = [self.stringify_rich_cell(cell) for cell in row]
            if not any(cells):
                return ""
            return "".join(f"<td>{cell}</td>" for cell in cells)

        cell = self.stringify_rich_cell(row)
        if not cell:
            return ""
        return f"<td>{cell}</td>"

    def stringify_rich_cell(self, cell: Any) -> str:
        """Convert one rich table cell into HTML text.

        Nested list cells are flattened with ``<br/>`` separators so multi-part
        rich cells still remain readable after conversion.
        """
        if cell is None:
            return ""
        if isinstance(cell, list):
            cell_parts = [self.stringify_rich_cell(item) for item in cell]
            cell_parts = [part for part in cell_parts if part]
            return "<br/>".join(cell_parts)
        return self.coerce_text(cell).strip()

    def coerce_text(self, value: Any) -> str:
        """Convert parser input values to text while preserving Unicode characters.

        This helper accepts bytes, strings, and other scalar values and returns
        a best-effort string representation suitable for later HTML assembly.
        """
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, str):
            return value
        return str(value)

    def looks_like_wrapped_html(self, text: str, tag_name: str) -> bool:
        """Return True when text already appears to have one outer wrapper tag.

        This prevents helpers such as ``render_paragraph_html(...)`` from
        double-wrapping content that already arrived as complete HTML.
        """
        normalized = text.strip().lower()
        return normalized.startswith(f"<{tag_name}") and normalized.endswith(f"</{tag_name}>")
