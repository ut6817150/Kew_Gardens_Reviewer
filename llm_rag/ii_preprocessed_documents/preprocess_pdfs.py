"""PDF preprocessing for the RAG reference corpus.

Purpose:
    This module converts raw IUCN reference PDFs into retrieval-ready JSONL
    records. It keeps page provenance, section context, extracted table
    parents, row-level table records, and supporting-information fallback rows
    so the vector-db build stage can preserve evidence structure.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import fitz
import pandas as pd
import pdfplumber

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
INPUT_DIR = ROOT_DIR / "i_raw_documents"
OUTPUT_DIR = ROOT_DIR / "ii_preprocessed_documents"

MIN_TEXT_BLOCK_CHARS = 220
MAX_TEXT_BLOCK_CHARS = 1200
HEADER_FOOTER_REPEAT_THRESHOLD = 0.6

NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")
ENUM_ENTRY_RE = re.compile(r"(?<!\d)(\d{1,2})\.\s+")

SUPPORTING_INFO_TABLE_TITLES = {
    "Table 1": "Table 1: Required supporting information for all assessments",
    "Table 2": "Table 2: Required supporting information under specific conditions",
    "Table 3": "Table 3: Recommended supporting information",
}
SUPPORTING_INFO_TABLE_SCHEMAS = {
    "Table 1": "Required Information | Purpose | Guidance | Notes",
    "Table 2": "Required Information | Specific Condition | Purpose | Guidance | Notes",
    "Table 3": "Recommended Supporting Information | Specific Condition | Purpose | Guidance | Notes",
}


@dataclass
class LayoutBlock:
    """One layout-aware text block extracted from a PDF page.

    Each block keeps:
    - the source page number
    - cleaned text
    - a bounding box
    - an average font size
    - a simple bold/non-bold signal

    Those fields are later used for heading detection and text merging.
    """

    page: int
    text: str
    bbox: list[float]
    font_size: float
    is_bold: bool


@dataclass
class RetrievalRecord:
    """One retrieval-ready record written to the preprocessing outputs.

    This is the main record type emitted by preprocessing. It stores:
    - provenance fields such as page and source file
    - block type such as ``text``, ``table``, or ``table_row``
    - contextualized retrieval text
    - optional table and section metadata
    """

    doc_id: str
    source_file: str
    page: int
    block_type: str
    text: str
    raw_text: str
    section_title: Optional[str] = None
    section_path: Optional[list[str]] = None
    table_id: Optional[str] = None
    table_title: Optional[str] = None
    row_id: Optional[int] = None
    parent_id: Optional[str] = None
    priority: float = 1.0
    metadata: Optional[dict[str, Any]] = None


def make_retrieval_record(
    pdf_path: Path,
    doc_id: str,
    page: int,
    block_type: str,
    raw_text: str,
    section_path: Optional[list[str]],
    *,
    priority: float = 1.0,
    table_id: Optional[str] = None,
    table_title: Optional[str] = None,
    row_id: Optional[int] = None,
    parent_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> RetrievalRecord:
    """Create one contextualized retrieval record from raw extracted content.

    The helper wraps raw content with provenance and section metadata, then
    returns a structured ``RetrievalRecord`` ready to be serialized into the
    preprocessing outputs.
    """
    return RetrievalRecord(
        doc_id=doc_id,
        source_file=pdf_path.name,
        page=page,
        block_type=block_type,
        text=contextualize_text(raw_text, pdf_path.name, page, section_path, block_type, table_title),
        raw_text=raw_text,
        section_title=section_path[-1] if section_path else None,
        section_path=section_path,
        table_id=table_id,
        table_title=table_title,
        row_id=row_id,
        parent_id=parent_id,
        priority=priority,
        metadata=metadata,
    )


def clean_text(text: str) -> str:
    """Normalize whitespace and common PDF punctuation artifacts.

    This helper removes null characters, soft hyphens, and a few recurring
    punctuation artifacts seen in PDF extraction output, then collapses
    repeated whitespace into one normalized string.
    """
    text = text.replace("\x00", " ").replace("\u00ad", "")
    text = text.replace("\u2010", "-").replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("•", " • ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_repeat_text(text: str) -> str:
    """Normalize block text for repeated-header and repeated-footer detection.

    The function lowercases the text and neutralizes page numbers so repeated
    running headers or footers can be matched across pages even when the page
    number changes.
    """
    text = clean_text(text).lower()
    text = re.sub(r"\bpage\s+\d+\b", "page", text)
    text = re.sub(r"\d+", "#", text)
    return text


def looks_like_heading(text: str, font_size: float, is_bold: bool, body_font: float) -> bool:
    """Return True when a block looks like a heading rather than body text.

    The current heuristic favors:
    - numbered headings
    - unusually large text
    - short bold headings
    - short all-caps headings

    It intentionally rejects long prose-like blocks.
    """
    text = clean_text(text)
    if not text or len(text) > 160:
        return False
    if NUMBERED_HEADING_RE.match(text):
        return True
    if font_size >= body_font * 1.22 and len(text.split()) <= 16:
        return True
    if is_bold and len(text.split()) <= 12 and not text.endswith((".", ";", ":")):
        return True
    if text.isupper() and 2 <= len(text.split()) <= 10:
        return True
    return False


def heading_level(text: str) -> int:
    """Infer a nesting level from a numbered heading.

    For example:
    - ``1 Title`` -> level 1
    - ``2.1 Title`` -> level 2
    - unnumbered headings -> level 1 by default
    """
    m = NUMBERED_HEADING_RE.match(clean_text(text))
    return m.group(1).count(".") + 1 if m else 1


def update_section_path(current: list[str], heading: str) -> list[str]:
    """Update the active section path after encountering a new heading.

    The function trims the current path to the inferred heading depth, appends
    the new cleaned heading text, and returns a copy of the updated path.
    """
    level = heading_level(heading)
    while len(current) >= level:
        current.pop()
    current.append(clean_text(heading))
    return current.copy()


def contextualize_text(raw_text: str, source_file: str, page: int, section_path: Optional[list[str]], block_type: str, table_title: Optional[str] = None) -> str:
    """Wrap raw content with provenance fields used later during retrieval.

    The resulting text includes source file, page, block type, optional section
    path, optional table title, and a trailing ``Content:`` block that holds
    the original extracted text.
    """
    lines = [f"Source: {source_file}", f"Page: {page}", f"Block type: {block_type}"]
    if section_path:
        lines.append("Section path: " + " > ".join(section_path))
    if table_title:
        lines.append(f"Table title: {table_title}")
    lines.append("Content:")
    lines.append(raw_text.strip())
    return "\n".join(lines)


def extract_layout_blocks(pdf_path: Path) -> tuple[list[LayoutBlock], float]:
    """Extract layout-aware text blocks and estimate the body font size.

    This pass reads the PDF with PyMuPDF, collects block-level text spans, and
    preserves simple layout cues that help the later heading-detection step.
    It also estimates the median body font size for the document.
    """
    doc = fitz.open(pdf_path)
    blocks: list[LayoutBlock] = []
    font_sizes: list[float] = []
    try:
        for page_num, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict", sort=True)
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                line_texts = []
                block_sizes = []
                is_bold = False
                for line in block.get("lines", []):
                    parts = []
                    for span in line.get("spans", []):
                        text = clean_text(span.get("text", ""))
                        if not text:
                            continue
                        parts.append(text)
                        size = float(span.get("size", 0.0) or 0.0)
                        if size:
                            block_sizes.append(size)
                            font_sizes.append(size)
                        font_name = str(span.get("font", "")).lower()
                        if "bold" in font_name or int(span.get("flags", 0)) & 16:
                            is_bold = True
                    if parts:
                        line_texts.append(" ".join(parts))
                text = clean_text(" ".join(line_texts))
                if not text:
                    continue
                bbox = list(map(float, block.get("bbox", [0, 0, 0, 0])))
                blocks.append(
                    LayoutBlock(
                        page=page_num,
                        text=text,
                        bbox=bbox,
                        font_size=sum(block_sizes) / len(block_sizes) if block_sizes else 0.0,
                        is_bold=is_bold,
                    )
                )
    finally:
        doc.close()

    body_font = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 10.0
    return blocks, body_font


def detect_repeated_header_footer_text(raw_blocks: list[LayoutBlock]) -> set[str]:
    """Detect repeated edge text that should be treated as headers or footers.

    The function looks at the first and last few blocks on each page, counts
    normalized repeats across the document, and returns the texts whose repeat
    rate crosses the configured threshold.
    """
    by_page: dict[int, list[LayoutBlock]] = defaultdict(list)
    for block in raw_blocks:
        by_page[block.page].append(block)

    num_pages = len(by_page)
    if num_pages <= 1:
        return set()

    candidates: Counter[str] = Counter()
    for blocks in by_page.values():
        ordered = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
        for block in ordered[:2] + ordered[-2:]:
            norm = normalize_repeat_text(block.text)
            if len(norm) >= 6:
                candidates[norm] += 1

    return {text for text, count in candidates.items() if count / num_pages >= HEADER_FOOTER_REPEAT_THRESHOLD}


def merge_text_blocks(pdf_path: Path, raw_blocks: list[LayoutBlock], repeated_edge_text: set[str], body_font: float) -> tuple[list[RetrievalRecord], dict[int, list[str]]]:
    """Merge narrative layout blocks into contextualized text retrieval records.

    This stage:
    - removes repeated edge text
    - updates section paths when headings are encountered
    - accumulates narrative blocks into longer text passages
    - emits ``text`` retrieval records
    - tracks the best-known section path for each page
    """
    by_page: dict[int, list[LayoutBlock]] = defaultdict(list)
    for block in raw_blocks:
        if normalize_repeat_text(block.text) in repeated_edge_text:
            continue
        by_page[block.page].append(block)

    page_section_map: dict[int, list[str]] = {}
    records: list[RetrievalRecord] = []
    section_path: list[str] = []
    record_index = 0

    for page, blocks in sorted(by_page.items()):
        blocks = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
        buffer: list[str] = []
        page_heading_path = section_path.copy()

        def flush_buffer() -> None:
            nonlocal buffer, record_index
            if not buffer:
                return
            raw_text = clean_text(" ".join(buffer))
            if len(raw_text) < 40:
                buffer = []
                return

            record_index += 1
            records.append(
                make_retrieval_record(
                    pdf_path,
                    doc_id=f"{pdf_path.stem}::text::{record_index:05d}",
                    page=page,
                    block_type="text",
                    raw_text=raw_text,
                    section_path=section_path.copy() if section_path else None,
                )
            )
            buffer = []

        for block in blocks:
            text = block.text
            if looks_like_heading(text, block.font_size, block.is_bold, body_font):
                flush_buffer()
                section_path = update_section_path(section_path, text)
                page_heading_path = section_path.copy()
                continue

            prospective = clean_text(" ".join(buffer + [text])) if buffer else text
            if buffer and len(prospective) > MAX_TEXT_BLOCK_CHARS and len(clean_text(" ".join(buffer))) >= MIN_TEXT_BLOCK_CHARS:
                flush_buffer()
            buffer.append(text)

        flush_buffer()
        page_section_map[page] = page_heading_path or section_path.copy()

    return records, page_section_map


def normalize_table_matrix(table: list[list[Any]]) -> list[list[str]]:
    """Normalize one extracted table matrix into cleaned string rows.

    Empty rows are dropped and every non-empty cell is converted to a cleaned
    string so later table serialization works with one consistent shape.
    """
    rows = []
    for row in table:
        norm = [clean_text(str(cell)) if cell is not None else "" for cell in row]
        if any(norm):
            rows.append(norm)
    return rows


def make_headers(rows: list[list[str]]) -> list[str]:
    """Build header labels for an extracted table.

    The first row is treated as headers when available. Empty header cells fall
    back to generic names such as ``column_2`` so every column remains
    addressable during row serialization.
    """
    return [cell or f"column_{idx + 1}" for idx, cell in enumerate(rows[0])] if rows else []


def row_to_pairs(headers: list[str], row: list[str]) -> str:
    """Serialize one table row as ``header: value`` pairs for retrieval.

    This turns a structured row into a flat text representation that still
    preserves the semantic meaning of each cell.
    """
    out = []
    for idx, value in enumerate(row):
        header = headers[idx] if idx < len(headers) else f"column_{idx + 1}"
        out.append(f"{header}: {value}")
    return " | ".join(out)


def nearest_section_path(page: int, page_section_map: dict[int, list[str]]) -> Optional[list[str]]:
    """Return the closest known section path for a page.

    If the current page has no direct section-path entry, the helper walks
    backwards to the nearest earlier page that does. This is mainly used for
    assigning tables to nearby narrative section context.
    """
    if page in page_section_map and page_section_map[page]:
        return page_section_map[page]
    for lookback in range(page - 1, 0, -1):
        if page_section_map.get(lookback):
            return page_section_map[lookback]
    return None


def infer_table_title(section_path: Optional[list[str]], headers: list[str]) -> str:
    """Infer a readable title for an extracted table.

    The helper prefers:
    - the current section title when available
    - otherwise a short title built from the header row
    - otherwise a generic fallback title
    """
    if section_path:
        return section_path[-1]
    if headers:
        return "Table with headers: " + " | ".join(headers[:4])
    return "Untitled table"


def get_pdfplumber_page_text(pdf_path: Path) -> dict[int, str]:
    """Extract cleaned page text with pdfplumber for fallback table recovery.

    This page-level text is used by the supporting-information fallback parser
    when visually tabular content is not recovered as a true PDF table.
    """
    page_text = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            page_text[page_num] = clean_text(text)
    return page_text


def detect_supporting_info_table(page_text: str) -> Optional[str]:
    """Detect whether a page belongs to Table 1, Table 2, or Table 3.

    The helper looks for the table label plus the matching supporting-
    information phrasing so fallback-table reconstruction can keep rows grouped
    under the correct synthetic parent.
    """
    t = clean_text(page_text).lower()
    if "table 1" in t and "required supporting information" in t:
        return "Table 1"
    if "table 2" in t and "required supporting information" in t:
        return "Table 2"
    if "table 3" in t and "recommended supporting information" in t:
        return "Table 3"
    return None


def detect_flattened_table_schema(page_text: str) -> bool:
    """Return True when flattened page text looks like a table schema.

    This is used as a fallback signal for pages that appear to continue a
    required-supporting-information table even if no explicit table label is
    repeated on that page.
    """
    t = clean_text(page_text).lower()
    schemas = [
        "required information purpose guidance notes",
        "required information specific condition purpose guidance notes",
        "specific condition purpose guidance notes",
        "recommended supporting information specific condition purpose guidance notes",
    ]
    return any(s in t for s in schemas)


def split_enumerated_entries(page_text: str) -> list[tuple[int, str]]:
    """Split flattened supporting-information text into numbered entry segments.

    The helper looks for numbered entries such as ``1.`` or ``12.`` and then
    extracts the text span belonging to each entry. Very short segments are
    ignored as likely noise.
    """
    text = clean_text(page_text)
    matches = list(ENUM_ENTRY_RE.finditer(text))
    if len(matches) < 2:
        return []
    entries = []
    for idx, match in enumerate(matches):
        num = int(match.group(1))
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        segment = clean_text(text[start:end])
        if len(segment) >= 25:
            entries.append((num, segment))
    return entries


def add_required_info_fallback_rows(pdf_path: Path, page_section_map: dict[int, list[str]], out_records: list[RetrievalRecord], start_index: int) -> int:
    """Recover synthetic Table 1 to Table 3 rows when normal table extraction fails.

    This fallback is specific to the supporting-information PDF. It:
    - gathers page-level text with pdfplumber
    - groups numbered entries under Table 1, Table 2, or Table 3
    - creates one synthetic parent ``table`` record per recovered table
    - creates one synthetic ``table_row`` record per recovered entry
    """
    doc_index = start_index
    if "Required_and_Recommended_Supporting_Information_for_IUCN_Red_List_Assessments" not in pdf_path.name:
        return doc_index

    texts = get_pdfplumber_page_text(pdf_path)
    current_table: Optional[str] = None
    grouped: dict[str, list[tuple[int, int, str]]] = defaultdict(list)

    for page, page_text in sorted(texts.items()):
        explicit = detect_supporting_info_table(page_text)
        if explicit:
            current_table = explicit
        entries = split_enumerated_entries(page_text)
        if current_table and (entries or detect_flattened_table_schema(page_text)):
            for entry_num, segment in entries:
                grouped[current_table].append((page, entry_num, segment))

    for table_name in ["Table 1", "Table 2", "Table 3"]:
        entries = grouped.get(table_name, [])
        if not entries:
            continue
        start_page = min(page for page, _, _ in entries)
        section_path = nearest_section_path(start_page, page_section_map)
        table_id = table_name.lower().replace(" ", "_")
        table_title = SUPPORTING_INFO_TABLE_TITLES[table_name]
        schema = SUPPORTING_INFO_TABLE_SCHEMAS[table_name]
        raw_table = "\n".join(
            [table_title, f"Schema: {schema}"]
            + [f"Page {page} | Entry {entry}: {segment}" for page, entry, segment in entries]
        )

        doc_index += 1
        out_records.append(
            make_retrieval_record(
                pdf_path,
                doc_id=f"{pdf_path.stem}::synthetic_parent::{doc_index:05d}",
                page=start_page,
                block_type="table",
                raw_text=raw_table,
                section_path=section_path,
                priority=1.1,
                table_id=table_id,
                table_title=table_title,
                parent_id=table_id,
                metadata={"fallback": True, "table_name": table_name, "schema": schema},
            )
        )

        row_counter = 0
        for page, entry_num, segment in entries:
            row_counter += 1
            doc_index += 1
            section_path = nearest_section_path(page, page_section_map)
            out_records.append(
                make_retrieval_record(
                    pdf_path,
                    doc_id=f"{pdf_path.stem}::synthetic_row::{doc_index:05d}",
                    page=page,
                    block_type="table_row",
                    raw_text=f"Entry {entry_num}: {segment}",
                    section_path=section_path,
                    priority=1.55,
                    table_id=table_id,
                    table_title=table_title,
                    row_id=row_counter,
                    parent_id=table_id,
                    metadata={
                        "fallback": True,
                        "entry_number": entry_num,
                        "table_name": table_name,
                        "schema": schema,
                    },
                )
            )
    return doc_index


def extract_tables(pdf_path: Path, out_dir: Path, page_section_map: dict[int, list[str]], doc_id_start: int) -> tuple[list[RetrievalRecord], int]:
    """Extract standard tables and append any synthetic fallback tables.

    This function:
    - runs normal pdfplumber table extraction
    - writes CSV exports for recovered tables
    - creates parent ``table`` and child ``table_row`` records
    - then appends any synthetic supporting-information fallback rows
    """
    records: list[RetrievalRecord] = []
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    doc_index = doc_id_start
    table_counter = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            try:
                extracted_tables = page.extract_tables()
            except Exception:
                extracted_tables = []
            section_path = nearest_section_path(page_index, page_section_map)
            for table in extracted_tables or []:
                rows = normalize_table_matrix(table)
                if len(rows) < 2:
                    continue
                headers = make_headers(rows)
                body_rows = rows[1:]
                if not body_rows:
                    continue
                table_counter += 1
                table_id = f"table_{table_counter:03d}"
                table_title = infer_table_title(section_path, headers)
                width = max(len(headers), max(len(r) for r in body_rows))
                headers = headers + [f"column_{i + 1}" for i in range(len(headers), width)]
                body = [r + [""] * (width - len(r)) for r in body_rows]
                csv_path = tables_dir / f"{table_id}.csv"
                pd.DataFrame(body, columns=headers).to_csv(csv_path, index=False)

                raw_table = "\n".join([f"Table title: {table_title}", "Headers: " + " | ".join(headers)] + [row_to_pairs(headers, r) for r in body])
                doc_index += 1
                records.append(
                    make_retrieval_record(
                        pdf_path,
                        doc_id=f"{pdf_path.stem}::table::{doc_index:05d}",
                        page=page_index,
                        block_type="table",
                        raw_text=raw_table,
                        section_path=section_path,
                        table_id=table_id,
                        table_title=table_title,
                        parent_id=table_id,
                        metadata={"csv_path": str(csv_path)},
                    )
                )
                for row_id, row in enumerate(body, start=1):
                    doc_index += 1
                    raw_row = row_to_pairs(headers, row)
                    records.append(
                        make_retrieval_record(
                            pdf_path,
                            doc_id=f"{pdf_path.stem}::table_row::{doc_index:05d}",
                            page=page_index,
                            block_type="table_row",
                            raw_text=raw_row,
                            section_path=section_path,
                            priority=1.3,
                            table_id=table_id,
                            table_title=table_title,
                            row_id=row_id,
                            parent_id=table_id,
                            metadata={"csv_path": str(csv_path)},
                        )
                    )

    doc_index = add_required_info_fallback_rows(pdf_path, page_section_map, records, doc_index)
    return records, doc_index


def drop_overlapping_text_records(records: list[RetrievalRecord]) -> list[RetrievalRecord]:
    """Drop obvious text duplicates when table-derived records already cover the same content.

    The current logic prefers:
    - ``table_row`` over ``table``
    - ``table`` over ``text``

    This helps avoid duplicate evidence later in the reference corpus.
    """
    table_pages = defaultdict(set)
    for r in records:
        if r.block_type in {"table", "table_row"}:
            table_pages[r.source_file].add(r.page)

    filtered = []
    for r in records:
        if r.block_type == "text" and r.page in table_pages[r.source_file]:
            raw = r.raw_text.lower()
            sec = (r.section_title or "").lower()
            if "table 1" in raw or "table 2" in raw or "table 3" in raw:
                continue
            if "required information purpose guidance notes" in sec or "specific condition purpose guidance notes" in sec:
                continue
        filtered.append(r)

    best = {}
    rank = {"table_row": 3, "table": 2, "text": 1}
    for r in filtered:
        key = (r.source_file, r.page, normalize_repeat_text(r.raw_text))
        cur = best.get(key)
        if cur is None or rank.get(r.block_type, 0) > rank.get(cur.block_type, 0):
            best[key] = r
    return sorted(best.values(), key=lambda r: r.doc_id)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write an iterable of dictionaries to JSONL using UTF-8 encoding.

    Each record is serialized on its own line so later build steps can stream
    the file line by line.
    """
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_pdf(pdf_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run the full preprocessing pipeline for one raw reference PDF.

    The output for each document includes:
    - raw layout blocks
    - retrieval blocks
    - any extracted table CSVs
    - a per-document manifest summarizing counts and fallback activity
    """
    out_dir = output_dir / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_blocks, body_font = extract_layout_blocks(pdf_path)
    repeated = detect_repeated_header_footer_text(raw_blocks)
    text_records, page_section_map = merge_text_blocks(pdf_path, raw_blocks, repeated, body_font)
    table_records, _ = extract_tables(pdf_path, out_dir, page_section_map, len(text_records))
    all_records = drop_overlapping_text_records(text_records + table_records)

    write_jsonl(out_dir / "raw_page_blocks.jsonl", [{"page": b.page, "text": b.text, "bbox": b.bbox, "font_size": b.font_size, "is_bold": b.is_bold} for b in raw_blocks])
    write_jsonl(out_dir / "retrieval_blocks.jsonl", [asdict(r) for r in all_records])

    manifest = {
        "source_file": pdf_path.name,
        "raw_text_block_count": len(raw_blocks),
        "retrieval_block_count": len(all_records),
        "text_block_count": sum(1 for r in all_records if r.block_type == "text"),
        "table_block_count": sum(1 for r in all_records if r.block_type == "table"),
        "table_row_block_count": sum(1 for r in all_records if r.block_type == "table_row"),
        "fallback_table_row_count": sum(1 for r in all_records if r.block_type == "table_row" and (r.metadata or {}).get("fallback")),
        "synthetic_parent_count": sum(1 for r in all_records if r.block_type == "table" and (r.metadata or {}).get("fallback")),
        "repeated_edge_text_removed": sorted(repeated),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def main() -> None:
    """Preprocess every raw reference PDF and write the corpus-level summary.

    This is the entry point for the preprocessing stage that feeds the
    later reference-index build.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")
    pdf_paths = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {INPUT_DIR}")
    summary = []
    for pdf_path in pdf_paths:
        print(f"Processing {pdf_path.name} ...")
        summary.append(process_pdf(pdf_path, OUTPUT_DIR))
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Done. Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
