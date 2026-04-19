"""Reference retrieval asset builder.

Purpose:
    This module reads preprocessed retrieval blocks and builds the reference
    assets used by the RAG retriever: sparse JSONL corpus records, parent table
    contexts, build metadata, and the optional persistent Chroma dense index.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embedding_loader import build_huggingface_embeddings

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
PROCESSED_DIR = ROOT_DIR / "ii_preprocessed_documents"
PERSIST_DIR = SCRIPT_DIR / "chroma_db" / "reference_docs"
SUMMARY_PATH = SCRIPT_DIR / "build_summary.json"
CORPUS_PATH = SCRIPT_DIR / "reference_corpus.jsonl"
PARENT_CONTEXTS_PATH = SCRIPT_DIR / "parent_contexts.jsonl"

COLLECTION_NAME = "iucn_reference_docs"
EMBED_MODEL = "BAAI/bge-m3"
TABLE_CHUNK_SIZE = 2400
TABLE_CHUNK_OVERLAP = 200
CHUNK_SEPARATORS = ["\n\n", "\n", "Page ", ". ", " ", ""]


def extract_display_text(raw_text: str) -> str:
    """Return the human-facing content portion of a contextualized retrieval block.

    Many preprocessing records store provenance fields plus a ``Content:``
    section. This helper prefers the trailing content portion so chunking and
    parent-context exports operate on the main text rather than the surrounding
    wrapper fields.
    """
    text = str(raw_text or "").strip()
    if not text:
        return ""

    marker = "Content:"
    if marker in text:
        content = text.split(marker, 1)[1].strip()
        if content:
            return content
    return text


def build_search_text(piece_text: str, metadata: dict[str, Any]) -> str:
    """Build the enriched text string stored in the dense and sparse indexes.

    The search text intentionally includes retrieval metadata such as:
    - source file
    - section title
    - section path
    - table title
    - block type
    - page

    This gives both dense and sparse retrieval more structured cues than raw
    content alone.
    """
    parts = [
        f"Source file: {metadata.get('source_file')}",
        f"Section title: {metadata.get('section_title')}",
        f"Section path: {metadata.get('section_path')}",
        f"Table title: {metadata.get('table_title')}",
        f"Block type: {metadata.get('block_type')}",
        f"Page: {metadata.get('page')}",
        f"Content: {piece_text}",
    ]
    return "\n".join(
        str(part)
        for part in parts
        if part and not str(part).endswith(": None") and not str(part).endswith(": ")
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the reference-index build step.

    The build currently supports:
    - ``--reset`` to rebuild the Chroma directory from scratch
    - ``--skip-chroma`` to export only JSONL assets
    - configurable chunk size and chunk overlap
    """
    parser = argparse.ArgumentParser(description="Build the reference Chroma DB from retrieval_blocks.jsonl files.")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--skip-chroma", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    return parser.parse_args()


def load_retrieval_records(processed_dir: Path) -> list[dict[str, Any]]:
    """Load every retrieval record produced by preprocessing across all documents.

    This function walks the per-document folders in
    ``ii_preprocessed_documents/``, reads each ``retrieval_blocks.jsonl`` file,
    and returns one combined record list for later chunking and indexing.
    """
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    records: list[dict[str, Any]] = []
    for pdf_dir in sorted(processed_dir.iterdir()):
        if not pdf_dir.is_dir():
            continue

        blocks_path = pdf_dir / "retrieval_blocks.jsonl"
        if not blocks_path.exists():
            continue

        with blocks_path.open("r", encoding="utf-8") as file_handle:
            for line_number, line in enumerate(file_handle, start=1):
                record = _parse_record_line(line, pdf_dir.name, line_number)
                if record is not None:
                    records.append(record)

    if not records:
        raise RuntimeError("No retrieval records found.")
    return records


def _parse_record_line(line: str, pdf_stem: str, line_number: int) -> dict[str, Any] | None:
    """Parse one JSONL record and attach normalized build-time defaults.

    The helper skips blank lines and empty-text records, then adds:
    - a scalar ``section_path_str`` representation
    - a fallback ``doc_id`` when needed
    - the source PDF stem
    - a default retrieval priority
    """
    line = line.strip()
    if not line:
        return None

    record = json.loads(line)
    text = str(record.get("text", "")).strip()
    if not text:
        return None

    section_path = record.get("section_path")
    record["section_path_str"] = " > ".join(section_path) if isinstance(section_path, list) else section_path
    record.setdefault("doc_id", f"{pdf_stem}::line::{line_number:05d}")
    record.setdefault("pdf_stem", pdf_stem)
    record.setdefault("priority", 1.0)
    return record


def build_record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Extract the metadata fields that should travel with each built chunk.

    These values are stored on the LangChain document so they remain available
    later for:
    - retrieval reranking
    - debug output
    - parent-context lookup
    - prompt construction
    """
    metadata = record.get("metadata") or {}
    return {
        "doc_id": record.get("doc_id"),
        "source_file": record.get("source_file"),
        "pdf_stem": record.get("pdf_stem"),
        "page": record.get("page"),
        "block_type": str(record.get("block_type", "text")),
        "table_id": record.get("table_id"),
        "table_title": record.get("table_title"),
        "row_id": record.get("row_id"),
        "parent_id": record.get("parent_id"),
        "section_title": record.get("section_title"),
        "section_path": record.get("section_path_str"),
        "priority": float(record.get("priority", 1.0)),
        "fallback": bool(metadata.get("fallback", False)),
        "table_name": metadata.get("table_name"),
    }


def _splitter_for_block(block_type: str, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Return the appropriate text splitter for the given retrieval block type.

    The current policy is:
    - ``table_row`` records stay atomic elsewhere and do not use a splitter
    - ``table`` records get a larger chunk window
    - narrative text records use the caller-provided standard chunk settings
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=TABLE_CHUNK_SIZE if block_type == "table" else chunk_size,
        chunk_overlap=TABLE_CHUNK_OVERLAP if block_type == "table" else chunk_overlap,
        separators=CHUNK_SEPARATORS,
    )


def _build_document(piece_text: str, metadata: dict[str, Any], chunk_index: int) -> Document:
    """Wrap one chunk of retrieval text and metadata in a LangChain document.

    Each built document gets a stable ``chunk_id`` derived from the parent
    record plus the chunk index so dense and sparse outputs can reference the
    same chunk consistently.
    """
    chunk_metadata = metadata.copy()
    chunk_metadata["chunk_id"] = f"{metadata['doc_id']}::{chunk_index}"
    chunk_metadata["display_text"] = piece_text
    return Document(
        page_content=build_search_text(piece_text, chunk_metadata),
        metadata=chunk_metadata,
    )


def split_record(record: dict[str, Any], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split one retrieval record into one or more indexable documents.

    The current behavior is:
    - ``table_row`` stays as one atomic document
    - longer ``table`` and ``text`` records are split with the configured
      block-aware splitter
    """
    metadata = build_record_metadata(record)
    block_type = str(metadata["block_type"])
    raw_text = str(record["text"]).strip()
    text = extract_display_text(raw_text) or raw_text

    if block_type == "table_row":
        return [_build_document(text, metadata, chunk_index=0)]

    splitter = _splitter_for_block(block_type, chunk_size, chunk_overlap)
    return [
        _build_document(piece, metadata, chunk_index=index)
        for index, piece in enumerate(splitter.split_text(text))
    ]


def build_documents(records: list[dict[str, Any]], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Convert all retrieval records into the document list used for indexing.

    This helper simply applies ``split_record(...)`` across the full record
    list and concatenates the resulting document chunks.
    """
    documents: list[Document] = []
    for record in records:
        documents.extend(split_record(record, chunk_size, chunk_overlap))
    return documents


def build_embeddings():
    """Load the embedding model used for both build-time and query-time dense retrieval.

    The build step is strict here: if the embedding model cannot be loaded, the
    build fails rather than silently producing an incomplete dense index.
    """
    embeddings = build_huggingface_embeddings(EMBED_MODEL, device="cpu", normalize_embeddings=True, strict=True)
    if embeddings is None:
        raise RuntimeError(f"Could not load embeddings for model '{EMBED_MODEL}'.")
    return embeddings


def write_reference_corpus(documents: list[Document]) -> None:
    """Write the sparse-retrieval JSONL corpus from the built documents.

    The sparse corpus stores both:
    - the human-facing display text
    - the richer search text used for lexical retrieval
    """
    with CORPUS_PATH.open("w", encoding="utf-8") as file_handle:
        for document in documents:
            payload = {
                "text": document.metadata.get("display_text", document.page_content),
                "search_text": document.page_content,
                **document.metadata,
            }
            file_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_parent_contexts(records: list[dict[str, Any]]) -> None:
    """Write one unchunked parent-context record per source table.

    This avoids relying on chunked table parents later during inference and
    lets the retriever attach larger table context to row-level hits.
    """
    seen: set[tuple[str, str]] = set()
    with PARENT_CONTEXTS_PATH.open("w", encoding="utf-8") as file_handle:
        for record in records:
            if str(record.get("block_type", "")) != "table":
                continue

            source_file = str(record.get("source_file", ""))
            table_id = str(record.get("table_id", ""))
            if not source_file or not table_id:
                continue

            key = (source_file, table_id)
            if key in seen:
                continue

            seen.add(key)
            payload = {
                "source_file": source_file,
                "table_id": table_id,
                "table_title": record.get("table_title"),
                "page": record.get("page"),
                "text": extract_display_text(record.get("text")),
                "section_title": record.get("section_title"),
                "section_path": record.get("section_path_str"),
                "metadata": record.get("metadata"),
            }
            file_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_chroma(documents: list[Document], reset: bool) -> None:
    """Build or rebuild the persistent Chroma collection from the prepared documents.

    When ``reset`` is True, the existing Chroma directory is deleted first so
    the new dense index starts from a clean state.
    """
    if reset and PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        documents=documents,
        embedding=build_embeddings(),
        ids=[str(document.metadata["chunk_id"]) for document in documents],
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION_NAME,
    )


def write_summary(records: list[dict[str, Any]], documents: list[Document], chunk_size: int, chunk_overlap: int, chroma_built: bool) -> None:
    """Write a compact summary describing the built reference assets.

    The summary captures:
    - record and chunk counts
    - embedding settings
    - whether Chroma was built
    - counts for fallback table rows and synthetic parents
    """
    payload = {
        "processed_dir": str(PROCESSED_DIR),
        "persist_dir": str(PERSIST_DIR),
        "corpus_path": str(CORPUS_PATH),
        "parent_contexts_path": str(PARENT_CONTEXTS_PATH),
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBED_MODEL,
        "raw_record_count": len(records),
        "chunk_count": len(documents),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chroma_built": chroma_built,
        "table_row_count": sum(1 for record in records if record.get("block_type") == "table_row"),
        "fallback_table_row_count": sum(
            1
            for record in records
            if record.get("block_type") == "table_row" and (record.get("metadata") or {}).get("fallback")
        ),
        "synthetic_parent_count": sum(
            1
            for record in records
            if record.get("block_type") == "table" and (record.get("metadata") or {}).get("fallback")
        ),
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)


def main() -> None:
    """Run the full reference-index build pipeline.

    The build order is:
    - load processed retrieval records
    - split them into indexable documents
    - write sparse JSONL assets
    - optionally build Chroma
    - write the build summary
    """
    args = parse_args()
    records = load_retrieval_records(PROCESSED_DIR)
    documents = build_documents(records, args.chunk_size, args.chunk_overlap)

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_reference_corpus(documents)
    write_parent_contexts(records)

    if not args.skip_chroma:
        build_chroma(documents, args.reset)

    write_summary(
        records,
        documents,
        args.chunk_size,
        args.chunk_overlap,
        chroma_built=not args.skip_chroma,
    )

    if args.skip_chroma:
        print(f"Saved reference corpus to: {CORPUS_PATH}")
    else:
        print(f"Saved Chroma DB to: {PERSIST_DIR}")


if __name__ == "__main__":
    main()
