from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from langchain_chroma import Chroma

from embedding_loader import build_huggingface_embeddings
from threshold_lookup import answer_threshold_query, is_threshold_query

SCRIPT_DIR = Path(__file__).resolve().parent
PERSIST_DIR = SCRIPT_DIR / "chroma_db" / "reference_docs"
CORPUS_PATH = SCRIPT_DIR / "reference_corpus.jsonl"
PARENT_CONTEXTS_PATH = SCRIPT_DIR / "parent_contexts.jsonl"
SUMMARY_PATH = SCRIPT_DIR / "build_summary.json"
COLLECTION_NAME = "iucn_reference_docs"
EMBED_MODEL = "BAAI/bge-m3"
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9()\[\].,/%:+\-_–—]*")


@dataclass
class Candidate:
    """One retrieved reference candidate plus its ranking signals.

    A candidate carries:
    - the retrieved chunk text
    - search text used for overlap scoring
    - metadata from the build step
    - dense, sparse, fused, and rerank scores
    - optional attached parent table context
    """

    chunk_id: str
    text: str
    search_text: str
    metadata: dict[str, Any]
    dense_score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None
    bm25_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    parent_text: Optional[str] = None
    parent_metadata: Optional[dict[str, Any]] = None
    forced_for_coverage: bool = False


class BM25Index:
    """Lightweight BM25-style sparse index over the JSONL reference corpus.

    This sparse lane complements dense retrieval by rewarding:
    - exact symbolic terms
    - exact criterion labels
    - short table phrases
    - other lexical matches that embeddings may smooth over
    """

    def __init__(self, docs, k1=1.5, b=0.75):
        """Build sparse retrieval statistics for the provided document list.

        The constructor tokenizes the corpus, computes document lengths, and
        precomputes per-term inverse-document-frequency values used during
        BM25-style scoring.
        """
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.tokenized_docs = [tokenize(doc.get("search_text") or doc["text"]) for doc in docs]
        self.doc_lens = [len(tokens) for tokens in self.tokenized_docs]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.term_freqs = [Counter(tokens) for tokens in self.tokenized_docs]
        df = Counter()
        for tf in self.term_freqs:
            for term in tf:
                df[term] += 1
        total_docs = len(self.docs)
        self.idf = {term: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def search(self, query, k=20):
        """Return the top sparse hits for a query using BM25-style scoring.

        The method scores every tokenized document against the tokenized query
        and returns the top ``k`` results as ``(doc_index, score)`` pairs.
        """
        query_terms = tokenize(query)
        scores = []
        for idx, tf in enumerate(self.term_freqs):
            score = 0.0
            dl = self.doc_lens[idx] or 1
            for term in query_terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                score += idf * (freq * (self.k1 + 1) / max(denom, 1e-9))
            if score > 0:
                scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


def tokenize(text: str):
    """Tokenize text into lowercase terms for sparse retrieval and overlap scoring.

    The regex keeps compact IUCN-style tokens such as criterion codes,
    percentages, and punctuation-bearing table fragments better than a simpler
    whitespace split would.
    """
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def has_phrase(text: str, phrase: str) -> bool:
    """Return True when a phrase appears with simple word-boundary handling.

    For plain alphanumeric phrases, the helper uses regex word boundaries so
    matches behave more like whole-term phrase checks. Punctuation-heavy
    phrases fall back to substring matching.
    """
    phrase = str(phrase or "").strip().lower()
    if not phrase:
        return False
    if all(char.isalnum() or char.isspace() for char in phrase):
        pattern = r"\b" + r"\s+".join(re.escape(part) for part in phrase.split()) + r"\b"
        return re.search(pattern, text) is not None
    return phrase in text


def reciprocal_rank_fusion(rank: int, k: int = 60) -> float:
    """Convert a rank position into a reciprocal-rank-fusion contribution.

    Higher-ranked dense or sparse hits therefore contribute more to the fused
    score used before final reranking.
    """
    return 1.0 / (k + rank)


def detect_query_mode(query: str):
    """Classify the query into high-level retrieval modes used by reranking.

    The current mode flags capture whether a question is mainly about:
    - supporting information
    - threshold-like criterion codes
    - threatened-category language
    """
    q = query.lower()
    return {
        "supporting_info": looks_like_supporting_info_query(q),
        "criterion_codes": any(token in q for token in ["aoo", "eoo", "b1", "b2", "d1", "d2", "threshold", "cutoff", "cut-off"]),
        "threatened": any(token in q for token in ["threatened", "critically endangered", "endangered", "vulnerable", "near threatened", "data deficient"]),
    }


def looks_like_supporting_info_query(q: str) -> bool:
    """Return True when the prompt appears to ask about supporting-information requirements.

    This is the switch that enables the more specialized retrieval behavior
    for Table 1 and Table 2 style questions.
    """
    direct_phrases = [
        "supporting information",
        "supporting info",
        "required information",
        "required data",
        "additional information",
        "additional data",
        "documentation",
        "what data do i need",
        "what must i include",
        "what do i need to include",
        "what should i include",
        "what needs to be included",
        "include in my assessment",
        "required for all assessments",
        "required under specific conditions",
        "beyond the basics",
        "beyond the basic requirements",
        "extra supporting info",
        "extra supporting information",
    ]
    if any(has_phrase(q, phrase) for phrase in direct_phrases):
        return True

    support_nouns = ["information", "info", "data", "documentation"]
    support_verbs = ["include", "included", "required", "need", "needed", "submit", "missing", "extra", "additional"]
    return any(has_phrase(q, noun) for noun in support_nouns) and any(has_phrase(q, verb) for verb in support_verbs)


def is_all_conditions_subquery(query: str) -> bool:
    """Return True when an internal subquery targets baseline requirements.

    These subqueries are intended to retrieve evidence corresponding to the
    "required for all assessments" side of the supporting-information logic.
    """
    q = query.lower()
    return has_phrase(q, "all conditions") or has_phrase(q, "all assessments") or has_phrase(q, "basic requirements")


def is_specific_conditions_subquery(query: str) -> bool:
    """Return True when an internal subquery targets condition-specific requirements.

    These subqueries are intended to favor Table 2 style evidence covering
    additional requirements under specific conditions.
    """
    q = query.lower()
    return (
        has_phrase(q, "specific conditions")
        or has_phrase(q, "threatened taxa")
        or has_phrase(q, "threatened species")
        or has_phrase(q, "beyond the basics")
        or has_phrase(q, "additional required supporting information")
        or has_phrase(q, "extra supporting info")
    )


@lru_cache(maxsize=4)
def _load_jsonl_cached(path_str: str, mtime: float):
    """Load a JSONL file once per path and modification time.

    The modification time is part of the cache key so edits on disk invalidate
    the cached representation automatically.
    """
    path = Path(path_str)
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return tuple(out)


def load_corpus(path: Path = CORPUS_PATH):
    """Load the sparse reference corpus from disk.

    The returned objects are plain dictionaries produced during the build step
    and used by the sparse retrieval lane and corpus backfilling.
    """
    return list(_load_jsonl_cached(str(path), path.stat().st_mtime))


def load_parent_contexts(path: Path = PARENT_CONTEXTS_PATH):
    """Load parent table contexts keyed by ``(source_file, table_id)``.

    This lookup is what allows row-level hits to recover larger table context
    during inference.
    """
    rows = list(_load_jsonl_cached(str(path), path.stat().st_mtime))
    return {(str(row["source_file"]), str(row["table_id"])): dict(row) for row in rows}


@lru_cache(maxsize=4)
def _build_bm25_cached(path_str: str, mtime: float):
    """Build and cache the BM25 index for the reference corpus.

    The cache avoids rebuilding sparse statistics on every question during one
    app session.
    """
    docs = load_corpus(Path(path_str))
    return BM25Index(docs)


def get_bm25_index(path: Path = CORPUS_PATH):
    """Return the cached BM25 index for the current reference corpus.

    Callers should use this helper rather than building a fresh BM25 index per
    request.
    """
    return _build_bm25_cached(str(path), path.stat().st_mtime)


@lru_cache(maxsize=1)
def get_embeddings():
    """Load the dense embedding model, allowing sparse-only fallback when needed.

    Unlike the build step, query-time inference may continue with sparse-only
    retrieval if the embedding model cannot be loaded.
    """
    return build_huggingface_embeddings(
        EMBED_MODEL,
        device="cpu",
        normalize_embeddings=True,
        strict=False,
    )


@lru_cache(maxsize=1)
def get_db():
    """Return the persistent Chroma handle used for dense reference retrieval.

    If embeddings are unavailable, this helper returns ``None`` so the dense
    lane can be skipped cleanly.
    """
    embeddings = get_embeddings()
    if embeddings is None:
        return None
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
    )


@lru_cache(maxsize=1)
def dense_index_available() -> bool:
    """Return True when the on-disk dense index appears to be present and non-empty.

    The helper checks both the build summary and the Chroma directory contents
    before the runtime attempts dense retrieval.
    """
    if SUMMARY_PATH.exists():
        try:
            summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
            if summary.get("chroma_built") is False:
                return False
        except Exception:
            pass
    if not PERSIST_DIR.exists():
        return False
    return any(path.is_file() and path.stat().st_size > 0 for path in PERSIST_DIR.rglob("*"))


def safe_dense_search(query: str, dense_k: int):
    """Run dense retrieval defensively.

    If the dense index is unavailable, empty, or raises at runtime, the helper
    returns an empty list so the rest of the retrieval flow can continue with
    sparse evidence only.
    """
    if not dense_index_available():
        return []
    try:
        db = get_db()
        if db is None:
            return []
        collection = getattr(db, "_collection", None)
        if collection is not None:
            try:
                if collection.count() == 0:
                    return []
            except Exception:
                pass
        return db.similarity_search_with_score(query, k=dense_k)
    except Exception:
        return []


def table_label(candidate: Candidate):
    """Map a candidate to ``table1``, ``table2``, ``table3``, or ``other``.

    The mapping is driven by the stored table title and is mainly used by the
    supporting-information reranking and coverage logic.
    """
    title = str(candidate.metadata.get("table_title", "") or "")
    if "Table 1" in title:
        return "table1"
    if "Table 2" in title:
        return "table2"
    if "Table 3" in title:
        return "table3"
    return "other"


def priority_boost(candidate: Candidate, query_mode):
    """Apply metadata-based boosts that favor high-value table and source records.

    This helper rewards:
    - table rows and parent tables
    - records from supporting-information sources
    - records already marked as high priority during preprocessing
    """
    meta = candidate.metadata
    block_type = str(meta.get("block_type", "text"))
    source_file = str(meta.get("source_file", ""))
    table_title = str(meta.get("table_title", "") or "")
    section_title = str(meta.get("section_title", "") or "")
    priority = float(meta.get("priority", 1.0))
    boost = 0.06 * max(priority - 1.0, 0.0)

    if block_type == "table_row":
        boost += 0.18
    elif block_type == "table":
        boost += 0.08

    if query_mode["supporting_info"]:
        if "Required_and_Recommended_Supporting_Information" in source_file:
            boost += 0.18
        if "RL_Standards_Consistency" in source_file:
            boost += 0.12
        if "Table 1" in table_title or "Table 2" in table_title:
            boost += 0.14
        if "Required Supporting Information" in section_title or "Specific Condition" in section_title:
            boost += 0.08
        if block_type == "text":
            boost -= 0.03

    return boost


def content_intent_boost(candidate: Candidate, query: str, query_mode) -> float:
    """Apply query-intent boosts that favor likely reference sources and sections.

    The current rules add targeted boosts for questions about:
    - categories and status
    - criterion thresholds
    - countries and distribution
    - threats, conservation, habitats, and supporting information
    """
    q = query.lower()
    source_file = str(candidate.metadata.get("source_file", "") or "")
    section_title = str(candidate.metadata.get("section_title", "") or "")
    table_title = str(candidate.metadata.get("table_title", "") or "")
    combined = f"{section_title} {table_title}".lower()
    boost = 0.0

    def in_source(*names: str) -> bool:
        return any(name in source_file for name in names)

    def in_combined(*terms: str) -> bool:
        return any(term in combined for term in terms)

    if any(term in q for term in ["red list status", "red list category", "category is", "currently in"]):
        if in_source("RL_categories_and_criteria", "RedListGuidelines"):
            boost += 0.16
        if in_combined("categories", "category", "figure 2.1"):
            boost += 0.12

    if any(term in q for term in ["d2", "criterion d", "criterion b", "eoo", "aoo", "cutoff", "cut-off", "threshold"]):
        if in_source("RedListGuidelines", "RL_categories_and_criteria", "RL_criteria_summary_sheet"):
            boost += 0.18
        if in_combined("criterion", "guidelines for applying criterion", "structure of the iucn red list categories"):
            boost += 0.10

    if any(term in q for term in ["country", "countries", "occur", "recorded from", "elevation", "location", "locations"]):
        if in_source("RL_Standards_Consistency", "Mapping_Standards"):
            boost += 0.12
        if in_combined("distribution", "map", "location", "occurrence", "country"):
            boost += 0.10

    if any(term in q for term in ["threat", "conservation", "research", "use", "trade", "habitat", "system", "supporting information"]):
        if in_source("Required_and_Recommended_Supporting_Information", "RL_Standards_Consistency"):
            boost += 0.10

    if query_mode["supporting_info"] and any(term in q for term in ["beyond the basics", "extra", "additional"]):
        if table_label(candidate) == "table2":
            boost += 0.18
        elif table_label(candidate) == "table1":
            boost -= 0.04

    return boost


def subquery_specific_boost(candidate: Candidate, query: str) -> float:
    """Apply extra boosts when an internal subquery targets a specific requirement table.

    For example:
    - "all assessments" subqueries favor Table 1
    - "specific conditions" subqueries favor Table 2
    """
    label = table_label(candidate)
    block_type = str(candidate.metadata.get("block_type", ""))
    boost = 0.0

    if is_all_conditions_subquery(query):
        if label == "table1":
            boost += 0.24
            if block_type == "table":
                boost += 0.08
        elif label == "table2":
            boost -= 0.06

    if is_specific_conditions_subquery(query):
        if label == "table2":
            boost += 0.20
            if block_type == "table_row":
                boost += 0.04
        elif label == "table1":
            boost -= 0.04

    return boost


def dedup_key(candidate: Candidate):
    """Build the deduplication key used when merging overlapping retrieval results.

    The key keeps one logical record per source/page/block/table-row identity
    even if multiple retrieval lanes surfaced it.
    """
    meta = candidate.metadata
    return (
        meta.get("source_file"),
        meta.get("page"),
        meta.get("block_type"),
        meta.get("table_id"),
        meta.get("row_id"),
    )


def token_coverage(query: str, text: str):
    """Measure how much of the query vocabulary appears in candidate search text.

    This score helps the reranker reward candidates that cover more of the
    query terms directly.
    """
    q_terms = set(tokenize(query))
    if not q_terms:
        return 0.0
    doc_terms = set(tokenize(text))
    return len(q_terms & doc_terms) / len(q_terms)


def candidate_from_doc(doc):
    """Convert one sparse-corpus document dictionary into a ``Candidate``.

    This is mainly used for sparse retrieval hits and corpus backfill records,
    which start as JSON-style dictionaries rather than Chroma document objects.
    """
    metadata = {k: v for k, v in doc.items() if k not in {"text", "search_text"}}
    return Candidate(
        chunk_id=str(doc["chunk_id"]),
        text=doc.get("text") or doc.get("display_text") or "",
        search_text=doc.get("search_text") or doc.get("text") or "",
        metadata=metadata,
    )


def build_supporting_info_queries(query: str):
    """Expand supporting-information questions into the internal subqueries used for coverage.

    The expanded set typically includes:
    - a baseline all-assessments query
    - a specific-conditions query
    - the original user query
    """
    q = query.lower()
    if not looks_like_supporting_info_query(q):
        return [query]

    specific_conditions_query = "required supporting information under specific conditions for threatened taxa"
    if has_phrase(q, "beyond the basics") or has_phrase(q, "extra") or has_phrase(q, "additional"):
        specific_conditions_query = (
            "additional required supporting information under specific conditions for threatened taxa beyond the basics"
        )
    return [
        "required supporting information under all conditions for IUCN Red List assessments",
        specific_conditions_query,
        query,
    ]


def pick_best_table_candidate(candidates, table_name: str, preferred_block_type: str | None = None):
    """Pick the best-ranked candidate for a target requirement table.

    When a preferred block type is provided, the helper prefers that block type
    first and falls back to any candidate from the requested table only if
    necessary.
    """
    matches = [c for c in candidates if table_label(c) == table_name]
    if not matches:
        return None
    if preferred_block_type is not None:
        typed = [c for c in matches if str(c.metadata.get("block_type", "")) == preferred_block_type]
        if typed:
            matches = typed
    matches.sort(key=lambda c: c.rerank_score, reverse=True)
    return matches[0]


def pick_corpus_backfill(corpus, table_name: str, preferred_block_type: str | None = None):
    """Backfill a missing table candidate directly from the sparse corpus.

    This is used only when the normal ranked results fail to include required
    coverage for key supporting-information tables.
    """
    desired = {"table1": "Table 1", "table2": "Table 2", "table3": "Table 3"}[table_name]
    matches = []
    for doc in corpus:
        title = str(doc.get("table_title", "") or "")
        if desired not in title:
            continue
        if preferred_block_type is not None and str(doc.get("block_type", "")) != preferred_block_type:
            continue
        matches.append(candidate_from_doc(doc))
    if not matches and preferred_block_type is not None:
        for doc in corpus:
            title = str(doc.get("table_title", "") or "")
            if desired in title:
                matches.append(candidate_from_doc(doc))
    if not matches:
        return None
    matches.sort(
        key=lambda c: (
            0 if str(c.metadata.get("block_type", "")) == (preferred_block_type or str(c.metadata.get("block_type", ""))) else 1,
            int(c.metadata.get("page") or 999),
            int(c.metadata.get("row_id") or 999),
        )
    )
    return matches[0]


def _upsert_dense_hits(by_id, dense_hits):
    """Merge dense search hits into the working candidate map.

    Dense distances are converted into similarity-style scores, then each hit
    contributes both a dense score and a reciprocal-rank-fusion term.
    """
    max_dense = 0.0
    processed = []
    for doc, distance in dense_hits:
        dense_score = 1.0 / (1.0 + float(distance))
        processed.append((doc, dense_score))
        max_dense = max(max_dense, dense_score)

    for rank, (doc, dense_score) in enumerate(processed, start=1):
        chunk_id = str(doc.metadata["chunk_id"])
        metadata = doc.metadata.copy()
        display_text = metadata.pop("display_text", doc.page_content)
        by_id.setdefault(
            chunk_id,
            Candidate(
                chunk_id=chunk_id,
                text=display_text,
                search_text=doc.page_content,
                metadata=metadata,
            ),
        )
        candidate = by_id[chunk_id]
        candidate.dense_rank = rank
        candidate.dense_score = dense_score
        candidate.fused_score += reciprocal_rank_fusion(rank) + 0.10 * (dense_score / max(max_dense, 1e-9))


def _upsert_sparse_hits(by_id, sparse_hits, corpus):
    """Merge sparse BM25 hits into the working candidate map.

    Sparse hits contribute:
    - the BM25 score itself
    - the sparse rank
    - a reciprocal-rank-fusion term
    """
    max_bm25 = max((score for _, score in sparse_hits), default=1.0)
    for rank, (doc_idx, score) in enumerate(sparse_hits, start=1):
        doc = corpus[doc_idx]
        chunk_id = str(doc["chunk_id"])
        by_id.setdefault(chunk_id, candidate_from_doc(doc))
        candidate = by_id[chunk_id]
        candidate.sparse_rank = rank
        candidate.bm25_score = score
        candidate.fused_score += reciprocal_rank_fusion(rank)
        candidate.fused_score += 0.06 * (score / max_bm25)


def _rerank_candidates(candidates, query: str, query_mode):
    """Compute the final heuristic rerank score for each candidate.

    The current reranker combines:
    - normalized dense score
    - normalized lexical score
    - token coverage
    - metadata boosts
    - subquery-aware boosts
    - content-intent boosts
    """
    max_dense = max((candidate.dense_score for candidate in candidates), default=1.0)
    max_bm25 = max((candidate.bm25_score for candidate in candidates), default=1.0)

    for candidate in candidates:
        dense_norm = candidate.dense_score / max(max_dense, 1e-9)
        lexical = candidate.bm25_score / max(max_bm25, 1e-9)
        coverage = token_coverage(query, candidate.search_text)
        candidate.rerank_score = (
            0.42 * dense_norm
            + 0.22 * lexical
            + 0.16 * coverage
            + priority_boost(candidate, query_mode)
            + subquery_specific_boost(candidate, query)
            + content_intent_boost(candidate, query, query_mode)
        )


def hybrid_search_single(query, corpus, bm25, dense_k=24, sparse_k=24, final_k=20):
    """Run one dense-plus-sparse retrieval pass for a single query string.

    This helper is used both for the original user query and for any internal
    supporting-information subqueries.
    """
    by_id = {}
    query_mode = detect_query_mode(query)

    _upsert_dense_hits(by_id, safe_dense_search(query, dense_k))
    _upsert_sparse_hits(by_id, bm25.search(query, k=sparse_k), corpus)

    candidates = list(by_id.values())
    _rerank_candidates(candidates, query, query_mode)
    candidates.sort(key=lambda c: c.rerank_score, reverse=True)
    return candidates[:final_k]


def attach_parent_context(candidates, parent_map):
    """Attach parent table context to any candidate that belongs to a known table.

    This supports the small-to-big retrieval pattern where a specific row hit
    can be paired with the broader parent table context.
    """
    for cand in candidates:
        table_id = cand.metadata.get("table_id")
        source_file = cand.metadata.get("source_file")
        if table_id and source_file:
            parent = parent_map.get((str(source_file), str(table_id)))
            if parent:
                cand.parent_text = parent["text"]
                cand.parent_metadata = {k: v for k, v in parent.items() if k != "text"}


def _merge_candidate_metrics(target: Candidate, source: Candidate):
    """Merge ranking signals from duplicate candidates selected by different subqueries.

    The merged candidate keeps the strongest observed score from each ranking
    signal.
    """
    target.rerank_score = max(target.rerank_score, source.rerank_score)
    target.fused_score = max(target.fused_score, source.fused_score)
    target.dense_score = max(target.dense_score, source.dense_score)
    target.bm25_score = max(target.bm25_score, source.bm25_score)


def _dedupe_merged_candidates(merged, per_source_limit: int):
    """Deduplicate merged candidates while enforcing a per-source result cap.

    This prevents one document from dominating the final selection while still
    allowing strong hits from multiple sources to appear.
    """
    deduped = []
    seen = set()
    per_source_counts = Counter()
    for candidate in sorted(merged.values(), key=lambda item: item.rerank_score, reverse=True):
        key = dedup_key(candidate)
        if key in seen:
            continue

        source_file = str(candidate.metadata.get("source_file", ""))
        if per_source_counts[source_file] >= per_source_limit:
            continue

        seen.add(key)
        per_source_counts[source_file] += 1
        deduped.append(candidate)
    return deduped


def _force_supporting_info_coverage(deduped, corpus):
    """Ensure supporting-information answers include key Table 1 and Table 2 evidence.

    If either table is missing from the ranked candidates, the helper tries to
    backfill a representative result directly from the sparse corpus.
    """
    forced = []
    table1 = pick_best_table_candidate(deduped, "table1", preferred_block_type="table")
    table2 = pick_best_table_candidate(deduped, "table2", preferred_block_type="table_row")

    if table1 is None:
        table1 = pick_corpus_backfill(corpus, "table1", preferred_block_type="table")
        if table1 is not None:
            table1.forced_for_coverage = True
    if table2 is None:
        table2 = pick_corpus_backfill(corpus, "table2", preferred_block_type="table_row")
        if table2 is not None:
            table2.forced_for_coverage = True

    for candidate in [table1, table2]:
        if candidate is not None:
            forced.append(candidate)
    return forced


def merge_and_select(results_by_query, corpus, final_k=8, support_info_mode=True, per_source_limit=8):
    """Merge subquery results into the final selected candidate list.

    In supporting-information mode, the selector first injects any forced
    coverage results, then fills the remainder from the deduplicated ranked
    list up to ``final_k``.
    """
    merged = {}
    for _, results in results_by_query:
        for candidate in results:
            if candidate.chunk_id not in merged:
                merged[candidate.chunk_id] = candidate
            else:
                _merge_candidate_metrics(merged[candidate.chunk_id], candidate)

    deduped = _dedupe_merged_candidates(merged, per_source_limit)

    if support_info_mode:
        selected = []
        used = set()
        for candidate in _force_supporting_info_coverage(deduped, corpus):
            if candidate.chunk_id not in used:
                selected.append(candidate)
                used.add(candidate.chunk_id)

        for candidate in deduped:
            if candidate.chunk_id in used:
                continue
            selected.append(candidate)
            used.add(candidate.chunk_id)
            if len(selected) >= final_k:
                break
        return selected[:final_k]

    return deduped[:final_k]


def extract_parent_rows(parent_text: str, max_rows: int = 4):
    """Extract a few readable parent-table rows for scaffold construction.

    The helper drops table titles and schema lines, then keeps only the first
    few row lines that start with ``Page`` so the scaffold stays compact.
    """
    if not parent_text:
        return []
    content = parent_text.split("Content:", 1)[-1].strip()
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    rows = []
    for line in lines:
        if line.startswith("Table ") or line.startswith("Schema:"):
            continue
        if line.startswith("Page "):
            rows.append(line)
        if len(rows) >= max_rows:
            break
    return rows


def _build_general_answer_summary(results):
    """Build a short scaffold for non-supporting-information queries.

    The scaffold summarizes the top few selected candidates in a compact,
    prompt-friendly form for later generation.
    """
    lines = ["Answer scaffold", "---------------", "Most relevant retrieved reference evidence:"]
    if not results:
        lines.append("- No retrieved reference evidence was available.")
        return "\n".join(lines)

    for candidate in results[:4]:
        metadata = candidate.metadata
        snippet = candidate.text.split("Content:")[-1].strip().replace("\n", " ")
        lines.append(
            f"- Section: {str(metadata.get('section_title', '') or metadata.get('table_title', '') or 'Unknown section')} "
            f"| Source: {str(metadata.get('source_file', ''))} | Page: {metadata.get('page')} | Evidence: {snippet[:260]}"
        )
    return "\n".join(lines)


def _append_supporting_info_section(lines, heading: str, results, *, prefer_parent_rows: bool):
    """Append one supporting-information scaffold section from selected table results.

    For Table 1 style sections the helper prefers cleaner parent-table rows.
    For other sections it falls back to clipped candidate snippets.
    """
    lines.append(heading)
    if not results:
        fallback = "- No clean Table 1 evidence selected." if prefer_parent_rows else "- No clean Table 2 evidence selected."
        lines.append(fallback)
        return

    if prefer_parent_rows:
        parent_rows = []
        for candidate in results:
            if candidate.parent_text:
                parent_rows = extract_parent_rows(candidate.parent_text, max_rows=4)
                if parent_rows:
                    break
            elif str(candidate.metadata.get("block_type", "")) == "table":
                parent_rows = extract_parent_rows(candidate.text, max_rows=4)
                if parent_rows:
                    break
        if parent_rows:
            for row in parent_rows:
                lines.append(f"- {row[:260]}")
            return

    for candidate in results[:4]:
        snippet = candidate.text.split("Content:")[-1].strip()
        lines.append(f"- {snippet[:260]}")


def build_answer_summary(results, support_info_mode: bool = True, query: str = ""):
    """Build the reference-evidence scaffold that bridges retrieval and generation.

    Supporting-information queries get a two-part scaffold separating:
    - requirements for all assessments
    - additional requirements under specific conditions

    Other queries get a simpler top-results summary.
    """
    q = (query or "").lower()
    if not support_info_mode:
        return _build_general_answer_summary(results)

    table1_items = [candidate for candidate in results if table_label(candidate) == "table1"]
    table2_items = [candidate for candidate in results if table_label(candidate) == "table2"]
    specific_first = any(term in q for term in ["beyond the basics", "extra", "additional", "specific conditions"])
    lines = ["Answer scaffold", "---------------"]

    if specific_first:
        _append_supporting_info_section(
            lines,
            "Additional required under specific conditions relevant to threatened taxa:",
            table2_items,
            prefer_parent_rows=False,
        )
        lines.append("")
        _append_supporting_info_section(
            lines,
            "Required for all assessments:",
            table1_items,
            prefer_parent_rows=True,
        )
    else:
        _append_supporting_info_section(
            lines,
            "Required for all assessments:",
            table1_items,
            prefer_parent_rows=True,
        )
        lines.append("")
        _append_supporting_info_section(
            lines,
            "Additional required under specific conditions relevant to threatened taxa:",
            table2_items,
            prefer_parent_rows=False,
        )
    return "\n".join(lines)


def answer_query(
    query: str,
    dense_k: int = 24,
    sparse_k: int = 24,
    final_k: int = 8,
    allow_threshold_short_circuit: bool = True,
):
    """Answer one reference-side query using deterministic lookup or hybrid retrieval.

    The method first tries the threshold route when allowed. If no exact
    deterministic answer is available, it runs hybrid retrieval, attaches
    parent table context, builds the answer scaffold, and returns the final
    reference payload used later by inference.
    """
    threshold_answer = None
    if allow_threshold_short_circuit and is_threshold_query(query):
        threshold_answer = answer_threshold_query(query)
        if threshold_answer is not None:
            return {
                "route": "deterministic_threshold_lookup",
                "threshold_answer": threshold_answer,
                "query": query,
            }

    corpus = load_corpus()
    bm25 = get_bm25_index()
    parent_map = load_parent_contexts()
    query_mode = detect_query_mode(query)
    subqueries = build_supporting_info_queries(query) if query_mode["supporting_info"] else [query]

    results_by_query = []
    for subquery in subqueries:
        res = hybrid_search_single(
            subquery,
            corpus,
            bm25,
            dense_k=dense_k,
            sparse_k=sparse_k,
            final_k=max(final_k * 4, 24),
        )
        results_by_query.append((subquery, res))

    final_results = merge_and_select(
        results_by_query,
        corpus,
        final_k=final_k,
        support_info_mode=query_mode["supporting_info"],
        per_source_limit=8,
    )
    attach_parent_context(final_results, parent_map)
    return {
        "route": "hybrid_rag",
        "query": query,
        "subqueries": subqueries,
        "answer_scaffold": build_answer_summary(
            final_results,
            support_info_mode=query_mode["supporting_info"],
            query=query,
        ),
        "results": final_results,
        "threshold_fallback_miss": is_threshold_query(query) and threshold_answer is None,
    }
