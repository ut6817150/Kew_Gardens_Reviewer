from __future__ import annotations

import re
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9()\[\].,/%:+\-_]*")
HTML_TAG_RE = re.compile(r"<[^>]+>")
BLOCK_MARKER_RE = re.compile(r"\s+\[(?:paragraph|table|row)\s+\d+\]$", flags=re.IGNORECASE)
STOPWORDS = {
    "about",
    "a",
    "according",
    "already",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "beyond",
    "by",
    "does",
    "draft",
    "extra",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "its",
    "known",
    "main",
    "many",
    "of",
    "official",
    "on",
    "or",
    "roughly",
    "say",
    "separate",
    "species",
    "that",
    "the",
    "this",
    "to",
    "what",
    "where",
    "which",
}


def _normalize_text(text: str) -> str:
    """Collapse repeated whitespace so draft text compares more consistently.

    This helper is used before tokenization and scoring so minor formatting
    differences in the parsed draft do not create separate retrieval behavior.
    """
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _strip_html_tags(text: str) -> str:
    """Remove simple HTML tags from a draft section before tokenization.

    The draft store keeps the original HTML text for prompt assembly, but
    retrieval scoring works better on a lightly stripped plain-text view.
    """
    return HTML_TAG_RE.sub(" ", str(text or ""))


def _normalize_token(token: str) -> set[str]:
    """Normalize one token into retrieval-friendly lowercase variants.

    This helper:
    - lowercases the token
    - removes empty and stopword-only cases
    - adds a few simple singularization variants such as ``species`` -> ``specy``
      style fallbacks where useful for fuzzy overlap
    """
    token = token.lower().strip()
    if not token or token in STOPWORDS:
        return set()

    variants = {token}
    if token.endswith("ies") and len(token) > 4:
        variants.add(token[:-3] + "y")
    if token.endswith("es") and len(token) > 4:
        variants.add(token[:-2])
    if token.endswith("s") and len(token) > 3:
        variants.add(token[:-1])
    return {variant for variant in variants if variant and variant not in STOPWORDS}


def _tokenize(text: str) -> list[str]:
    """Tokenize draft text into normalized retrieval terms.

    Each regex token is expanded through ``_normalize_token(...)`` so draft
    retrieval can match simple inflection variants more reliably.
    """
    tokens: list[str] = []
    for match in TOKEN_RE.finditer(text):
        for variant in sorted(_normalize_token(match.group(0))):
            tokens.append(variant)
    return tokens


def _normalize_section_path(section_path: str) -> str:
    """Remove parser block markers from a section path.

    The parsed report may contain suffixes such as ``[paragraph 2]`` or
    ``[table 1]``. This helper removes them so related draft content groups
    under one cleaner section label.
    """
    return BLOCK_MARKER_RE.sub("", section_path or "").strip() or "Assessment"


def _contains_phrase(text: str, phrase: str) -> bool:
    """Return True when a phrase appears with simple word-boundary handling.

    The helper prefers word-boundary regex matching for plain alphanumeric
    phrases, and falls back to a simpler substring match for phrases that
    include punctuation.
    """
    phrase = str(phrase or "").strip().lower()
    if not phrase:
        return False
    if all(char.isalnum() or char.isspace() for char in phrase):
        pattern = r"\b" + r"\s+".join(re.escape(part) for part in phrase.split()) + r"\b"
        return re.search(pattern, text) is not None
    return phrase in text


def _looks_like_supporting_info_query(query_lower: str) -> bool:
    """Return True when the prompt appears to ask about supporting-information requirements.

    This is used to activate extra section-path boosts for draft parts such as:
    - locations information
    - population information
    - threats
    - conservation actions
    """
    direct_phrases = [
        "supporting information",
        "supporting info",
        "required information",
        "required data",
        "additional information",
        "additional data",
        "what do i need to include",
        "what must i include",
        "what should i include",
        "beyond the basics",
        "extra supporting info",
        "extra supporting information",
        "required under specific conditions",
    ]
    if any(_contains_phrase(query_lower, phrase) for phrase in direct_phrases):
        return True

    support_nouns = ["information", "info", "data", "documentation"]
    support_verbs = ["include", "included", "required", "need", "needed", "submit", "extra", "additional"]
    return any(_contains_phrase(query_lower, noun) for noun in support_nouns) and any(
        _contains_phrase(query_lower, verb) for verb in support_verbs
    )


def build_draft_store_from_report(full_report: dict[str, str]) -> list[dict[str, Any]]:
    """Turn the parsed draft report into a deduplicated section-level retrieval store.

    Each stored chunk contains:
    - a normalized section path
    - the original source key from the parsed report
    - the original HTML text
    - normalized retrieval tokens built from the section path plus stripped text

    The current implementation stores one chunk per parsed section.
    """
    chunks: list[dict[str, Any]] = []

    if not isinstance(full_report, dict):
        return chunks

    for raw_section_path, raw_text in full_report.items():
        section_path = _normalize_section_path(str(raw_section_path))
        text = _normalize_text(raw_text)
        if not text:
            continue

        token_text = _normalize_text(_strip_html_tags(text))
        combined = f"{section_path}\n{token_text}"
        chunks.append(
            {
                "section_path": section_path,
                "source_key": str(raw_section_path),
                "text": text,
                "tokens": _tokenize(combined),
            }
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        key = (chunk["source_key"], chunk["text"])
        if key not in seen:
            seen.add(key)
            deduped.append(chunk)

    return deduped


def _draft_hit(chunk: dict[str, Any], score: float) -> dict[str, Any]:
    """Format one scored draft chunk for prompt assembly and debug output.

    The returned object keeps the rounded score plus the section and source
    identifiers needed later by the prompt builder and debug view.
    """
    return {
        "score": round(score, 3),
        "section_path": chunk["section_path"],
        "source_key": chunk["source_key"],
        "text": chunk["text"],
    }


def retrieve_from_draft(query: str, draft_store: list[dict[str, Any]], top_k: int = 6) -> list[dict[str, Any]]:
    """Return the highest-scoring draft sections for the current user question.

    The score combines:
    - token overlap between query and chunk
    - extra weight for query tokens appearing in the section path
    - intent-specific heuristic boosts for common Red List question types

    If the query tokenizes to nothing, the method falls back to the first
    ``top_k`` draft chunks with a score of ``0.0``.
    """
    if not draft_store:
        return []

    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return [_draft_hit(chunk, 0.0) for chunk in draft_store[:top_k]]

    scored: list[tuple[float, int, dict[str, Any]]] = []
    query_lower = query.lower()

    for index, chunk in enumerate(draft_store):
        chunk_tokens = set(chunk.get("tokens", []))
        overlap = len(query_tokens & chunk_tokens)

        section_path = chunk.get("section_path", "").lower()
        text_lower = str(chunk.get("text", "")).lower()
        score = float(overlap)

        for token in query_tokens:
            if token in section_path:
                score += 0.75

        score += _intent_boost(query_lower, section_path, text_lower)

        scored.append((score, index, chunk))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [_draft_hit(chunk, score) for score, _, chunk in scored[:top_k]]


def _supporting_info_boost(
    query_lower: str,
    is_root_section: bool,
    path_match,
    has_any,
) -> float:
    """Boost draft sections that look especially relevant to supporting-information prompts.

    This helper favors sections that often contain required-supporting-info
    evidence, and it adds further boosts when the question asks for
    ``extra`` or ``additional`` information beyond the basics.
    """
    if not _looks_like_supporting_info_query(query_lower):
        return 0.0

    boost = 0.0
    if path_match("assessment information"):
        boost += 2.2
    if path_match("assessment rationale"):
        boost += 1.8
    if path_match("locations information"):
        boost += 2.2
    if path_match("population information"):
        boost += 2.0
    if path_match("threats classification scheme"):
        boost += 2.4
    if path_match("conservation actions in- place"):
        boost += 2.0
    if path_match("important conservation actions needed"):
        boost += 2.3
    if path_match("research needed"):
        boost += 2.0
    if path_match("countries of occurrence"):
        boost += 1.5
    if path_match("iucn habitats classification scheme", "> systems"):
        boost += 1.4
    if is_root_section:
        boost -= 1.2

    if has_any("beyond the basics", "extra", "additional"):
        if path_match("threats classification scheme"):
            boost += 1.2
        if path_match("important conservation actions needed", "conservation actions in- place"):
            boost += 1.2
        if path_match("locations information", "population information"):
            boost += 1.0
        if path_match("assessment information"):
            boost -= 0.4

    return boost


def _intent_boost(query_lower: str, section_path: str, text_lower: str) -> float:
    """Apply query-intent heuristics so draft retrieval favors the most relevant sections.

    The current rules cover common assessment questions about:
    - Red List status and category
    - rationale and data deficiency
    - EOO, locations, and D2
    - population, threats, habitats, and systems
    - conservation, research, bibliography, and related metadata

    These boosts help draft retrieval behave more like section-aware review
    than plain bag-of-words matching.
    """
    boost = 0.0
    is_root_section = ">" not in section_path

    def has_any(*phrases: str) -> bool:
        return any(_contains_phrase(query_lower, phrase) for phrase in phrases)

    def path_match(*phrases: str) -> bool:
        return any(phrase in section_path for phrase in phrases)

    def text_match(*phrases: str) -> bool:
        return any(phrase in text_lower for phrase in phrases)

    if has_any("red list status", "assessment status") or ("status" in query_lower and "red list" in query_lower):
        if path_match("assessment information"):
            boost += 3.4
        elif text_match("red list status", "<b>status:</b>"):
            boost += 1.8
        if "assessment rationale" in section_path:
            boost -= 0.4
        if is_root_section:
            boost -= 1.2

    if (has_any("red list category", "category") and has_any("red list", "iucn")) or has_any(
        "currently listed",
        "currently in",
    ):
        if path_match("assessment information"):
            boost += 3.4
        elif text_match("red list status", "vulnerable", "endangered", "critically endangered", "data deficient"):
            boost += 1.8
        if path_match("assessment rationale"):
            boost -= 0.5

    if has_any("data deficient", "assessment rationale", "why is"):
        if path_match("assessment rationale"):
            boost += 2.8
        elif text_match("data deficient"):
            boost += 1.6

    if has_any("extent of occurrence", "eoo"):
        if path_match("extent of occurrence"):
            boost += 3.0
        elif text_match("eoo"):
            boost += 1.6

    if has_any("location", "locations"):
        if path_match("locations information"):
            boost += 3.0
        elif path_match("very restricted aoo or number of locations"):
            boost += 2.2
        elif text_match("number of locations", "very restricted"):
            boost += 1.4

    if has_any("d2", "very restricted"):
        if path_match("triggers vu d2"):
            boost += 3.0
        elif text_match("very restricted", "d2"):
            boost += 1.6

    if has_any("population trend"):
        if path_match("population information"):
            boost += 3.0
        elif text_match("current population trend"):
            boost += 1.6
    elif "population" in query_lower:
        if path_match("population information"):
            boost += 2.4
        elif path_match("> population"):
            boost += 1.8
        elif text_match("population"):
            boost += 0.5

    if has_any("where is", "geographic", "distribution", "found geographically"):
        if path_match("geographic range"):
            boost += 2.4
        elif path_match("> distribution"):
            boost += 1.2
        elif text_match("distribution"):
            boost += 0.5

    if has_any("country", "countries", "occur in", "occurrence"):
        if path_match("countries of occurrence"):
            boost += 3.2
        elif text_match("<b>country</b>"):
            boost += 1.8
    if has_any("recorded from"):
        if path_match("countries of occurrence"):
            boost += 2.8

    if has_any("elevation", "altitude", "depth zone", "depth zones"):
        if path_match("elevation / depth / depth zones"):
            boost += 3.0
        elif text_match("<b>elevation"):
            boost += 1.6

    if has_any("habitat", "ecology"):
        if path_match("iucn habitats classification scheme"):
            boost += 2.8
        elif path_match("> habitats and ecology"):
            boost += 1.8
        elif text_match("habitat", "system:"):
            boost += 0.8

    if "system" in query_lower:
        if path_match("> systems"):
            boost += 3.0
        elif text_match("<b>system: </b>", "<b>system:</b>"):
            boost += 1.6

    if has_any("use", "trade", "utilized"):
        if path_match("use and trade"):
            boost += 3.0
        elif text_match("species not utilized", "known uses"):
            boost += 1.4

    if "threat" in query_lower:
        if path_match("threats classification scheme"):
            boost += 2.8
        elif path_match("> threats"):
            boost += 1.8
        elif text_match("threat"):
            boost += 0.6

    if has_any("conservation actions are in place", "in place"):
        if path_match("conservation actions in- place"):
            boost += 3.0
        elif text_match("occur in at least one pa"):
            boost += 1.6

    if has_any("conservation actions are needed", "actions needed"):
        if path_match("important conservation actions needed"):
            boost += 3.0
    if has_any("conservation work", "conservation measures needed", "extra conservation work"):
        if path_match("important conservation actions needed"):
            boost += 3.2
        if path_match("research needed") and "research" not in query_lower:
            boost -= 0.5

    if "research" in query_lower:
        if path_match("research needed"):
            boost += 2.8
        elif text_match("<b>research</b>"):
            boost += 1.5

    if has_any("ecosystem service", "ecosystem services"):
        if path_match("ecosystem services provided by the species"):
            boost += 3.0
        elif text_match("insufficient information available"):
            boost += 1.4

    if has_any("biogeographic", "realm", "realms"):
        if path_match("biogeographic realms"):
            boost += 3.0
        elif text_match("biogeographic realm"):
            boost += 1.6

    if has_any("map status", "map") and "status" in query_lower:
        if path_match("map status"):
            boost += 3.0
        elif text_match("<b>map status</b>"):
            boost += 1.6

    if has_any("growth form", "growth forms", "plant growth"):
        if path_match("plant specific"):
            boost += 3.0
        elif text_match("plant growth forms"):
            boost += 1.6

    if has_any("bibliography", "references", "reference", "cited", "citation", "citations", "literature"):
        if path_match("bibliography"):
            boost += 3.6

    if is_root_section and has_any(
        "red list status",
        "data deficient",
        "extent of occurrence",
        "eoo",
        "location",
        "locations",
        "population",
        "country",
        "countries",
        "elevation",
        "habitat",
        "ecology",
        "system",
        "use",
        "trade",
        "threat",
        "conservation",
        "research",
        "ecosystem service",
        "biogeographic",
        "realm",
        "map status",
        "bibliography",
        "references",
        "citation",
        "cited",
    ):
        boost -= 0.8

    return boost + _supporting_info_boost(query_lower, is_root_section, path_match, has_any)
