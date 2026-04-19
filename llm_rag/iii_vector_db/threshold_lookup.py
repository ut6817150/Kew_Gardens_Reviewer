"""Deterministic Red List threshold lookup.

Purpose:
    This module answers stable threshold-style questions from the curated
    `thresholds.json` data instead of relying on generative retrieval. It covers
    common Red List threshold facts such as EOO, AOO, number of locations,
    mature individuals, and extinction probability.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
THRESHOLDS_PATH = SCRIPT_DIR / "thresholds.json"

CRITERION_RE = re.compile(r"\bcriterion\s*([A-E])\b", re.IGNORECASE)
SUBCRITERION_RE = re.compile(r"\b([A-E])([12])\b", re.IGNORECASE)

THRESHOLD_TERMS = (
    "threshold",
    "thresholds",
    "cutoff",
    "cut-off",
    "criterion a",
    "criterion b",
    "criterion c",
    "criterion d",
    "criterion e",
    "b1",
    "b2",
    "d1",
    "d2",
    "extinction probability",
    "qualif",
)
CONTEXTUAL_THRESHOLD_TERMS = ("threshold", "cutoff", "criterion", "qualif")
FIELD_TERMS = {
    "aoo": ("aoo",),
    "eoo": ("eoo",),
    "locations": ("locations", "number of locations"),
    "mature_individuals": ("mature individuals",),
    "extinction_probability": ("extinction probability",),
}


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    """Return True when any candidate term appears in the normalized query.

    This helper is used throughout the threshold route to keep simple
    substring-based intent checks readable and consistent.
    """
    return any(term in text for term in terms)


@lru_cache(maxsize=1)
def load_thresholds() -> dict:
    """Load the deterministic threshold data from ``thresholds.json``.

    The result is cached so repeated queries do not re-read the JSON file
    during one app session.
    """
    with THRESHOLDS_PATH.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def is_threshold_query(query: str) -> bool:
    """Return True when a query looks like a threshold-style question.

    This function checks for:
    - explicit threshold words such as ``threshold`` or ``cutoff``
    - criterion references such as ``Criterion B`` or ``D2``
    - field names such as ``AOO``, ``EOO``, ``locations``, or
      ``mature individuals`` when they appear in threshold-like contexts
    """
    q = query.lower()
    if _contains_any(q, THRESHOLD_TERMS):
        return True

    if _contains_any(q, FIELD_TERMS["aoo"] + FIELD_TERMS["eoo"]) and _contains_any(q, CONTEXTUAL_THRESHOLD_TERMS):
        return True

    location_terms = FIELD_TERMS["locations"] + FIELD_TERMS["mature_individuals"]
    location_context = CONTEXTUAL_THRESHOLD_TERMS + ("d1", "d2", "b1", "b2")
    return _contains_any(q, location_terms) and _contains_any(q, location_context)


def infer_criterion(query: str) -> str | None:
    """Infer the most likely Red List criterion referenced by the query.

    It first prefers explicit criterion mentions such as ``Criterion B`` or
    ``D2``. If none are present, it falls back to field-based heuristics such
    as:
    - ``AOO`` or ``EOO`` -> Criterion B
    - ``mature individuals`` -> Criterion D
    - ``extinction probability`` -> Criterion E
    """
    match = CRITERION_RE.search(query) or SUBCRITERION_RE.search(query)
    if match:
        return match.group(1).upper()

    q = query.lower()
    if _contains_any(q, FIELD_TERMS["aoo"] + FIELD_TERMS["eoo"] + FIELD_TERMS["locations"]):
        return "B"
    if _contains_any(q, FIELD_TERMS["mature_individuals"]):
        return "D"
    if _contains_any(q, FIELD_TERMS["extinction_probability"]):
        return "E"
    return None


def _requested_fields(query_lower: str) -> dict[str, bool]:
    """Mark which threshold fields the user explicitly asked about.

    The returned flags help the deterministic route choose between:
    - full criterion summaries
    - narrower field-specific answers
    """
    return {
        field_name: _contains_any(query_lower, terms)
        for field_name, terms in FIELD_TERMS.items()
    }


def _join_non_empty(lines: list[str]) -> str | None:
    """Join non-empty answer lines while preserving their original order.

    This helper lets the deterministic route compose field-specific answers
    without introducing blank lines for fields the user did not request.
    """
    filtered = [line for line in lines if line]
    return "\n".join(filtered) if filtered else None


def answer_threshold_query(query: str) -> str | None:
    """Return a deterministic threshold answer when the query maps cleanly.

    This method uses the threshold JSON plus light criterion inference to
    answer questions about:
    - AOO
    - EOO
    - number of locations
    - mature individuals
    - extinction probability

    It returns ``None`` when the query does not map cleanly enough, allowing
    the caller to fall back to hybrid retrieval instead.
    """
    data = load_thresholds()
    criteria = data["criteria"]
    q = query.lower()
    requested_fields = _requested_fields(q)
    criterion = infer_criterion(query)

    if criterion:
        crit_data = criteria.get(criterion)
        if not crit_data:
            return None

        if criterion == "B":
            return _join_non_empty(
                [
                    crit_data["fields"]["eoo"] if requested_fields["eoo"] else "",
                    crit_data["fields"]["aoo"] if requested_fields["aoo"] else "",
                    crit_data["fields"]["locations"] if requested_fields["locations"] else "",
                ]
            ) or "\n".join([f"Criterion {criterion}: {crit_data['title']}"] + crit_data["summary"])

        if criterion == "D":
            return _join_non_empty(
                [
                    crit_data["fields"]["mature_individuals"] if requested_fields["mature_individuals"] else "",
                    crit_data["fields"]["d2"] if "d2" in q else "",
                ]
            ) or "\n".join([f"Criterion {criterion}: {crit_data['title']}"] + crit_data["summary"])

        if criterion == "E" and requested_fields["extinction_probability"]:
            return crit_data["fields"]["extinction_probability"]

        return "\n".join([f"Criterion {criterion}: {crit_data['title']}"] + crit_data["summary"])

    if requested_fields["aoo"] and requested_fields["eoo"]:
        return "\n".join(
            [
                criteria["B"]["fields"]["eoo"],
                criteria["B"]["fields"]["aoo"],
            ]
        )
    if requested_fields["aoo"]:
        return criteria["B"]["fields"]["aoo"]
    if requested_fields["eoo"]:
        return criteria["B"]["fields"]["eoo"]
    if requested_fields["locations"] and _contains_any(q, CONTEXTUAL_THRESHOLD_TERMS):
        return criteria["B"]["fields"]["locations"]
    if requested_fields["mature_individuals"]:
        return criteria["D"]["fields"]["mature_individuals"]
    if requested_fields["extinction_probability"]:
        return criteria["E"]["fields"]["extinction_probability"]

    return None
