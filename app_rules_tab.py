"""Rules-tab UI helpers for the Streamlit app.

Purpose:
    This module renders the deterministic rules-based review tab. It accepts
    the shared parsed assessment dictionary prepared by ``app.py``, runs the
    parser and reviewer pipeline on demand, and reshapes the resulting
    violations for display and JSON export.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from iucn_rules_checker.assessment_parser import AssessmentParser
from iucn_rules_checker.assessment_reviewer import IUCNAssessmentReviewer
from llm_rag.iv_inference.ui_helpers import normalize_display_section_name


def build_ordered_grouped_violations(
    full_report: dict[str, str],
    cleaned_violations_dicts: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Group rules-based violations by display section while preserving report order.

    This helper keeps the section order from the parsed full report so the UI
    mirrors the original document structure as closely as possible, even after
    the reviewer has produced a flat list of violations.

    Args:
        full_report (dict[str, str]): Parsed ``section_name -> text`` mapping
            used to preserve the source document's original section order.
        cleaned_violations_dicts (list[dict[str, Any]]): JSON-serialisable
            rules-based violations produced by the reviewer.

    Returns:
        dict[str, list[dict[str, Any]]]: Ordered grouped violations ready for
            UI display and workbook export.
    """

    grouped_violations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for violation in cleaned_violations_dicts:
        section_name = normalize_display_section_name(
            violation.get("section_name") or "Whole document"
        )
        grouped_violations[section_name].append(violation)

    # Walk the parsed report first so displayed sections follow document order.
    ordered_grouped_violations: dict[str, list[dict[str, Any]]] = {}
    seen_sections = set()
    for section_name in full_report:
        normalized_name = normalize_display_section_name(section_name)
        if normalized_name in seen_sections:
            continue
        seen_sections.add(normalized_name)
        if normalized_name in grouped_violations:
            ordered_grouped_violations[normalized_name] = grouped_violations[normalized_name]

    # Append any residual sections that were not present in the parsed mapping.
    for section_name, rows in grouped_violations.items():
        if section_name not in ordered_grouped_violations:
            ordered_grouped_violations[section_name] = rows

    return ordered_grouped_violations


def render_rules_tab(
    *,
    input_ready: bool,
    assessment: dict[str, Any] | None,
    uploaded_name: str,
) -> None:
    """
    Render the rules-based feedback tab.

    Args:
        input_ready (bool): Whether the uploaded file is in a supported format
            and ready for processing.
        assessment (dict[str, Any] | None): Parsed assessment dictionary shared
            by ``app.py``, or ``None`` when no supported upload is ready.
        uploaded_name (str): Original uploaded filename used for downloads.

    Returns:
        None: Value produced by this method.
    """

    st.subheader("Rules-based feedback")
    st.write("Run the deterministic checker suite on the uploaded assessment.")

    if st.button(
        "Generate feedback",
        key="generate_rules_feedback",
        type="primary",
        disabled=not input_ready,
    ):
        with st.spinner("Generating rules-based feedback..."):
            if assessment is None:
                raise ValueError("Rules-based feedback requires a parsed assessment dictionary.")

            # Rebuild the full report from the shared assessment dict, then run
            # the reviewer and keep both raw and cleaned outputs for different
            # downstream export/display needs.
            parser = AssessmentParser()
            reviewer = IUCNAssessmentReviewer()
            full_report = parser.parse(assessment)
            raw_violations = reviewer.review_full_report(full_report)
            raw_violations_dicts = [violation.to_dict() for violation in raw_violations]
            cleaned_violations = reviewer.clean_up_violations(list(raw_violations))
            cleaned_violations_dicts = [
                violation.to_dict() for violation in cleaned_violations
            ]

            ordered_grouped_violations = build_ordered_grouped_violations(
                full_report,
                cleaned_violations_dicts,
            )

            # Cache the tab output so reruns preserve the last generated review
            # until the user uploads a new document.
            st.session_state["rules_feedback"] = {
                "violations": cleaned_violations_dicts,
                "raw_violations": raw_violations_dicts,
                "grouped_violations": ordered_grouped_violations,
            }

    if st.session_state["rules_feedback"] is None:
        st.info("Click `Generate feedback` in this tab to run the rules-based reviewer.")
    else:
        feedback = st.session_state["rules_feedback"]
        cleaned_violations_dicts = feedback["violations"]
        grouped_violations = feedback["grouped_violations"]

        if not cleaned_violations_dicts:
            st.success("No rules-based violations were found for this document.")
        else:
            st.metric("Violations found", len(cleaned_violations_dicts))

            for section_name, rows in grouped_violations.items():
                # Keep the on-screen table focused on the reader-facing fields
                # most useful for triage inside the app.
                table = pd.DataFrame(
                    [
                        {
                            "Matched text": row.get("matched_text", ""),
                            "Context": row.get("matched_snippet", ""),
                            "Message": row.get("message", ""),
                            "Suggested fix": row.get("suggested_fix", ""),
                        }
                        for row in rows
                    ]
                )
                table = table[
                    [
                        "Matched text",
                        "Context",
                        "Message",
                        "Suggested fix",
                    ]
                ]

                error_label = "error" if len(rows) == 1 else "errors"
                with st.expander(f"{section_name} ({len(rows)} {error_label})", expanded=False):
                    st.dataframe(table, width="stretch", hide_index=True)

            st.download_button(
                label="Download rules feedback (JSON)",
                data=json.dumps(
                    feedback.get("raw_violations", []),
                    indent=2,
                    ensure_ascii=False,
                ).encode("utf-8"),
                file_name=f"{Path(uploaded_name).stem}_rules_feedback.json",
                mime="application/json",
            )
