"""Download-tab UI helpers for the Streamlit app.

Purpose:
    This module renders the Excel export tab and builds the downloadable
    workbook from whichever feedback outputs are currently available in
    Streamlit session state.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def build_downloadable_feedback_excel(
    rules_feedback: dict[str, Any] | None,
    llm_feedback: dict[str, Any] | None,
) -> bytes:
    """
    Return an Excel workbook for any available rules-based and/or LLM feedback.

    The workbook can contain:
    - a ``Rules`` sheet built from grouped rules-based violations
    - an ``LLM`` sheet built from the simplified LLM review findings

    Either payload may be omitted. The helper writes only the sheets for which
    feedback data is available.

    Args:
        rules_feedback (dict[str, Any] | None): Rules-based feedback payload
            stored in Streamlit session state, or ``None`` when no rules output
            has been generated.
        llm_feedback (dict[str, Any] | None): LLM feedback payload stored in
            Streamlit session state, or ``None`` when no LLM output has been
            generated.

    Returns:
        bytes: Excel workbook bytes ready to pass into
            ``st.download_button(...)``.
    """

    rules_columns = [
        "Section title",
        "Matched text",
        "Context",
        "Message",
        "Suggested fix",
    ]
    llm_columns = [
        "Rule name",
        "Report section",
        "Severity",
        "Feedback",
        "Suggestion",
    ]

    # Flatten the grouped rules output into one worksheet row per violation.
    rules_rows: list[dict[str, str]] = []
    grouped_violations = (rules_feedback or {}).get("grouped_violations") or {}
    for section_name, rows in grouped_violations.items():
        for row in rows:
            rules_rows.append(
                {
                    "Section title": section_name,
                    "Matched text": row.get("matched_text", ""),
                    "Context": row.get("matched_snippet", ""),
                    "Message": row.get("message", ""),
                    "Suggested fix": row.get("suggested_fix", ""),
                }
            )

    # Flatten the nested LLM rule/findings structure into one row per finding.
    llm_rows: list[dict[str, str]] = []
    if llm_feedback is not None:
        for llm_result in llm_feedback.get("llm_results", []):
            rule_name = llm_result.get("rule_name") or ""
            findings = llm_result.get("findings") or []
            for finding in findings:
                llm_rows.append(
                    {
                        "Rule name": rule_name,
                        "Report section": finding.get("section_path") or "",
                        "Severity": finding.get("severity") or "",
                        "Feedback": finding.get("issue") or "",
                        "Suggestion": finding.get("suggestion") or "",
                    }
                )

    # Only write the sheets for outputs that currently exist so the workbook
    # matches the feedback the user has actually generated.
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        if rules_feedback is not None:
            pd.DataFrame(rules_rows, columns=rules_columns).to_excel(
                writer,
                index=False,
                sheet_name="Rules",
            )

        if llm_feedback is not None:
            pd.DataFrame(llm_rows, columns=llm_columns).to_excel(
                writer,
                index=False,
                sheet_name="LLM",
            )

    excel_buffer.seek(0)
    return excel_buffer.getvalue()


def render_download_tab(*, uploaded_name: str) -> None:
    """
    Render the downloadable-feedback tab.

    Args:
        uploaded_name (str): Original uploaded filename used for the workbook
            download name.

    Returns:
        None: Value produced by this method.
    """

    st.subheader("Download feedback")
    st.write(
        "Generate a downloadable Excel file containing whichever feedback is ready "
        "(rules-based, LLM, or both)."
    )

    has_rules_output = st.session_state.get("rules_feedback") is not None
    has_llm_output = st.session_state.get("llm_feedback") is not None
    has_any_output = has_rules_output or has_llm_output

    status_col_1, status_col_2 = st.columns(2)
    with status_col_1:
        if has_rules_output:
            st.success("Rules-based feedback ready")
        else:
            st.warning("Rules-based feedback not ready")
    with status_col_2:
        if has_llm_output:
            st.success("LLM feedback ready")
        else:
            st.warning("LLM feedback not ready")
    st.caption(
        "At least one of these feedbacks must be ready for the download to be available."
    )

    if not has_any_output:
        st.download_button(
            label="Download feedback",
            data=b"",
            file_name=f"{Path(uploaded_name).stem or 'feedback'}_feedback.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            disabled=True,
        )
    else:
        excel_bytes = build_downloadable_feedback_excel(
            rules_feedback=st.session_state.get("rules_feedback") if has_rules_output else None,
            llm_feedback=st.session_state.get("llm_feedback") if has_llm_output else None,
        )
        st.download_button(
            label="Download feedback",
            data=excel_bytes,
            file_name=f"{Path(uploaded_name).stem or 'feedback'}_feedback.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
