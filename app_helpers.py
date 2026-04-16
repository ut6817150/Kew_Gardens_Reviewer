"""Helpers shared by the Streamlit app.

Purpose:
    This module contains small workbook-building helpers used by ``app.py`` so
    the tab rendering logic can stay focused on UI flow rather than Excel
    export details.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd


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
