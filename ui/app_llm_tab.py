"""LLM-tab UI helpers for the Streamlit app.

Purpose:
    This module renders the model-driven feedback tab. It receives the shared
    parsed assessment dictionary from ``app.py``, uses the shared sidebar
    model and API-key configuration, runs the simplified LLM reviewer, and
    formats the returned findings for display and JSON export. It also gates
    the primary action when the sidebar model configuration is incomplete.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import streamlit as st

from simplified_llm_api_script.llm_checker_v2 import ReviewDocumentError
from simplified_llm_api_script.llm_checker_v2 import provider_from_config
from simplified_llm_api_script.llm_checker_v2 import review_document


def _format_llm_error_message(error_text: str | None) -> str | None:
    """
    Convert raw provider errors into clearer user-facing messages.

    Args:
        error_text (str | None): Raw exception text captured from the LLM
            provider flow.

    Returns:
        str | None: Friendly message to display in the UI, or ``None`` when
            there is no error text.
    """

    if not error_text:
        return None

    lowered = error_text.lower()
    if "503" in lowered or "502" in lowered:
        return (
            "The selected OpenRouter model is temporarily unavailable or overloaded. "
            "Please try again later, or switch to a different model."
        )
    if "429" in lowered:
        return (
            "OpenRouter rate limit reached or quota exceeded for this model or API key. "
            "Please wait and try again, or switch model or API key."
        )
    return error_text


def render_llm_tab(
    *,
    input_ready: bool,
    assessment: dict[str, Any] | None,
    uploaded_name: str,
    selected_llm_label: str,
    selected_llm_config: dict[str, Any],
    custom_model_missing: bool,
) -> None:
    """
    Render the LLM feedback tab.

    Args:
        input_ready (bool): Whether the uploaded file is in a supported format
            and ready for processing.
        assessment (dict[str, Any] | None): Parsed assessment dictionary shared
            by ``app.py``, or ``None`` when no supported upload is ready.
        uploaded_name (str): Original uploaded filename used for downloads.
        selected_llm_label (str): Sidebar-selected LLM label used for the
            current tab run.
        selected_llm_config (dict[str, Any]): Fully resolved LLM config chosen
            in the shared sidebar.
        custom_model_missing (bool): Whether the custom-model branch is active
            but still missing a model slug.

    Returns:
        None: Value produced by this method.
    """

    llm_config_ready = bool(
        selected_llm_config.get("api_key") and selected_llm_config.get("model")
    )

    st.subheader("LLM feedback")
    st.write("Run the simplified LLM reviewer separately from the rules-based checks.")

    st.caption(
        f"Selected config: OpenRouter model `{selected_llm_config['model']}` "
        f"with reasoning `{'on' if selected_llm_config['reasoning_enabled'] else 'off'}`."
    )

    if custom_model_missing:
        st.info("Enter an OpenRouter model slug in the sidebar to enable the custom LLM option.")
    elif not llm_config_ready:
        st.info("Configure a valid API key and model in the sidebar to use this tab.")

    if st.button(
        "Generate feedback",
        key="generate_llm_feedback",
        type="primary",
        disabled=not input_ready or custom_model_missing or not llm_config_ready,
    ):
        with st.spinner("Generating LLM feedback..."):
            if assessment is None:
                raise ValueError("LLM feedback requires a parsed assessment dictionary.")

            llm_results = []
            llm_error = None

            try:
                # Build the provider from the shared sidebar config and run the
                # existing sequential review workflow over the shared input dict.
                provider = provider_from_config(selected_llm_config)
                llm_results = asyncio.run(
                    review_document(assessment, provider=provider, mode="sequential")
                )
            except ReviewDocumentError as exc:
                llm_error = str(exc)
                llm_results = exc.results
            except Exception as exc:
                llm_error = str(exc)

            # Preserve both the results and the execution metadata so the tab
            # can survive reruns and the download tab can export the output.
            st.session_state["llm_feedback"] = {
                "status": "success" if llm_error is None else "error",
                "model_label": selected_llm_label,
                "model_slug": selected_llm_config["model"],
                "error": llm_error,
                "display_error": _format_llm_error_message(llm_error),
                "llm_results": [result.model_dump() for result in llm_results],
            }

    if st.session_state["llm_feedback"] is None:
        st.info("Click `Generate feedback` in this tab to run the LLM workflow.")
    else:
        feedback = st.session_state["llm_feedback"]
        st.caption(
            f"Generated with `{feedback.get('model_label', 'Unknown')}` "
            f"using model `{feedback.get('model_slug', 'Unknown')}`."
        )
        if feedback.get("error"):
            st.warning(
                "The LLM review returned partial or empty results. "
                f"Last error: {feedback.get('display_error') or feedback['error']}"
            )

        if not feedback.get("llm_results"):
            st.info("No LLM rule results were returned.")

        for llm_result in feedback.get("llm_results", []):
            rule_name = llm_result.get("rule_name") or "unknown_rule"
            findings = llm_result.get("findings") or []
            with st.expander(
                f"{rule_name} ({len(findings)} findings)",
                expanded=False,
            ):
                if not findings:
                    st.write("No findings or external LLM failed")
                else:
                    # Show each finding as a short triage block rather than as
                    # a wide table because the response text can vary a lot.
                    for index, finding in enumerate(findings, start=1):
                        section_path = finding.get("section_path") or "Whole document"
                        severity = finding.get("severity") or "unknown"
                        st.markdown(
                            f"**{section_path} clause (Severity: {severity})**"
                        )
                        st.write(finding.get("issue") or "No issue provided.")
                        st.caption(
                            "Suggested fix: "
                            + (finding.get("suggestion") or "No suggestion provided.")
                        )
                        if index < len(findings):
                            st.divider()

        st.download_button(
            label="Download LLM feedback (JSON)",
            data=json.dumps(
                feedback.get("llm_results", []),
                indent=2,
                ensure_ascii=False,
            ).encode("utf-8"),
            file_name=f"{Path(uploaded_name).stem}_llm_feedback.json",
            mime="application/json",
        )
