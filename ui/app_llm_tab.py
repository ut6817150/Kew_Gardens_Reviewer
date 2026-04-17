"""LLM-tab UI helpers for the Streamlit app.

Purpose:
    This module renders the model-driven feedback tab. It receives the shared
    parsed assessment dictionary from ``app.py``, lets the user choose between
    preset and custom OpenRouter models, runs the simplified LLM reviewer, and
    formats the returned findings for display and JSON export.
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


def render_llm_tab(
    *,
    input_ready: bool,
    assessment: dict[str, Any] | None,
    uploaded_name: str,
    llm_tab_configs: dict[str, dict[str, Any]],
    custom_llm_option: str,
    openrouter_api_key: str | None,
) -> None:
    """
    Render the LLM feedback tab.

    Args:
        input_ready (bool): Whether the uploaded file is in a supported format
            and ready for processing.
        assessment (dict[str, Any] | None): Parsed assessment dictionary shared
            by ``app.py``, or ``None`` when no supported upload is ready.
        uploaded_name (str): Original uploaded filename used for downloads.
        llm_tab_configs (dict[str, dict[str, Any]]): Preset LLM configurations
            shown in the model dropdown.
        custom_llm_option (str): Label for the custom-model dropdown option.
        openrouter_api_key (str | None): OpenRouter API key reused by the
            custom-model branch.

    Returns:
        None: Value produced by this method.
    """

    st.subheader("LLM feedback")
    st.write("Run the simplified LLM reviewer separately from the rules-based checks.")

    # Keep model selection local to the tab while reusing the app-level config
    # payloads passed in from ``app.py``.
    selected_llm_label = st.selectbox(
        "Choose LLM",
        options=[*llm_tab_configs.keys(), custom_llm_option],
        index=0,
        key="llm_tab_model_choice",
    )

    custom_llm_model = ""
    if selected_llm_label == custom_llm_option:
        # The custom branch still uses the shared OpenRouter key and reasoning
        # settings; only the model slug is user-specified.
        custom_llm_model = st.text_input(
            "Enter OpenRouter model slug",
            key="llm_custom_model_slug",
            placeholder="e.g. openai/gpt-oss-120b:free",
        ).strip()
        selected_llm_config = {
            "base_url": "https://openrouter.ai/api/v1/chat/completions",
            "model": custom_llm_model,
            "api_key": openrouter_api_key,
            "reasoning_enabled": True,
        }
    else:
        selected_llm_config = llm_tab_configs[selected_llm_label]

    st.caption(
        f"Selected config: OpenRouter model `{selected_llm_config['model']}` "
        f"with reasoning `{'on' if selected_llm_config['reasoning_enabled'] else 'off'}`."
    )

    if selected_llm_label == custom_llm_option and not custom_llm_model:
        st.info("Enter an OpenRouter model slug to enable the custom LLM option.")

    if st.button(
        "Generate feedback",
        key="generate_llm_feedback",
        type="primary",
        disabled=not input_ready or (selected_llm_label == custom_llm_option and not custom_llm_model),
    ):
        with st.spinner("Generating LLM feedback..."):
            if assessment is None:
                raise ValueError("LLM feedback requires a parsed assessment dictionary.")

            llm_results = []
            llm_error = None

            try:
                # Build the provider from the selected config and run the
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
                f"Last error: {feedback['error']}"
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
