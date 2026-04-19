"""Sidebar UI helpers for the Streamlit app.

Purpose:
    This module renders the shared sidebar controls used across the app. It
    owns document upload, upload validation, temporary staging of supported
    files, API key selection, model selection, and the user-facing validation
    messages that belong directly under those controls. The returned
    ``SidebarState`` is shared by both the LLM-feedback and RAG tabs.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st


@dataclass
class SidebarState:
    """
    Store the shared sidebar selections needed by ``app.py``.

    Attributes:
        tmp_path (Path | None): Temporary path containing the uploaded file, or
            ``None`` when no supported upload is ready yet.
        uploaded_name (str): Original uploaded filename, or an empty string
            when no file is available.
        file_signature (str | None): Stable identifier for the current upload
            used to reset upload-scoped caches.
        input_ready (bool): Whether a supported upload has been validated and
            staged to disk for downstream parsing.
        selected_openrouter_api_key (str | None): Resolved OpenRouter API key
            selected in the sidebar, either from a preset env var or from the
            current session's custom API-key field.
        selected_llm_label (str): User-facing label for the selected model
            option.
        selected_llm_config (dict[str, Any]): Fully resolved config for the
            currently selected model, shared across the LLM and RAG workflows.
        custom_model_missing (bool): Whether the custom-model branch is active
            but still missing a model slug.
    """

    tmp_path: Path | None
    uploaded_name: str
    file_signature: str | None
    input_ready: bool
    selected_openrouter_api_key: str | None
    selected_llm_label: str
    selected_llm_config: dict[str, Any]
    custom_model_missing: bool


def render_sidebar_controls(
    *,
    openrouter_key_options: dict[str, str],
    custom_openrouter_key_option: str,
    model_specs: dict[str, dict[str, Any]],
    custom_llm_option: str,
) -> SidebarState:
    """
    Render the shared sidebar controls and return the selected values.

    Args:
        openrouter_key_options (dict[str, str]): Mapping from display labels to
            environment-variable names for the preset API keys.
        custom_openrouter_key_option (str): Label for the session-only custom
            API-key option.
        model_specs (dict[str, dict[str, Any]]): Shared preset model
            configurations available to the app.
        custom_llm_option (str): Label for the custom-model option.

    Returns:
        SidebarState: Structured sidebar selections and validated upload state
            used by the main app shell.
    """

    tmp_path: Path | None = None
    uploaded_name = ""
    file_signature = None
    input_ready = False

    with st.sidebar:
        st.subheader("Inputs")
        st.caption(
            "Choose a document, API key, and LLM here, then run each "
            "workflow from the tabs in the main workspace."
        )
        st.markdown("##### <u>Upload assessment</u>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload assessment",
            type=["docx", "html", "doc"],
            label_visibility="collapsed",
        )
        # Keep upload validation and readiness messaging anchored directly
        # under the uploader instead of letting it fall below the other
        # sidebar selectors.
        upload_feedback_placeholder = st.empty()
        st.markdown("##### <u>Choose API key</u>", unsafe_allow_html=True)
        selected_key_label = st.selectbox(
            "Choose API key",
            options=[*openrouter_key_options.keys(), custom_openrouter_key_option],
            index=2,
            key="openrouter_api_key_choice",
            label_visibility="collapsed",
        )
        # Keep key-selection errors anchored directly under the key selector.
        api_key_feedback_placeholder = st.empty()
        custom_openrouter_api_key = ""
        if selected_key_label == custom_openrouter_key_option:
            custom_openrouter_api_key = st.text_input(
                "Enter your own API key",
                key="custom_openrouter_api_key",
                type="password",
                placeholder="sk-or-v1-...",
            ).strip()

        # The selected model config is shared by both the LLM-feedback and RAG
        # workflows, so it lives alongside the shared API-key controls.
        st.markdown("##### <u>Choose model</u>", unsafe_allow_html=True)
        selected_llm_label = st.selectbox(
            "Choose model",
            options=[*model_specs.keys(), custom_llm_option],
            index=0,
            key="sidebar_llm_model_choice",
            label_visibility="collapsed",
        )
        custom_llm_model = ""
        if selected_llm_label == custom_llm_option:
            custom_llm_model = st.text_input(
                "Enter OpenRouter model slug",
                key="sidebar_custom_model_slug",
                placeholder="e.g. openai/gpt-oss-120b:free",
            ).strip()

    if uploaded:
        uploaded_name = uploaded.name
        uploaded_ext = Path(uploaded.name).suffix.lower()
        file_signature = f"{uploaded.name}:{len(uploaded.getbuffer())}"

        # Validate supported document types before the app shell attempts to
        # parse the staged file.
        if uploaded_ext == ".doc":
            upload_feedback_placeholder.error(
                "The feedback tool only supports .docx files. Please convert .doc to .docx and re-upload. Or, upload a HTML file."
            )
        elif uploaded_ext not in [".docx", ".html"]:
            upload_feedback_placeholder.error(
                "The feedback tool only supports .docx or HTML files."
            )
        else:
            tmp_dir = Path(tempfile.mkdtemp(prefix="upload_"))
            safe_name = Path(uploaded.name).name
            tmp_path = tmp_dir / safe_name
            tmp_path.write_bytes(uploaded.getbuffer())
            input_ready = True
    else:
        upload_feedback_placeholder.caption(
            "Upload a file to enable the rules-based, LLM, and RAG workflows."
        )

    if selected_key_label == custom_openrouter_key_option:
        selected_openrouter_api_key = custom_openrouter_api_key or None
        if not selected_openrouter_api_key:
            api_key_feedback_placeholder.error(
                "Enter your own API key to use the custom API-key option."
            )
    else:
        # Preset key labels resolve to env vars so the app can keep a few
        # shared team-managed keys alongside the session-only custom option.
        selected_key_env_var = openrouter_key_options[selected_key_label]
        selected_openrouter_api_key = os.getenv(selected_key_env_var)
        if not selected_openrouter_api_key:
            api_key_feedback_placeholder.error(
                f"The selected OpenRouter API key, `{selected_key_label}`, is not configured "
                "in the environment."
            )

    if selected_llm_label == custom_llm_option:
        # The custom-model branch still assumes OpenRouter; only the model slug
        # changes while the base URL and reasoning settings stay fixed.
        selected_llm_config = {
            "base_url": "https://openrouter.ai/api/v1/chat/completions",
            "model": custom_llm_model,
            "api_key": selected_openrouter_api_key,
            "reasoning_enabled": True,
        }
    else:
        selected_llm_config = {
            **model_specs[selected_llm_label],
            "api_key": selected_openrouter_api_key,
        }

    return SidebarState(
        tmp_path=tmp_path,
        uploaded_name=uploaded_name,
        file_signature=file_signature,
        input_ready=input_ready,
        selected_openrouter_api_key=selected_openrouter_api_key,
        selected_llm_label=selected_llm_label,
        selected_llm_config=selected_llm_config,
        custom_model_missing=(
            selected_llm_label == custom_llm_option and not custom_llm_model
        ),
    )
