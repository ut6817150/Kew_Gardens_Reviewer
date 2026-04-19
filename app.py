"""Streamlit entry point for the combined IUCN review app.

Purpose:
    This module owns the shared Streamlit page shell for the feedback tool.
    It delegates sidebar rendering and upload validation to the sidebar helper,
    caches the parsed assessment input once per uploaded file, and passes that
    shared dictionary into the rules-based, LLM, and RAG tabs. It is the
    orchestration layer for the Streamlit app rather than the home of
    tab-specific UI logic.
"""

import streamlit as st
from preprocessing.assessment_processor import parse_to_dict
from llm_rag.iv_inference.rag_runtime import init_rag_session_state
from llm_rag.iv_inference.rag_runtime import sync_rag_state_with_upload

from ui.app_download_tab import render_download_tab
from ui.app_config import CUSTOM_LLM_OPTION
from ui.app_config import CUSTOM_OPENROUTER_KEY_OPTION
from ui.app_config import MODEL_SPECS
from ui.app_config import OPENROUTER_KEY_OPTIONS
from ui.app_llm_tab import render_llm_tab
from ui.app_rag_tab import render_rag_tab
from ui.app_rules_tab import render_rules_tab
from ui.app_sidebar import render_sidebar_controls

# Import the shared sidebar/app option lists from the dedicated config module
# so this entry point stays focused on runtime orchestration.
# Configure the shared page shell before any widgets are created.
st.set_page_config(
    page_title="IUCN Assessment Feedback Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("IUCN Assessment Feedback Tool")

st.caption(
    "Use the arrow on the top left of the screen to access the sidebar for "
    "document upload, API key entry, and LLM selection. Upload one assessment, "
    "then run the rules-based, LLM, and RAG systems separately from the tabs below."
)

# Track upload-scoped outputs and the shared parsed assessment in session state
# so reruns preserve the last generated feedback until the user changes the
# uploaded document.
if "uploaded_file_signature" not in st.session_state:
    st.session_state["uploaded_file_signature"] = None
if "rules_feedback" not in st.session_state:
    st.session_state["rules_feedback"] = None
if "llm_feedback" not in st.session_state:
    st.session_state["llm_feedback"] = None
if "assessment_input_dict" not in st.session_state:
    st.session_state["assessment_input_dict"] = None
if "assessment_input_signature" not in st.session_state:
    st.session_state["assessment_input_signature"] = None

# The RAG runtime manages its own larger chat/session state payload.
init_rag_session_state(st.session_state)

# Render the shared sidebar once, then let the main app shell orchestrate
# parsing, cache resets, and tab-specific workflows from the returned state.
sidebar_state = render_sidebar_controls(
    openrouter_key_options=OPENROUTER_KEY_OPTIONS,
    custom_openrouter_key_option=CUSTOM_OPENROUTER_KEY_OPTION,
    model_specs=MODEL_SPECS,
    custom_llm_option=CUSTOM_LLM_OPTION,
)

tmp_path = sidebar_state.tmp_path
uploaded_name = sidebar_state.uploaded_name
file_signature = sidebar_state.file_signature
input_ready = sidebar_state.input_ready
selected_llm_label = sidebar_state.selected_llm_label
selected_llm_config = sidebar_state.selected_llm_config

# Clear all upload-scoped cached outputs when the user switches documents so
# results from an earlier file do not leak into the next review session.
if st.session_state["uploaded_file_signature"] != file_signature:
    st.session_state["uploaded_file_signature"] = file_signature
    st.session_state["rules_feedback"] = None
    st.session_state["llm_feedback"] = None
    st.session_state["assessment_input_dict"] = None
    st.session_state["assessment_input_signature"] = None

# Keep the RAG runtime's upload-specific caches aligned with the current file.
sync_rag_state_with_upload(st.session_state, file_signature)

assessment_input = None
if input_ready:
    # Parse the staged upload once per file signature and share the resulting
    # assessment dictionary across the tab render helpers.
    if st.session_state.get("assessment_input_signature") != file_signature:
        st.session_state["assessment_input_dict"] = parse_to_dict(str(tmp_path))
        st.session_state["assessment_input_signature"] = file_signature
    assessment_input = st.session_state.get("assessment_input_dict")

# Build the four top-level workflows from the shared page shell.
rules_tab, llm_tab, rag_tab, download_tab = st.tabs(
    ["Rules-based feedback", "LLM feedback", "RAG chat (prototype)", "Download feedback"]
)

# Render the deterministic rules-based review workflow.
with rules_tab:
    render_rules_tab(
        input_ready=input_ready,
        assessment=assessment_input,
        uploaded_name=uploaded_name,
    )

# Render the separate LLM review workflow.
with llm_tab:
    render_llm_tab(
        input_ready=input_ready,
        assessment=assessment_input,
        uploaded_name=uploaded_name,
        selected_llm_label=selected_llm_label,
        selected_llm_config=selected_llm_config,
        custom_model_missing=sidebar_state.custom_model_missing,
    )

# Render the prototype RAG chat workflow over the same parsed assessment and
# sidebar-selected model configuration.
with rag_tab:
    render_rag_tab(
        input_ready=input_ready,
        assessment=assessment_input,
        file_signature=file_signature,
        selected_llm_config=selected_llm_config,
    )

# Render the export tab for whichever feedback outputs are currently available.
with download_tab:
    render_download_tab(uploaded_name=uploaded_name)
