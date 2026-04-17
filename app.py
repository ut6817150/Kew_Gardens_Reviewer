"""Streamlit entry point for the combined IUCN review app.

Purpose:
    This module owns the shared Streamlit page shell for the feedback tool.
    It handles upload validation, caches the parsed assessment input once per
    uploaded file, and passes that shared dictionary into the rules-based, LLM,
    and RAG tabs so each workflow can focus on its own UI and processing.
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from preprocessing.assessment_processor import parse_to_dict
from llm_rag.iv_inference.rag_runtime import init_rag_session_state
from llm_rag.iv_inference.rag_runtime import sync_rag_state_with_upload

from ui.app_download_tab import render_download_tab
from ui.app_llm_tab import render_llm_tab
from ui.app_rag_tab import render_rag_tab
from ui.app_rules_tab import render_rules_tab

# Read the shared OpenRouter credential once so the tab helpers can reuse it.
OPENROUTER_API_KEY = os.getenv("OR_TOKEN")

# Keep the prototype RAG model configuration centralised at the app entry point.
RAG_LLM_CONFIG = {
    "base_url": "https://openrouter.ai/api/v1",
    "model": "openai/gpt-oss-120b:free",
    "api_key": OPENROUTER_API_KEY,
    "reasoning_enabled": True,
}

# Expose the preset models for the LLM feedback tab in one place.
LLM_TAB_CONFIGS = {
    "GPT OSS 120b (free and zero data retention)": {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "openai/gpt-oss-120b:free",
        "api_key": OPENROUTER_API_KEY,
        "reasoning_enabled": True,
    },
    "Qwen 3.5 Plus 02-15 (Paid and retains data)": {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "qwen/qwen3.5-plus-02-15",
        "api_key": OPENROUTER_API_KEY,
        "reasoning_enabled": True,
    },
}
CUSTOM_LLM_OPTION = "configure your own LLM"

# Configure the shared page shell before any widgets are created.
st.set_page_config(page_title="IUCN Assessment Feedback Tool", layout="wide")
st.title("IUCN Assessment Feedback Tool")
st.caption(
    "Upload one assessment, then run the rules-based, LLM, and RAG systems separately "
    "from the tabs below."
)

# Track upload-scoped outputs and the shared parsed assessment in session state.
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

# Accept the document formats currently supported by the app workflows.
uploaded = st.file_uploader(
    "Upload a .docx, .html or .htm file",
    type=["docx", "html", "doc"],
)

tmp_path: Path | None = None
uploaded_ext: str | None = None
uploaded_name = ""
file_signature = None
input_ready = False

# Validate the upload and materialise a temporary file for the parser.
if uploaded:
    uploaded_name = uploaded.name
    uploaded_ext = Path(uploaded.name).suffix.lower()
    file_signature = f"{uploaded.name}:{len(uploaded.getbuffer())}"

    # Legacy `.doc` uploads must be converted before they can be parsed safely.
    if uploaded_ext == ".doc":
        st.error(
            "The feedback tool only supports .docx files. Please convert .doc to .docx and re-upload. Or, upload a HTML file."
        )
    # Reject any other unsupported extension early so the tabs stay disabled.
    elif uploaded_ext not in [".docx", ".html"]:
        st.error("The feedback tool only supports .docx or HTML files.")

    # Persist the supported upload to a temporary path because the parser
    # expects a filesystem location rather than an in-memory upload object.
    elif uploaded_ext == ".docx" or uploaded_ext == ".html":
        tmp_dir = Path(tempfile.mkdtemp(prefix="upload_"))
        safe_name = Path(uploaded.name).name
        tmp_path = tmp_dir / safe_name
        tmp_path.write_bytes(uploaded.getbuffer())

        st.success(f"Uploaded: {uploaded.name}")

# Clear all upload-scoped cached outputs when the user switches documents.
if st.session_state["uploaded_file_signature"] != file_signature:
    st.session_state["uploaded_file_signature"] = file_signature
    st.session_state["rules_feedback"] = None
    st.session_state["llm_feedback"] = None
    st.session_state["assessment_input_dict"] = None
    st.session_state["assessment_input_signature"] = None

# Keep the RAG runtime's upload-specific caches aligned with the current file.
sync_rag_state_with_upload(st.session_state, file_signature)

# Enable the downstream tabs only when a supported upload has been staged.
if uploaded is None:
    st.info("Upload a file to enable both feedback systems.")
elif tmp_path is not None:
    input_ready = True

assessment_input = None
if input_ready:
    # Parse the uploaded document once per file signature and share the
    # resulting assessment dictionary across the tab render helpers.
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
        llm_tab_configs=LLM_TAB_CONFIGS,
        custom_llm_option=CUSTOM_LLM_OPTION,
        openrouter_api_key=OPENROUTER_API_KEY,
    )

# Render the prototype RAG chat workflow over the same parsed assessment.
with rag_tab:
    render_rag_tab(
        input_ready=input_ready,
        assessment=assessment_input,
        file_signature=file_signature,
        rag_llm_config=RAG_LLM_CONFIG,
    )

# Render the export tab for whichever feedback outputs are currently available.
with download_tab:
    render_download_tab(uploaded_name=uploaded_name)
