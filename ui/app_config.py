"""Shared UI configuration for the Streamlit app.

Purpose:
    This module stores the static option lists and labels used by the sidebar
    and app shell. Keeping these definitions in one place keeps ``app.py``
    focused on orchestration rather than embedded configuration data and gives
    the sidebar helper one shared source of truth for preset labels.
"""

from __future__ import annotations

# Shared preset OpenRouter keys offered in the sidebar. The values are env-var
# names rather than raw keys so the app can keep team-managed keys out of the
# codebase.
OPENROUTER_KEY_OPTIONS = {
    "Steve Bachman's Key": "Openrouter_API_key_Steve_Bachman",
    "Jack Plummer's Key": "JackAPIKey",
    "Khalid Alahmadi's Key": "OR_TOKEN",
}
# Session-only custom API-key option shown alongside the preset team keys.
CUSTOM_OPENROUTER_KEY_OPTION = "Enter your own OpenRouter API key"

# Shared model definitions used by both the LLM feedback and RAG workflows.
MODEL_SPECS = {
    "GPT OSS 120b (free and zero data retention)": {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "openai/gpt-oss-120b:free",
        "reasoning_enabled": True,
    },
    "Qwen 3.5 Plus 02-15 (Paid and retains data)": {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "qwen/qwen3.5-plus-02-15",
        "reasoning_enabled": True,
    },
}
# Custom-model option shown in the sidebar model selector.
CUSTOM_LLM_OPTION = "configure your own LLM"
