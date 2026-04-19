"""RAG-tab UI helpers for the Streamlit app.

Purpose:
    This module renders the prototype retrieval-augmented chat workflow. It
    reuses the shared parsed assessment dictionary prepared by ``app.py``,
    derives the report blocks needed by the draft-store pipeline, and manages
    the chat-specific UI around the existing RAG runtime helpers using the
    shared sidebar-selected model and API-key configuration. It adapts the
    shared sidebar config into the form expected by the RAG runtime and gates
    the chat input when the config is incomplete.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from llm_rag.iv_inference.rag_runtime import answer_rag_question
from llm_rag.iv_inference.rag_runtime import build_rag_debug_payload
from llm_rag.iv_inference.rag_runtime import build_report_from_assessment
from llm_rag.iv_inference.rag_runtime import ensure_draft_store_from_report


OPENROUTER_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"


def _build_rag_llm_config(selected_llm_config: dict[str, Any]) -> dict[str, Any]:
    """
    Adapt the shared selected LLM config for the RAG runtime.

    Args:
        selected_llm_config (dict[str, Any]): Sidebar-selected config shared by
            the app shell.

    Returns:
        dict[str, Any]: RAG-ready config with the provider base URL normalized
            to the format expected by the inference runtime.
    """

    rag_base_url = selected_llm_config["base_url"]
    if rag_base_url.endswith(OPENROUTER_CHAT_COMPLETIONS_SUFFIX):
        rag_base_url = rag_base_url[: -len(OPENROUTER_CHAT_COMPLETIONS_SUFFIX)]

    return {
        "base_url": rag_base_url,
        "model": selected_llm_config["model"],
        "api_key": selected_llm_config["api_key"],
        "reasoning_enabled": selected_llm_config["reasoning_enabled"],
    }


def render_rag_tab(
    *,
    input_ready: bool,
    assessment: dict[str, Any] | None,
    file_signature: str | None,
    selected_llm_config: dict[str, Any],
) -> None:
    """
    Render the RAG chat tab.

    Args:
        input_ready (bool): Whether the uploaded file is in a supported format
            and ready for processing.
        assessment (dict[str, Any] | None): Parsed assessment dictionary shared
            by ``app.py``, or ``None`` when no supported upload is ready.
        file_signature (str | None): Stable signature for the current upload.
        selected_llm_config (dict[str, Any]): Shared sidebar-selected LLM
            configuration used to derive the RAG runtime config.

    Returns:
        None: Value produced by this method.
    """

    rag_llm_config = _build_rag_llm_config(selected_llm_config)
    rag_config_ready = bool(rag_llm_config.get("api_key") and rag_llm_config.get("model"))

    st.subheader("RAG chat")
    st.write("Ask grounded questions about the uploaded assessment and the IUCN reference documents.")
    st.caption(
        "For each question, the app fetches deterministic threshold facts, retrieves reference evidence, "
        "retrieves the top draft chunks, and sends that combined prompt to the external LLM when configured."
    )
    show_debug = st.checkbox(
        "Show retrieval debug output",
        value=False,
        key="rag_show_debug",
    )

    if not input_ready:
        st.info("Upload a file to enable the RAG chat.")
    else:
        if assessment is None:
            raise ValueError("RAG chat requires a parsed assessment dictionary.")

        # Rebuild the draft-store inputs only when the upload changes so chat
        # turns can reuse the same cached parsed report and vector store.
        if st.session_state.get("rag_assessment_input_signature") != file_signature:
            st.session_state["rag_assessment_input_dict"] = assessment
            st.session_state["rag_assessment_input_signature"] = file_signature
            st.session_state["rag_report_dict"] = build_report_from_assessment(assessment)
            ensure_draft_store_from_report(
                st.session_state,
                st.session_state.get("rag_report_dict"),
                file_signature,
            )

        if st.button(
            "Clear RAG chat",
            key="clear_rag_chat",
            disabled=not st.session_state["rag_messages"],
            width="stretch",
        ):
            st.session_state["rag_messages"] = []
            st.rerun()

        if rag_config_ready and rag_llm_config["base_url"].strip():
            st.caption(
                f"External LLM uses the sidebar-selected model `{rag_llm_config['model']}`."
            )
        else:
            st.caption(
                "External LLM is not fully configured yet. Check the sidebar model and API key selection."
            )
            st.info("Configure a valid API key and model in the sidebar to use RAG chat.")

        if st.session_state.get("rag_draft_store") is None:
            st.caption("RAG context is prepared automatically when a new file is uploaded.")
        else:
            st.caption(
                f"Draft store ready: {len(st.session_state['rag_draft_store'])} chunks "
                f"from {len(st.session_state.get('rag_report_dict', {}))} parsed report blocks"
            )

        chat_history = st.container(height=520, border=True)
        with chat_history:
            if not st.session_state["rag_messages"]:
                st.caption(
                    "Start the conversation below. The chat history stays in this pane so the input box remains in place."
                )

            for message in st.session_state["rag_messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant":
                        # Surface a compact summary of how the answer was
                        # produced so the prototype remains easier to debug.
                        route = message.get("route", "unknown")
                        used_external_llm = "yes" if message.get("used_external_llm") else "no"
                        deterministic_used = "yes" if message.get("deterministic_answer") else "no"
                        reference_count = message.get("reference_count", 0)
                        draft_count = message.get("draft_count", 0)
                        st.caption(
                            f"Route: `{route}` | External LLM: `{used_external_llm}` | "
                            f"Deterministic facts: `{deterministic_used}` | "
                            f"Reference chunks: `{reference_count}` | Draft chunks: `{draft_count}`"
                        )
                    if show_debug and message["role"] == "assistant" and message.get("debug"):
                        with st.expander("Debug details", expanded=False):
                            if message.get("debug_error"):
                                st.error(message["debug_error"])
                            st.code(message["debug"], language="json")

        user_prompt = st.chat_input(
            "Ask about the uploaded assessment or the IUCN requirements.",
            key="rag_chat_input",
            disabled=not rag_config_ready,
        )

        if user_prompt:
            # Persist the user turn immediately so reruns keep the chat log in
            # the same order as the rendered conversation.
            st.session_state["rag_messages"].append({"role": "user", "content": user_prompt})

            with chat_history:
                with st.chat_message("user"):
                    st.markdown(user_prompt)

                with st.chat_message("assistant"):
                    with st.status("Running RAG inference...", expanded=True) as status:
                        status.write("Loading the uploaded draft context.")
                        draft_store = ensure_draft_store_from_report(
                            st.session_state,
                            st.session_state.get("rag_report_dict"),
                            file_signature,
                        )

                        status.write("Fetching deterministic facts, reference evidence, and draft evidence.")
                        response = answer_rag_question(
                            query=user_prompt,
                            draft_store=draft_store,
                            llm_config=rag_llm_config,
                        )

                        if response.get("llm_configured"):
                            status.write("Sending the combined prompt to the external LLM.")
                        else:
                            status.write("No external LLM is configured, so the chat returns a configuration message.")
                        status.update(label="RAG inference complete", state="complete", expanded=False)

                    st.markdown(response["answer"])
                    st.caption(
                        f"Route: `{response.get('route', 'unknown')}` | "
                        f"LLM configured: `{'yes' if response.get('llm_configured') else 'no'}` | "
                        f"External LLM: `{'yes' if response.get('used_external_llm') else 'no'}` | "
                        f"Deterministic facts: `{'yes' if response.get('deterministic_answer') else 'no'}` | "
                        f"Reference chunks: `{len(response.get('reference_payload', {}).get('results') or [])}` | "
                        f"Draft chunks: `{len(response.get('draft_hits') or [])}`"
                    )

                    debug_data = build_rag_debug_payload(response)
                    debug_payload = json.dumps(debug_data, ensure_ascii=False, indent=2)
                    if show_debug:
                        with st.expander("Debug details", expanded=False):
                            if debug_data.get("request_error"):
                                st.error(debug_data["request_error"])
                            st.code(debug_payload, language="json")

            # Save the assistant turn after rendering so the chat history and
            # debug payload are both available on the next rerun.
            st.session_state["rag_messages"].append(
                {
                    "role": "assistant",
                    "content": response["answer"],
                    "debug": debug_payload,
                    "debug_error": debug_data.get("request_error"),
                    "route": response.get("route"),
                    "llm_configured": response.get("llm_configured"),
                    "used_external_llm": response.get("used_external_llm"),
                    "deterministic_answer": response.get("deterministic_answer"),
                    "reference_count": len(response.get("reference_payload", {}).get("results") or []),
                    "draft_count": len(response.get("draft_hits") or []),
                }
            )
            st.rerun()
