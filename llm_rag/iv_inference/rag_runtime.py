"""RAG runtime orchestration for uploaded assessment review.

Purpose:
    This module coordinates the inference-time RAG workflow. It combines
    deterministic threshold facts, reference retrieval, uploaded-draft
    retrieval, prompt assembly, optional external LLM calls, and debug-payload
    formatting for the Streamlit app and evaluation notebooks.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover - depends on local environment
    OpenAI = None
    OPENAI_IMPORT_ERROR = exc
else:
    OPENAI_IMPORT_ERROR = None

SCRIPT_DIR = Path(__file__).resolve().parent
LLM_RAG_DIR = SCRIPT_DIR.parent
REFERENCE_DIR = LLM_RAG_DIR / "iii_vector_db"

RAG_STATE_DEFAULTS: dict[str, Any] = {
    "rag_messages": [],
    "rag_draft_store": None,
    "rag_draft_signature": None,
    "rag_assessment_input_dict": None,
    "rag_assessment_input_signature": None,
    "rag_report_dict": None,
}
RAG_UPLOAD_RESET_KEYS = tuple(RAG_STATE_DEFAULTS)

SYSTEM_MESSAGE = "You answer using only the retrieved evidence provided."
DEFAULT_REFERENCE_EXCERPT_LIMIT = 4
DEFAULT_PARENT_TEXT_LIMIT = 700
DEFAULT_DRAFT_TEXT_LIMIT = 800


def _load_module(module_name: str, file_path: Path):
    """Import one local helper module from a file path.

    The RAG app loads several inference helpers dynamically so they can live in
    the `llm_rag` package without requiring package-style imports from the repo
    root. This helper registers the loaded module in ``sys.modules`` so later
    imports can reuse it normally.
    """
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_openai_base_url(base_url: str) -> str:
    """Normalize an OpenAI-compatible base URL.

    Some configs may already include the full ``/chat/completions`` suffix,
    while the OpenAI client expects the base API URL. This helper trims the
    suffix when present and otherwise leaves the URL untouched.
    """
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    return url


def _build_llm_request_kwargs(prompt: str, model: str, reasoning_enabled: bool) -> dict[str, Any]:
    """Build the chat-completions payload used for the external LLM request.

    The request always includes:
    - one system message
    - one user message containing the full grounded prompt
    - a low temperature

    When reasoning is enabled, the helper also passes the provider-specific
    reasoning configuration through ``extra_body``.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
    }
    if reasoning_enabled:
        kwargs["extra_body"] = {"reasoning": {"enabled": True}}
    return kwargs


def _completion_to_dict(completion: Any) -> dict[str, Any]:
    """Convert SDK completion objects into a plain dictionary.

    The OpenAI client may return:
    - native dictionaries
    - Pydantic-style objects with ``model_dump()``
    - older objects with ``to_dict()``

    This helper normalizes those shapes before later response parsing.
    """
    if isinstance(completion, dict):
        return completion
    if hasattr(completion, "model_dump"):
        return completion.model_dump()
    if hasattr(completion, "to_dict"):
        return completion.to_dict()
    if hasattr(completion, "model_dump_json"):
        return json.loads(completion.model_dump_json())
    raise TypeError("Unsupported completion response type")


def _llm_error_result(error: Any) -> dict[str, Any]:
    """Return a normalized error payload for failed external LLM requests.

    The rest of the runtime expects model responses to share one shape, even
    on failure, so this helper fills the missing output fields with ``None``
    and records the error text in one predictable key.
    """
    return {
        "output": None,
        "thinking": None,
        "reasoning_details": None,
        "error": str(error),
    }


def _stringify_response_text(value: Any) -> str | None:
    """Convert nested response content fragments into one optional text string.

    Different providers can return message content as:
    - one plain string
    - a list of structured parts
    - nested dictionaries

    This helper collapses those shapes into one trimmed text string whenever
    possible.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = _stringify_response_text(item)
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts)
        return None
    if isinstance(value, dict):
        for key in ("text", "content", "summary"):
            text = _stringify_response_text(value.get(key))
            if text:
                return text
        return None
    text = str(value).strip()
    return text or None


def _extract_reasoning_text(message: dict[str, Any]) -> str | None:
    """Extract any reasoning text returned alongside the assistant message.

    The runtime first checks for a direct ``reasoning`` field. If that is not
    available, it falls back to ``reasoning_details`` and joins supported
    summary/text blocks into one debug-friendly string.
    """
    reasoning_text = _stringify_response_text(message.get("reasoning"))
    if reasoning_text:
        return reasoning_text

    reasoning_details = message.get("reasoning_details") or []
    if not isinstance(reasoning_details, list):
        return None

    parts: list[str] = []
    for detail in reasoning_details:
        if not isinstance(detail, dict):
            continue
        detail_type = str(detail.get("type", ""))
        if detail_type == "reasoning.summary":
            summary = _stringify_response_text(detail.get("summary"))
            if summary:
                parts.append(f"Summary: {summary}")
        elif detail_type == "reasoning.text":
            text = _stringify_response_text(detail.get("text"))
            if text:
                parts.append(text)
        elif detail_type == "reasoning.encrypted":
            parts.append("[Encrypted reasoning block]")

    if not parts:
        return None
    return "\n\n".join(parts)


def _extract_message_parts(response_json: dict[str, Any]) -> dict[str, Any] | None:
    """Pull output text, reasoning text, and reasoning metadata from one completion response.

    This helper reads only the first returned choice and normalizes it into the
    fields used by the rest of the app:
    - ``output``
    - ``thinking``
    - ``reasoning_details``
    """
    choices = response_json.get("choices") or []
    if not choices:
        return None

    message = choices[0].get("message") or {}
    output = _stringify_response_text(message.get("content"))
    thinking = _extract_reasoning_text(message)
    reasoning_details = message.get("reasoning_details")

    if output is None and thinking is None and not reasoning_details:
        return None

    return {
        "output": output,
        "thinking": thinking,
        "reasoning_details": reasoning_details,
    }


def _build_answer_text(
    llm_configured: bool,
    llm_result: dict[str, Any] | None,
) -> str:
    """Choose the final answer text shown in the UI.

    This helper translates several runtime states into one visible answer:
    - no LLM config
    - request failure
    - empty model output
    - normal assistant output
    """
    if not llm_configured:
        return "External LLM not configured"
    if llm_result is None:
        return "External LLM request failed"
    if llm_result.get("error"):
        return f"External LLM request failed: {llm_result['error']}"
    if not llm_result.get("output"):
        return "External LLM returned no final answer"
    return llm_result["output"]


def _build_model_response_steps(
    model_response: dict[str, Any],
    answer_text: str,
) -> list[dict[str, Any]]:
    """Split the model response into debug-friendly thinking and output steps.

    The returned list is used by the debug payload so the UI can show model
    reasoning separately from the final answer when the provider exposes it.
    """
    return [
        {"step": "thinking", "content": model_response.get("thinking")},
        {"step": "output", "content": model_response.get("output") or answer_text},
    ]


def build_rag_debug_payload(response: dict[str, Any]) -> dict[str, Any]:
    """Build the structured debug payload shown in the RAG tab.

    The payload includes:
    - routing information
    - deterministic threshold output
    - retrieval subqueries
    - the answer scaffold
    - retrieved reference evidence excerpts
    - retrieved draft hits
    - the full final prompt
    - model response steps
    - any request failure details
    """
    reference_payload = response.get("reference_payload", {})
    model_response = response.get("model_response") or {}
    answer_text = str(response.get("answer") or "")
    model_response_error = model_response.get("error")
    return {
        "route": response.get("route"),
        "llm_configured": response.get("llm_configured"),
        "used_external_llm": response.get("used_external_llm"),
        "deterministic_answer": response.get("deterministic_answer"),
        "request_failed": bool(model_response_error),
        "request_error": model_response_error,
        "subqueries": reference_payload.get("subqueries"),
        "answer_scaffold": reference_payload.get("answer_scaffold"),
        "retrieved_reference_evidence_excerpts": build_reference_result_lines(reference_payload),
        "draft_hits": response.get("draft_hits"),
        "full_final_prompt": response.get("prompt"),
        "model_response_steps": _build_model_response_steps(model_response, answer_text),
        "model_response_reasoning_details": model_response.get("reasoning_details"),
    }


_draft_module = _load_module("draft_retrieval_local", SCRIPT_DIR / "draft_retrieval.py")
_parser_module = _load_module("inference_assessment_parser_local", SCRIPT_DIR / "inference_assessment_parser.py")
_load_module("embedding_loader", REFERENCE_DIR / "embedding_loader.py")
_threshold_module = _load_module("threshold_lookup", REFERENCE_DIR / "threshold_lookup.py")
_retrieval_module = _load_module("retrieval_engine_local", REFERENCE_DIR / "retrieval_engine.py")

build_draft_store_from_report = _draft_module.build_draft_store_from_report
retrieve_from_draft = _draft_module.retrieve_from_draft
InferenceAssessmentParser = _parser_module.InferenceAssessmentParser
answer_query = _retrieval_module.answer_query
answer_threshold_query = _threshold_module.answer_threshold_query
is_threshold_query = _threshold_module.is_threshold_query


def init_rag_session_state(session_state: dict[str, Any]) -> None:
    """Populate any missing Streamlit session-state keys used by the RAG workflow.

    The defaults cover:
    - visible chat history
    - cached uploaded-draft state
    - parsed draft input
    - cached section-level report output
    """
    for key, value in RAG_STATE_DEFAULTS.items():
        session_state.setdefault(key, value.copy() if isinstance(value, list) else value)


def sync_rag_state_with_upload(session_state: dict[str, Any], file_signature: str | None) -> None:
    """Reset cached RAG state when the uploaded draft file changes.

    This ensures old draft chunks, parsed report content, and prior draft
    signatures do not leak across uploads.
    """
    session_state.setdefault("rag_upload_signature", None)
    if session_state["rag_upload_signature"] == file_signature:
        return

    session_state["rag_upload_signature"] = file_signature
    for key in RAG_UPLOAD_RESET_KEYS:
        default_value = RAG_STATE_DEFAULTS[key]
        session_state[key] = default_value.copy() if isinstance(default_value, list) else default_value


def build_report_from_assessment(assessment: dict[str, Any]) -> dict[str, str]:
    """Convert the parsed assessment tree into the section-level report used for inference.

    The heavy lifting is delegated to ``InferenceAssessmentParser`` so the
    runtime can work from a flatter, retrieval-friendly report structure.
    """
    return InferenceAssessmentParser().parse(assessment)


def ensure_draft_store_from_report(
    session_state: dict[str, Any],
    report_dict: dict[str, str] | None,
    report_signature: str | None,
) -> list[dict[str, Any]]:
    """Build and cache the draft retrieval store for the current uploaded report.

    If the report signature matches the cached one, the existing draft store is
    reused. Otherwise the helper rebuilds the store and updates the cached
    signature in session state.
    """
    if report_dict is None or report_signature is None:
        return []

    if (
        session_state.get("rag_draft_signature") == report_signature
        and session_state.get("rag_draft_store") is not None
    ):
        return session_state["rag_draft_store"]

    draft_store = build_draft_store_from_report(report_dict)
    session_state["rag_draft_store"] = draft_store
    session_state["rag_draft_signature"] = report_signature
    return draft_store


def build_reference_result_lines(
    reference_payload: dict[str, Any],
    max_results: int = DEFAULT_REFERENCE_EXCERPT_LIMIT,
) -> list[str]:
    """Format retrieved reference chunks into prompt-ready evidence lines.

    Each line includes the source file, page, block type, section title, and
    the retrieved chunk text. When parent table context is available, the
    helper also appends a clipped parent-context line.
    """
    results = reference_payload.get("results") or []
    if not results:
        return ["- No retrieved reference chunks were available."]

    lines: list[str] = []
    for index, candidate in enumerate(results[:max_results], start=1):
        metadata = getattr(candidate, "metadata", {}) or {}
        lines.append(
            f"- Result {index} | Source: {metadata.get('source_file', '')} | "
            f"Page: {metadata.get('page', '')} | Block type: {metadata.get('block_type', '')} | "
            f"Section: {metadata.get('section_title', '')} | "
            f"Text: {getattr(candidate, 'text', '')}"
        )
        parent_text = getattr(candidate, "parent_text", None)
        if parent_text:
            lines.append(f"  Parent context: {parent_text[:DEFAULT_PARENT_TEXT_LIMIT]}")

    return lines


def build_generation_prompt(
    query: str,
    deterministic_answer: str | None,
    reference_payload: dict[str, Any],
    draft_hits: list[dict[str, Any]],
    has_draft: bool,
) -> str:
    """Assemble the grounded prompt sent to the external LLM.

    The final prompt keeps evidence sources separate by including:
    - deterministic threshold facts
    - the reference-evidence scaffold
    - retrieved reference excerpts
    - top retrieved draft chunks
    - answer-formatting instructions
    """
    prompt_parts = [
        "You are reviewing an IUCN assessment draft using retrieved evidence.",
        f"User question: {query}",
        "",
        "Deterministic reference facts:",
        deterministic_answer or "No deterministic reference facts matched this question.",
        "",
        "Retrieved reference evidence scaffold:",
        reference_payload.get("answer_scaffold", "No reference scaffold available."),
        "",
        "Retrieved reference evidence excerpts:",
        *build_reference_result_lines(reference_payload),
        "",
    ]

    if has_draft and draft_hits:
        prompt_parts.append("Draft assessment evidence (top 6 chunks):")
        for hit in draft_hits:
            prompt_parts.append(
                f"- Section: {hit['section_path']} | Source key: {hit.get('source_key', '')} | "
                f"Score: {hit['score']} | Text: {hit['text'][:DEFAULT_DRAFT_TEXT_LIMIT]}"
            )
    elif has_draft:
        prompt_parts.append("Draft assessment evidence: No draft chunks were available for this prompt.")
    else:
        prompt_parts.append("No draft has been uploaded for review.")

    prompt_parts.extend(
        [
            "",
            "Instructions:",
            "- Deterministic reference facts are authoritative when present.",
            "- Use retrieved reference evidence to add context and detail.",
            "- Use draft assessment evidence only to describe what the uploaded draft appears to contain.",
            "- If the draft is silent on a point, say that it is not evident from the retrieved draft evidence.",
            "",
            "Write a grounded answer with these headings:",
            "1. Deterministic reference facts",
            "2. What the IUCN reference documents require",
            "3. What the uploaded draft appears to contain",
            "4. Any uncertainty",
            "",
            "Do not invent requirements that are not supported by the evidence.",
        ]
    )
    return "\n".join(prompt_parts)


def external_llm_is_configured(llm_config: dict[str, Any] | None) -> bool:
    """Return True when the runtime has enough information to call an external LLM.

    The current check requires only:
    - a non-empty base URL
    - a non-empty model name
    """
    config = llm_config or {}
    return bool(str(config.get("base_url") or "").strip() and str(config.get("model") or "").strip())


def maybe_call_external_llm(prompt: str, llm_config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Call the external LLM and return normalized output, reasoning, or error fields.

    This helper:
    - validates the configured endpoint and model
    - instantiates the OpenAI client against the configured base URL
    - sends the grounded prompt as one chat completion request
    - normalizes the returned output or error into one shared result shape
    """
    config = llm_config or {}
    base_url = str(config.get("base_url") or "").strip()
    model = str(config.get("model") or "").strip()
    api_key = str(config.get("api_key") or "").strip()
    reasoning_enabled = bool(config.get("reasoning_enabled"))

    if not base_url or not model:
        return None

    if OpenAI is None:
        return _llm_error_result(f"OpenAI client is not installed: {OPENAI_IMPORT_ERROR}")

    try:
        client = OpenAI(
            base_url=_build_openai_base_url(base_url),
            api_key=api_key,
        )
        completion = client.chat.completions.create(
            **_build_llm_request_kwargs(prompt, model, reasoning_enabled),
        )
        result = _extract_message_parts(_completion_to_dict(completion))
        if result is None:
            return _llm_error_result("OpenAI client returned an empty completion payload")
        return result
    except Exception as exc:
        return _llm_error_result(repr(exc))


def answer_rag_question(
    query: str,
    draft_store: list[dict[str, Any]] | None,
    top_k_draft: int = 6,
    llm_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full inference flow for one RAG question against the current draft.

    The flow is:
    - get deterministic threshold facts when relevant
    - retrieve reference evidence
    - retrieve draft evidence
    - build the full generation prompt
    - call the external LLM when configured
    - return the final answer plus debug-friendly metadata
    """
    has_draft = bool(draft_store)
    deterministic_answer = answer_threshold_query(query) if is_threshold_query(query) else None
    reference_payload = answer_query(query, allow_threshold_short_circuit=False)
    draft_hits = retrieve_from_draft(query, draft_store or [], top_k=top_k_draft) if has_draft else []
    prompt = build_generation_prompt(
        query=query,
        deterministic_answer=deterministic_answer,
        reference_payload=reference_payload,
        draft_hits=draft_hits,
        has_draft=has_draft,
    )

    llm_configured = external_llm_is_configured(llm_config)
    llm_result = maybe_call_external_llm(prompt, llm_config=llm_config) if llm_configured else None

    answer_text = _build_answer_text(llm_configured, llm_result)

    route = "deterministic_plus_hybrid_rag" if deterministic_answer else reference_payload.get("route", "hybrid_rag")
    return {
        "answer": answer_text,
        "route": route,
        "deterministic_answer": deterministic_answer,
        "reference_payload": reference_payload,
        "draft_hits": draft_hits,
        "has_draft": has_draft,
        "llm_configured": llm_configured,
        "used_external_llm": llm_result is not None,
        "model_response": llm_result,
        "prompt": prompt,
    }


def format_rag_debug_payload(response: dict[str, Any]) -> str:
    """Serialize the structured RAG debug payload as pretty-printed JSON.

    This helper is mainly used by the Streamlit UI so debug mode can show one
    consistent JSON view for each assistant response.
    """
    return json.dumps(build_rag_debug_payload(response), ensure_ascii=False, indent=2)
