# UI

This folder contains the Streamlit UI helpers used by the repo-level app.
The helpers in this folder do not run as standalone scripts. They are imported
and orchestrated by the repo-root `app.py`, which remains the single entry
point for the interface.

The repository root is the parent folder of this `ui/` subfolder.

## Overview

The current app flow is built around one shared uploaded assessment and one
shared sidebar configuration.

At a high level:

1. `app.py` sets up the Streamlit page and shared session state.
2. `ui/app_sidebar.py` renders the sidebar, validates the uploaded file,
   stages supported uploads to a temporary path, and returns the selected API
   key and model configuration.
3. `app.py` uses the returned upload state to reset any upload-scoped caches
   when the file changes.
4. `app.py` calls `parse_to_dict(...)` once per upload signature and caches
   the parsed assessment dictionary in `st.session_state`.
5. The cached assessment dictionary is passed into the rules-based, LLM, and
   RAG tabs so all three workflows operate on the same parsed input.
6. The download tab reads the tab outputs already stored in session state and
   exports them as a workbook.

That split is deliberate:

- `app.py` is the orchestration layer.
- `ui/app_sidebar.py` owns shared input collection and validation.
- the tab helpers each focus on one workflow.

## Runtime Flow

### 1. Page setup and shared app shell

`app.py` is responsible for:

- configuring the Streamlit page
- rendering the app title and top-level caption
- initializing session-state keys used across reruns
- calling the sidebar helper once
- resetting upload-scoped state when a new file is uploaded
- caching the parsed assessment dictionary once per uploaded file
- creating the top-level tabs
- passing the shared parsed assessment into each tab helper

This means `app.py` is intentionally thin in UI detail, but it still owns the
overall lifecycle of the app.

### 2. Sidebar input collection

`ui/app_sidebar.py` renders the shared sidebar controls:

- document upload
- API key choice
- optional custom API-key entry
- LLM choice
- optional custom OpenRouter model slug entry

It also owns the validation and staging logic that belongs directly under
those controls:

- upload-extension validation
- upload feedback messages
- API-key feedback messages
- temporary-file staging for supported uploads

The sidebar helper returns a `SidebarState` object that includes:

- `tmp_path`
- `uploaded_name`
- `file_signature`
- `input_ready`
- `selected_openrouter_api_key`
- `selected_llm_label`
- `selected_llm_config`
- `custom_model_missing`

That returned state is then consumed by `app.py`.

The option lists used by the sidebar, such as preset API keys, preset models,
and custom-option labels, now live in `ui/app_config.py`.

### 3. Shared parsing

Once the sidebar helper has produced a valid staged upload, `app.py` calls
`parse_to_dict(...)` from `preprocessing/assessment_processor.py`.

Important details:

- parsing happens only when `input_ready` is true
- parsing is cached by `file_signature`
- the parsed dictionary is stored in `st.session_state["assessment_input_dict"]`
- the same parsed dictionary is reused by all three main workflows

This avoids reparsing the uploaded document separately in each tab.

### 4. Rules-based feedback

`ui/app_rules_tab.py` handles the deterministic checker workflow.

Its responsibilities are:

- running `AssessmentParser.parse(...)`
- running `IUCNAssessmentReviewer.review_full_report(...)`
- cleaning and grouping violations for display
- preserving source-section order where possible
- storing the results in `st.session_state["rules_feedback"]`
- offering JSON download for the rules-based output

This tab is the deterministic, non-LLM path.

### 5. LLM feedback

`ui/app_llm_tab.py` handles the model-driven review workflow.

Its responsibilities are:

- reading the shared sidebar-selected LLM config
- gating the tab action if the sidebar config is incomplete
- creating the provider via `provider_from_config(...)`
- running `review_document(...)`
- translating common provider errors into clearer user-facing messages
- storing results in `st.session_state["llm_feedback"]`
- offering JSON download for the LLM output

The selected model is not chosen inside this tab anymore. It is chosen once in
the sidebar and passed in through `app.py`.

### 6. RAG chat

`ui/app_rag_tab.py` handles the prototype retrieval-augmented workflow.

Its responsibilities are:

- reading the shared sidebar-selected LLM config
- adapting that shared config into the format expected by the RAG runtime
- rebuilding draft-store inputs only when the upload changes
- rendering the chat interface
- handling the "clear chat" interaction
- running `answer_rag_question(...)`
- showing optional debug output
- storing the chat history in session state

The RAG tab uses the same sidebar-selected model as the LLM feedback tab.
Only the final config shape is adapted locally for the runtime helper.

### 7. Download workflow

`ui/app_download_tab.py` handles export of whatever feedback has already been generated.

Its responsibilities are:

- checking whether rules-based feedback exists
- checking whether LLM feedback exists
- building an Excel workbook in memory
- flattening the stored feedback payloads into tabular rows
- exposing a single download button to the user

This tab does not generate new review output. It only exports existing
session-state results.

## Shared State and Data Handoffs

The UI relies on a small number of shared session-state values.

Key examples:

- `uploaded_file_signature`
  Tracks when the uploaded file changes, so cached outputs can be reset.
- `assessment_input_dict`
  Shared parsed assessment dictionary from `parse_to_dict(...)`.
- `assessment_input_signature`
  Lets the app know whether the cached parsed dict still matches the current
  file.
- `rules_feedback`
  Cached output from the rules-based tab.
- `llm_feedback`
  Cached output from the LLM tab.
- RAG-specific session keys
  Managed through the RAG runtime helpers to preserve draft-store and chat
  state across reruns.

This shared-state model is what lets Streamlit rerun the script while still
keeping the app usable from the user's perspective.

## Script-by-Script Responsibilities

### Repo root

- `app.py`
  Main Streamlit entry point. Owns page setup, session-state initialization,
  upload-scoped cache resets, parse-to-dict caching, tab construction, and
  passing the shared assessment/config state into the tab helpers.

### This folder

- `ui/app_sidebar.py`
  Renders the shared sidebar, validates uploads, stages supported files to a
  temporary path, validates API-key entry, and returns the selected API-key
  and model configuration used by the rest of the app.

- `ui/app_config.py`
  Stores the shared static configuration used by the app shell and sidebar,
  including preset API-key labels, preset model definitions, and custom-option
  labels.

- `ui/app_rules_tab.py`
  Renders the rules-based feedback tab, runs the deterministic parser and
  reviewer flow, groups violations for display, and exposes JSON export.

- `ui/app_llm_tab.py`
  Renders the LLM feedback tab, uses the shared sidebar-selected model and API
  key, runs the simplified LLM reviewer, translates common provider errors,
  and exposes JSON export.

- `ui/app_rag_tab.py`
  Renders the prototype RAG chat tab, derives the RAG-ready model config from
  the shared sidebar-selected config, manages chat/debug UI, and calls the RAG
  runtime helpers.

- `ui/app_download_tab.py`
  Renders the export tab and builds the downloadable Excel workbook from the
  rules-based and/or LLM outputs currently available in session state.

- `ui/README.md`
  This document.

## Imports

The modules in this folder are imported by `app.py` using paths such as:

- `from ui.app_sidebar import render_sidebar_controls`
- `from ui.app_config import MODEL_SPECS`
- `from ui.app_rules_tab import render_rules_tab`
- `from ui.app_llm_tab import render_llm_tab`
- `from ui.app_rag_tab import render_rag_tab`
- `from ui.app_download_tab import render_download_tab`

Inside the UI modules themselves, imports from the rest of the codebase still
use absolute repo-root package paths such as:

- `from iucn_rules_checker.assessment_parser import AssessmentParser`
- `from llm_rag.iv_inference.rag_runtime import answer_rag_question`
- `from simplified_llm_api_script.llm_checker_v2 import review_document`

Those imports do not need to become relative just because the UI modules live
inside `ui/`. They are still executed as part of the repo-root app, not as
standalone scripts launched directly from the `ui/` folder.

## Current Design Notes

There are a few important current design assumptions:

- the app assumes OpenRouter-backed model access
- the selected model in the sidebar is shared by both the LLM and RAG tabs
- custom model entry currently assumes an OpenRouter model slug
- custom API-key entry is session-only
- upload validation and staging happen before parsing
- parsing is shared rather than re-run independently in each tab

These choices keep the app simpler, but they also shape the current
limitations.

## Future Work

### 1. Authentication and per-user API-key management

A natural next step for the Streamlit interface is user authentication plus
per-user API-key management.

At the moment, the app lets users:

- choose from a small set of shared environment-variable-backed OpenRouter keys
- enter a session-only user-provided OpenRouter key

A future version could instead let each signed-in user:

- log in with their own account
- save their own API key
- reuse that key across sessions
- manage or replace stored keys without editing app configuration

If this is implemented for a deployed app, the API keys should not be stored
in Streamlit session state or local files. The safer design is to identify the
user through an authentication provider and store each user's API key in an
external persistent backend with encryption.

### 2. Stronger secret handling

The current custom API-key option is suitable for session-level use, but it is
not a full secret-management solution.

Possible future improvements:

- explicit secret-redaction rules in logging and error handling
- stronger separation between UI state and runtime credential handling
- optional support for cloud secret managers or encrypted backend storage

### 3. Richer workflow status and progress handling

The app currently gives useful inline status messages, but longer-running
workflows could be made clearer with:

- more explicit step progress for LLM and RAG actions
- clearer success/failure states after reruns
- better surfacing of partial-result scenarios

### 4. History and persistence

At the moment, results live in the current session unless the user downloads
them.

A future app could optionally support:

- saved run history
- named review sessions
- reloading previous results without recomputing everything

That would need careful design so it does not conflict with privacy and secret
handling.

## Review

- Reviewed by Dilip on 17.04.2026. Code will be stress tested once all branches have been merged.
- Re-reviewed by Dilip on 18.04.2026. Changes made to allow users to use different API keys. As above, will be stress tested once all branches have been merged.
