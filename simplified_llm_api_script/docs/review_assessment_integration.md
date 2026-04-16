# `review_assessment` Integration Guide

This document explains how to call `review_assessment` from another codebase — what you need to import, how to configure the provider and model, and what the function returns.

---

## Function Signature

```python
async def review_assessment(
    assessment: dict,
    provider: LLMProvider | None = None,
    *,
    config: dict | None = None,
) -> dict[str, list[dict]]
```

Defined in `llm_checker_v2.py`.

---

## What It Returns

A dict keyed by rule name. Each value is a list of finding dicts:

```python
{
    "rule_01_consistency_threats": [
        {
            "section_path": "Threats > Classification Scheme",
            "issue": "Threat X appears here but not in the Redlist Assessment.",
            "severity": "high",
            "suggestion": "Add threat X to the Redlist Assessment narrative."
        }
    ],
    "rule_02_consistency_habitats": [],   # no findings
    "rule_03_consistency_bibliography": []  # rule failed — empty list, no exception
}
```

- Rules with no findings return `[]`
- Rules that failed (LLM error, parse error) also return `[]` — they do not raise
- On a hard failure (e.g. all rules fail), `review_document` raises `ReviewDocumentError`; `review_assessment` lets this propagate

---

## Required Supporting Code

`review_assessment` depends on everything else in `llm_checker_v2.py`. You must import or include the whole module — you cannot cherry-pick the function alone.

**Classes and objects that must be present:**

| Name | Type | Role |
|---|---|---|
| `LLMProvider` | Protocol | Base interface all providers implement |
| `AnthropicProvider` | Class | Calls the Anthropic API |
| `OpenRouterProvider` | Class | Calls OpenRouter (OpenAI-compatible) |
| `HuggingFaceProvider` | Class | Calls HuggingFace router |
| `CompletionResult` | Pydantic model | LLM response + token usage |
| `Finding` | Pydantic model | Single issue found by a rule |
| `RuleResult` | Dataclass | Rule name + findings list |
| `RuleMetrics` | Dataclass | Timing and token stats for one rule run |
| `RuleEvaluationResult` | Dataclass | Combined rule result + metrics |
| `ReviewDocumentError` | Exception | Raised on hard failure; has `.results` attribute |
| `provider_from_config()` | Function | Builds a provider from a config dict |
| `get_provider()` | Function | Factory: build provider by name string |
| `load_rules()` | Function | Reads all rule markdown files |
| `load_system_prompt()` | Function | Reads `system_prompt.md` |
| `select_sections()` | Function | Filters assessment tree by rule scope |
| `evaluate_rule()` | Function | Single LLM call for one rule |
| `review_document()` | Function | Orchestrates all rule evaluations |

---

## Environment Variables

Set in `.env` (loaded automatically via `python-dotenv`) or in your environment:

| Variable | Required for |
|---|---|
| `ANTHROPIC_API_KEY` | `AnthropicProvider` |
| `OPENROUTER_KEY` | `OpenRouterProvider` |
| `HUGGINGFACE_TOKEN` | `HuggingFaceProvider` |

Only the key for your chosen provider is required.

---

## Provider and Model Configuration

### Option 1: Default (no arguments)

Uses `OpenRouterProvider` with `DEFAULT_OPENROUTER_MODEL = "google/gemma-4-31b-it"`.

```python
import asyncio, json
from llm_checker_v2 import review_assessment

with open("assessment.json") as f:
    assessment = json.load(f)

results = asyncio.run(review_assessment(assessment))
```

Requires `OPENROUTER_KEY` in your environment.

---

### Option 2: `config` dict (recommended for external callers)

Pass a `config` dict to specify the provider endpoint, model, and API key without constructing a provider object yourself.

```python
import asyncio, json, os
from llm_checker_v2 import review_assessment

config = {
    "base_url": "https://openrouter.ai/api/v1/chat/completions",
    "model": "google/gemma-4-31b-it",           # any OpenRouter model slug
    "api_key": os.environ["OPENROUTER_KEY"],
    "reasoning_enabled": False,                  # True for reasoning/chain-of-thought models
}

with open("assessment.json") as f:
    assessment = json.load(f)

results = asyncio.run(review_assessment(assessment, config=config))
```

**`config` keys:**

| Key | Type | Description |
|---|---|---|
| `base_url` | `str` | Chat completions endpoint URL |
| `model` | `str` | Provider-specific model identifier |
| `api_key` | `str \| None` | Bearer token; falls back to `OPENROUTER_KEY` env var if `None` |
| `reasoning_enabled` | `bool` | Sends OpenRouter's reasoning flag when `True` |

**Note:** `provider_from_config` currently routes only OpenRouter endpoints (detected via `"openrouter.ai"` in the URL). To support other providers via config, add an `elif` branch in `provider_from_config()`.

---

### Option 3: Explicit provider object

Construct a provider directly and pass it in. This gives full control over provider settings.

```python
import asyncio, json
from llm_checker_v2 import review_assessment, AnthropicProvider

provider = AnthropicProvider(model="claude-sonnet-4-5")

with open("assessment.json") as f:
    assessment = json.load(f)

results = asyncio.run(review_assessment(assessment, provider=provider))
```

Available provider classes and their init signatures:

```python
AnthropicProvider(model: str = DEFAULT_ANTHROPIC_MODEL)
OpenRouterProvider(model: str, api_key: str | None = None, base_url: str = ..., reasoning_enabled: bool = False)
HuggingFaceProvider(model: str = DEFAULT_HUGGINGFACE_MODEL)
```

---

### Option 4: Handling partial failures

```python
import asyncio, json
from llm_checker_v2 import review_assessment, ReviewDocumentError

with open("assessment.json") as f:
    assessment = json.load(f)

try:
    results = asyncio.run(review_assessment(assessment))
except ReviewDocumentError as exc:
    # Hard failure — some rules may still have run
    results = {
        r.rule_name: [f.model_dump() for f in r.findings]
        for r in exc.results
    }
    print("Partial results recovered:", len(results), "rules")
```

---

## Prompt Library Dependency

`load_rules()` and `load_system_prompt()` resolve paths relative to `llm_checker_v2.py` itself:

```python
BASE_DIR = Path(__file__).parent
RULES_DIR = BASE_DIR / "prompt_library" / "rules"
SYSTEM_PROMPT_PATH = BASE_DIR / "prompt_library" / "system_prompt.md"
```

When importing `llm_checker_v2` from another location, the `prompt_library/` directory must exist alongside `llm_checker_v2.py` — not alongside your calling script. If you move or copy the module, bring `prompt_library/` with it.

---

## Assessment Input Format

`assessment` must be a dict in the hierarchical tree format produced by `assessment_processor.parse_docx_to_dict()`:

```python
{
    "title": "species_name",
    "level": 0,
    "path": [],
    "blocks": [
        {"type": "paragraph", "text": "..."},
        {"type": "table", "rows": [[...]]}
    ],
    "children": [
        {
            "title": "Threats",
            "level": 1,
            "path": ["Threats"],
            "blocks": [...],
            "children": [...]
        }
    ]
}
```

To produce this from a DOCX file:

```python
from assessment_processor import parse_docx_to_dict

assessment = parse_docx_to_dict("path/to/assessment.docx")
results = asyncio.run(review_assessment(assessment))
```
