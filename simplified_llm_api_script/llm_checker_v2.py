import asyncio
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol

import anthropic
import frontmatter
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

try:
    import tiktoken
except ImportError:  # pragma: no cover - fallback for environments pending dependency install
    tiktoken = None

load_dotenv()

BASE_DIR = Path(__file__).parent
RULES_DIR = BASE_DIR / "prompt_library" / "rules"
SYSTEM_PROMPT_PATH = BASE_DIR / "prompt_library" / "system_prompt.md"

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4.6"
DEFAULT_OPENROUTER_MODEL = "google/gemma-4-31b-it"
DEFAULT_HUGGINGFACE_MODEL = "deepseek-ai/DeepSeek-R1:novita"

# Example provider config dict for the handover entry point.
# Pass into review_assessment(assessment, config=LLM_CONFIG) or
# provider_from_config(LLM_CONFIG) to build a provider.
LLM_CONFIG = {
    "base_url": "https://openrouter.ai/api/v1/chat/completions",
    "model": "nvidia/nemotron-3-super-120b-a12b:free",
    "api_key": os.environ.get("OPENROUTER_KEY"),
    "reasoning_enabled": True,
}

# ── LLM Provider protocol & implementations ───────────────────────────────────

class LLMProvider(Protocol):
    async def complete(self, system_prompt: str, user_message: str) -> "CompletionResult":
        """Single-turn completion. Returns text plus token usage metadata."""
        ...


def _coerce_token_count(value: Any) -> int | None:
    """Normalise provider usage counts into integers when present."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(round(value))
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _extract_text_content(content: Any) -> str:
    """Flatten provider message content into a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif hasattr(item, "text") and isinstance(item.text, str):
                parts.append(item.text)
        return "".join(parts)
    if hasattr(content, "text") and isinstance(content.text, str):
        return content.text
    return str(content)


def _estimate_token_count(text: str) -> int:
    """Estimate tokens for arbitrary text using tiktoken, with a simple fallback."""
    if not text:
        return 0
    if tiktoken is not None:
        try:
            encoding = tiktoken.get_encoding("o200k_base")
            return len(encoding.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def _serialize_prompt_payload(
    provider_name: str,
    model: str,
    system_prompt: str,
    user_message: str,
) -> str:
    """Serialize the exact prompt payload shape used for token estimation."""
    if provider_name == "anthropic":
        payload = {
            "model": model,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }
    else:
        payload = {
            "model": model,
            "max_tokens": 4096,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _estimate_prompt_tokens(
    provider_name: str,
    model: str,
    system_prompt: str,
    user_message: str,
) -> int:
    return _estimate_token_count(
        _serialize_prompt_payload(provider_name, model, system_prompt, user_message)
    )


class CompletionResult(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    token_source: Literal["provider", "estimated"]
    raw_usage: dict[str, Any] | None = None
    provider_details: dict[str, Any] | None = None


class AnthropicProvider:
    provider_name = "anthropic"

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL):
        self._client = anthropic.AsyncAnthropic()
        self._model = model

    async def complete(self, system_prompt: str, user_message: str) -> CompletionResult:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        text = _extract_text_content(response.content)
        usage = getattr(response, "usage", None)
        input_tokens = _coerce_token_count(getattr(usage, "input_tokens", None))
        output_tokens = _coerce_token_count(getattr(usage, "output_tokens", None))
        if input_tokens is not None and output_tokens is not None:
            return CompletionResult(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                token_source="provider",
                raw_usage={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
                provider_details={"provider": self.provider_name, "model": self._model},
            )

        estimated_input = _estimate_prompt_tokens(
            self.provider_name,
            self._model,
            system_prompt,
            user_message,
        )
        estimated_output = _estimate_token_count(text)
        return CompletionResult(
            text=text,
            input_tokens=estimated_input,
            output_tokens=estimated_output,
            total_tokens=estimated_input + estimated_output,
            token_source="estimated",
            provider_details={"provider": self.provider_name, "model": self._model},
        )


class OpenRouterProvider:
    _BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    provider_name = "openrouter"

    def __init__(
        self,
        model: str = DEFAULT_OPENROUTER_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning_enabled: bool = False,
        min_interval_seconds: float | None = None,
        max_retries: int = 4,
    ):
        self._model = model
        self._api_key = api_key or os.environ["OPENROUTER_KEY"]
        self._endpoint = base_url or self._BASE_URL
        self._reasoning_enabled = reasoning_enabled
        self._client = httpx.AsyncClient(timeout=120)
        self._request_lock = asyncio.Lock()
        self._last_request_started = 0.0
        self._max_retries = max_retries
        if min_interval_seconds is None:
            # OpenRouter documents 20 RPM for free variants, so keep a small buffer above 3s.
            min_interval_seconds = 3.1 if model.endswith(":free") else 0.0
        self._min_interval_seconds = max(0.0, min_interval_seconds)

    async def _wait_for_turn(self) -> None:
        """Throttle requests from this provider instance to avoid free-tier 429 errors."""
        if self._min_interval_seconds <= 0:
            return

        async with self._request_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait_seconds = self._last_request_started + self._min_interval_seconds - now
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            self._last_request_started = loop.time()

    def _retry_delay_seconds(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return max(float(retry_after), self._min_interval_seconds)
            except ValueError:
                pass

        return max(self._min_interval_seconds, 2 ** attempt)

    @staticmethod
    def _response_error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if message:
                    return str(message)
                return json.dumps(error, ensure_ascii=False)
            if error:
                return str(error)
            return json.dumps(payload, ensure_ascii=False)

        text = response.text.strip()
        if not text:
            return f"{response.reason_phrase or 'Unknown error'}"
        if len(text) > 500:
            text = f"{text[:500]}..."
        return text

    async def complete(self, system_prompt: str, user_message: str) -> CompletionResult:
        for attempt in range(self._max_retries + 1):
            await self._wait_for_turn()
            payload: dict[str, Any] = {
                "model": self._model,
                "max_tokens": 4096,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            }
            if self._reasoning_enabled:
                payload["reasoning"] = {"enabled": True}
            resp = await self._client.post(
                self._endpoint,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )

            if resp.status_code == 429 and attempt < self._max_retries:
                wait_seconds = self._retry_delay_seconds(resp, attempt)
                print(
                    f"       [retry] OpenRouter 429 for {self._model}; "
                    f"sleeping {wait_seconds:.1f}s before retry {attempt + 1}/{self._max_retries}"
                )
                await asyncio.sleep(wait_seconds)
                continue

            if resp.is_error:
                detail = self._response_error_detail(resp)
                raise RuntimeError(
                    f"OpenRouter request failed ({resp.status_code}) for {self._model}: {detail}"
                )

            body = resp.json()
            if "error" in body:
                raise RuntimeError(f"OpenRouter error: {body['error']}")
            text = _extract_text_content(body["choices"][0]["message"]["content"])
            usage = body.get("usage") or {}
            input_tokens = _coerce_token_count(usage.get("prompt_tokens"))
            output_tokens = _coerce_token_count(usage.get("completion_tokens"))
            total_tokens = _coerce_token_count(usage.get("total_tokens"))
            if input_tokens is not None and output_tokens is not None:
                return CompletionResult(
                    text=text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens or (input_tokens + output_tokens),
                    token_source="provider",
                    raw_usage=usage,
                    provider_details={"provider": self.provider_name, "model": self._model},
                )

            estimated_input = _estimate_prompt_tokens(
                self.provider_name,
                self._model,
                system_prompt,
                user_message,
            )
            estimated_output = _estimate_token_count(text)
            return CompletionResult(
                text=text,
                input_tokens=estimated_input,
                output_tokens=estimated_output,
                total_tokens=estimated_input + estimated_output,
                token_source="estimated",
                raw_usage=usage or None,
                provider_details={"provider": self.provider_name, "model": self._model},
            )

        raise RuntimeError(f"OpenRouter retries exhausted for {self._model}")


class HuggingFaceProvider:
    provider_name = "huggingface"

    def __init__(self, model: str = DEFAULT_HUGGINGFACE_MODEL):
        self._model = model
        self._client = AsyncOpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ["HUGGINGFACE_TOKEN"],
        )

    async def complete(self, system_prompt: str, user_message: str) -> CompletionResult:
        completion = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        text = _extract_text_content(completion.choices[0].message.content)
        usage = getattr(completion, "usage", None)
        input_tokens = _coerce_token_count(getattr(usage, "prompt_tokens", None))
        output_tokens = _coerce_token_count(getattr(usage, "completion_tokens", None))
        total_tokens = _coerce_token_count(getattr(usage, "total_tokens", None))
        if input_tokens is not None and output_tokens is not None:
            return CompletionResult(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens or (input_tokens + output_tokens),
                token_source="provider",
                raw_usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": total_tokens or (input_tokens + output_tokens),
                },
                provider_details={"provider": self.provider_name, "model": self._model},
            )

        estimated_input = _estimate_prompt_tokens(
            self.provider_name,
            self._model,
            system_prompt,
            user_message,
        )
        estimated_output = _estimate_token_count(text)
        return CompletionResult(
            text=text,
            input_tokens=estimated_input,
            output_tokens=estimated_output,
            total_tokens=estimated_input + estimated_output,
            token_source="estimated",
            provider_details={"provider": self.provider_name, "model": self._model},
        )


def get_provider(provider_name: str, model: str | None = None) -> LLMProvider:
    if provider_name == "anthropic":
        return AnthropicProvider(model or DEFAULT_ANTHROPIC_MODEL)
    if provider_name == "openrouter":
        return OpenRouterProvider(model or DEFAULT_OPENROUTER_MODEL)
    if provider_name == "huggingface":
        return HuggingFaceProvider(model or DEFAULT_HUGGINGFACE_MODEL)
    raise ValueError(f"Unknown provider: {provider_name!r}")


# ── Pydantic models ──────────────────────────────────────────────────────────

class Finding(BaseModel):
    section_path: str
    issue: str
    severity: Literal["high", "medium", "low"]
    suggestion: str

    @field_validator("section_path", mode="before")
    @classmethod
    def coerce_section_path(cls, v):
        if isinstance(v, list):
            return " > ".join(str(x) for x in v)
        return v


class RuleResult(BaseModel):
    rule_name: str
    findings: list[Finding]


class RuleMetrics(BaseModel):
    duration_seconds: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    token_source: Literal["provider", "estimated"]
    status: Literal["success", "error"]
    error: str | None = None


class RuleEvaluationResult(BaseModel):
    rule_name: str
    findings: list[Finding]
    metrics: RuleMetrics


class ReviewDocumentError(RuntimeError):
    def __init__(self, message: str, results: list[RuleEvaluationResult]):
        super().__init__(message)
        self.results = results


# ── Rule loading ──────────────────────────────────────────────────────────────

REQUIRED_FRONTMATTER = {"scope", "severity", "category"}


def load_rules() -> list[dict]:
    """Load and validate all rule markdown files from the rules directory."""
    rules = []
    for path in sorted(RULES_DIR.glob("*.md")):
        post = frontmatter.load(str(path))
        missing = REQUIRED_FRONTMATTER - set(post.metadata.keys())
        if missing:
            raise ValueError(f"Rule {path.name} missing frontmatter fields: {missing}")
        rules.append({
            "name": path.stem,
            "scope": post.metadata["scope"],
            "severity": post.metadata["severity"],
            "category": post.metadata["category"],
            "body": post.content,
        })
    return rules


def load_system_prompt() -> str:
    """Load the shared system prompt."""
    return SYSTEM_PROMPT_PATH.read_text()


# ── Section selection ─────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Normalize a string for fuzzy matching: lowercase, strip whitespace."""
    return s.replace(" ", "").lower().strip()


def select_sections(document_tree: dict, scope: str) -> list[dict]:
    """
    Extract the relevant parts of the document tree based on a rule's scope.

    Returns a list of section dicts. For 'section_type:*', each element is a
    single section (the rule will be called once per section).
    """
    scope_stripped = scope.strip()

    if scope_stripped == "full_document":
        return [document_tree]

    if scope_stripped.startswith("relevant_sections:"):
        names_raw = scope_stripped.split(":", 1)[1]
        target_names = {_normalize(n) for n in names_raw.split(",")}
        # "all" means the whole document
        if "all" in target_names:
            return [document_tree]
        matched = [
            child for child in document_tree.get("children", [])
            if _normalize(child.get("title", "")) in target_names
        ]
        return matched

    if scope_stripped == "section_type:*":
        return document_tree.get("children", [])

    raise ValueError(f"Unknown scope format: {scope}")


# ── Prompt assembly & LLM call ───────────────────────────────────────────────

def _build_user_message(rule_body: str, sections: list[dict]) -> str:
    """Assemble the user message with rule instructions and document content."""
    doc_json = json.dumps(sections, indent=2, ensure_ascii=False)
    return (
        f"<rule>\n{rule_body}\n</rule>\n\n"
        f"<document>\n{doc_json}\n</document>"
    )


async def evaluate_rule(
    provider: LLMProvider,
    system_prompt: str,
    rule: dict,
    sections: list[dict],
) -> RuleEvaluationResult:
    """Make a single-turn LLM call to evaluate one rule against document sections."""
    user_message = _build_user_message(rule["body"], sections)
    provider_name = getattr(provider, "provider_name", "unknown")
    model_name = getattr(provider, "_model", "unknown")
    estimated_input_tokens = _estimate_prompt_tokens(
        provider_name,
        model_name,
        system_prompt,
        user_message,
    )

    print(f"       [rule] sending: {rule['name']}")
    started_at = perf_counter()
    completion: CompletionResult | None = None

    try:
        completion = await provider.complete(system_prompt, user_message)
        raw_text = completion.text

        # Strip markdown code fences if the model wraps the JSON
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]  # remove opening fence line
            cleaned = cleaned.rsplit("```", 1)[0]  # remove closing fence
            cleaned = cleaned.strip()

        data = json.loads(cleaned)
        parsed = RuleResult(**data)
        metrics = RuleMetrics(
            duration_seconds=perf_counter() - started_at,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            total_tokens=completion.total_tokens,
            token_source=completion.token_source,
            status="success",
            error=None,
        )
        print(
            f"       [rule] done: {rule['name']}  "
            f"{metrics.duration_seconds:.2f}s  "
            f"{metrics.input_tokens} in / {metrics.output_tokens} out"
        )
        return RuleEvaluationResult(
            rule_name=parsed.rule_name,
            findings=parsed.findings,
            metrics=metrics,
        )
    except Exception as exc:
        duration_seconds = perf_counter() - started_at
        if completion is None:
            input_tokens = estimated_input_tokens
            output_tokens = 0
            total_tokens = input_tokens
            token_source: Literal["provider", "estimated"] = "estimated"
        else:
            input_tokens = completion.input_tokens
            output_tokens = completion.output_tokens
            total_tokens = completion.total_tokens
            token_source = completion.token_source

        metrics = RuleMetrics(
            duration_seconds=duration_seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            token_source=token_source,
            status="error",
            error=str(exc),
        )
        print(
            f"       [rule] failed: {rule['name']}  "
            f"{metrics.duration_seconds:.2f}s  "
            f"{metrics.input_tokens} in / {metrics.output_tokens} out"
        )
        return RuleEvaluationResult(
            rule_name=rule["name"],
            findings=[],
            metrics=metrics,
        )


# ── Orchestration ─────────────────────────────────────────────────────────────

async def review_document(
    document_tree: dict,
    provider: LLMProvider | None = None,
    mode: str = "concurrent",
) -> list[RuleEvaluationResult]:
    """
    Evaluate a document tree against all rules.

    mode="sequential": awaits each rule call one at a time — safe
    for rate-limited providers like OpenRouter free tier.
    mode="concurrent" (default): fires all rule calls at once with asyncio.gather.

    This is the main entry point, designed to be called from other scripts
    with a dict (not a file path). Defaults to OpenRouterProvider if no
    provider is supplied.
    """
    if provider is None:
        provider = OpenRouterProvider()

    rules = load_rules()
    system_prompt = load_system_prompt()

    if mode == "sequential":
        results = []
        for rule in rules:
            sections = select_sections(document_tree, rule["scope"])
            if rule["scope"].strip() == "section_type:*":
                for section in sections:
                    result = await evaluate_rule(provider, system_prompt, rule, [section])
                    results.append(result)
                    if result.metrics.status == "error":
                        raise ReviewDocumentError(result.metrics.error or "Rule evaluation failed", results)
            else:
                result = await evaluate_rule(provider, system_prompt, rule, sections)
                results.append(result)
                if result.metrics.status == "error":
                    raise ReviewDocumentError(result.metrics.error or "Rule evaluation failed", results)
        return results

    coroutines = []
    for rule in rules:
        sections = select_sections(document_tree, rule["scope"])
        if rule["scope"].strip() == "section_type:*":
            for section in sections:
                coroutines.append(evaluate_rule(provider, system_prompt, rule, [section]))
        else:
            coroutines.append(evaluate_rule(provider, system_prompt, rule, sections))

    results = await asyncio.gather(*coroutines)
    if any(result.metrics.status == "error" for result in results):
        first_error = next(result.metrics.error for result in results if result.metrics.status == "error")
        raise ReviewDocumentError(first_error or "Rule evaluation failed", results)

    return results


def summarize_rule_results(results: list[RuleEvaluationResult]) -> dict:
    """Summarise per-rule outcomes into run/grid-friendly aggregate metrics."""
    rules_attempted = len(results)
    rules_succeeded = sum(result.metrics.status == "success" for result in results)
    rules_failed = rules_attempted - rules_succeeded
    total_input_tokens = sum(result.metrics.input_tokens for result in results)
    total_output_tokens = sum(result.metrics.output_tokens for result in results)
    total_tokens = sum(result.metrics.total_tokens for result in results)
    token_estimation_used = any(result.metrics.token_source == "estimated" for result in results)
    total_findings = sum(len(result.findings) for result in results)
    return {
        "rules_attempted": rules_attempted,
        "rules_succeeded": rules_succeeded,
        "rules_failed": rules_failed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "token_estimation_used": token_estimation_used,
        "total_findings": total_findings,
    }


def results_to_json(results: list[RuleEvaluationResult]) -> list[dict]:
    """Convert results to the feedback-only JSON output structure."""
    return [
        {
            "rule_name": result.rule_name,
            "findings": [finding.model_dump() for finding in result.findings],
        }
        for result in results
    ]


def provider_from_config(config: dict) -> LLMProvider:
    """
    Build a provider from a RAG-style config dict.

    Expected keys:
      - base_url: full chat-completions endpoint URL
      - model: provider-specific model identifier
      - api_key: bearer token (may be None to fall back to env var)
      - reasoning_enabled: bool, sends OpenRouter's reasoning flag when True

    Currently dispatches OpenRouter endpoints. Extend with elif branches
    if other providers need the same config-driven entry point.
    """
    base_url = config.get("base_url") or ""
    if "openrouter.ai" in base_url:
        return OpenRouterProvider(
            model=config["model"],
            api_key=config.get("api_key"),
            base_url=base_url,
            reasoning_enabled=bool(config.get("reasoning_enabled", False)),
        )
    raise ValueError(f"Unsupported base_url for provider config: {base_url!r}")


async def review_assessment(
    assessment: dict,
    provider: LLMProvider | None = None,
    *,
    config: dict | None = None,
) -> dict[str, list[dict]]:
    """
    Handover entry point: dict in, dict out.

    Takes a parsed assessment tree (the same shape produced by
    assessment_processor.parse_docx_to_dict) and returns feedback as a
    dict keyed by rule_name, where each value is a list of finding dicts
    with keys: section_path, issue, severity, suggestion.

    Provide either a built `provider` or a `config` dict (see
    LLM_CONFIG / provider_from_config). If neither is given,
    review_document falls back to its default provider.

    On per-rule failures the rule still appears in the output with an
    empty findings list, matching review_document's semantics. Callers
    that want partial results on hard failure can catch ReviewDocumentError
    and inspect exc.results.
    """
    if provider is None and config is not None:
        provider = provider_from_config(config)
    results = await review_document(assessment, provider=provider)
    return {
        result.rule_name: [finding.model_dump() for finding in result.findings]
        for result in results
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    PROVIDERS = {
        "anthropic": (AnthropicProvider, DEFAULT_ANTHROPIC_MODEL),
        "openrouter": (OpenRouterProvider, DEFAULT_OPENROUTER_MODEL),
        "huggingface": (HuggingFaceProvider, DEFAULT_HUGGINGFACE_MODEL),
    }

    parser = argparse.ArgumentParser(description="IUCN LLM document checker")
    parser.add_argument(
        "json_file",
        nargs="?",
        default=str(BASE_DIR / "Myrcia almasensis_draft_status_Apr2022.json"),
        help="Path to document JSON (default: bundled example)",
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDERS.keys(),
        default="openrouter",
        help="LLM provider to use (default: openrouter)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override (provider default used if omitted)",
    )
    parser.add_argument(
        "--mode",
        choices=["sequential", "concurrent"],
        default="sequential",
        help="sequential (default): await each rule one at a time; concurrent: fire all rules at once",
    )
    args = parser.parse_args()

    provider_cls, default_model = PROVIDERS[args.provider]
    provider = provider_cls(model=args.model or default_model)

    with open(args.json_file, encoding="utf-8") as f:
        document_tree = json.load(f)

    exit_code = 0
    try:
        results = asyncio.run(review_document(document_tree, provider=provider, mode=args.mode))
    except ReviewDocumentError as exc:
        results = exc.results
        exit_code = 1
        print(f"ERROR: {exc}", file=sys.stderr)
    print(json.dumps(results_to_json(results), indent=2, ensure_ascii=False))
    raise SystemExit(exit_code)
