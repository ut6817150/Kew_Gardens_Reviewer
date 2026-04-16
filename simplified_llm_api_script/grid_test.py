"""
grid_test.py — Run a matrix of (provider, model) combinations across all
documents in json_converted/ and save structured results to grid_outputs/.

Grid entries are processed sequentially (one model after the next).
Within a single run, rules fire concurrently for normal models and
sequentially for OpenRouter ":free" models, which have tight rate limits.

Output layout:
  grid_outputs/
    {doc_stem}/
      {provider}__{model_slug}.json       <- list of feedback-only rule results
      {provider}__{model_slug}_meta.json  <- run status + per-rule metrics + totals
    _grid_summary.json                    <- aggregate metrics for the whole run
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from llm_checker_v2 import (
    AnthropicProvider,
    HuggingFaceProvider,
    OpenRouterProvider,
    ReviewDocumentError,
    load_rules,
    results_to_json,
    review_document,
    select_sections,
    summarize_rule_results,
)

load_dotenv()

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "json_converted"
OUTPUT_DIR = BASE_DIR / "grid_outputs"

GRID = [
    #("huggingface", "deepseek-ai/DeepSeek-R1:novita"),
    ("openrouter", "google/gemma-4-31b-it"), # zero retention
    ("openrouter", "minimax/minimax-m2.5"), # zero retention
    ("openrouter", "google/gemini-2.5-flash-lite"), # zero retention
    ("openrouter", "mistralai/ministral-14b-2512"), # zero retention
    ("openrouter", "qwen/qwen3.5-plus-02-15"),
    ("openrouter", "deepseek/deepseek-r1-distill-llama-70b"),
    ("openrouter", "qwen/qwen3.6-plus:free"),
    ("openrouter", "nvidia/nemotron-3-super-120b-a12b:free"), # zero retention
    ("openrouter", "openai/gpt-oss-120b:free") # zero retention
]

_PROVIDER_CLASSES = {
    "anthropic": AnthropicProvider,
    "openrouter": OpenRouterProvider,
    "huggingface": HuggingFaceProvider,
}


def _model_slug(model: str) -> str:
    """Make a model string safe for use as a filename component."""
    return model.replace("/", "_").replace(":", "_")


def _display_path(path: Path) -> str:
    """Prefer repo-relative paths in logs, with an absolute fallback for redirected outputs."""
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def _load_docs(filter_stems: list[str] | None) -> list[tuple[str, dict]]:
    """Return (stem, document_tree) pairs from json_converted/, skipping _errors.json."""
    docs = []
    for path in sorted(DOCS_DIR.glob("*.json")):
        if path.stem == "_errors":
            continue
        if filter_stems and not any(f in path.stem for f in filter_stems):
            continue
        with open(path, encoding="utf-8") as f:
            docs.append((path.stem, json.load(f)))
    return docs


def _estimate_rule_calls(document_tree: dict) -> int:
    """Estimate how many LLM API requests one document review will make."""
    calls = 0
    for rule in load_rules():
        sections = select_sections(document_tree, rule["scope"])
        calls += len(sections) if rule["scope"].strip() == "section_type:*" else 1
    return calls


async def _run_one(
    doc_stem: str,
    document_tree: dict,
    provider_name: str,
    model: str,
    delay: float,
) -> dict:
    slug = _model_slug(model)
    out_dir = OUTPUT_DIR / doc_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{provider_name}__{slug}.json"
    meta_path = out_dir / f"{provider_name}__{slug}_meta.json"

    run_started_at = datetime.now(timezone.utc)
    run_started = perf_counter()
    meta: dict = {
        "provider": provider_name,
        "model": model,
        "document": doc_stem,
        "started_at": run_started_at.isoformat(),
        "error": None,
    }

    print(f"  -> {provider_name}/{model}  [{doc_stem}]")
    results = []
    status = "success"
    try:
        provider_cls = _PROVIDER_CLASSES[provider_name]
        provider = provider_cls(model=model)
        mode = "sequential" if model.endswith(":free") else "concurrent"
        results = await review_document(document_tree, provider=provider, mode=mode)
    except ReviewDocumentError as exc:
        results = exc.results
        meta["error"] = str(exc)
        status = "error"
        print(f"     ERROR  {exc}")
    except Exception as exc:
        meta["error"] = str(exc)
        status = "error"
        print(f"     ERROR  {exc}")

    run_duration_seconds = perf_counter() - run_started
    completed_at = datetime.now(timezone.utc)
    summary = summarize_rule_results(results)
    findings_count = summary["total_findings"]
    rule_metrics = [
        {
            "rule_name": result.rule_name,
            "duration_seconds": result.metrics.duration_seconds,
            "input_tokens": result.metrics.input_tokens,
            "output_tokens": result.metrics.output_tokens,
            "total_tokens": result.metrics.total_tokens,
            "token_source": result.metrics.token_source,
            "status": result.metrics.status,
            "error": result.metrics.error,
        }
        for result in results
    ]

    meta["completed_at"] = completed_at.isoformat()
    meta["status"] = status
    meta["rules_evaluated"] = summary["rules_attempted"]
    meta["rule_metrics"] = rule_metrics
    meta["overall"] = {
        "rules_attempted": summary["rules_attempted"],
        "rules_succeeded": summary["rules_succeeded"],
        "rules_failed": summary["rules_failed"],
        "run_duration_seconds": round(run_duration_seconds, 3),
        "total_findings": findings_count,
        "total_input_tokens": summary["total_input_tokens"],
        "total_output_tokens": summary["total_output_tokens"],
        "total_tokens": summary["total_tokens"],
        "token_estimation_used": summary["token_estimation_used"],
    }

    results_path.write_text(
        json.dumps(results_to_json(results), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    status_label = "OK" if status == "success" else "PARTIAL"
    print(
        f"     {status_label}  {findings_count} findings  "
        f"{summary['total_input_tokens']} in / {summary['total_output_tokens']} out  "
        f"{run_duration_seconds:.2f}s  -> {_display_path(results_path)}"
    )

    if delay > 0:
        await asyncio.sleep(delay)

    return {
        "document": doc_stem,
        "provider": provider_name,
        "model": model,
        "status": status,
        "error": meta["error"],
        "started_at": meta["started_at"],
        "completed_at": meta["completed_at"],
        "run_duration_seconds": round(run_duration_seconds, 3),
        "rules_attempted": summary["rules_attempted"],
        "rules_succeeded": summary["rules_succeeded"],
        "rules_failed": summary["rules_failed"],
        "total_findings": findings_count,
        "total_input_tokens": summary["total_input_tokens"],
        "total_output_tokens": summary["total_output_tokens"],
        "total_tokens": summary["total_tokens"],
        "token_estimation_used": summary["token_estimation_used"],
        "results_path": _display_path(results_path),
        "meta_path": _display_path(meta_path),
    }


async def main(filter_stems: list[str] | None, delay: float) -> None:
    docs = _load_docs(filter_stems)
    if not docs:
        print("No documents found. Check json_converted/ or your --docs filter.")
        return

    grid_started_at = datetime.now(timezone.utc)
    grid_started = perf_counter()
    total = len(docs) * len(GRID)
    doc_request_counts = [_estimate_rule_calls(tree) for _, tree in docs]
    min_requests_per_run = min(doc_request_counts)
    max_requests_per_run = max(doc_request_counts)
    total_api_requests = sum(doc_request_counts) * len(GRID)
    free_openrouter_models = sum(
        1 for provider_name, model in GRID
        if provider_name == "openrouter" and model.endswith(":free")
    )
    free_openrouter_requests = sum(doc_request_counts) * free_openrouter_models

    print(f"Documents     : {len(docs)}")
    print(f"Grid          : {len(GRID)} models")
    print(f"Total runs    : {total}  (sequential, one model at a time)")
    if min_requests_per_run == max_requests_per_run:
        print(f"Rule calls    : {min_requests_per_run} per run")
    else:
        print(f"Rule calls    : {min_requests_per_run}-{max_requests_per_run} per run")
    print(f"API requests  : {total_api_requests} total")
    if free_openrouter_requests:
        print(f"OpenRouter :free requests: {free_openrouter_requests}")
    if delay > 0:
        print(f"Delay         : {delay}s between grid runs")
    print("Request mode  : concurrent rules (sequential for :free models)")
    print()

    done = 0
    run_summaries = []
    for doc_stem, tree in docs:
        for provider_name, model in GRID:
            run_summary = await _run_one(doc_stem, tree, provider_name, model, delay)
            run_summaries.append(run_summary)
            done += 1
            print(f"  [{done}/{total}]\n")

    grid_duration_seconds = perf_counter() - grid_started
    grid_completed_at = datetime.now(timezone.utc)
    grid_summary = {
        "started_at": grid_started_at.isoformat(),
        "completed_at": grid_completed_at.isoformat(),
        "grid_duration_seconds": round(grid_duration_seconds, 3),
        "documents": len(docs),
        "models": len(GRID),
        "runs_attempted": len(run_summaries),
        "runs_succeeded": sum(run["status"] == "success" for run in run_summaries),
        "runs_failed": sum(run["status"] != "success" for run in run_summaries),
        "rules_attempted": sum(run["rules_attempted"] for run in run_summaries),
        "rules_succeeded": sum(run["rules_succeeded"] for run in run_summaries),
        "rules_failed": sum(run["rules_failed"] for run in run_summaries),
        "total_input_tokens": sum(run["total_input_tokens"] for run in run_summaries),
        "total_output_tokens": sum(run["total_output_tokens"] for run in run_summaries),
        "total_tokens": sum(run["total_tokens"] for run in run_summaries),
        "token_estimation_used": any(run["token_estimation_used"] for run in run_summaries),
        "runs": run_summaries,
    }
    summary_path = OUTPUT_DIR / "_grid_summary.json"
    summary_path.write_text(
        json.dumps(grid_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        "Totals        : "
        f"{grid_summary['total_input_tokens']} in / "
        f"{grid_summary['total_output_tokens']} out / "
        f"{grid_summary['total_tokens']} total tokens"
    )
    print(
        "Time          : "
        f"{grid_summary['grid_duration_seconds']:.2f}s across "
        f"{grid_summary['runs_attempted']} runs"
    )
    print(f"Done. Results saved to {_display_path(OUTPUT_DIR)}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid test: run multiple LLM providers × models across documents"
    )
    parser.add_argument(
        "--docs",
        nargs="+",
        default=None,
        metavar="STEM",
        help="Filter documents by stem substring (e.g. --docs Test1 Test2). Default: all.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between each (doc, model) grid run (default: 2.0)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.docs, args.delay))
