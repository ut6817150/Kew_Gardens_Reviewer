"""
Shared test helper factories used across multiple test modules.
Not pytest fixtures — plain functions that test code can call inline.
"""
from unittest.mock import AsyncMock, MagicMock

from llm_checker_v2 import CompletionResult, Finding, RuleEvaluationResult, RuleMetrics


def make_mock_provider(response_text: str, input_tokens: int = 10, output_tokens: int = 20):
    """Return an async mock provider that returns a successful CompletionResult."""
    provider = MagicMock()
    provider.provider_name = "mock"
    provider._model = "mock-model"
    provider.complete = AsyncMock(
        return_value=CompletionResult(
            text=response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            token_source="provider",
        )
    )
    return provider


def make_failing_provider(exc: Exception):
    """Return an async mock provider whose complete() raises exc."""
    provider = MagicMock()
    provider.provider_name = "mock"
    provider._model = "mock-model"
    provider.complete = AsyncMock(side_effect=exc)
    return provider


def make_rule_result(
    rule_name: str = "rule_01",
    findings=None,
    status: str = "success",
    input_tokens: int = 100,
    output_tokens: int = 50,
    token_source: str = "provider",
    error=None,
) -> RuleEvaluationResult:
    """Build a RuleEvaluationResult for use in orchestration tests."""
    findings = findings or []
    metrics = RuleMetrics(
        duration_seconds=1.0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        token_source=token_source,
        status=status,
        error=error,
    )
    return RuleEvaluationResult(rule_name=rule_name, findings=findings, metrics=metrics)


def make_finding(
    section_path: str = "Threats",
    issue: str = "Missing data",
    severity: str = "high",
    suggestion: str = "Add data",
) -> Finding:
    return Finding(section_path=section_path, issue=issue, severity=severity, suggestion=suggestion)
