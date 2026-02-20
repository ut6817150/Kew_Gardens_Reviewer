"""Base classes for IUCN rule checkers."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Callable, Union
import re

from ..models import Violation, TextPosition, Severity, RuleDefinition

class BaseChecker(ABC):
    """Abstract base class for all rule checkers."""

    def __init__(
        self,
        rule_id: str,
        rule_name: str,
        category: str,
        severity: Severity = Severity.WARNING,
        assessment_section: str = "Whole Document"
    ):
        self.rule_id = rule_id
        self.rule_name = rule_name
        self.category = category
        self.severity = severity
        self.assessment_section = assessment_section

    @abstractmethod
    def check(self, text: str) -> List[Violation]:
        """Check text and return all violations found."""
        pass

    def _calculate_position(self, text: str, start: int, end: int) -> TextPosition:
        """Calculate line and column numbers from character offset."""
        line = text[:start].count('\n') + 1
        last_newline = text.rfind('\n', 0, start)
        column = start - last_newline if last_newline >= 0 else start + 1
        return TextPosition(start=start, end=end, line=line, column=column)

    def _extract_context(self, text: str, start: int, end: int, context_chars: int = 40) -> str:
        """Extract surrounding text for context."""
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)

        prefix = "..." if context_start > 0 else ""
        suffix = "..." if context_end < len(text) else ""

        return prefix + text[context_start:context_end] + suffix

    def _create_violation(
        self,
        text: str,
        matched_text: str,
        start: int,
        end: int,
        message: str,
        suggested_fix: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> Violation:
        """Helper to create a Violation with position calculation."""
        position = self._calculate_position(text, start, end)
        context = self._extract_context(text, start, end)

        return Violation(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            category=self.category,
            matched_text=matched_text,
            position=position,
            severity=self.severity,
            message=message,
            suggested_fix=suggested_fix,
            context=context,
            assessment_section=self.assessment_section,
            metadata=metadata or {}
        )


class PatternChecker(BaseChecker):
    """Checker for rules that use regex patterns."""

    def __init__(
        self,
        rule_id: str,
        rule_name: str,
        category: str,
        pattern: str,
        message_template: str = "Found '{matched}' - violates rule: {rule_name}",
        fix_map: Optional[Dict[str, str]] = None,
        fix_function: Optional[Callable[[re.Match], str]] = None,
        flags: int = re.IGNORECASE,
        severity: Severity = Severity.WARNING,
        assessment_section: str = "Whole Document"
    ):
        super().__init__(rule_id, rule_name, category, severity, assessment_section)
        self.compiled_pattern = re.compile(pattern, flags)
        self.message_template = message_template
        self.fix_map = fix_map or {}
        self.fix_function = fix_function

    def check(self, text: str) -> List[Violation]:
        """Check text against the pattern and return violations."""
        violations = []

        for match in self.compiled_pattern.finditer(text):
            matched_text = match.group(0)

            # Generate suggested fix
            suggested_fix = None
            if self.fix_function:
                suggested_fix = self.fix_function(match)
            elif matched_text.lower() in self.fix_map:
                # Preserve case when applying fix
                fix = self.fix_map[matched_text.lower()]
                if matched_text[0].isupper() and fix:
                    fix = fix[0].upper() + fix[1:] if len(fix) > 1 else fix.upper()
                if matched_text.isupper() and fix:
                    fix = fix.upper()
                suggested_fix = fix

            message = self.message_template.format(
                matched=matched_text,
                rule_name=self.rule_name,
                fix=suggested_fix or ""
            )

            violations.append(self._create_violation(
                text=text,
                matched_text=matched_text,
                start=match.start(),
                end=match.end(),
                message=message,
                suggested_fix=suggested_fix
            ))

        return violations


class MultiPatternChecker(BaseChecker):
    """Checker that applies multiple patterns with individual messages/fixes."""

    def __init__(
        self,
        rule_id: str,
        rule_name: str,
        category: str,
        patterns: List[Dict],  # List of {pattern, message, fix, flags}
        severity: Severity = Severity.WARNING,
        assessment_section: str = "Whole Document"
    ):
        super().__init__(rule_id, rule_name, category, severity, assessment_section)
        self.patterns = []
        for p in patterns:
            flags = p.get('flags', re.IGNORECASE)
            self.patterns.append({
                'compiled': re.compile(p['pattern'], flags),
                'message': p.get('message', f"Violates: {rule_name}"),
                'fix': p.get('fix'),
                'fix_function': p.get('fix_function')
            })

    def check(self, text: str) -> List[Violation]:
        """Check text against all patterns."""
        violations = []

        for pattern_def in self.patterns:
            for match in pattern_def['compiled'].finditer(text):
                matched_text = match.group(0)

                # Determine fix
                fix = None
                if pattern_def['fix_function']:
                    fix = pattern_def['fix_function'](match)
                elif pattern_def['fix']:
                    fix = pattern_def['fix']

                violations.append(self._create_violation(
                    text=text,
                    matched_text=matched_text,
                    start=match.start(),
                    end=match.end(),
                    message=pattern_def['message'].format(matched=matched_text, fix=fix or ""),
                    suggested_fix=fix
                ))

        return violations
