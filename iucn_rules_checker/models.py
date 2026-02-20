"""Core data models for IUCN rule violation checking."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List
import json


class Severity(Enum):
    """Severity levels for violations."""
    ERROR = "error"      # Must fix - document won't be accepted
    WARNING = "warning"  # Should fix - strongly recommended
    INFO = "info"        # Suggestion - nice to have


class JudgmentType(Enum):
    """Type of rule validation."""
    RULE_BASED = "Rule-based"
    LLM_JUDGED = "LLM-judged"
    HYBRID = "Hybrid"


@dataclass
class TextPosition:
    """Represents a position in the text."""
    start: int              # Character offset from beginning
    end: int                # Character offset end
    line: Optional[int] = None      # Line number (1-indexed)
    column: Optional[int] = None    # Column in line (1-indexed)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "start": self.start,
            "end": self.end,
            "line": self.line,
            "column": self.column
        }


@dataclass
class Violation:
    """Represents a single rule violation found in text."""
    rule_id: str                          # Unique rule identifier
    rule_name: str                        # Human-readable rule name
    category: str                         # e.g., "Language", "Numbers", "IUCN Terms"
    matched_text: str                     # The actual text that violated the rule
    position: TextPosition                # Where in the text
    severity: Severity                    # How serious
    message: str                          # Explanation of the violation
    suggested_fix: Optional[str] = None   # Recommended correction
    context: Optional[str] = None         # Surrounding text for context
    assessment_section: Optional[str] = None  # Which IUCN section applies
    metadata: dict = field(default_factory=dict)  # Additional info

    def to_dict(self) -> dict:
        """Serialize violation to dictionary for JSON export."""
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "category": self.category,
            "matched_text": self.matched_text,
            "position": self.position.to_dict(),
            "severity": self.severity.value,
            "message": self.message,
            "suggested_fix": self.suggested_fix,
            "context": self.context,
            "assessment_section": self.assessment_section,
            "metadata": self.metadata
        }


@dataclass
class RuleDefinition:
    """Definition of a single IUCN rule."""
    rule_id: str                          # Unique ID
    name: str                             # Rule/Standard from JSON
    category: str                         # Section/Category from JSON
    document_source: str                  # e.g., "Doc 3: Formatting"
    assessment_section: str               # e.g., "Whole Document", "Taxonomy"
    judgment_type: JudgmentType
    rationale: str                        # Why this rule exists
    severity: Severity = Severity.WARNING
    enabled: bool = True


@dataclass
class ViolationReport:
    """Aggregated report of all violations found."""
    text_length: int
    total_violations: int
    violations_by_severity: Dict[Severity, int]
    violations_by_category: Dict[str, int]
    violations: List[Violation]
    checked_rules: List[str]
    skipped_rules: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize report to dictionary."""
        return {
            "summary": {
                "text_length": self.text_length,
                "total_violations": self.total_violations,
                "by_severity": {s.value: c for s, c in self.violations_by_severity.items()},
                "by_category": self.violations_by_category,
                "rules_checked": len(self.checked_rules),
                "rules_skipped": len(self.skipped_rules)
            },
            "violations": [v.to_dict() for v in self.violations],
            "checked_rules": self.checked_rules,
            "skipped_rules": self.skipped_rules
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def get_violations_by_line(self) -> Dict[int, List[Violation]]:
        """Group violations by line number."""
        by_line: Dict[int, List[Violation]] = {}
        for v in self.violations:
            line = v.position.line or 0
            if line not in by_line:
                by_line[line] = []
            by_line[line].append(v)
        return by_line
