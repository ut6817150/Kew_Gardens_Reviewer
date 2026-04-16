"""Violation model for IUCN rule checking."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Violation:
    """
    Represents a single rule violation found in text.

    Purpose:
        This data container stores the fields returned for one rules-based finding.
    """
    rule_class: str
    rule_method: str
    matched_text: str
    matched_snippet: str
    message: str
    suggested_fix: Optional[str] = None
    section_name: Optional[str] = None

    def to_dict(self) -> dict:
        """
        Serialize violation to dictionary for JSON export.

        Args:
            None.

        Returns:
            dict: Derived value produced by this helper.
        """
        data = {
            "rule_class": self.rule_class,
            "rule_method": self.rule_method,
            "section_name": self.section_name,
            "matched_text": self.matched_text,
            "matched_snippet": self.matched_snippet,
            "message": self.message,
            "suggested_fix": self.suggested_fix,
        }
        return data
