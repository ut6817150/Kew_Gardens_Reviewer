"""Shared base class for section-level checkers."""

import inspect
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from ..violation import Violation

SectionItem = Tuple[str, str]


class BaseChecker(ABC):
    """Shared checker interface and violation-construction helper."""

    def begin_sweep(self) -> None:
        """Prepare any temporary state before reviewing a full parsed report."""

    def end_sweep(self) -> None:
        """Clear any temporary state after reviewing a full parsed report."""

    def check(self, section_item: SectionItem) -> List[Violation]:
        """Run this checker against one parsed ``(section_name, text)`` pair."""
        section_name, text = section_item
        return self.check_text(section_name, text)

    def strip_style_markers(
        self,
        text: str,
        *,
        italics: bool = True,
        bold: bool = True,
        superscript: bool = False,
        subscript: bool = False,
    ) -> Tuple[str, List[int]]:
        """Strip selected inline style tags and map cleaned indexes back.

        By default this matches the old per-checker helper behavior and strips:
        - italic tags: ``<i>``, ``</i>``, ``<em>``, ``</em>``
        - bold tags: ``<b>``, ``</b>``, ``<strong>``, ``</strong>``

        Optional flags also allow stripping:
        - superscript tags: ``<sup>``, ``</sup>``
        - subscript tags: ``<sub>``, ``</sub>``

        Returns:
        - the cleaned text with the selected markers removed
        - an ``index_map`` where each position in the cleaned text points back
          to the matching character index in the original text
        """
        tag_names = []
        if italics:
            tag_names.extend(["i", "em"])
        if bold:
            tag_names.extend(["b", "strong"])
        if superscript:
            tag_names.append("sup")
        if subscript:
            tag_names.append("sub")

        if not tag_names:
            return text, list(range(len(text)))

        pattern = re.compile(
            r"</?(?:" + "|".join(re.escape(tag_name) for tag_name in tag_names) + r")>",
            re.IGNORECASE,
        )

        cleaned_parts: List[str] = []
        index_map: List[int] = []
        last_index = 0

        for match in pattern.finditer(text):
            if match.start() > last_index:
                chunk = text[last_index:match.start()]
                cleaned_parts.append(chunk)
                index_map.extend(range(last_index, match.start()))
            last_index = match.end()

        if last_index < len(text):
            chunk = text[last_index:]
            cleaned_parts.append(chunk)
            index_map.extend(range(last_index, len(text)))

        return "".join(cleaned_parts), index_map

    @abstractmethod
    def check_text(self, section_name: str, text: str) -> List[Violation]:
        """Apply this rule to one parsed section."""

    def create_violation(
        self,
        section_name: str,
        text: str,
        span: Tuple[int, int],
        message: str,
        suggested_fix: Optional[str] = None,
    ) -> Violation:
        """Build a violation with exact match text plus nearby context snippet."""
        start, end = span
        matched_text = text[start:end]
        tokens = list(re.finditer(r"\S+", text))

        if not tokens:
            snippet = matched_text
        else:
            overlapping = [
                index for index, token in enumerate(tokens)
                if token.start() < end and token.end() > start
            ]

            if overlapping:
                first_index = max(0, overlapping[0] - 2)
                last_index = min(len(tokens) - 1, overlapping[-1] + 2)
            else:
                nearest_index = min(
                    range(len(tokens)),
                    key=lambda index: min(
                        abs(tokens[index].start() - start),
                        abs(tokens[index].end() - end),
                    ),
                )
                first_index = max(0, nearest_index - 2)
                last_index = min(len(tokens) - 1, nearest_index + 2)

            snippet = text[tokens[first_index].start():tokens[last_index].end()]

        return Violation(
            rule_class=type(self).__name__,
            rule_method=self.get_rule_method_name(),
            matched_text=matched_text,
            matched_snippet=snippet,
            message=message,
            suggested_fix=suggested_fix,
            section_name=self.normalize_section_name(section_name),
        )

    def normalize_section_name(self, section_name: str) -> str:
        """Hide paragraph suffixes in violations while keeping table suffixes."""
        return re.sub(r"\s+\[paragraph\s+\d+\]$", "", section_name, flags=re.IGNORECASE)

    def get_rule_method_name(self) -> str:
        """Return the checker method name that directly created the violation."""
        caller_frame = inspect.currentframe()
        if (
            caller_frame is None
            or caller_frame.f_back is None
            or caller_frame.f_back.f_back is None
        ):
            return "unknown_rule_method"

        caller_name = caller_frame.f_back.f_back.f_code.co_name
        normalized_name = caller_name[1:] if caller_name.startswith("_") else caller_name
        return f"{type(self).__name__}.{normalized_name}"
