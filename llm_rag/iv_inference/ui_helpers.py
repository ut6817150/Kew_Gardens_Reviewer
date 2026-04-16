
from __future__ import annotations

import re


def normalize_display_section_name(section_name: str) -> str:
    """Remove parser-added block markers from a section label.

    The parsed assessment can include suffixes such as:
    - ``[paragraph 3]``
    - ``[table 1]``
    - ``[row 4]``

    This helper removes those suffixes so the UI can group related draft
    content under one cleaner human-readable section heading.
    """
    return re.sub(
        r"\s+\[(?:paragraph|table|row)\s+\d+\]",
        "",
        section_name,
        flags=re.IGNORECASE,
    )
