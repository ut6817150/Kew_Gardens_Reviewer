"""HTML processing utilities for IUCN checker."""

from typing import List, Tuple, Dict
from bs4 import BeautifulSoup
import re


class HTMLProcessor:
    """Process HTML input for the IUCN checker."""
    
    def __init__(self, html_content: str):
        """Initialize with HTML content.
        
        Args:
            html_content: Raw HTML string
        """
        self.html = html_content
        self.soup = BeautifulSoup(html_content, 'html.parser')
        self.text = self.soup.get_text()
        self._build_formatting_map()
    
    def _build_formatting_map(self):
        """Build a map of which text ranges are formatted (italic, bold, etc.)."""
        self.italic_ranges = []
        self.bold_ranges = []
        
        # Find all italic tags
        for tag in self.soup.find_all(['i', 'em']):
            text = tag.get_text()
            if text:
                # Find all occurrences of this text in the plain text
                start = 0
                while True:
                    pos = self.text.find(text, start)
                    if pos == -1:
                        break
                    self.italic_ranges.append((pos, pos + len(text)))
                    start = pos + 1
        
        # Find all bold tags
        for tag in self.soup.find_all(['b', 'strong']):
            text = tag.get_text()
            if text:
                start = 0
                while True:
                    pos = self.text.find(text, start)
                    if pos == -1:
                        break
                    self.bold_ranges.append((pos, pos + len(text)))
                    start = pos + 1
    
    def is_italicized(self, start: int, end: int) -> bool:
        """Check if a text range is italicized.
        
        Args:
            start: Start position in plain text
            end: End position in plain text
            
        Returns:
            True if the range is italicized
        """
        for italic_start, italic_end in self.italic_ranges:
            # Check if the range is completely within an italic range
            if italic_start <= start and end <= italic_end:
                return True
        return False
    
    def is_bold(self, start: int, end: int) -> bool:
        """Check if a text range is bold."""
        for bold_start, bold_end in self.bold_ranges:
            if bold_start <= start and end <= bold_end:
                return True
        return False
    
    def get_plain_text(self) -> str:
        """Get plain text without HTML tags."""
        return self.text


def detect_input_type(content: str) -> str:
    """Detect if input is HTML or plain text.
    
    Args:
        content: Input string
        
    Returns:
        'html' or 'text'
    """
    # Simple heuristic: if it has HTML tags, treat as HTML
    html_tag_pattern = re.compile(r'<[^>]+>')
    if html_tag_pattern.search(content):
        return 'html'
    return 'text'