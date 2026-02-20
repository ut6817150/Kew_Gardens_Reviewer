import pytest
from pathlib import Path

@pytest.fixture
def sample_text():
    """Provide sample text for testing"""
    return """
    The species Homo sapiens has a population of 5000.
    Found in the Blue mountains region.
    """

@pytest.fixture
def temp_test_file(tmp_path):
    """Create a temporary test file"""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")
    return test_file

@pytest.fixture
def formatting_checker():
    """Provide a FormattingChecker instance"""
    from checkers.formatting import FormattingChecker
    return FormattingChecker()