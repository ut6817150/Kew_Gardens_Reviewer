import subprocess
import pytest

def test_cli_runs_with_file():
    """Test main.py runs successfully"""
    result = subprocess.run(
        ["python", "main.py", "test_assessment.txt"],
        capture_output=True,
        text=True
    )
    # Exit code 1 means violations found, which is fine
    # Exit code 2+ would be an error
    assert result.returncode in [0, 1], f"Unexpected return code: {result.returncode}"
    assert "violations" in result.stdout.lower()


def test_cli_pretty_flag():
    """Test --pretty flag works"""
    result = subprocess.run(
        ["python", "main.py", "test_assessment.txt", "--pretty"],
        capture_output=True,
        text=True
    )
    assert result.returncode in [0, 1]
    assert "IUCN Rule Checker Report" in result.stdout
