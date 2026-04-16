#!/usr/bin/env bash
# Run unit tests and save results to test_results.txt in this folder.
# Usage: bash unit_tests/run_tests.sh   (from project root)
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
VENV="$PROJECT_ROOT/.venv/bin/python"

echo "Running unit tests..."
"$VENV" -m pytest "$SCRIPT_DIR" -v --tb=short 2>&1 | tee "$SCRIPT_DIR/test_results.txt"
echo ""
echo "Results saved to unit_tests/test_results.txt"
