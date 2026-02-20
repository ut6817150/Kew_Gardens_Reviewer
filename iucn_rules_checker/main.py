#!/usr/bin/env python3
"""CLI entry point for IUCN rule checker."""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

from engine import IUCNRuleChecker
from models import Severity

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Check text against IUCN formatting rules.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check a text file
  python -m iucn_checker input.txt

  # Check from stdin
  echo "The species is found in Vietnam" | python -m iucn_checker

  # Output to file
  python -m iucn_checker input.txt -o report.json

  # Only check specific categories
  python -m iucn_checker input.txt --categories Language Numbers

  # Show summary only
  python -m iucn_checker input.txt --summary
        """
    )

    parser.add_argument(
        'input',
        nargs='?',
        type=str,
        help='Input file path (reads from stdin if not provided)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file path (writes to stdout if not provided)'
    )

    parser.add_argument(
        '--categories',
        nargs='+',
        type=str,
        help='Only check these categories (e.g., Language Numbers IUCN)'
    )

    parser.add_argument(
        '--severity',
        choices=['info', 'warning', 'error'],
        default='info',
        help='Minimum severity level to report (default: info)'
    )

    parser.add_argument(
        '--summary',
        action='store_true',
        help='Only show summary, not individual violations'
    )

    parser.add_argument(
        '--list-categories',
        action='store_true',
        help='List available categories and exit'
    )

    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty print violations to terminal'
    )

    parser.add_argument(
        '--plain-text',
        action='store_true',
        help='Skip formatting checks that require HTML tags (use when input is plain text, not HTML)'
    )
    
    parser.add_argument(
        '--input-type',
        choices=['text', 'html', 'auto'],
        default='auto',
        help='Input format (default: auto-detect)'
    )

    args = parser.parse_args()

    # List categories if requested
    if args.list_categories:
        checker = IUCNRuleChecker()
        print("Available categories:")
        for cat in sorted(checker.get_available_categories()):
            print(f"  - {cat}")
        return 0

    # Read input text
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: File not found: {args.input}", file=sys.stderr)
            return 1
        text = input_path.read_text(encoding='utf-8')
    else:
        # Read from stdin
        if sys.stdin.isatty():
            print("Reading from stdin (Ctrl+D to end):", file=sys.stderr)
        text = sys.stdin.read()

    if not text.strip():
        print("Error: No input text provided", file=sys.stderr)
        return 1

    # Parse severity
    severity_map = {
        'info': Severity.INFO,
        'warning': Severity.WARNING,
        'error': Severity.ERROR
    }
    min_severity = severity_map[args.severity]

    # Parse categories
    enabled_categories = set(args.categories) if args.categories else None

    # Exclude formatting checks for plain text input
    if args.plain_text:
        if enabled_categories is None:
            checker_tmp = IUCNRuleChecker()
            enabled_categories = checker_tmp.get_available_categories() - {'Formatting'}
        else:
            enabled_categories.discard('Formatting')

    # Create checker and run
    checker = IUCNRuleChecker(
        enabled_categories=args.categories,
        min_severity=severity_map.get(args.severity) if args.severity else None
    )
    
    report = checker.check(text)

    # Output results
    if args.pretty:
        output = format_pretty(report, args.summary)
    elif args.summary:
        output = json.dumps(report.to_dict()['summary'], indent=2)
    else:
        output = report.to_json()

    if args.output:
        Path(args.output).write_text(output, encoding='utf-8')
        print(f"Report written to: {args.output}", file=sys.stderr)
    else:
        print(output)

    # Return exit code based on violations
    if report.violations_by_severity.get(Severity.ERROR, 0) > 0:
        return 2  # Errors found
    elif report.total_violations > 0:
        return 1  # Warnings/info found
    return 0  # Clean


def format_pretty(report, summary_only: bool = False) -> str:
    """Format report for terminal display."""
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append("IUCN Rule Checker Report")
    lines.append("=" * 60)
    lines.append("")

    # Summary
    lines.append(f"Text length: {report.text_length:,} characters")
    lines.append(f"Total violations: {report.total_violations}")
    lines.append("")

    if report.violations_by_severity:
        lines.append("By severity:")
        for sev, count in report.violations_by_severity.items():
            if count > 0:
                lines.append(f"  {sev.value}: {count}")
        lines.append("")

    if report.violations_by_category:
        lines.append("By category:")
        for cat, count in sorted(report.violations_by_category.items()):
            lines.append(f"  {cat}: {count}")
        lines.append("")

    lines.append(f"Rules checked: {len(report.checked_rules)}")
    if report.skipped_rules:
        lines.append(f"Rules skipped: {len(report.skipped_rules)}")
    lines.append("")

    # Individual violations
    if not summary_only and report.violations:
        lines.append("-" * 60)
        lines.append("Violations:")
        lines.append("-" * 60)

        for i, v in enumerate(report.violations, 1):
            lines.append(f"\n[{i}] Line {v.position.line}, Col {v.position.column}")
            lines.append(f"    Category: {v.category}")
            lines.append(f"    Severity: {v.severity.value}")
            lines.append(f"    Rule: {v.rule_name}")
            lines.append(f"    Found: \"{v.matched_text}\"")
            lines.append(f"    Message: {v.message}")
            if v.suggested_fix:
                lines.append(f"    Suggestion: {v.suggested_fix}")
            if v.context:
                # Truncate long context
                ctx = v.context[:100] + "..." if len(v.context) > 100 else v.context
                lines.append(f"    Context: {ctx}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == '__main__':
    sys.exit(main())
