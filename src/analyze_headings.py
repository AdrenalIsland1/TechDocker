"""Command-line paragraph-analysis / inspection tool.

Usage::

    python3 -m src.analyze_headings path/to/document.docx
    python3 -m src.analyze_headings path/to/document.docx --output analysis.csv

Prints a compact, human-readable table of every paragraph's heading score and
classification, and optionally writes the full feature table to CSV.

basicallyaccepts the file path, runs the analyzer, prints a compact table, and writes the CSV.
"""

from __future__ import annotations

import argparse
import sys

from src.docx_heading_analyzer import analysis_to_dataframe, analyze_docx_file

# Columns shown in the compact terminal table.
_TERMINAL_COLUMNS = [
    "text",
    "style",
    "score",
    "classification",
    "predicted_level",
    "detection_method",
    "signals",
]

_MAX_TEXT_WIDTH = 45


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m src.analyze_headings",
        description="Inspect DOCX paragraph heading scores and classifications.",
    )
    parser.add_argument("document", help="Path to the .docx file to analyze.")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional path to write the full feature table as CSV.",
    )
    return parser


def _truncate(text: str, width: int = _MAX_TEXT_WIDTH) -> str:
    text = " ".join(str(text).split())
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def main(argv: list[str] | None = None) -> int:
    """Entry point for the inspection command."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        analysis = analyze_docx_file(args.document)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    dataframe = analysis_to_dataframe(analysis)

    if args.output:
        dataframe.to_csv(args.output, index=False)
        print(f"Wrote full analysis for {len(dataframe)} paragraphs to {args.output}\n")

    if dataframe.empty:
        print("No non-empty paragraphs found.")
        return 0

    import pandas as pd

    terminal_view = dataframe[_TERMINAL_COLUMNS].copy()
    terminal_view["text"] = terminal_view["text"].map(_truncate)

    with pd.option_context(
        "display.max_rows", None,
        "display.max_colwidth", 60,
        "display.width", None,
    ):
        print(terminal_view.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
