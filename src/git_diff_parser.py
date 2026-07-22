"""Pure, offline parsing of unified git diffs and Python symbol detection.

No subprocess, git, or network — every function here operates on strings, so
it is deterministic and unit-testable with plain fixtures. Git execution lives
in :mod:`src.git_change_detector`; JSON assembly in
:mod:`src.change_summary_generator`.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Optional

# Cap on changed lines *stored* per hunk. True counts are always preserved for
# summaries; only the stored line arrays are trimmed, and the hunk is flagged
# with ``hunk_text_truncated`` so truncation is never silent.
#
# Three independent, conservative caps guard the stored text (true additions/
# deletions counts are always preserved for metadata and summaries):
#   * lines stored per side of a hunk,
#   * characters stored for a single changed line (guards a huge minified line),
#   * total changed-text characters stored across a hunk.
# The defaults are generous for ordinary source diffs and only bite on
# pathological input.
DEFAULT_MAX_LINES_PER_HUNK = 500
DEFAULT_MAX_LINE_CHARS = 2000
DEFAULT_MAX_HUNK_TEXT_CHARS = 200_000

# Unified-diff hunk header: @@ -oldStart[,oldCount] +newStart[,newCount] @@ [section]
_HUNK_HEADER_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: (.*))?$"
)
_DEF_RE = re.compile(r"(?:^|\b)(?:async\s+def|def)\s+([A-Za-z_]\w*)")
_CLASS_RE = re.compile(r"(?:^|\b)class\s+([A-Za-z_]\w*)")


@dataclass
class ChangedLine:
    """One added or removed line with its line number on the relevant side.

    ``text_truncated`` is ``True`` when ``text`` was shortened to the per-line
    character cap (the omitted text is never stored elsewhere).
    """

    line_number: int
    text: str
    text_truncated: bool = False


@dataclass
class DiffHunk:
    """One parsed unified-diff hunk."""

    hunk_header: str
    old_start_line: int
    old_line_count: int
    old_end_line: Optional[int]
    new_start_line: int
    new_line_count: int
    new_end_line: Optional[int]
    change_type: str  # "added" | "deleted" | "modified"
    section_heading: str = ""
    added_lines: list[ChangedLine] = field(default_factory=list)
    removed_lines: list[ChangedLine] = field(default_factory=list)
    # True counts of changed content lines (may exceed the stored arrays when
    # a very large hunk is truncated).
    added_count: int = 0
    removed_count: int = 0
    hunk_text_truncated: bool = False


@dataclass
class PySymbol:
    """A Python function/class/method with its source line range."""

    qualified_name: str
    kind: str  # "function" | "async function" | "class" | "method"
    start_line: int
    end_line: int


# ---------------------------------------------------------------------------
# Hunk header / range maths
# ---------------------------------------------------------------------------
def end_line(start: int, count: int) -> Optional[int]:
    """End line for one diff side; ``None`` when that side has no lines.

    ``end = start + count - 1`` when ``count > 0``; a count of zero means the
    side has no real line range, so its end is ``None`` (never an invented line).
    """
    if count <= 0:
        return None
    return start + count - 1


def hunk_change_type(old_count: int, new_count: int) -> str:
    """Stable hunk vocabulary from the two side counts."""
    if old_count == 0 and new_count > 0:
        return "added"
    if new_count == 0 and old_count > 0:
        return "deleted"
    return "modified"


def parse_hunk_header(header: str) -> Optional[tuple[int, int, int, int, str]]:
    """Parse a ``@@ ... @@`` header. An omitted count defaults to 1.

    Returns ``(old_start, old_count, new_start, new_count, section)`` or
    ``None`` when the line is not a hunk header.
    """
    match = _HUNK_HEADER_RE.match((header or "").rstrip("\n"))
    if match is None:
        return None
    old_start = int(match.group(1))
    old_count = int(match.group(2)) if match.group(2) is not None else 1
    new_start = int(match.group(3))
    new_count = int(match.group(4)) if match.group(4) is not None else 1
    section = (match.group(5) or "").strip()
    return old_start, old_count, new_start, new_count, section


def parse_unified_diff(
    diff_text: str,
    max_lines_per_hunk: int = DEFAULT_MAX_LINES_PER_HUNK,
    max_line_chars: int = DEFAULT_MAX_LINE_CHARS,
    max_hunk_text_chars: int = DEFAULT_MAX_HUNK_TEXT_CHARS,
) -> list[DiffHunk]:
    """Parse the hunks of a single-file unified diff.

    File headers (``diff --git``, ``---``, ``+++``, ``index`` ...) and the
    ``\\ No newline at end of file`` marker appear *outside* hunks and are
    ignored there. Inside a hunk, the leading ``+``/``-`` is the diff marker
    and the remainder is content, so an added line ``+++value`` is real content
    ``++value`` (never mistaken for a file header). Only ``+``/``-`` content
    lines count as changes, tracked with their line numbers, subject to the
    stored-text caps.
    """
    hunks: list[DiffHunk] = []
    lines = (diff_text or "").split("\n")
    index = 0
    total = len(lines)

    while index < total:
        parsed = parse_hunk_header(lines[index])
        if parsed is None:
            index += 1
            continue

        old_start, old_count, new_start, new_count, section = parsed
        hunk = DiffHunk(
            hunk_header=lines[index].rstrip("\n"),
            old_start_line=old_start,
            old_line_count=old_count,
            old_end_line=end_line(old_start, old_count),
            new_start_line=new_start,
            new_line_count=new_count,
            new_end_line=end_line(new_start, new_count),
            change_type=hunk_change_type(old_count, new_count),
            section_heading=section,
        )

        stored_chars = 0
        old_ln = old_start
        new_ln = new_start
        index += 1
        while index < total:
            body = lines[index]
            # A new hunk or a new file's header ends this hunk. These only
            # occur between hunks, never as in-hunk content.
            if body.startswith("@@") or body.startswith("diff --git "):
                break
            if body.startswith("\\"):  # "\ No newline at end of file"
                index += 1
                continue

            if body.startswith("+"):
                hunk.added_count += 1
                stored_chars = _store_changed_line(
                    hunk, hunk.added_lines, new_ln, body[1:], stored_chars,
                    max_lines_per_hunk, max_line_chars, max_hunk_text_chars,
                )
                new_ln += 1
            elif body.startswith("-"):
                hunk.removed_count += 1
                stored_chars = _store_changed_line(
                    hunk, hunk.removed_lines, old_ln, body[1:], stored_chars,
                    max_lines_per_hunk, max_line_chars, max_hunk_text_chars,
                )
                old_ln += 1
            else:
                # Context line (starts with a space, or a stray blank line).
                old_ln += 1
                new_ln += 1
            index += 1

        hunks.append(hunk)

    return hunks


def _store_changed_line(
    hunk: DiffHunk,
    target: list[ChangedLine],
    line_number: int,
    text: str,
    stored_chars: int,
    max_lines_per_hunk: int,
    max_line_chars: int,
    max_hunk_text_chars: int,
) -> int:
    """Store one changed line under the stored-text caps; return new char total.

    Truncation is never silent: any cap that bites sets ``hunk_text_truncated``,
    and a shortened line is flagged via ``ChangedLine.text_truncated``. The true
    ``added_count``/``removed_count`` are updated by the caller regardless.
    """
    if len(target) >= max_lines_per_hunk:
        hunk.hunk_text_truncated = True
        return stored_chars

    stored_text = text
    text_truncated = False
    if len(stored_text) > max_line_chars:
        stored_text = stored_text[:max_line_chars]
        text_truncated = True
        hunk.hunk_text_truncated = True

    if stored_chars + len(stored_text) > max_hunk_text_chars:
        # Storing this line would exceed the per-hunk text budget; stop here.
        hunk.hunk_text_truncated = True
        return stored_chars

    target.append(ChangedLine(line_number, stored_text, text_truncated))
    return stored_chars + len(stored_text)


# ---------------------------------------------------------------------------
# Python symbol detection (best-effort, never raises)
# ---------------------------------------------------------------------------
def extract_python_symbols(source: str) -> list[PySymbol]:
    """Functions/classes/methods in Python source, with line ranges.

    Best-effort: syntax errors (including partially edited files) yield an
    empty list rather than raising.
    """
    try:
        tree = ast.parse(source or "")
    except (SyntaxError, ValueError):
        return []

    symbols: list[PySymbol] = []

    def end_of(node: ast.AST) -> int:
        return getattr(node, "end_lineno", None) or node.lineno

    def visit(node: ast.AST, prefix: str, in_class: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                qualified = prefix + child.name
                symbols.append(
                    PySymbol(qualified, "class", child.lineno, end_of(child))
                )
                visit(child, qualified + ".", True)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = prefix + child.name
                if isinstance(child, ast.AsyncFunctionDef):
                    kind = "async function"
                elif in_class:
                    kind = "method"
                else:
                    kind = "function"
                symbols.append(
                    PySymbol(qualified, kind, child.lineno, end_of(child))
                )
                visit(child, qualified + ".", False)

    visit(tree, "", False)
    return symbols


def enclosing_symbol_objects(
    symbols: list[PySymbol], line_numbers: list[int]
) -> list[PySymbol]:
    """Innermost symbols enclosing the given line numbers, deduped.

    Order is deterministic (by the line at which each was first matched).
    """
    ordered: list[PySymbol] = []
    seen: set[str] = set()
    for line in sorted(set(line_numbers)):
        candidates = [
            symbol
            for symbol in symbols
            if symbol.start_line <= line <= symbol.end_line
        ]
        if not candidates:
            continue
        innermost = min(
            candidates, key=lambda s: (s.end_line - s.start_line, s.start_line)
        )
        if innermost.qualified_name not in seen:
            seen.add(innermost.qualified_name)
            ordered.append(innermost)
    return ordered


# ---------------------------------------------------------------------------
# Deterministic hunk summaries
# ---------------------------------------------------------------------------
def header_symbol(section_heading: str) -> Optional[tuple[str, str]]:
    """(kind, name) parsed from a hunk-header section heading, or ``None``.

    Git puts the enclosing function/class after the closing ``@@``; this is a
    safe fallback when AST symbol detection is unavailable.
    """
    if not section_heading:
        return None
    match = _DEF_RE.search(section_heading)
    if match is not None:
        kind = "async function" if section_heading.lstrip().startswith("async") else "function"
        return kind, match.group(1)
    match = _CLASS_RE.search(section_heading)
    if match is not None:
        return "class", match.group(1)
    return None


def build_hunk_summary(
    hunk: DiffHunk,
    primary_symbol: Optional[PySymbol] = None,
    header_sym: Optional[tuple[str, str]] = None,
) -> str:
    """Deterministic, conservative one-line summary from metadata only.

    Uses only mechanically-known facts (counts and the enclosing symbol); it
    never asserts intent or quality.
    """
    if primary_symbol is not None:
        phrase = f"{primary_symbol.kind} {primary_symbol.qualified_name}"
    elif header_sym is not None:
        phrase = f"{header_sym[0]} {header_sym[1]}"
    else:
        phrase = ""

    added = hunk.added_count
    removed = hunk.removed_count

    if hunk.change_type == "added":
        location = f" in {phrase}" if phrase else ""
        return f"Added {added} lines{location}."
    if hunk.change_type == "deleted":
        location = f" from {phrase}" if phrase else ""
        return f"Deleted {removed} lines{location}."
    location = f" {phrase} with" if phrase else ""
    return f"Modified{location} {added} added and {removed} removed lines."
