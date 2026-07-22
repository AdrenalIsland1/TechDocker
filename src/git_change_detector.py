"""Detect which files changed between two commits.

Second piece of the GitHub automation detection layer: given a before/after
commit pair (as delivered by a push event), list the changed files with their
change type. Only file-level metadata is collected here — full diff parsing
and summarization belong to a later phase.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.git_diff_parser import (
    build_hunk_summary,
    enclosing_symbol_objects,
    extract_python_symbols,
    header_symbol,
    parse_unified_diff,
)

# First letter of a `git diff --name-status` status -> change type.
_STATUS_MAP: dict[str, str] = {
    "A": "added",
    "M": "modified",
    "D": "deleted",
    "R": "renamed",
    "C": "copied",
}


@dataclass
class ChangedFile:
    """One file touched between two commits."""

    path: str
    change_type: str
    old_path: Optional[str] = None


@dataclass
class GitChangeSet:
    """All changed files for one push, plus its commit metadata."""

    repository: str
    branch: str
    before_sha: str
    after_sha: str
    changed_files: list[ChangedFile] = field(default_factory=list)
    commit_message: Optional[str] = None
    author: Optional[str] = None


def parse_name_status_output(output: str) -> list[ChangedFile]:
    """Parse ``git diff --name-status BEFORE AFTER`` output.

    Lines are tab-separated: a status column followed by one path (A/M/D) or
    two paths (R/C carry old and new names, e.g. ``R100``). Unknown statuses
    keep their path with ``change_type="unknown"``. Blank and malformed lines
    that carry no usable path are skipped rather than raising.
    """
    changed_files: list[ChangedFile] = []

    for line in (output or "").splitlines():
        if not line.strip():
            continue

        parts = line.split("\t")
        status = parts[0].strip()
        change_type = _STATUS_MAP.get(status[:1].upper()) if status else None

        if change_type in ("renamed", "copied"):
            old_path = parts[1].strip() if len(parts) > 1 else ""
            new_path = parts[2].strip() if len(parts) > 2 else ""
            if new_path:
                changed_files.append(
                    ChangedFile(
                        path=new_path,
                        change_type=change_type,
                        old_path=old_path or None,
                    )
                )
            elif old_path:
                # Malformed rename/copy with a single path: keep what we have.
                changed_files.append(
                    ChangedFile(path=old_path, change_type=change_type)
                )
            continue

        path = parts[1].strip() if len(parts) > 1 else ""
        if not path:
            continue  # malformed line without a path

        changed_files.append(
            ChangedFile(path=path, change_type=change_type or "unknown")
        )

    return changed_files


def detect_changed_files(
    before_sha: str,
    after_sha: str,
    repo_path: str | Path = ".",
) -> list[ChangedFile]:
    """List the files changed between two commits of a local repository.

    Runs ``git diff --name-status before_sha after_sha`` in ``repo_path``.
    A failing git command raises ``subprocess.CalledProcessError``.
    """
    result = subprocess.run(
        ["git", "diff", "--name-status", before_sha, after_sha],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_name_status_output(result.stdout)


def build_change_set(
    repository: str,
    branch: str,
    before_sha: str,
    after_sha: str,
    commit_message: Optional[str] = None,
    author: Optional[str] = None,
    repo_path: str | Path = ".",
) -> GitChangeSet:
    """Detect changed files and wrap them with the push metadata."""
    changed_files = detect_changed_files(before_sha, after_sha, repo_path)

    return GitChangeSet(
        repository=repository,
        branch=branch,
        before_sha=before_sha,
        after_sha=after_sha,
        changed_files=changed_files,
        commit_message=commit_message,
        author=author,
    )


# ---------------------------------------------------------------------------
# Detailed file-level change collection (schema v2 `what_changed`)
# ---------------------------------------------------------------------------
_BINARY_NOTE = "Binary file changed; textual hunks were not extracted."


@dataclass
class FileChange:
    """Rich, per-file change record for the change package."""

    path: str
    old_path: Optional[str]
    status: str  # added | modified | deleted | renamed
    additions: Optional[int]
    deletions: Optional[int]
    binary: bool
    binary_note: Optional[str] = None
    what_changed: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        # ``change_type`` mirrors ``status`` for backward compatibility with
        # schema-v1 readers (pr_summary_report, llm_change_analyzer) that key
        # on ``change_type``; ``status`` is the canonical v2 field.
        return {
            "path": self.path,
            "old_path": self.old_path,
            "change_type": self.status,
            "status": self.status,
            "additions": self.additions,
            "deletions": self.deletions,
            "binary": self.binary,
            "binary_note": self.binary_note,
            "what_changed": self.what_changed,
        }


@dataclass
class _Numstat:
    additions: Optional[int]
    deletions: Optional[int]
    binary: bool
    old_path: Optional[str]


def _run_git(args: list[str], repo_path: str | Path) -> str:
    """Run a git command and return stdout, decoding invalid UTF-8 safely."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return result.stdout


def _safe_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_name_status_z(output: str) -> list[tuple[str, str, Optional[str]]]:
    """Parse ``git diff --name-status -z -M`` into (status, path, old_path)."""
    tokens = (output or "").split("\x00")
    records: list[tuple[str, str, Optional[str]]] = []
    index = 0
    while index < len(tokens):
        code = tokens[index]
        index += 1
        if not code:
            continue
        letter = code[0].upper()
        if letter in ("R", "C"):
            old_path = tokens[index] if index < len(tokens) else ""
            new_path = tokens[index + 1] if index + 1 < len(tokens) else ""
            index += 2
            # A copy produces a new file; canonical status set has no "copied".
            status = "renamed" if letter == "R" else "added"
            records.append((status, new_path, old_path or None))
        else:
            path = tokens[index] if index < len(tokens) else ""
            index += 1
            status = {"A": "added", "M": "modified", "D": "deleted"}.get(
                letter, "modified"
            )
            records.append((status, path, None))
    return records


def _parse_numstat_z(output: str) -> dict[str, _Numstat]:
    """Parse ``git diff --numstat -z -M`` keyed by (new) path.

    Binary files report ``-`` counts; renames carry the path in two extra
    NUL-delimited fields after an empty path field.
    """
    tokens = (output or "").split("\x00")
    result: dict[str, _Numstat] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token:
            continue
        parts = token.split("\t")
        if len(parts) < 3:
            continue
        add_s, del_s = parts[0], parts[1]
        path = "\t".join(parts[2:])
        old_path: Optional[str] = None
        if path == "":
            old_path = tokens[index] if index < len(tokens) else ""
            path = tokens[index + 1] if index + 1 < len(tokens) else ""
            index += 2
            old_path = old_path or None
        binary = add_s == "-" or del_s == "-"
        result[path] = _Numstat(
            additions=None if binary else _safe_int(add_s),
            deletions=None if binary else _safe_int(del_s),
            binary=binary,
            old_path=old_path,
        )
    return result


def _blob_content(sha: str, path: Optional[str], repo_path: str | Path) -> str:
    """Content of ``path`` at ``sha`` via ``git show``; ``""`` when absent."""
    if not sha or not path:
        return ""
    try:
        return _run_git(["show", f"{sha}:{path}"], repo_path)
    except (subprocess.CalledProcessError, OSError):
        return ""


def _changed_line_to_dict(line) -> dict:
    # ``text_truncated`` is only emitted when true, so ordinary line entries
    # keep the stable {line_number, text} shape.
    entry = {"line_number": line.line_number, "text": line.text}
    if line.text_truncated:
        entry["text_truncated"] = True
    return entry


def _hunk_to_dict(hunk, summary: str, symbols: list[str]) -> dict:
    return {
        "old_start_line": hunk.old_start_line,
        "old_line_count": hunk.old_line_count,
        "old_end_line": hunk.old_end_line,
        "new_start_line": hunk.new_start_line,
        "new_line_count": hunk.new_line_count,
        "new_end_line": hunk.new_end_line,
        "hunk_header": hunk.hunk_header,
        "change_type": hunk.change_type,
        "summary": summary,
        "symbols": symbols,
        "added_lines": [_changed_line_to_dict(line) for line in hunk.added_lines],
        "removed_lines": [_changed_line_to_dict(line) for line in hunk.removed_lines],
        "hunk_text_truncated": hunk.hunk_text_truncated,
    }


def _build_what_changed(
    file_change: FileChange,
    before_sha: str,
    after_sha: str,
    repo_path: str | Path,
) -> list[dict]:
    """Parse textual hunks for one file and attach symbols + summaries."""
    diff_paths = (
        [file_change.old_path, file_change.path]
        if file_change.status == "renamed" and file_change.old_path
        else [file_change.path]
    )
    try:
        diff_text = _run_git(
            ["diff", "--no-color", "-M", before_sha, after_sha, "--", *diff_paths],
            repo_path,
        )
    except (subprocess.CalledProcessError, OSError):
        diff_text = ""

    hunks = parse_unified_diff(diff_text)
    if not hunks:
        return []

    is_python = file_change.path.endswith(".py")
    pre_path = file_change.old_path or file_change.path
    post_content = (
        "" if file_change.status == "deleted"
        else _blob_content(after_sha, file_change.path, repo_path)
    )
    pre_content = (
        "" if file_change.status == "added"
        else _blob_content(before_sha, pre_path, repo_path)
    )
    post_symbols = extract_python_symbols(post_content) if is_python else []
    pre_symbols = (
        extract_python_symbols(pre_content)
        if is_python and pre_path.endswith(".py")
        else []
    )

    what_changed: list[dict] = []
    for hunk in hunks:
        new_symbols = enclosing_symbol_objects(
            post_symbols, [line.line_number for line in hunk.added_lines]
        )
        old_symbols = enclosing_symbol_objects(
            pre_symbols, [line.line_number for line in hunk.removed_lines]
        )
        symbol_names: list[str] = []
        seen: set[str] = set()
        for symbol in (*new_symbols, *old_symbols):
            if symbol.qualified_name not in seen:
                seen.add(symbol.qualified_name)
                symbol_names.append(symbol.qualified_name)

        if hunk.change_type == "deleted":
            primary = old_symbols[0] if old_symbols else (
                new_symbols[0] if new_symbols else None
            )
        else:
            primary = new_symbols[0] if new_symbols else (
                old_symbols[0] if old_symbols else None
            )

        summary = build_hunk_summary(hunk, primary, header_symbol(hunk.section_heading))
        what_changed.append(_hunk_to_dict(hunk, summary, symbol_names))
    return what_changed


def collect_file_changes(
    before_sha: str,
    after_sha: str,
    repo_path: str | Path = ".",
) -> list[FileChange]:
    """Collect rich per-file change details between two commits.

    Uses ``git diff`` name-status and numstat (null-delimited, rename-aware)
    for file metadata, then parses per-file unified diffs into hunks with
    Python symbols and deterministic summaries. A failing top-level git
    command raises ``subprocess.CalledProcessError``; individual per-file or
    blob failures degrade to empty hunk lists rather than raising.
    """
    name_status = _run_git(
        ["diff", "--name-status", "-z", "-M", before_sha, after_sha], repo_path
    )
    numstat = _run_git(
        ["diff", "--numstat", "-z", "-M", before_sha, after_sha], repo_path
    )
    stats = _parse_numstat_z(numstat)

    changes: list[FileChange] = []
    for status, path, old_path in _parse_name_status_z(name_status):
        stat = stats.get(path)
        binary = stat.binary if stat else False
        if old_path is None and stat is not None and stat.old_path:
            old_path = stat.old_path

        file_change = FileChange(
            path=path,
            old_path=old_path,
            status=status,
            additions=stat.additions if stat else None,
            deletions=stat.deletions if stat else None,
            binary=binary,
        )
        if binary:
            file_change.binary_note = _BINARY_NOTE
        else:
            file_change.what_changed = _build_what_changed(
                file_change, before_sha, after_sha, repo_path
            )
        changes.append(file_change)

    return changes
