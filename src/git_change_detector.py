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
