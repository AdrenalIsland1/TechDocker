"""Create a structured change package from git-diff results.

Formats a concise deterministic summary of a push's changed files and saves
the whole package to ``artifacts/change_packages/latest_change_summary.json``.
Detection itself uses :mod:`src.git_change_detector` (callers pass the changed
files in, so unit tests never run real git). A future phase will replace the
deterministic text with a Copilot/LLM summary of the actual diffs.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.git_change_detector import ChangedFile

CHANGE_PACKAGE_DIRECTORY = Path("artifacts") / "change_packages"
CHANGE_PACKAGE_NAME = "latest_change_summary.json"

# Schema history: implicit v1 (file names only) -> v2 (per-file additions/
# deletions, binary status, hunk ``what_changed`` with per-line added_lines/
# removed_lines) -> v3, which replaces the verbose per-line arrays with coherent
# ``change_blocks`` (multiline added_text/removed_text, true ranges, per-block
# symbols and truncation). Readers tolerate v1/v2/v3 via
# :mod:`src.change_package_reader`.
SCHEMA_VERSION = 3


def change_package_path(repo_path: str | Path = ".") -> Path:
    return Path(repo_path) / CHANGE_PACKAGE_DIRECTORY / CHANGE_PACKAGE_NAME


def generate_change_summary(changed_files: list[ChangedFile]) -> str:
    """Concise deterministic text describing the changed files."""
    if not changed_files:
        return "No changed files were available for this run."

    by_type = Counter(changed.change_type for changed in changed_files)
    counts = ", ".join(
        f"{count} {change_type}" for change_type, count in sorted(by_type.items())
    )
    listed = "; ".join(
        f"{changed.change_type} {changed.path}" for changed in changed_files[:10]
    )
    suffix = "" if len(changed_files) <= 10 else f" (and {len(changed_files) - 10} more)"
    return (
        f"{len(changed_files)} file(s) changed ({counts}): {listed}{suffix}."
    )


def create_change_package(
    repository: str,
    branch: str,
    actor: str,
    before_sha: Optional[str],
    after_sha: str,
    changed_files: list[ChangedFile],
    repo_path: str | Path = ".",
    file_details: Optional[list[dict]] = None,
) -> tuple[dict, Path]:
    """Build the change package and write it as JSON; returns (package, path).

    When ``file_details`` is provided (the enriched schema-v2 entries produced
    by :func:`src.git_change_detector.collect_file_changes`), those are used
    for ``changed_files``. Otherwise the entries fall back to the basic
    ``ChangedFile`` shape (``path``/``change_type``/``old_path``), keeping
    backward compatibility with callers and tests that pass no details.
    """
    if file_details is not None:
        changed_entries: list[dict] = file_details
    else:
        changed_entries = [asdict(changed) for changed in changed_files]

    package = {
        "schema_version": SCHEMA_VERSION,
        "repository": repository,
        "branch": branch,
        "actor": actor,
        "before_sha": before_sha,
        "after_sha": after_sha,
        "changed_files": changed_entries,
        "generated_summary": generate_change_summary(changed_files),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    path = change_package_path(repo_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return package, path
