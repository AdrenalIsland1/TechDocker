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
) -> tuple[dict, Path]:
    """Build the change package and write it as JSON; returns (package, path)."""
    package = {
        "repository": repository,
        "branch": branch,
        "actor": actor,
        "before_sha": before_sha,
        "after_sha": after_sha,
        "changed_files": [asdict(changed) for changed in changed_files],
        "generated_summary": generate_change_summary(changed_files),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    path = change_package_path(repo_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return package, path
