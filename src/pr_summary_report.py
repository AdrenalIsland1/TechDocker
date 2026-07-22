"""Build the Pull Request body for a proposed summary update.

Run as ``python3 -m src.pr_summary_report`` — prints Markdown for the PR the
workflow opens after ``summary_updater`` produced new artifacts. Reads
``artifacts/change_packages/latest_change_summary.json`` when available and
degrades gracefully when it is missing (e.g. a manual dispatch with no
detectable diff). No network, no token, no file modification.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Optional

from src.change_summary_generator import change_package_path


def load_change_package(repo_path: str | Path = ".") -> Optional[dict]:
    """Return the latest change package, or ``None`` when absent/unreadable."""
    path = change_package_path(repo_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def build_pr_body(
    package: Optional[dict],
    env: Mapping[str, str] | None = None,
) -> str:
    """Render the PR body Markdown from the change package (env as fallback)."""
    env = env if env is not None else os.environ
    package = package or {}

    source_sha = package.get("after_sha") or env.get("GITHUB_SHA", "") or "(unknown)"
    actor = package.get("actor") or env.get("GITHUB_ACTOR", "") or "(unknown)"
    branch = package.get("branch") or env.get("GITHUB_REF_NAME", "") or "(unknown)"

    lines = [
        "## Suggested project summary update",
        "",
        "TechDocker generated an updated project summary for review.",
        "",
        f"- **Source commit:** `{source_sha}`",
        f"- **Actor:** {actor}",
        f"- **Branch:** {branch}",
        "",
        "### Changed files in the triggering push",
        "",
    ]

    changed_files = package.get("changed_files") or []
    if changed_files:
        for entry in changed_files:
            change_type = entry.get("change_type", "changed")
            path = entry.get("path", "(unknown)")
            old_path = entry.get("old_path")
            if old_path:
                lines.append(f"- {change_type}: `{old_path}` -> `{path}`")
            else:
                lines.append(f"- {change_type}: `{path}`")
    else:
        lines.append("- (change details unavailable — no change package found)")

    generated_summary = package.get("generated_summary")
    if generated_summary:
        lines += ["", "### Generated change summary", "", generated_summary]

    lines += [
        "",
        "---",
        "",
        "`artifacts/summaries/base_original_summary.md` remains the unchanged "
        "baseline; this PR only proposes changes to the reviewable artifacts.",
        "",
        "**Reviewer actions:** merge to accept, edit "
        "`artifacts/summaries/base_updated_summary.md` in this branch before "
        "merging, or close the PR to reject the suggestion.",
    ]
    return "\n".join(lines)


def main() -> int:
    """Print the PR body for the current repository state."""
    print(build_pr_body(load_change_package(".")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
