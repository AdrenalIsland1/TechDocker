"""Push-triggered automation demo runner.

Run as ``python3 -m src.automation_demo`` — either inside GitHub Actions
(where the ``GITHUB_*`` environment variables are provided) or locally, where
sensible fallbacks are used (repository ``TechDocker``, the current branch,
``HEAD~1..HEAD``).

The demo wires the two existing automation pieces together and prints what a
real documentation-update run would start from: which project and master
document the pushed repository maps to, and which files changed between the
before/after commits. SharePoint, LLM placement, and document writing are
future phases.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Mapping, Optional

from src.git_change_detector import ChangedFile, build_change_set
from src.project_resolver import ProjectConfig, resolve_project

# Git reports "no previous commit" (e.g. a branch's first push) as a zero SHA.
_DEFAULT_REPOSITORY = "TechDocker"


@dataclass
class DemoSummary:
    """Everything the automation demo learned about one push."""

    repository: str
    branch: str
    actor: str
    before_sha: Optional[str]
    after_sha: str
    project: Optional[ProjectConfig] = None
    changed_files: list[ChangedFile] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def extract_repository_name(full_name: str) -> str:
    """Return the short repository name from ``owner/name`` (or the name itself)."""
    if not full_name or not full_name.strip():
        return _DEFAULT_REPOSITORY
    return full_name.strip().split("/")[-1]


def is_missing_sha(sha: Optional[str]) -> bool:
    """True for absent, blank, or all-zero SHAs (git's "no commit" marker)."""
    if sha is None or not sha.strip():
        return True
    stripped = sha.strip()
    return set(stripped) == {"0"}


def _local_git_output(args: list[str], repo_path: str) -> Optional[str]:
    """Run a git query for local fallbacks; ``None`` when it fails."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    return result.stdout.strip() or None


def build_demo_summary_from_env(
    env: Mapping[str, str],
    repo_path: str = ".",
) -> DemoSummary:
    """Build a :class:`DemoSummary` from GitHub Actions-style environment vars.

    Handles a missing/zero before-SHA (first push, manual ``workflow_dispatch``)
    and an unconfigured repository gracefully: a warning is recorded and the
    rest of the summary is still produced.
    """
    warnings: list[str] = []

    repository = extract_repository_name(env.get("GITHUB_REPOSITORY", ""))
    running_in_ci = bool(env.get("GITHUB_SHA", "").strip())

    if running_in_ci:
        after_sha = env["GITHUB_SHA"].strip()
        branch = env.get("GITHUB_REF_NAME", "").strip() or "main"
        before_sha: Optional[str] = env.get("GITHUB_EVENT_BEFORE", "").strip() or None
    else:
        # Local fallback: compare the last two commits of the working repo.
        after_sha = "HEAD"
        branch = (
            _local_git_output(["rev-parse", "--abbrev-ref", "HEAD"], repo_path)
            or "main"
        )
        before_sha = (
            "HEAD~1"
            if _local_git_output(["rev-parse", "--verify", "HEAD~1"], repo_path)
            else None
        )
        if before_sha is None:
            warnings.append(
                "HEAD~1 is not available (single-commit repository?); "
                "continuing with an empty changed-file list."
            )

    actor = env.get("GITHUB_ACTOR", "").strip() or "local-user"

    summary = DemoSummary(
        repository=repository,
        branch=branch,
        actor=actor,
        before_sha=before_sha,
        after_sha=after_sha,
        warnings=warnings,
    )

    try:
        summary.project = resolve_project(repository)
    except (KeyError, FileNotFoundError, ValueError) as error:
        warnings.append(f"Could not resolve project for {repository!r}: {error}")

    if is_missing_sha(summary.before_sha):
        if running_in_ci:
            warnings.append(
                "Before SHA is missing or all zeroes (first push or manual "
                "workflow_dispatch); continuing with an empty changed-file list."
            )
        summary.before_sha = None
        return summary

    try:
        change_set = build_change_set(
            repository=repository,
            branch=branch,
            before_sha=summary.before_sha,
            after_sha=after_sha,
            repo_path=repo_path,
        )
        summary.changed_files = change_set.changed_files
    except subprocess.CalledProcessError as error:
        warnings.append(
            f"git diff {summary.before_sha}..{after_sha} failed "
            f"({error.stderr.strip() if error.stderr else error}); "
            "continuing with an empty changed-file list."
        )

    return summary


def format_summary(summary: DemoSummary) -> str:
    """Render the demo summary as a readable multi-line report."""
    lines = [
        "=" * 60,
        "TechDocker Automation Demo",
        "=" * 60,
        f"Repository: {summary.repository}",
        f"Branch:     {summary.branch}",
        f"Actor:      {summary.actor}",
        f"Before SHA: {summary.before_sha or '(none)'}",
        f"After SHA:  {summary.after_sha}",
        "",
        "Resolved Project",
        "-" * 60,
    ]

    if summary.project is not None:
        lines += [
            f"Project ID:        {summary.project.project_id}",
            f"Document:          {summary.project.document_name}",
            f"Document Location: {summary.project.document_location}",
        ]
    else:
        lines.append("(no project configuration resolved)")

    lines += ["", "Changed Files", "-" * 60]

    if summary.changed_files:
        for changed in summary.changed_files:
            if changed.old_path:
                lines.append(
                    f"- {changed.change_type}: {changed.old_path} -> {changed.path}"
                )
            else:
                lines.append(f"- {changed.change_type}: {changed.path}")
    else:
        lines.append("(no changed files detected)")

    if summary.warnings:
        lines += ["", "Warnings", "-" * 60]
        lines += [f"! {warning}" for warning in summary.warnings]

    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> int:
    """Entry point for ``python3 -m src.automation_demo``."""
    summary = build_demo_summary_from_env(os.environ)
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
