"""Demo DOCX updater for the GitHub automation pipeline.

Run as ``python3 -m src.demo_docx_updater`` — inside GitHub Actions (where the
``GITHUB_*`` environment variables are provided) or locally with sensible
fallbacks (repository ``TechDocker``, current branch, ``HEAD~1..HEAD``).

The updater resolves the pushed repository to its configured document via
``config/projects.json``, detects the changed files of the push, and appends a
clearly marked "Automated Documentation Update" section to the demo DOCX. In
the real pipeline the document would come from SharePoint and the update would
be LLM-placed; this demo proves the trigger-to-document write path end to end
with a local sample file.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

from docx import Document

from src.automation_demo import extract_repository_name, is_missing_sha
from src.git_change_detector import ChangedFile, build_change_set
from src.project_resolver import ProjectConfig, resolve_project

AUTOMATED_SECTION_TITLE = "Automated Documentation Update"
AUTOMATED_SECTION_NOTE = (
    "This section was generated automatically by the TechDocker GitHub "
    "automation demo."
)


@dataclass
class UpdateResult:
    """What one updater run did, for printing and for tests."""

    document_path: Path
    project_id: str
    repository: str
    branch: str
    changed_files: list[ChangedFile] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


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


def _style_exists(document: Document, style_name: str) -> bool:
    """True when the document defines the named style.

    Must be checked *before* adding content: ``add_paragraph``/``add_heading``
    insert the paragraph first and only then resolve the style, so catching
    their ``KeyError`` afterwards would leave a stray paragraph behind.
    """
    try:
        document.styles[style_name]
    except KeyError:
        return False
    return True


def _add_heading_safely(document: Document, text: str, level: int):
    """Add a heading, falling back to a bold paragraph.

    Real-world documents do not always define the ``Heading N`` styles.
    """
    if _style_exists(document, f"Heading {level}"):
        return document.add_heading(text, level=level)
    paragraph = document.add_paragraph()
    paragraph.add_run(text).bold = True
    return paragraph


def _add_bullet_safely(document: Document, text: str):
    """Add a bulleted line, falling back to a plain ``- `` prefixed paragraph.

    The demo sample document defines no ``List Bullet`` style, so the style
    must not be assumed to exist.
    """
    if _style_exists(document, "List Bullet"):
        return document.add_paragraph(text, style="List Bullet")
    return document.add_paragraph(f"- {text}")


def append_update_section(
    document_path: str | Path,
    *,
    repository: str,
    branch: str,
    actor: str,
    before_sha: Optional[str],
    after_sha: str,
    project: ProjectConfig,
    changed_files: list[ChangedFile],
) -> None:
    """Append the marked update section to the DOCX and save it in place."""
    document = Document(str(document_path))

    _add_heading_safely(document, AUTOMATED_SECTION_TITLE, level=1)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    for label, value in (
        ("Timestamp", timestamp),
        ("Repository", repository),
        ("Branch", branch),
        ("Actor", actor),
        ("Before SHA", before_sha or "(none)"),
        ("After SHA", after_sha),
        ("Project ID", project.project_id),
        ("Document", project.document_name),
    ):
        document.add_paragraph(f"{label}: {value}")

    _add_heading_safely(document, "Changed Files", level=2)
    if changed_files:
        for changed in changed_files:
            if changed.old_path:
                text = f"{changed.change_type}: {changed.old_path} -> {changed.path}"
            else:
                text = f"{changed.change_type}: {changed.path}"
            _add_bullet_safely(document, text)
    else:
        document.add_paragraph(
            "No changed files were available for this run "
            "(first push or manual trigger)."
        )

    document.add_paragraph(AUTOMATED_SECTION_NOTE)

    document.save(str(document_path))


def run_update(env: Mapping[str, str], repo_path: str = ".") -> UpdateResult:
    """Resolve project + changed files from the environment, update the DOCX."""
    warnings: list[str] = []

    repository = extract_repository_name(env.get("GITHUB_REPOSITORY", ""))
    running_in_ci = bool(env.get("GITHUB_SHA", "").strip())

    if running_in_ci:
        after_sha = env["GITHUB_SHA"].strip()
        branch = env.get("GITHUB_REF_NAME", "").strip() or "main"
        before_sha: Optional[str] = env.get("GITHUB_EVENT_BEFORE", "").strip() or None
    else:
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

    actor = env.get("GITHUB_ACTOR", "").strip() or "local-user"

    project = resolve_project(repository)

    document_path = Path(repo_path) / project.document_location
    if not document_path.exists():
        raise FileNotFoundError(
            f"Configured demo document not found: {document_path} "
            f"(from document_location of {repository!r})"
        )

    changed_files: list[ChangedFile] = []
    if is_missing_sha(before_sha):
        before_sha = None
        warnings.append(
            "Before SHA is missing or all zeroes; updating the document "
            "with an empty changed-file list."
        )
    else:
        try:
            change_set = build_change_set(
                repository=repository,
                branch=branch,
                before_sha=before_sha,
                after_sha=after_sha,
                repo_path=repo_path,
            )
            changed_files = change_set.changed_files
        except subprocess.CalledProcessError as error:
            warnings.append(
                f"git diff {before_sha}..{after_sha} failed "
                f"({error.stderr.strip() if error.stderr else error}); "
                "updating the document with an empty changed-file list."
            )

    append_update_section(
        document_path,
        repository=repository,
        branch=branch,
        actor=actor,
        before_sha=before_sha,
        after_sha=after_sha,
        project=project,
        changed_files=changed_files,
    )

    return UpdateResult(
        document_path=document_path,
        project_id=project.project_id,
        repository=repository,
        branch=branch,
        changed_files=changed_files,
        warnings=warnings,
    )


def format_result(result: UpdateResult) -> str:
    """Render a readable terminal confirmation of the update."""
    lines = [
        "=" * 60,
        "TechDocker Demo DOCX Updater",
        "=" * 60,
        f"Updated document:    {result.document_path}",
        f"Project ID:          {result.project_id}",
        f"Repository / branch: {result.repository} / {result.branch}",
        f"Changed files:       {len(result.changed_files)}",
    ]
    if result.warnings:
        lines += [f"! {warning}" for warning in result.warnings]
    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> int:
    """Entry point for ``python3 -m src.demo_docx_updater``."""
    result = run_update(os.environ)
    print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
