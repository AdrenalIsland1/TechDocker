"""Resolve and validate a repository's canonical technical document.

Every push looks for a repository-specific baseline::

    artifacts/summaries/{RepoName}_TechnicalDocument.md

(e.g. ``TechDocker_TechnicalDocument.md``, ``ProfitPulse_TechnicalDocument.md``).
When that document exists and validates it is the permanent canonical baseline:
the skeleton is built from it and Copilot is never called. When it is missing,
the baseline-initialization path (:mod:`src.baseline_initializer`) generates it
through a provider and proposes it via a pull request.

This module owns three concerns so filename checks are not scattered:

* **repository name** — resolved from ``GITHUB_REPOSITORY`` → an optional git
  remote → the repository directory name, then sanitized to a safe filename
  component (path traversal, separators, empty and unsafe values are rejected);
* **document validation** — never a bare ``Path.exists()``: a real file (no
  directory or symlink), non-empty UTF-8 Markdown, and the existing structural/
  grounding rules of :func:`src.project_summary_generator.validate_llm_summary`;
* **baseline resolution** — the preferred existing baseline among the canonical
  document, the legacy ``base_original_summary.md`` fallback, and the reviewable
  ``base_updated_summary.md``.

The module writes no documents. Its CLI is read-only and can emit machine-
readable JSON and GitHub Actions outputs without mixing diagnostics into stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional

from src.project_summary_generator import (
    SUMMARIES_DIRECTORY,
    SummaryValidationResult,
    original_summary_path,
    updated_summary_path,
    validate_llm_summary,
)
from src.repo_context_collector import RepoContext, collect_repo_context

# The canonical repository technical-document suffix (spelled exactly).
TECHNICAL_DOCUMENT_SUFFIX = "_TechnicalDocument.md"

GITHUB_REPOSITORY_ENV_VAR = "GITHUB_REPOSITORY"

# Structured statuses returned by the check/initialization flow.
STATUS_EXISTING_VALID = "existing_valid"
STATUS_MISSING = "missing"
STATUS_EXISTING_INVALID = "existing_invalid"
STATUS_GENERATED_COPILOT = "generated_copilot"
STATUS_GENERATED_DETERMINISTIC = "generated_deterministic_fallback"

# Filename safety.
_UNSAFE_NAME_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]")
_MAX_NAME_LENGTH = 100
_DEFAULT_REPOSITORY_NAME = "Repository"


class CanonicalDocumentError(ValueError):
    """A repository name or canonical document path could not be resolved."""


# ---------------------------------------------------------------------------
# Repository name resolution
# ---------------------------------------------------------------------------
def _git_remote_repo_name(repo_path: str | Path) -> Optional[str]:
    """Best-effort short repo name from ``git remote`` (offline, never raises).

    Only ``git config`` is consulted — a local read, no network. Any failure
    (no git, no remote, timeout) yields ``None`` so resolution falls through to
    the directory name.
    """
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    url = (result.stdout or "").strip()
    if result.returncode != 0 or not url:
        return None
    # Strip a trailing ".git" and take the final path/scp component.
    tail = re.split(r"[/:]", url.rstrip("/"))[-1]
    if tail.endswith(".git"):
        tail = tail[: -len(".git")]
    tail = tail.strip()
    return tail or None


def resolve_repository_name(
    env: Optional[Mapping[str, str]] = None,
    repo_path: str | Path = ".",
    *,
    remote_reader: Callable[[str | Path], Optional[str]] = _git_remote_repo_name,
) -> str:
    """Resolve the repository name (raw, pre-sanitization).

    Order: ``GITHUB_REPOSITORY`` final component → git remote helper →
    repository directory name. ``remote_reader`` is injectable so tests never
    shell out. The result is *not* yet filename-safe; call
    :func:`sanitize_repository_name` for that.
    """
    env = env if env is not None else os.environ

    github_repository = (env.get(GITHUB_REPOSITORY_ENV_VAR) or "").strip()
    if github_repository:
        candidate = github_repository.split("/")[-1].strip()
        if candidate:
            return candidate

    remote = remote_reader(repo_path)
    if remote:
        return remote

    directory = Path(repo_path).resolve().name
    return directory or _DEFAULT_REPOSITORY_NAME


def sanitize_repository_name(name: str) -> str:
    """Return a safe filename component, or raise :class:`CanonicalDocumentError`.

    Rejects empty names, path traversal (``.``/``..``), separators (``/``/``\\``)
    and null bytes outright. Other unsafe characters are replaced with ``_`` so
    a sensible display name is preserved where safe; hyphens and dots (e.g.
    ``my-repo``, ``my.repo``) are kept. If nothing safe remains, it raises.
    """
    raw = (name or "").strip()
    if not raw:
        raise CanonicalDocumentError("repository name is empty")
    if "\x00" in raw:
        raise CanonicalDocumentError("repository name contains a null byte")
    if "/" in raw or "\\" in raw:
        raise CanonicalDocumentError(
            f"repository name {name!r} contains a path separator"
        )
    if raw in (".", ".."):
        raise CanonicalDocumentError(
            f"repository name {name!r} is a path-traversal token"
        )

    sanitized = _UNSAFE_NAME_CHARS_RE.sub("_", raw)
    # Trim leading/trailing separators so no hidden file or dangling dash/dot
    # survives (e.g. ".hidden" -> "hidden", "repo." -> "repo").
    sanitized = sanitized.strip("._-")
    if not sanitized:
        raise CanonicalDocumentError(
            f"repository name {name!r} has no filename-safe characters"
        )
    if len(sanitized) > _MAX_NAME_LENGTH:
        sanitized = sanitized[:_MAX_NAME_LENGTH].rstrip("._-")
    # Defensive: a safe component can never itself be a traversal token.
    if sanitized in (".", ".."):
        raise CanonicalDocumentError(
            f"repository name {name!r} sanitizes to a path-traversal token"
        )
    return sanitized


def canonical_document_filename(
    env: Optional[Mapping[str, str]] = None,
    repo_path: str | Path = ".",
    repository_name: Optional[str] = None,
) -> str:
    """The ``{RepoName}_TechnicalDocument.md`` filename (no directory)."""
    name = repository_name or resolve_repository_name(env, repo_path)
    return f"{sanitize_repository_name(name)}{TECHNICAL_DOCUMENT_SUFFIX}"


def canonical_document_path(
    repo_path: str | Path = ".",
    env: Optional[Mapping[str, str]] = None,
    repository_name: Optional[str] = None,
) -> Path:
    """Absolute path of the canonical technical document for this repository.

    Always lands inside ``artifacts/summaries`` of ``repo_path``; a name that
    would escape that directory raises (defence in depth over sanitization).
    """
    filename = canonical_document_filename(env, repo_path, repository_name)
    summaries_dir = Path(repo_path) / SUMMARIES_DIRECTORY
    candidate = summaries_dir / filename
    # The resolved path must stay within the summaries directory.
    resolved_parent = candidate.resolve().parent
    if resolved_parent != summaries_dir.resolve():
        raise CanonicalDocumentError(
            f"canonical document path escapes the summaries directory: {candidate}"
        )
    return candidate


# ---------------------------------------------------------------------------
# Document validation (never a bare Path.exists())
# ---------------------------------------------------------------------------
def _is_safe_regular_file(path: Path) -> bool:
    """True only for an existing, non-symlink regular file."""
    try:
        return path.is_file() and not path.is_symlink()
    except OSError:
        return False


@dataclass
class CanonicalDocumentCheck:
    """Result of checking a repository's canonical document."""

    status: str  # existing_valid | missing | existing_invalid
    repository_name: str
    path: Path
    exists: bool
    problems: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status == STATUS_EXISTING_VALID

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "repository_name": self.repository_name,
            "path": str(self.path),
            "filename": self.path.name,
            "exists": self.exists,
            "problems": list(self.problems),
        }


def validate_canonical_document(
    path: str | Path,
    context: Optional[RepoContext] = None,
) -> SummaryValidationResult:
    """Validate an existing canonical document beyond ``Path.exists()``.

    Checks a real file (no directory, no symlink), non-empty UTF-8 Markdown,
    then delegates the structural/grounding rules (exactly one H1; a reasonable,
    unique, non-generic set of H2 sections with real content; no whole-document
    fence; minimum size; unsupported-technology grounding) to
    :func:`validate_llm_summary`.
    """
    path = Path(path)
    if not path.exists():
        return SummaryValidationResult(False, [f"document does not exist: {path}"])
    if path.is_symlink():
        return SummaryValidationResult(False, ["document is a symlink, not a file"])
    if not path.is_file():
        return SummaryValidationResult(False, ["document is not a regular file"])

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return SummaryValidationResult(False, ["document is not valid UTF-8"])
    except OSError as error:
        return SummaryValidationResult(False, [f"document is unreadable: {error}"])

    if not text.strip():
        return SummaryValidationResult(False, ["document is empty"])

    return validate_llm_summary(text, context)


def check_canonical_document(
    repo_path: str | Path = ".",
    env: Optional[Mapping[str, str]] = None,
    *,
    context: Optional[RepoContext] = None,
    repository_name: Optional[str] = None,
    remote_reader: Callable[[str | Path], Optional[str]] = _git_remote_repo_name,
) -> CanonicalDocumentCheck:
    """Resolve the canonical path and classify it: valid / missing / invalid.

    An existing but invalid document is reported as ``existing_invalid`` (never
    silently overwritten). Grounding checks run only when ``context`` is given.
    """
    resolved_name = repository_name or resolve_repository_name(
        env, repo_path, remote_reader=remote_reader
    )
    safe_name = sanitize_repository_name(resolved_name)
    path = canonical_document_path(repo_path, env, repository_name=safe_name)

    if not _is_safe_regular_file(path):
        # A directory or symlink at the canonical path is an invalid document,
        # not a "missing" one — refuse to treat it as absent.
        if path.exists():
            return CanonicalDocumentCheck(
                status=STATUS_EXISTING_INVALID,
                repository_name=safe_name,
                path=path,
                exists=True,
                problems=["canonical path is a directory or symlink, not a file"],
            )
        return CanonicalDocumentCheck(
            status=STATUS_MISSING,
            repository_name=safe_name,
            path=path,
            exists=False,
        )

    result = validate_canonical_document(path, context)
    if result.ok:
        return CanonicalDocumentCheck(
            status=STATUS_EXISTING_VALID,
            repository_name=safe_name,
            path=path,
            exists=True,
        )
    return CanonicalDocumentCheck(
        status=STATUS_EXISTING_INVALID,
        repository_name=safe_name,
        path=path,
        exists=True,
        problems=result.problems,
    )


# ---------------------------------------------------------------------------
# Preferred baseline resolution (canonical -> legacy -> none)
#
# The reviewable ``base_updated_summary.md`` is deliberately NOT a baseline
# candidate: it accumulates generated update blocks, so using it to (re)build
# the permanent skeleton would let generated content pollute the baseline. It
# remains only the reviewable output.
# ---------------------------------------------------------------------------
BASELINE_CANONICAL = "canonical"
BASELINE_LEGACY_ORIGINAL = "legacy_original"
BASELINE_NONE = "none"


@dataclass
class ResolvedBaseline:
    """The baseline document a normal push should read from."""

    path: Path
    kind: str  # canonical | legacy_original | none
    exists: bool


def resolve_canonical_baseline(
    repo_path: str | Path = ".",
    env: Optional[Mapping[str, str]] = None,
    *,
    remote_reader: Callable[[str | Path], Optional[str]] = _git_remote_repo_name,
) -> ResolvedBaseline:
    """The preferred *existing* baseline document, most-preferred first.

    Order: the repository canonical ``{RepoName}_TechnicalDocument.md`` → the
    legacy ``base_original_summary.md``. The reviewable ``base_updated_summary
    .md`` is never a baseline. When neither exists, returns the canonical path
    with ``exists=False`` so the caller routes to initialization. Never raises
    for an unsafe repository name — it simply skips the canonical candidate.
    """
    try:
        canonical = canonical_document_path(
            repo_path,
            env,
            repository_name=sanitize_repository_name(
                resolve_repository_name(env, repo_path, remote_reader=remote_reader)
            ),
        )
    except CanonicalDocumentError:
        canonical = None

    if canonical is not None and _is_safe_regular_file(canonical):
        return ResolvedBaseline(canonical, BASELINE_CANONICAL, True)

    legacy = original_summary_path(repo_path)
    if _is_safe_regular_file(legacy):
        return ResolvedBaseline(legacy, BASELINE_LEGACY_ORIGINAL, True)

    fallback = canonical if canonical is not None else legacy
    return ResolvedBaseline(fallback, BASELINE_NONE, False)


# ---------------------------------------------------------------------------
# Read-only CLI
# ---------------------------------------------------------------------------
def _write_github_outputs(github_output_path: str, check: CanonicalDocumentCheck) -> None:
    """Append GitHub Actions outputs — kept off stdout so JSON stays clean."""
    lines = [
        f"status={check.status}",
        f"exists={'true' if check.exists else 'false'}",
        f"repository_name={check.repository_name}",
        f"path={check.path}",
        f"filename={check.path.name}",
        f"is_valid={'true' if check.is_valid else 'false'}",
    ]
    with open(github_output_path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(
    argv: Optional[list[str]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> int:
    """Read-only check. Prints one JSON object on stdout; nothing is written."""
    environment = env if env is not None else os.environ

    parser = argparse.ArgumentParser(
        description="Resolve and check a repository's canonical technical document "
        "(read-only; writes no documents)."
    )
    parser.add_argument("--repo-path", default=".")
    parser.add_argument(
        "--with-context",
        action="store_true",
        help="Collect repository context so unsupported-technology claims are "
        "checked (slower; still read-only).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Dry run: print the JSON result only; emit no GitHub Actions "
        "outputs and write nothing at all.",
    )
    arguments = parser.parse_args(argv)

    context = (
        collect_repo_context(arguments.repo_path) if arguments.with_context else None
    )
    try:
        check = check_canonical_document(
            arguments.repo_path, environment, context=context
        )
    except CanonicalDocumentError as error:
        print(f"[canonical-document] {error}", file=sys.stderr)
        return 2

    # stdout carries ONLY the machine-readable JSON result.
    print(json.dumps(check.to_dict(), indent=2))

    github_output = environment.get("GITHUB_OUTPUT")
    if github_output and not arguments.preview:
        _write_github_outputs(github_output, check)

    print(
        f"[canonical-document] {check.status}: {check.path} "
        f"(read-only, nothing written)",
        file=sys.stderr,
    )
    for problem in check.problems:
        print(f"[canonical-document] problem: {problem}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
