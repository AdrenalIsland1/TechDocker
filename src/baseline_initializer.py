"""Baseline initialization: create a repository's canonical technical document.

Run as ``python3 -m src.baseline_initializer`` (``--preview`` for a dry run,
``--plan`` for a read-only routing decision, ``--force`` to regenerate an
invalid document).

Creating a permanent baseline is a deliberate, gated action — a normal push
never silently writes one:

1. resolve the repository name and canonical document path,
2. validate any existing document,
3. **valid** → do nothing (no provider call, no write),
4. **missing** → only initialize when initialization is *explicitly enabled*
   (``TECHDOCKER_ENABLE_CANONICAL_INITIALIZATION=true``) **and** a provider is
   *explicitly named* (``TECHDOCKER_BASE_SUMMARY_PROVIDER``, e.g.
   ``deterministic`` or ``copilot-cli``). An empty provider is never permission.
   A directly-injected ``provider`` (tests/manual previews) is itself explicit
   permission. When not enabled/selected, return ``initialization_pending`` and
   write nothing. When enabled: generate Markdown, validate it, and install
   three outputs together with rollback — the canonical
   ``{RepoName}_TechnicalDocument.md``, ``base_skeleton.json`` built from it,
   and ``base_updated_summary.md`` initialized from it,
5. a non-deterministic provider that fails (missing/auth/timeout/invalid) does
   **not** silently fall back to deterministic; fallback requires the explicit
   ``TECHDOCKER_ALLOW_BASELINE_PROVIDER_FALLBACK=true``. Otherwise the result is
   ``generation_failed`` with no writes,
6. **existing but invalid** → ``existing_invalid``; refuse to overwrite unless
   ``--force``.

Initialization never runs the incremental change updater, never modifies the
legacy ``base_original_summary.md``, and in preview mode writes nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional

from src.canonical_document import (
    STATUS_EXISTING_INVALID,
    STATUS_EXISTING_VALID,
    STATUS_GENERATED_COPILOT,
    STATUS_GENERATED_DETERMINISTIC,
    STATUS_MISSING,
    check_canonical_document,
    original_summary_path,
    resolve_canonical_baseline,
    validate_canonical_document,
)
from src.copilot_summary_provider import (
    BASE_SUMMARY_PROVIDER_ENV_VAR,
    COPILOT_CLI_PROVIDER_NAME,
    CopilotCliSummaryProvider,
    CopilotProviderError,
)
from src.project_summary_generator import (
    LocalDeterministicSummaryProvider,
    SummaryProvider,
    validate_llm_summary,
    updated_summary_path,
)
from src.repo_context_collector import RepoContext, collect_repo_context
from src.summary_skeleton_builder import build_summary_skeleton, summary_skeleton_path
from src.summary_skeleton_store import save_summary_skeleton

# Additional initializer statuses (the canonical-document statuses are reused).
STATUS_GENERATION_FAILED = "generation_failed"
STATUS_INITIALIZATION_PENDING = "initialization_pending"

# Explicit, gated production behaviour. Creating a permanent canonical baseline
# requires BOTH of these; empty/unset means "do not initialize".
ENABLE_INITIALIZATION_ENV_VAR = "TECHDOCKER_ENABLE_CANONICAL_INITIALIZATION"
# A non-deterministic provider failure only falls back to deterministic when
# this is explicitly true (default false, per Problem 2).
ALLOW_PROVIDER_FALLBACK_ENV_VAR = "TECHDOCKER_ALLOW_BASELINE_PROVIDER_FALLBACK"

PROVIDER_DETERMINISTIC_NAME = "deterministic"

_PROVIDER_COPILOT = "copilot"
_PROVIDER_DETERMINISTIC = "deterministic"
_TRUTHY = {"true", "1", "yes", "on"}


def _is_truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in _TRUTHY


@dataclass
class BaselineInitResult:
    """Structured outcome of one initialization attempt."""

    status: str
    repository_name: str
    canonical_path: Path
    wrote_files: bool = False
    created: list[Path] = field(default_factory=list)
    proposed_paths: list[Path] = field(default_factory=list)
    provider_used: Optional[str] = None
    problems: list[str] = field(default_factory=list)
    content_metadata: dict = field(default_factory=dict)

    @property
    def is_baseline_initialization(self) -> bool:
        return self.status in (STATUS_GENERATED_COPILOT, STATUS_GENERATED_DETERMINISTIC)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "repository_name": self.repository_name,
            "canonical_path": str(self.canonical_path),
            "canonical_filename": self.canonical_path.name,
            "wrote_files": self.wrote_files,
            "created": [str(path) for path in self.created],
            "proposed_paths": [str(path) for path in self.proposed_paths],
            "provider_used": self.provider_used,
            "problems": list(self.problems),
            "content_metadata": dict(self.content_metadata),
            "baseline_initialization": self.is_baseline_initialization,
        }


# ---------------------------------------------------------------------------
# Provider selection (gated) and generation
# ---------------------------------------------------------------------------
def _provider_kind(provider: SummaryProvider) -> str:
    return (
        _PROVIDER_COPILOT
        if getattr(provider, "name", "") == COPILOT_CLI_PROVIDER_NAME
        else _PROVIDER_DETERMINISTIC
    )


def _build_named_provider(
    name: str, env: Optional[Mapping[str, str]]
) -> tuple[Optional[SummaryProvider], Optional[str]]:
    """Build a provider for an EXPLICIT provider name, or ``(None, None)``."""
    normalized = (name or "").strip().lower()
    if normalized == COPILOT_CLI_PROVIDER_NAME:
        return CopilotCliSummaryProvider(env=env), _PROVIDER_COPILOT
    if normalized == PROVIDER_DETERMINISTIC_NAME:
        return LocalDeterministicSummaryProvider(), _PROVIDER_DETERMINISTIC
    return None, None


def _resolve_initialization_provider(
    env: Mapping[str, str],
    provider: Optional[SummaryProvider],
) -> tuple[Optional[SummaryProvider], Optional[str], Optional[str]]:
    """Decide the provider for a MISSING document, enforcing the production gate.

    Returns ``(provider, provider_kind, pending_reason)``. A directly-injected
    provider is itself explicit permission. Otherwise initialization must be
    explicitly enabled AND an explicit provider named — an empty provider value
    is never permission to write a deterministic baseline.
    """
    if provider is not None:
        return provider, _provider_kind(provider), None

    if not _is_truthy(env.get(ENABLE_INITIALIZATION_ENV_VAR)):
        return None, None, (
            "canonical initialization is not enabled; set "
            f"{ENABLE_INITIALIZATION_ENV_VAR}=true and an explicit "
            f"{BASE_SUMMARY_PROVIDER_ENV_VAR} to create a permanent baseline"
        )

    provider_name = (env.get(BASE_SUMMARY_PROVIDER_ENV_VAR) or "").strip()
    if not provider_name:
        return None, None, (
            f"no base summary provider selected; set {BASE_SUMMARY_PROVIDER_ENV_VAR} "
            "explicitly (an empty value is not permission to write a deterministic "
            "baseline)"
        )

    built, kind = _build_named_provider(provider_name, env)
    if built is None:
        return None, None, f"unknown base summary provider {provider_name!r}"
    return built, kind, None


def _resolve_allow_fallback(
    env: Mapping[str, str], allow_fallback: Optional[bool]
) -> bool:
    if allow_fallback is not None:
        return allow_fallback
    return _is_truthy(env.get(ALLOW_PROVIDER_FALLBACK_ENV_VAR))


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def _deterministic_summary(
    context: RepoContext,
) -> tuple[Optional[str], Optional[str], Optional[str], list[str]]:
    text = LocalDeterministicSummaryProvider().generate_summary(context)
    result = validate_llm_summary(text, context)
    if not result.ok:
        return None, None, None, [
            "deterministic summary failed validation: " + "; ".join(result.problems)
        ]
    return (
        _ensure_trailing_newline(text),
        STATUS_GENERATED_DETERMINISTIC,
        _PROVIDER_DETERMINISTIC,
        [],
    )


def _generate_validated_summary(
    provider: SummaryProvider,
    provider_kind: str,
    context: RepoContext,
    allow_fallback: bool,
) -> tuple[Optional[str], Optional[str], Optional[str], list[str]]:
    """Return (markdown, status, provider_used, problems).

    A NON-deterministic provider that fails (transport error or invalid output)
    only falls back to the deterministic template when ``allow_fallback`` is
    True. Otherwise the failure is clean: ``(None, ...)`` and no writes.
    """
    non_deterministic = provider_kind != _PROVIDER_DETERMINISTIC

    try:
        text = provider.generate_summary(context)
    except CopilotProviderError as error:
        if non_deterministic and allow_fallback:
            return _deterministic_summary(context)
        return None, None, None, [f"{provider_kind} provider failed: {error}"]

    result = validate_llm_summary(text, context)
    if result.ok:
        status = (
            STATUS_GENERATED_COPILOT if provider_kind == _PROVIDER_COPILOT
            else STATUS_GENERATED_DETERMINISTIC
        )
        return _ensure_trailing_newline(text), status, provider_kind, []

    if non_deterministic and allow_fallback:
        return _deterministic_summary(context)
    return None, None, None, [
        f"{provider_kind} summary failed validation: " + "; ".join(result.problems)
    ]


# ---------------------------------------------------------------------------
# Rollback-safe multi-file installation
#
# Three independent filesystem paths CANNOT be committed by one OS-level atomic
# operation. Instead we prepare and validate every temporary output first, back
# up any destination that already exists, then attempt the replacements. If any
# replacement fails, every pre-existing destination is restored byte-for-byte
# and every newly-created destination is removed, so the set is all-or-nothing.
# ---------------------------------------------------------------------------
class InitializationCommitError(RuntimeError):
    """A destination replacement failed; all changes were rolled back."""


ReplaceFn = Callable[[Path, Path], None]


def _relative_source(path: Path, repo_path: str | Path) -> str:
    try:
        return str(path.resolve().relative_to(Path(repo_path).resolve()))
    except ValueError:
        return str(path)


def _temp_sibling(path: Path, suffix: str) -> Path:
    return path.parent / f".{path.name}.{suffix}-{os.getpid()}"


def _commit_outputs_with_rollback(
    installs: list[tuple[Path, Path]],
    *,
    replace: ReplaceFn = os.replace,
) -> None:
    """Move each ``(destination, temp)`` into place, or roll everything back.

    Not a single atomic OS operation: it backs up every pre-existing
    destination, replaces each in turn, and on any failure restores the
    pre-existing destinations byte-for-byte, removes destinations that did not
    previously exist, and cleans all temporary and backup files. On success no
    temporary or backup file remains.
    """
    existed: dict[Path, bool] = {}
    backups: dict[Path, Path] = {}
    replaced: list[Path] = []

    try:
        # Back up destinations that already exist (recoverable copies).
        for destination, _temp in installs:
            existed[destination] = destination.exists()
            if existed[destination]:
                backup = _temp_sibling(destination, "bak")
                shutil.copy2(destination, backup)
                backups[destination] = backup

        # Attempt the replacements; the injected ``replace`` may fail on any.
        for destination, temp in installs:
            replace(temp, destination)
            replaced.append(destination)
    except BaseException as error:
        # Roll back: restore pre-existing destinations, remove new ones. The
        # restore never uses the (possibly failing) injected ``replace``.
        for destination in replaced:
            if existed.get(destination):
                shutil.copy2(backups[destination], destination)
            else:
                _silent_unlink(destination)
        if isinstance(error, Exception):
            raise InitializationCommitError(
                f"failed to install initialization outputs: {error}"
            ) from error
        raise
    finally:
        # Always remove temps (replaced ones are already gone) and backups.
        for _destination, temp in installs:
            _silent_unlink(temp)
        for backup in backups.values():
            _silent_unlink(backup)


def _silent_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def install_initialization_outputs(
    canonical_path: Path,
    updated_path: Path,
    skeleton_path: Path,
    summary: str,
    source_rel: str,
    *,
    replace: ReplaceFn = os.replace,
) -> list[Path]:
    """Prepare, validate, then rollback-safely install the three outputs.

    Returns the installed paths. Raises :class:`InitializationCommitError` (with
    every change rolled back) if a replacement fails, or the underlying error if
    preparation fails before any replacement — either way no partial set, and no
    ``.tmp``/``.bak`` files, remain.
    """
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    updated_path.parent.mkdir(parents=True, exist_ok=True)
    skeleton_path.parent.mkdir(parents=True, exist_ok=True)

    canonical_tmp = _temp_sibling(canonical_path, "tmp")
    updated_tmp = _temp_sibling(updated_path, "tmp")
    skeleton_tmp = _temp_sibling(skeleton_path, "tmp")
    prep_temps = [canonical_tmp, updated_tmp, skeleton_tmp]

    # Phase 1: prepare and validate every temporary output. A failure here
    # installs nothing.
    try:
        canonical_tmp.write_text(summary, encoding="utf-8")
        updated_tmp.write_text(summary, encoding="utf-8")
        # Build the skeleton from the prepared canonical content, recording the
        # FINAL canonical path so metadata matches the document actually used.
        skeleton = build_summary_skeleton(canonical_tmp, source_summary_path=source_rel)
        if not skeleton.sections:
            raise ValueError("summary produced no skeleton sections")
        save_summary_skeleton(skeleton, skeleton_tmp)
    except BaseException:
        for temp in prep_temps:
            _silent_unlink(temp)
        raise

    # Phase 2: install the prepared outputs transactionally.
    _commit_outputs_with_rollback(
        [
            (canonical_path, canonical_tmp),
            (updated_path, updated_tmp),
            (skeleton_path, skeleton_tmp),
        ],
        replace=replace,
    )
    return [canonical_path, skeleton_path, updated_path]


def _content_metadata(summary: str, source_rel: str) -> dict:
    lines = summary.splitlines()
    return {
        "chars": len(summary),
        "lines": len(lines),
        "h1_count": sum(1 for line in lines if line.startswith("# ")),
        "h2_count": sum(1 for line in lines if line.startswith("## ")),
        "source_summary_path": source_rel,
    }


# ---------------------------------------------------------------------------
# Read-only push plan (routes a push without writing anything)
# ---------------------------------------------------------------------------
ACTION_INCREMENTAL_CANONICAL = "incremental_update"
ACTION_INCREMENTAL_LEGACY = "incremental_update_legacy"
ACTION_INITIALIZE = "initialize_baseline"
ACTION_INITIALIZATION_PENDING = "initialization_pending"
ACTION_MANUAL_REVIEW = "manual_review"


@dataclass
class PushPlan:
    """What a push should do, decided without writing anything."""

    action: str
    canonical_status: str
    repository_name: str
    canonical_path: Path
    legacy_baseline_valid: bool = False
    provider_kind: Optional[str] = None
    reason: str = ""
    problems: list[str] = field(default_factory=list)

    @property
    def run_updater(self) -> bool:
        return self.action in (ACTION_INCREMENTAL_CANONICAL, ACTION_INCREMENTAL_LEGACY)

    @property
    def run_initializer(self) -> bool:
        return self.action == ACTION_INITIALIZE

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "canonical_status": self.canonical_status,
            "repository_name": self.repository_name,
            "canonical_path": str(self.canonical_path),
            "canonical_filename": self.canonical_path.name,
            "legacy_baseline_valid": self.legacy_baseline_valid,
            "provider_kind": self.provider_kind,
            "run_updater": self.run_updater,
            "run_initializer": self.run_initializer,
            "reason": self.reason,
            "problems": list(self.problems),
        }


def _legacy_baseline_is_valid(repo_path: str | Path, context: RepoContext) -> bool:
    legacy = original_summary_path(repo_path)
    if not legacy.exists():
        return False
    return validate_canonical_document(legacy, context).ok


def resolve_push_plan(
    repo_path: str | Path = ".",
    env: Optional[Mapping[str, str]] = None,
    *,
    context: Optional[RepoContext] = None,
    provider: Optional[SummaryProvider] = None,
) -> PushPlan:
    """Decide the action for one push — read-only, writes nothing.

    * valid canonical            → incremental update,
    * invalid canonical          → manual review (no update, no overwrite),
    * missing + init permitted    → initialize baseline,
    * missing + not permitted + valid legacy → legacy incremental update,
    * missing + not permitted + no legacy     → initialization pending.
    """
    environment = env if env is not None else os.environ
    context = context if context is not None else collect_repo_context(repo_path)

    check = check_canonical_document(repo_path, environment, context=context)
    name, canonical_path = check.repository_name, check.path

    if check.status == STATUS_EXISTING_VALID:
        return PushPlan(
            ACTION_INCREMENTAL_CANONICAL, check.status, name, canonical_path,
            reason="canonical document is valid; running the incremental update",
        )
    if check.status == STATUS_EXISTING_INVALID:
        return PushPlan(
            ACTION_MANUAL_REVIEW, check.status, name, canonical_path,
            reason="canonical document exists but is invalid; manual review required",
            problems=check.problems,
        )

    # Missing: initialize only when explicitly permitted (or a provider was
    # injected, which is itself explicit permission).
    resolved_provider, provider_kind, pending_reason = _resolve_initialization_provider(
        environment, provider
    )
    if resolved_provider is not None:
        return PushPlan(
            ACTION_INITIALIZE, check.status, name, canonical_path,
            provider_kind=provider_kind,
            reason="canonical document missing and initialization is enabled",
        )

    if _legacy_baseline_is_valid(repo_path, context):
        return PushPlan(
            ACTION_INCREMENTAL_LEGACY, check.status, name, canonical_path,
            legacy_baseline_valid=True,
            reason=(
                "canonical initialization is deferred; continuing the incremental "
                "pipeline on the legacy base_original_summary.md"
            ),
        )

    return PushPlan(
        ACTION_INITIALIZATION_PENDING, check.status, name, canonical_path,
        reason="no valid canonical or legacy baseline; initialization is pending",
        problems=[pending_reason] if pending_reason else [],
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def initialize_baseline(
    repo_path: str | Path = ".",
    env: Optional[Mapping[str, str]] = None,
    *,
    provider: Optional[SummaryProvider] = None,
    context: Optional[RepoContext] = None,
    allow_fallback: Optional[bool] = None,
    force: bool = False,
    preview: bool = False,
    replace: ReplaceFn = os.replace,
) -> BaselineInitResult:
    """Initialize the canonical baseline for this repository (see module docs)."""
    environment = env if env is not None else os.environ
    context = context if context is not None else collect_repo_context(repo_path)

    check = check_canonical_document(repo_path, environment, context=context)
    canonical_path = check.path

    # Valid existing document: never call a provider, never write.
    if check.status == STATUS_EXISTING_VALID and not force:
        return BaselineInitResult(
            status=STATUS_EXISTING_VALID,
            repository_name=check.repository_name,
            canonical_path=canonical_path,
            wrote_files=False,
        )

    # Invalid existing document: refuse to overwrite unless forced.
    if check.status == STATUS_EXISTING_INVALID and not force:
        return BaselineInitResult(
            status=STATUS_EXISTING_INVALID,
            repository_name=check.repository_name,
            canonical_path=canonical_path,
            wrote_files=False,
            problems=check.problems,
        )

    # Missing (or forced regeneration): the production gate must permit writing
    # a permanent baseline. An injected provider is explicit permission.
    resolved_provider, provider_kind, pending_reason = _resolve_initialization_provider(
        environment, provider
    )
    if resolved_provider is None:
        # Not enabled / no explicit provider → do not initialize, write nothing.
        return BaselineInitResult(
            status=STATUS_INITIALIZATION_PENDING,
            repository_name=check.repository_name,
            canonical_path=canonical_path,
            wrote_files=False,
            problems=[pending_reason] if pending_reason else [],
        )

    allow_fallback = _resolve_allow_fallback(environment, allow_fallback)
    summary, status, provider_used, problems = _generate_validated_summary(
        resolved_provider, provider_kind, context, allow_fallback
    )
    if summary is None:
        return BaselineInitResult(
            status=STATUS_GENERATION_FAILED,
            repository_name=check.repository_name,
            canonical_path=canonical_path,
            wrote_files=False,
            provider_used=provider_kind,
            problems=problems,
        )

    updated_path = updated_summary_path(repo_path)
    skeleton_path = summary_skeleton_path(repo_path)
    source_rel = _relative_source(canonical_path, repo_path)
    metadata = _content_metadata(summary, source_rel)

    # Preview: report the proposal, write nothing at all.
    if preview:
        return BaselineInitResult(
            status=status,
            repository_name=check.repository_name,
            canonical_path=canonical_path,
            wrote_files=False,
            provider_used=provider_used,
            proposed_paths=[canonical_path, skeleton_path, updated_path],
            content_metadata=metadata,
        )

    try:
        created = install_initialization_outputs(
            canonical_path, updated_path, skeleton_path, summary, source_rel,
            replace=replace,
        )
    except (InitializationCommitError, OSError) as error:
        # The transactional install rolled everything back; report a clean
        # failure with no partial writes.
        return BaselineInitResult(
            status=STATUS_GENERATION_FAILED,
            repository_name=check.repository_name,
            canonical_path=canonical_path,
            wrote_files=False,
            provider_used=provider_used,
            problems=[f"initialization commit failed: {error}"],
        )

    return BaselineInitResult(
        status=status,
        repository_name=check.repository_name,
        canonical_path=canonical_path,
        wrote_files=True,
        created=created,
        provider_used=provider_used,
        content_metadata=metadata,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(
    argv: Optional[list[str]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> int:
    """Initialize the baseline; print a JSON result. ``--preview`` writes nothing."""
    environment = env if env is not None else os.environ

    parser = argparse.ArgumentParser(
        description="Initialize a repository's canonical technical document."
    )
    parser.add_argument("--repo-path", default=".")
    parser.add_argument(
        "--plan", action="store_true",
        help="Read-only: print the routing decision (action) and write nothing.",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Dry run: report the proposal and write nothing.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate even when an existing document is present/invalid.",
    )
    arguments = parser.parse_args(argv)

    # Read-only routing decision for the workflow — never writes documents.
    if arguments.plan:
        plan = resolve_push_plan(arguments.repo_path, environment)
        print(json.dumps(plan.to_dict(), indent=2))
        github_output = environment.get("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a", encoding="utf-8") as handle:
                handle.write(f"action={plan.action}\n")
                handle.write(f"canonical_status={plan.canonical_status}\n")
                handle.write(f"repository_name={plan.repository_name}\n")
                handle.write(f"canonical_path={plan.canonical_path}\n")
                handle.write(f"canonical_filename={plan.canonical_path.name}\n")
                handle.write(
                    f"legacy_baseline_valid={'true' if plan.legacy_baseline_valid else 'false'}\n"
                )
                handle.write(f"run_updater={'true' if plan.run_updater else 'false'}\n")
                handle.write(
                    f"run_initializer={'true' if plan.run_initializer else 'false'}\n"
                )
        print(f"[baseline-init] plan: {plan.action} — {plan.reason}", file=sys.stderr)
        return 0

    result = initialize_baseline(
        arguments.repo_path,
        environment,
        preview=arguments.preview,
        force=arguments.force,
    )

    print(json.dumps(result.to_dict(), indent=2))

    github_output = environment.get("GITHUB_OUTPUT")
    if github_output and not arguments.preview:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"status={result.status}\n")
            handle.write(
                "baseline_initialized="
                f"{'true' if result.is_baseline_initialization else 'false'}\n"
            )
            handle.write(f"canonical_path={result.canonical_path}\n")
            handle.write(f"repository_name={result.repository_name}\n")

    for problem in result.problems:
        print(f"[baseline-init] problem: {problem}", file=sys.stderr)
    if result.wrote_files:
        print(
            f"[baseline-init] {result.status}: wrote "
            f"{', '.join(str(p) for p in result.created)}",
            file=sys.stderr,
        )
    else:
        print(f"[baseline-init] {result.status}: no files written", file=sys.stderr)

    # A clean, safe outcome (including existing_invalid manual-review) exits 0;
    # only a hard generation failure is a non-zero exit.
    return 1 if result.status == STATUS_GENERATION_FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
