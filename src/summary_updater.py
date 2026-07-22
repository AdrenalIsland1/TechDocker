"""Update the reviewable project summary from a push.

Run as ``python3 -m src.summary_updater`` — in GitHub Actions (``GITHUB_*``
environment variables) or locally (falls back to the current branch and
``HEAD~1..HEAD``).

Flow, per push:

1. ensure ``base_original_summary.md`` exists (generate with the deterministic
   provider if missing) — it is **never modified** afterwards,
2. ensure ``base_updated_summary.md`` exists (copy of the baseline),
3. ensure ``base_skeleton.json`` exists (build from the updated summary),
4. detect changed files via git diff,
5. write ``artifacts/change_packages/latest_change_summary.json``,
6. route the change against the skeleton,
7. insert a marked update block into ``base_updated_summary.md`` under the
   routed section — the skeleton is untouched,
8. or, for ``create_new_section``: append the new heading with the block to
   ``base_updated_summary.md`` and append the new section to
   ``base_skeleton.json`` (the skeleton stays based on the original baseline
   plus explicit additions).

``base_original_summary.md`` is never modified by update runs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

from src.automation_demo import extract_repository_name, is_missing_sha
from src.canonical_document import resolve_canonical_baseline
from src.change_summary_generator import create_change_package
from src.git_change_detector import (
    ChangedFile,
    build_change_set,
    collect_file_changes,
)
from src.llm_change_analyzer import (
    NO_SUITABLE_SECTION,
    select_section_with_llm,
    selection_to_routing_decision,
)
from src.llm_provider import get_llm_provider_from_env
from src.section_candidate_scorer import (
    extract_change_signals,
    select_files_for_llm,
)
from src.markdown_summary_parser import normalize_heading
from src.project_summary_generator import (
    generate_original_summary,
    original_summary_path,
    updated_summary_path,
)
from src.summary_change_router import (
    CREATE_NEW,
    UPDATE_EXISTING,
    SummaryRoutingDecision,
    build_routing_context,
    decide_from_assessment,
    route_change,
)
from src.summary_skeleton_builder import (
    build_and_save_summary_skeleton,
    summary_skeleton_path,
)
from src.summary_skeleton_store import (
    append_section,
    load_summary_skeleton,
    save_summary_skeleton,
)

UPDATE_BLOCK_START = "<!-- TECHDOCKER_UPDATE_START -->"
UPDATE_BLOCK_END = "<!-- TECHDOCKER_UPDATE_END -->"

DEFAULT_LLM_MIN_CONFIDENCE = 0.75
LLM_MIN_CONFIDENCE_ENV_VAR = "TECHDOCKER_LLM_MIN_CONFIDENCE"


@dataclass
class SummaryUpdateResult:
    """What one updater run did, for printing and for tests."""

    original_summary: Path
    updated_summary: Path
    skeleton_path: Path
    change_package_path: Optional[Path]
    repository: str
    branch: str
    changed_files: list[ChangedFile] = field(default_factory=list)
    decision: Optional[SummaryRoutingDecision] = None
    placement: str = ""
    routing_source: str = "rule_based"  # or "llm"
    llm_confidence: Optional[float] = None
    original_generated: bool = False
    skeleton_created: bool = False
    skeleton_updated: bool = False
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


def build_update_block(
    *,
    repository: str,
    branch: str,
    actor: str,
    before_sha: Optional[str],
    after_sha: str,
    changed_files: list[ChangedFile],
    summary_text: str,
) -> list[str]:
    """The clearly marked Markdown block inserted into the updated summary."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        UPDATE_BLOCK_START,
        f"### Automated Change Update - {timestamp}",
        "",
        f"Repository: {repository}",
        f"Branch: {branch}",
        f"Actor: {actor}",
        f"Before SHA: {before_sha or '(none)'}",
        f"After SHA: {after_sha}",
        "",
        "Changed files:",
    ]
    if changed_files:
        for changed in changed_files:
            if changed.old_path:
                lines.append(
                    f"- {changed.change_type}: {changed.old_path} -> {changed.path}"
                )
            else:
                lines.append(f"- {changed.change_type}: {changed.path}")
    else:
        lines.append("- (no changed files were available for this run)")
    lines += ["", "Summary:", summary_text, UPDATE_BLOCK_END]
    return lines


def insert_block_under_heading(
    text: str, target_heading: str, block_lines: list[str]
) -> tuple[str, bool]:
    """Insert the block at the end of the target heading's section.

    The insertion point is just before the next heading of the same or higher
    level (fence-aware). Returns ``(new_text, found)``; when the heading is
    not found the text is returned unchanged with ``found=False``.
    """
    lines = text.splitlines()
    wanted = normalize_heading(target_heading)

    in_fence = False
    found_level: Optional[int] = None
    insert_index: Optional[int] = None

    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence or not stripped.startswith("#"):
            continue
        hashes = len(stripped) - len(stripped.lstrip("#"))
        if hashes < 1 or hashes > 6 or not stripped[hashes:].startswith(" "):
            continue
        heading_text = stripped[hashes:].strip()

        if found_level is None:
            if normalize_heading(heading_text) == wanted:
                found_level = hashes
        elif hashes <= found_level:
            insert_index = index
            break

    if found_level is None:
        return text, False
    if insert_index is None:
        insert_index = len(lines)

    new_lines = lines[:insert_index] + ["", *block_lines, ""] + lines[insert_index:]
    result = "\n".join(new_lines)
    if text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result, True


def run_update(env: Mapping[str, str], repo_path: str = ".") -> SummaryUpdateResult:
    """Execute the full summary-update flow for one push."""
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

    # 1-2. Baseline and reviewable copy. A normal push prefers the repository
    # canonical technical document and NEVER regenerates an accepted baseline;
    # only when no baseline exists at all does it fall back to the legacy
    # deterministic generation path (env selects the provider). Baseline
    # *creation* for the canonical document is the initializer's job, not this
    # incremental updater's.
    baseline = resolve_canonical_baseline(repo_path, env)
    updated_path = updated_summary_path(repo_path)
    original_path = original_summary_path(repo_path)
    if baseline.exists:
        # Canonical or legacy baseline already present: use it as-is.
        original_generated = False
        baseline_document = baseline.path
        if not updated_path.exists():  # initialize the reviewable copy once
            shutil.copyfile(baseline.path, updated_path)
    else:
        original_generated = not original_path.exists()
        # No-op when the legacy baseline exists; deterministic unless
        # TECHDOCKER_LLM_PROVIDER=ollama is set.
        generate_original_summary(repo_path, env=env)
        baseline_document = original_path
        if not updated_path.exists():  # defensive; generator normally created it
            shutil.copyfile(original_path, updated_path)

    # 3. Skeleton, built from the exact baseline document actually used.
    skeleton_path = summary_skeleton_path(repo_path)
    skeleton_created = not skeleton_path.exists()
    if skeleton_created:
        skeleton, skeleton_path = build_and_save_summary_skeleton(
            repo_path, source=baseline.path if baseline.exists else None
        )
    else:
        skeleton = load_summary_skeleton(skeleton_path)

    # 4. Changed files.
    changed_files: list[ChangedFile] = []
    if is_missing_sha(before_sha):
        before_sha = None
        warnings.append(
            "Before SHA is missing or all zeroes; continuing with an empty "
            "changed-file list."
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
                "continuing with an empty changed-file list."
            )

    # 5. Change package. Enrich with file-level and hunk-level details when a
    # real diff is available; degrade to file names only on any git failure.
    file_details: Optional[list[dict]] = None
    if changed_files and before_sha:
        try:
            file_details = [
                detail.to_dict()
                for detail in collect_file_changes(
                    before_sha, after_sha, repo_path
                )
            ]
        except (subprocess.CalledProcessError, OSError) as error:
            warnings.append(
                f"could not extract file-level diff details ({error}); the "
                "change package lists file names only."
            )

    change_package, change_package_file = create_change_package(
        repository=repository,
        branch=branch,
        actor=actor,
        before_sha=before_sha,
        after_sha=after_sha,
        changed_files=changed_files,
        repo_path=repo_path,
        file_details=file_details,
    )

    # 6. Route: rule-based by default; the LLM path is strictly opt-in via
    # TECHDOCKER_LLM_PROVIDER=ollama and only *suggests* — every suggestion is
    # validated, threshold-gated, and falls back to the rule-based router.
    summary_text = change_package["generated_summary"]
    routing_source = "rule_based"
    llm_confidence: Optional[float] = None

    # Deterministic scoring over the *actual* skeleton sections runs exactly
    # once; the optional LLM may only pick from the shortlist it produces.
    assessment, catalog = build_routing_context(
        summary_text,
        changed_files,
        skeleton,
        file_details=file_details,
        summary_text=updated_path.read_text(encoding="utf-8"),
    )
    decision = decide_from_assessment(assessment, catalog)

    if env.get("TECHDOCKER_LLM_PROVIDER", "").strip().lower() == "ollama":
        try:
            threshold = float(
                env.get(LLM_MIN_CONFIDENCE_ENV_VAR, "") or DEFAULT_LLM_MIN_CONFIDENCE
            )
        except ValueError:
            threshold = DEFAULT_LLM_MIN_CONFIDENCE

        candidates = assessment.candidates
        signals = extract_change_signals(summary_text, changed_files, file_details)
        prompt_paths, omitted = select_files_for_llm(signals, candidates)

        selection = select_section_with_llm(
            summary_text,
            candidates,
            provider=get_llm_provider_from_env(env),
            changed_paths=prompt_paths,
            changed_symbols=sorted(signals.symbols),
            additional_files_omitted=omitted,
        )
        if selection is None:
            warnings.append(
                "LLM selection was unavailable or invalid; using the "
                "deterministic routing result."
            )
        elif selection.decision == NO_SUITABLE_SECTION:
            warnings.append(
                f"LLM reported no suitable section ({selection.reasoning}); "
                "using the deterministic routing result."
            )
        elif selection.confidence < threshold:
            warnings.append(
                f"LLM confidence {selection.confidence:.2f} is below the "
                f"threshold {threshold:.2f}; using the deterministic routing "
                "result."
            )
        else:
            llm_decision = selection_to_routing_decision(selection, candidates)
            if llm_decision is not None:
                decision = llm_decision
                routing_source = "llm"
                llm_confidence = selection.confidence

    if decision.ambiguous and decision.decision == UPDATE_EXISTING:
        warnings.append(
            f"Section routing was ambiguous ({decision.reasoning}); the "
            "target section needs review."
        )

    # 7-8. Apply to the updated summary only.
    block = build_update_block(
        repository=repository,
        branch=branch,
        actor=actor,
        before_sha=before_sha,
        after_sha=after_sha,
        changed_files=changed_files,
        summary_text=summary_text,
    )

    text = updated_path.read_text(encoding="utf-8")
    skeleton_updated = False

    if decision.decision == CREATE_NEW:
        addition = ["", f"## {decision.new_heading}", "", *block, ""]
        text = text.rstrip("\n") + "\n" + "\n".join(addition) + "\n"
        updated_path.write_text(text, encoding="utf-8")
        placement = f"new section {decision.new_heading!r} appended"
        # Structure changed: append the new section to the skeleton. The
        # skeleton stays based on the original baseline plus explicit
        # additions — it is never rebuilt from the reviewable copy.
        append_section(skeleton, heading=decision.new_heading, level=2)
        save_summary_skeleton(skeleton, skeleton_path)
        skeleton_updated = True
    else:
        new_text, found = insert_block_under_heading(
            text, decision.target_heading, block
        )
        if found:
            updated_path.write_text(new_text, encoding="utf-8")
            placement = f"under existing section {decision.target_heading!r}"
        else:
            text = text.rstrip("\n") + "\n\n" + "\n".join(block) + "\n"
            updated_path.write_text(text, encoding="utf-8")
            placement = "appended to end (target heading not found)"
            warnings.append(
                f"Target heading {decision.target_heading!r} was not found in "
                "the updated summary; appended the update to the end instead."
            )

    return SummaryUpdateResult(
        original_summary=baseline_document,
        updated_summary=updated_path,
        skeleton_path=skeleton_path,
        change_package_path=change_package_file,
        repository=repository,
        branch=branch,
        changed_files=changed_files,
        decision=decision,
        placement=placement,
        routing_source=routing_source,
        llm_confidence=llm_confidence,
        original_generated=original_generated,
        skeleton_created=skeleton_created,
        skeleton_updated=skeleton_updated,
        warnings=warnings,
    )


def format_result(result: SummaryUpdateResult) -> str:
    """Readable terminal confirmation of one updater run."""
    lines = [
        "=" * 60,
        "TechDocker Summary Updater",
        "=" * 60,
        f"Original summary:  {result.original_summary}"
        + (" (generated)" if result.original_generated else " (unchanged)"),
        f"Updated summary:   {result.updated_summary}",
        f"Skeleton:          {result.skeleton_path}"
        + (" (created)" if result.skeleton_created else "")
        + (" (rebuilt)" if result.skeleton_updated else " (unchanged)"),
        f"Change package:    {result.change_package_path}",
        f"Repository/branch: {result.repository} / {result.branch}",
        f"Changed files:     {len(result.changed_files)}",
        f"Routing decision:  {result.decision.decision if result.decision else '(none)'}",
        f"Routing source:    {result.routing_source}"
        + (
            f" (confidence {result.llm_confidence:.2f})"
            if result.llm_confidence is not None
            else ""
        ),
        f"Placement:         {result.placement}",
    ]
    if result.decision is not None:
        lines.append(f"Reasoning:         {result.decision.reasoning}")
    if result.warnings:
        lines += [f"! {warning}" for warning in result.warnings]
    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> int:
    """Entry point for ``python3 -m src.summary_updater``."""
    result = run_update(os.environ)
    print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
