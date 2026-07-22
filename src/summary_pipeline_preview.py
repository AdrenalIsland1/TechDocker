"""Read-only end-to-end preview of the TechDocker decision pipeline.

Composes the existing components — it re-implements none of them:

    load change package
    -> load skeleton
    -> read base_updated_summary.md
    -> build the summary index IN MEMORY
    -> deterministic section scoring
    -> optional shortlist-only LLM section selection
    -> placement scoring inside the selected section
    -> optional LLM patch planning
    -> structured JSON report

The command stops before patch application and performs **no writes at all**:
no index artifact, no Markdown change, no skeleton or change-package change,
no branch, no PR, and it never calls ``summary_updater.run_update``. Every
input file is hashed before and after the run and the report fails loudly if
any hash moved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from src.llm_change_analyzer import (
    NO_SUITABLE_SECTION,
    select_section_with_llm,
    selection_to_routing_decision,
)
from src.placement_candidate_scorer import (
    PlacementIndexError,
    score_placement_candidates,
)
from src.section_candidate_scorer import (
    extract_change_signals,
    select_files_for_llm,
)
from src.summary_change_router import build_routing_context, decide_from_assessment
from src.summary_index_builder import build_summary_index
from src.summary_patch_planner import (
    STATUS_MANUAL_REVIEW,
    STATUS_NOT_INVOKED,
    allowed_operations,
    build_change_facts,
    build_patch_prompt,
    plan_summary_patch,
)
from src.summary_skeleton_store import SummarySkeleton, load_summary_skeleton
# The preview must predict what the real updater would do: it therefore gates
# an LLM section selection with the *same* minimum-confidence threshold the
# updater uses, so a selection the updater would reject is never reported here
# as having "resolved" the route.
from src.summary_updater import (
    DEFAULT_LLM_MIN_CONFIDENCE,
    LLM_MIN_CONFIDENCE_ENV_VAR,
)

PREVIEW_SCHEMA_VERSION = 1

# Default artifact locations (read-only).
DEFAULT_CHANGE_PACKAGE = Path("artifacts") / "change_packages" / "latest_change_summary.json"
DEFAULT_SUMMARY = Path("artifacts") / "summaries" / "base_updated_summary.md"
DEFAULT_SKELETON = Path("artifacts") / "skeletons" / "base_skeleton.json"

# LLM stages.
STAGE_NONE = "none"
STAGE_ROUTING = "routing"
STAGE_PATCH = "patch"
STAGE_ALL = "all"
LLM_STAGES = (STAGE_NONE, STAGE_ROUTING, STAGE_PATCH, STAGE_ALL)

_ROUTING_STAGES = frozenset({STAGE_ROUTING, STAGE_ALL})
_PATCH_STAGES = frozenset({STAGE_PATCH, STAGE_ALL})

# Exit codes.
EXIT_OK = 0
EXIT_INVALID_INPUT = 1
EXIT_STALE_DATA = 2
EXIT_HASH_MISMATCH = 3
EXIT_LLM_UNAVAILABLE = 4

# Report bounding (inputs are already bounded by the components; this is a
# final guard so a huge paragraph can never dump the summary into stdout).
MAX_REPORT_TEXT_CHARS = 600
MAX_REPORT_CANDIDATES = 3

# Deterministic routing at or above this confidence is considered resolved.
# Below it, an ambiguous route must be resolved by a validated LLM selection
# (or a manual override) before any executable patch may be planned.
SAFE_ROUTING_CONFIDENCE = 0.60

# Final-route ``strength`` values that only the preview can assign, because
# they describe *how* the route was settled rather than the deterministic
# score. The deterministic strengths (strong/reasonable/ambiguous/none) are
# always preserved separately under ``deterministic_strength``.
STRENGTH_LLM_RESOLVED = "llm_resolved"
STRENGTH_MANUAL_OVERRIDE = "manual_override"


class PreviewError(Exception):
    """A preview failed; carries the documented exit code."""

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass
class PreviewInputs:
    """Resolved, already-read inputs plus their before-hashes."""

    change_package_path: Path
    summary_path: Path
    skeleton_path: Path
    change_package: dict
    summary_markdown: str
    skeleton: SummarySkeleton
    hashes_before: dict[str, str]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bounded(text: str, limit: int = MAX_REPORT_TEXT_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"… [truncated, {len(text)} chars]"


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------
def load_inputs(
    repo_path: str | Path,
    change_package_path: Optional[str | Path] = None,
    summary_path: Optional[str | Path] = None,
    skeleton_path: Optional[str | Path] = None,
) -> PreviewInputs:
    """Read every input read-only and record its SHA-256."""
    root = Path(repo_path)
    package_file = Path(change_package_path) if change_package_path else root / DEFAULT_CHANGE_PACKAGE
    summary_file = Path(summary_path) if summary_path else root / DEFAULT_SUMMARY
    skeleton_file = Path(skeleton_path) if skeleton_path else root / DEFAULT_SKELETON

    for label, path in (
        ("change package", package_file),
        ("summary", summary_file),
        ("skeleton", skeleton_file),
    ):
        if not path.exists():
            raise PreviewError(f"Missing {label}: {path}", EXIT_INVALID_INPUT)

    try:
        change_package = json.loads(package_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PreviewError(
            f"Change package is not valid JSON: {error}", EXIT_INVALID_INPUT
        ) from error
    if not isinstance(change_package, dict):
        raise PreviewError(
            "Change package root must be a JSON object.", EXIT_INVALID_INPUT
        )

    schema_version = change_package.get("schema_version")
    if schema_version is not None and schema_version not in (1, 2, 3):
        raise PreviewError(
            f"Unsupported change-package schema_version {schema_version!r}.",
            EXIT_STALE_DATA,
        )

    try:
        skeleton = load_summary_skeleton(skeleton_file)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise PreviewError(
            f"Skeleton could not be loaded: {error}", EXIT_INVALID_INPUT
        ) from error

    summary_markdown = summary_file.read_text(encoding="utf-8")

    return PreviewInputs(
        change_package_path=package_file,
        summary_path=summary_file,
        skeleton_path=skeleton_file,
        change_package=change_package,
        summary_markdown=summary_markdown,
        skeleton=skeleton,
        hashes_before={
            "change_package": _sha256_file(package_file),
            "summary": _sha256_file(summary_file),
            "skeleton": _sha256_file(skeleton_file),
        },
    )


# ---------------------------------------------------------------------------
# Provider seam
# ---------------------------------------------------------------------------
def _resolve_llm_threshold(env: Mapping[str, str]) -> float:
    """The LLM minimum-confidence threshold, mirroring the updater exactly."""
    try:
        return float(
            env.get(LLM_MIN_CONFIDENCE_ENV_VAR, "") or DEFAULT_LLM_MIN_CONFIDENCE
        )
    except (TypeError, ValueError):
        return DEFAULT_LLM_MIN_CONFIDENCE


def default_provider_factory(env: Mapping[str, str]) -> Any:
    """Build the configured provider, or ``None`` when no LLM is configured.

    Only ever called when ``--llm-stage`` needs a model; a deterministic
    placeholder provider is treated as "not configured" so the preview reports
    it clearly instead of pretending an LLM ran.
    """
    if (env.get("TECHDOCKER_LLM_PROVIDER") or "").strip().lower() != "ollama":
        return None
    from src.llm_provider import get_llm_provider_from_env

    return get_llm_provider_from_env(env)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def _routing_stage(
    inputs: PreviewInputs,
    provider: Optional[Any],
    section_override: Optional[str],
    warnings: list[str],
    llm_threshold: float = DEFAULT_LLM_MIN_CONFIDENCE,
) -> tuple[dict, str]:
    """Deterministic section scoring, then optional shortlist-only selection."""
    entries = inputs.change_package.get("changed_files") or []
    change_summary = inputs.change_package.get("generated_summary") or ""

    assessment, catalog = build_routing_context(
        change_summary,
        [],
        inputs.skeleton,
        file_details=entries,
        summary_text=inputs.summary_markdown,
    )
    decision = decide_from_assessment(assessment, catalog)
    source = "deterministic"
    selected_section_id = decision.target_section_id

    if provider is not None and assessment.candidates:
        signals = extract_change_signals(change_summary, [], entries)
        prompt_paths, omitted = select_files_for_llm(signals, assessment.candidates)
        selection = select_section_with_llm(
            change_summary,
            assessment.candidates,
            provider=provider,
            changed_paths=prompt_paths,
            changed_symbols=sorted(signals.symbols),
            additional_files_omitted=omitted,
        )
        if selection is None:
            warnings.append(
                "LLM section selection was unavailable or invalid; using the "
                "deterministic result."
            )
        elif selection.decision == NO_SUITABLE_SECTION:
            warnings.append(
                f"LLM reported no suitable section ({selection.reasoning}); "
                "using the deterministic result."
            )
        elif selection.confidence < llm_threshold:
            warnings.append(
                f"LLM section selection confidence {selection.confidence:.2f} is "
                f"below the threshold {llm_threshold:.2f}; using the "
                "deterministic result."
            )
        else:
            llm_decision = selection_to_routing_decision(
                selection, assessment.candidates
            )
            if llm_decision is None:
                warnings.append(
                    "LLM section selection could not be converted to a routing "
                    "decision; using the deterministic result."
                )
            else:
                decision = llm_decision
                selected_section_id = llm_decision.target_section_id
                source = "llm"

    if section_override:
        known = {entry.section_id for entry in catalog}
        index_sections = None  # validated against the index later
        if section_override not in known:
            raise PreviewError(
                f"--section-id {section_override!r} is not an eligible section "
                f"in the skeleton; known: {sorted(known)}",
                EXIT_INVALID_INPUT,
            )
        selected_section_id = section_override
        source = "manual_override"
        warnings.append(
            f"Routing was overridden manually to {section_override!r}; the "
            "deterministic/LLM selection was not used."
        )

    # Routing provenance. The headline fields (``confidence``/``ambiguous``/
    # ``strength``) describe the *final selected route*; the ``deterministic_*``
    # fields always preserve the original Python assessment for diagnostics, so
    # a resolved tie is never erased and an unresolved one is never hidden.
    #
    #   * LLM-resolved (valid, above-threshold shortlist pick): the route is no
    #     longer ambiguous, its strength becomes ``llm_resolved``, and the
    #     headline confidence is the validated LLM confidence.
    #   * manual override: an explicit human decision — never ambiguous, its
    #     strength is ``manual_override``.
    #   * deterministic (LLM rejected, below threshold, or not consulted): the
    #     final ambiguity/strength/confidence stay exactly the deterministic
    #     assessment.
    resolved_by_llm = source == "llm"
    deterministic_strength = assessment.strength
    if resolved_by_llm:
        headline_confidence = decision.confidence
        final_ambiguous = False
        final_strength = STRENGTH_LLM_RESOLVED
    elif source == "manual_override":
        headline_confidence = assessment.confidence
        final_ambiguous = False
        final_strength = STRENGTH_MANUAL_OVERRIDE
    else:
        headline_confidence = assessment.confidence
        final_ambiguous = assessment.ambiguous
        final_strength = assessment.strength
    report = {
        "source": source,
        "selected_section_id": selected_section_id,
        "selected_heading": decision.target_heading
        if source != "manual_override"
        else None,
        "decision": decision.decision,
        "confidence": round(headline_confidence, 3),
        "ambiguous": final_ambiguous,
        "strength": final_strength,
        "deterministic_confidence": round(assessment.confidence, 3),
        "deterministic_ambiguous": assessment.ambiguous,
        "deterministic_strength": deterministic_strength,
        "resolved_by_llm": resolved_by_llm,
        "llm_confidence": round(decision.confidence, 3) if resolved_by_llm else None,
        "llm_reasoning": decision.reasoning if resolved_by_llm else None,
        "reasoning": decision.reasoning,
        "candidates": [
            candidate.to_dict()
            for candidate in assessment.candidates[:MAX_REPORT_CANDIDATES]
        ],
    }
    if selected_section_id is None:
        raise PreviewError(
            "Routing produced no target section (the skeleton may have no "
            "eligible sections); nothing to preview.",
            EXIT_STALE_DATA,
        )
    return report, selected_section_id


def _placement_report(assessment: Any) -> dict:
    candidates = []
    for candidate in assessment.candidates[:MAX_REPORT_CANDIDATES]:
        entry = candidate.to_dict()
        entry["text"] = _bounded(entry.get("text", ""))
        context = dict(entry.get("context") or {})
        for key in ("parent_block_text", "previous_excerpt", "next_excerpt"):
            if context.get(key):
                context[key] = _bounded(str(context[key]), 200)
        entry["context"] = context
        candidates.append(entry)
    return {
        "section_id": assessment.section_id,
        "recommendation": assessment.recommendation,
        "confidence": round(assessment.confidence, 3),
        "ambiguous": assessment.ambiguous,
        "reasoning": assessment.reasoning,
        "candidates": candidates,
    }


def _patch_report(result: Any) -> dict:
    plan = result.instruction.to_dict()
    plan["old_text"] = _bounded(plan.get("old_text", ""))
    plan["new_text"] = _bounded(plan.get("new_text", ""))
    return {
        "status": result.status,
        "reason": result.reason,
        "model_confidence": result.model_confidence,
        "model_reasoning": result.model_reasoning,
        "plan": plan,
    }


def _prompt_metadata(
    inputs: PreviewInputs, placement: Any, summary_index: Mapping[str, Any]
) -> dict:
    """Bounded prompt statistics — never the prompt text itself."""
    section = next(
        (
            section
            for section in summary_index.get("sections") or []
            if section.get("section_id") == placement.section_id
        ),
        {},
    )
    operations = allowed_operations(placement)
    prompt = build_patch_prompt(
        inputs.change_package, placement, summary_index, section, operations
    )
    facts, omissions = build_change_facts(inputs.change_package, placement)
    return {
        "patch_prompt_chars": len(prompt),
        "candidates_included": len(placement.candidates[:MAX_REPORT_CANDIDATES]),
        "change_fact_lines": len(facts),
        "files_omitted": omissions["files_omitted"],
        "hunks_omitted": omissions["hunks_omitted"],
        "lines_omitted": omissions["lines_omitted"],
        "allowed_operations": operations,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_preview(
    repo_path: str | Path = ".",
    *,
    change_package_path: Optional[str | Path] = None,
    summary_path: Optional[str | Path] = None,
    skeleton_path: Optional[str | Path] = None,
    section_override: Optional[str] = None,
    llm_stage: str = STAGE_NONE,
    env: Optional[Mapping[str, str]] = None,
    provider_factory: Callable[[Mapping[str, str]], Any] = default_provider_factory,
    include_prompt_metadata: bool = False,
) -> dict:
    """Run the read-only pipeline preview and return the structured report."""
    if llm_stage not in LLM_STAGES:
        raise PreviewError(
            f"Unknown --llm-stage {llm_stage!r}; expected one of {list(LLM_STAGES)}.",
            EXIT_INVALID_INPUT,
        )

    environment = env if env is not None else {}
    warnings: list[str] = []
    inputs = load_inputs(repo_path, change_package_path, summary_path, skeleton_path)

    # Providers are created only for the stages that need one.
    routing_provider = None
    patch_provider = None
    if llm_stage != STAGE_NONE:
        provider = provider_factory(environment)
        if provider is None:
            raise PreviewError(
                f"--llm-stage {llm_stage!r} requires an LLM provider, but none "
                "is configured (set TECHDOCKER_LLM_PROVIDER=ollama). No writes "
                "were performed.",
                EXIT_LLM_UNAVAILABLE,
            )
        if llm_stage in _ROUTING_STAGES:
            routing_provider = provider
        if llm_stage in _PATCH_STAGES:
            patch_provider = provider

    routing, section_id = _routing_stage(
        inputs,
        routing_provider,
        section_override,
        warnings,
        llm_threshold=_resolve_llm_threshold(environment),
    )

    # The index is built in memory only; nothing is ever written to disk.
    summary_index = build_summary_index(
        inputs.summary_markdown,
        inputs.skeleton,
        str(inputs.summary_path),
    )

    try:
        placement = score_placement_candidates(
            inputs.change_package,
            summary_index,
            section_id,
            source_markdown=inputs.summary_markdown,
        )
    except PlacementIndexError as error:
        raise PreviewError(
            f"Placement scoring could not run: {error}", EXIT_STALE_DATA
        ) from error

    # Gate: an unresolved ambiguous route must not yield an executable
    # mutation. A validated above-threshold LLM selection resolves the tie; a
    # manual override is an explicit human decision.
    routing_unresolved = (
        routing["ambiguous"]
        and not routing["resolved_by_llm"]
        and routing["source"] != "manual_override"
        and routing["deterministic_confidence"] < SAFE_ROUTING_CONFIDENCE
    )
    if routing_unresolved:
        warnings.append(
            "Section routing is ambiguous and unresolved "
            f"(deterministic confidence {routing['deterministic_confidence']} < "
            f"{SAFE_ROUTING_CONFIDENCE}); patch planning was skipped and the "
            "change needs manual review."
        )

    if patch_provider is not None and not routing_unresolved:
        planning = plan_summary_patch(
            inputs.change_package, placement, summary_index, provider=patch_provider
        )
        patch_planning = _patch_report(planning)
    elif routing_unresolved:
        patch_planning = {
            "status": STATUS_MANUAL_REVIEW,
            "reason": (
                "Routing remained ambiguous and unresolved; the patch planner "
                "was not called."
            ),
        }
    else:
        patch_planning = {"status": STATUS_NOT_INVOKED}

    report: dict[str, Any] = {
        "preview_schema_version": PREVIEW_SCHEMA_VERSION,
        "inputs": {
            "change_package": str(inputs.change_package_path),
            "summary": str(inputs.summary_path),
            "skeleton": str(inputs.skeleton_path),
            "llm_stage": llm_stage,
            "section_override": section_override,
        },
        "routing": routing,
        "placement": _placement_report(placement),
        "patch_planning": patch_planning,
        "warnings": warnings,
    }

    if include_prompt_metadata:
        report["prompt_metadata"] = _prompt_metadata(inputs, placement, summary_index)

    # Defensive invariant: this command writes nothing, so every input hash
    # must be unchanged. A mismatch is reported, never hidden.
    hashes_after = {
        "change_package": _sha256_file(inputs.change_package_path),
        "summary": _sha256_file(inputs.summary_path),
        "skeleton": _sha256_file(inputs.skeleton_path),
    }
    report["source_safety"] = {
        "summary_sha256_before": inputs.hashes_before["summary"],
        "summary_sha256_after": hashes_after["summary"],
        "skeleton_sha256_before": inputs.hashes_before["skeleton"],
        "skeleton_sha256_after": hashes_after["skeleton"],
        "change_package_sha256_before": inputs.hashes_before["change_package"],
        "change_package_sha256_after": hashes_after["change_package"],
        "writes_performed": False,
        "index_written": False,
    }
    changed = [
        name for name, digest in hashes_after.items()
        if digest != inputs.hashes_before[name]
    ]
    if changed:
        report["source_safety"]["writes_performed"] = True
        report["source_safety"]["changed_inputs"] = sorted(changed)
        raise PreviewSafetyError(
            f"Input file(s) changed during a read-only preview: {sorted(changed)}",
            report,
        )
    return report


class PreviewSafetyError(Exception):
    """An input file changed during the preview; carries the partial report."""

    def __init__(self, message: str, report: dict) -> None:
        super().__init__(message)
        self.report = report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None, env: Optional[Mapping[str, str]] = None) -> int:
    """Print the preview report as JSON on stdout; diagnostics on stderr."""
    import os

    parser = argparse.ArgumentParser(
        description="Read-only end-to-end preview of the TechDocker pipeline."
    )
    parser.add_argument("--repo-path", default=".")
    parser.add_argument("--change-package", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--skeleton", default=None)
    parser.add_argument("--section-id", default=None)
    parser.add_argument("--llm-stage", choices=list(LLM_STAGES), default=STAGE_NONE)
    parser.add_argument("--include-prompt-metadata", action="store_true")
    arguments = parser.parse_args(argv)

    environment = env if env is not None else os.environ

    try:
        report = run_preview(
            arguments.repo_path,
            change_package_path=arguments.change_package,
            summary_path=arguments.summary,
            skeleton_path=arguments.skeleton,
            section_override=arguments.section_id,
            llm_stage=arguments.llm_stage,
            env=environment,
            include_prompt_metadata=arguments.include_prompt_metadata,
        )
    except PreviewSafetyError as error:
        print(json.dumps(error.report, indent=2, ensure_ascii=False))
        print(f"[preview] SAFETY FAILURE: {error}", file=sys.stderr)
        return EXIT_HASH_MISMATCH
    except PreviewError as error:
        print(f"[preview] {error}", file=sys.stderr)
        return error.exit_code

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(
        "[preview] read-only: no artifact, index, summary, or skeleton was "
        "written; no patch was applied.",
        file=sys.stderr,
    )
    for warning in report["warnings"]:
        print(f"[preview] warning: {warning}", file=sys.stderr)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
