"""Route a change summary to the best section of the summary skeleton.

Rule-based today, LLM-ready by interface: the router receives the changed
files, the generated change summary text, and the skeleton, and returns a
:class:`SummaryRoutingDecision`. ``skeleton_should_change`` tells the caller
whether the skeleton JSON must be rewritten (only when a new heading is
needed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.summary_skeleton_store import (
    SummarySkeleton,
    find_section_by_heading,
)

UPDATE_EXISTING = "update_existing_section"
CREATE_NEW = "create_new_section"

_FALLBACK_HEADING = "System Overview"


@dataclass
class SummaryRoutingDecision:
    """Where an update belongs in the summary."""

    decision: str  # UPDATE_EXISTING or CREATE_NEW
    target_section_id: Optional[str] = None
    target_heading: Optional[str] = None
    new_heading: Optional[str] = None
    reasoning: str = ""
    skeleton_should_change: bool = False


def _matches_tests(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return (
        path.startswith("tests/")
        or "/tests/" in path
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _matches_ci(path: str) -> bool:
    return ".github/workflows" in path or path.startswith(".github/")


def _matches_config(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return (
        "config" in path
        or name in ("requirements.txt", "pytest.ini", ".gitignore")
        or name.endswith((".json", ".toml", ".ini", ".cfg", ".yml", ".yaml"))
    )


def _matches_docs(path: str) -> bool:
    name = path.rsplit("/", 1)[-1].lower()
    return name.startswith("readme") or path.startswith("docs/") or path.endswith(".md")


def _matches_automation(path: str) -> bool:
    return "automation" in path


def _matches_pipeline_modules(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return any(k in name for k in ("parser", "skeleton", "router", "updater"))


def _matches_source(path: str) -> bool:
    return path.startswith("src/") and path.endswith(".py")


# Checked in order; first rule any changed file matches wins. Preferred
# headings are tried in order; if none exists, a new section with the first
# one is created. A future LLM router replaces this table.
_ROUTING_RULES: list[dict[str, Any]] = [
    {"name": "tests", "match": _matches_tests,
     "preferred": ["Testing Strategy"]},
    {"name": "ci", "match": _matches_ci,
     "preferred": ["Deployment and CI", "Automation Pipeline"]},
    {"name": "config", "match": _matches_config,
     "preferred": ["Configuration"]},
    {"name": "docs", "match": _matches_docs,
     "preferred": ["System Overview", "Repository Structure"]},
    {"name": "automation", "match": _matches_automation,
     "preferred": ["Automation Pipeline"]},
    {"name": "pipeline-modules", "match": _matches_pipeline_modules,
     "preferred": ["Automation Pipeline", "Core Modules"]},
    {"name": "source", "match": _matches_source,
     "preferred": ["Core Modules"]},
]


def _file_paths(changed_files: list[Any]) -> list[str]:
    """Accept ChangedFile objects or plain strings."""
    return [getattr(changed, "path", str(changed)).lower() for changed in changed_files]


def _decide_for_preferred(
    skeleton: SummarySkeleton, preferred: list[str], reason: str
) -> SummaryRoutingDecision:
    for heading in preferred:
        section = find_section_by_heading(skeleton, heading)
        if section is not None:
            return SummaryRoutingDecision(
                decision=UPDATE_EXISTING,
                target_section_id=section.section_id,
                target_heading=section.heading,
                reasoning=f"{reason}; routing to existing section {section.heading!r}.",
                skeleton_should_change=False,
            )
    return SummaryRoutingDecision(
        decision=CREATE_NEW,
        new_heading=preferred[0],
        reasoning=(
            f"{reason}, but no matching section exists; "
            f"creating {preferred[0]!r}."
        ),
        skeleton_should_change=True,
    )


def route_change(
    change_summary: str,
    changed_files: list[Any],
    skeleton: SummarySkeleton,
) -> SummaryRoutingDecision:
    """Pick the target section for this change.

    ``change_summary`` is accepted for the future LLM router; the rule-based
    version inspects only the changed file paths.
    """
    paths = _file_paths(changed_files)

    for rule in _ROUTING_RULES:
        if any(rule["match"](path) for path in paths):
            return _decide_for_preferred(
                skeleton,
                rule["preferred"],
                f"Changed files match rule {rule['name']!r}",
            )

    # No rule matched: System Overview, then the first section, then create.
    fallback = find_section_by_heading(skeleton, _FALLBACK_HEADING)
    if fallback is None and skeleton.sections:
        fallback = skeleton.sections[0]
    if fallback is not None:
        return SummaryRoutingDecision(
            decision=UPDATE_EXISTING,
            target_section_id=fallback.section_id,
            target_heading=fallback.heading,
            reasoning=f"No routing rule matched; falling back to {fallback.heading!r}.",
            skeleton_should_change=False,
        )
    return SummaryRoutingDecision(
        decision=CREATE_NEW,
        new_heading=_FALLBACK_HEADING,
        reasoning=(
            "Skeleton has no sections; creating the fallback section "
            f"{_FALLBACK_HEADING!r}."
        ),
        skeleton_should_change=True,
    )
