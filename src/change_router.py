"""Route a code change to a document section (LLM-ready, rules for now).

Given a change summary, the changed files, and the stored document skeleton,
decide whether the documentation update belongs in an existing section or
needs a brand-new one. The interface (inputs and :class:`RoutingDecision`)
is what a future LLM router will implement; the current implementation is
deliberately simple keyword rules so the pipeline can run without any API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.skeleton_store import (
    DocumentSkeleton,
    find_section_by_heading,
    find_section_containing,
)

UPDATE_EXISTING = "update_existing_section"
CREATE_NEW = "create_new_section"


@dataclass
class RoutingDecision:
    """Where an update should land in the document."""

    decision: str  # UPDATE_EXISTING or CREATE_NEW
    target_section_id: Optional[str] = None
    target_heading: Optional[str] = None
    new_heading: Optional[str] = None
    reasoning: str = ""


# Keyword rules, checked in order. Each maps a path keyword to the headings it
# prefers (first existing wins) and the heading to create when none exist.
# A future LLM replaces this table with real content-based placement.
_KEYWORD_RULES: list[dict[str, Any]] = [
    {
        "keyword": "test",
        "preferred_headings": ["Validation Rules"],
        "heading_keywords": ["validation"],
        "create_heading": "Validation Rules",
    },
    {
        "keyword": "config",
        "preferred_headings": ["API Configuration", "System Overview"],
        "heading_keywords": ["configuration"],
        "create_heading": "API Configuration",
    },
    {
        "keyword": "docx",
        "preferred_headings": [],
        "heading_keywords": ["parser", "document processing", "docx"],
        "create_heading": "Document Processing",
    },
    {
        "keyword": "parser",
        "preferred_headings": [],
        "heading_keywords": ["parser", "document processing", "docx"],
        "create_heading": "Document Processing",
    },
]

_FALLBACK_HEADING = "System Overview"


def _file_paths(changed_files: list[Any]) -> list[str]:
    """Accept ChangedFile objects or plain strings."""
    return [
        getattr(changed, "path", str(changed)).lower() for changed in changed_files
    ]


def _existing_target(
    skeleton: DocumentSkeleton, rule: dict[str, Any]
) -> Optional[Any]:
    """Find a section matching the rule's preferred headings or keywords."""
    for heading in rule["preferred_headings"]:
        section = find_section_by_heading(skeleton, heading)
        if section is not None:
            return section
    if rule["heading_keywords"]:
        return find_section_containing(skeleton, rule["heading_keywords"])
    return None


def route_change(
    change_summary: str,
    changed_files: list[Any],
    skeleton: DocumentSkeleton,
) -> RoutingDecision:
    """Decide where the update belongs, using keyword rules over file paths.

    ``change_summary`` is accepted (and will drive the future LLM router) but
    the rule-based version only inspects the changed file paths.
    """
    paths = _file_paths(changed_files)

    for rule in _KEYWORD_RULES:
        if not any(rule["keyword"] in path for path in paths):
            continue

        section = _existing_target(skeleton, rule)
        if section is not None:
            return RoutingDecision(
                decision=UPDATE_EXISTING,
                target_section_id=section.section_id,
                target_heading=section.heading,
                reasoning=(
                    f"Changed files match keyword {rule['keyword']!r}; "
                    f"routing to existing section {section.heading!r}."
                ),
            )
        return RoutingDecision(
            decision=CREATE_NEW,
            new_heading=rule["create_heading"],
            reasoning=(
                f"Changed files match keyword {rule['keyword']!r} but no "
                f"matching section exists; creating "
                f"{rule['create_heading']!r}."
            ),
        )

    # No keyword matched: prefer System Overview, then the first heading.
    fallback = find_section_by_heading(skeleton, _FALLBACK_HEADING)
    if fallback is None and skeleton.sections:
        fallback = skeleton.sections[0]

    if fallback is not None:
        return RoutingDecision(
            decision=UPDATE_EXISTING,
            target_section_id=fallback.section_id,
            target_heading=fallback.heading,
            reasoning=(
                "No routing keyword matched; falling back to "
                f"{fallback.heading!r}."
            ),
        )

    return RoutingDecision(
        decision=CREATE_NEW,
        new_heading=_FALLBACK_HEADING,
        reasoning=(
            "Skeleton has no sections; creating the fallback section "
            f"{_FALLBACK_HEADING!r}."
        ),
    )
