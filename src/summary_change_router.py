"""Route a change to the best *actual* section of the summary skeleton.

The router scores whatever sections the skeleton really contains (see
:mod:`src.section_candidate_scorer`) instead of looking for a fixed heading
vocabulary, so summaries using "Quality and Tests" or "CI/CD Review Flow"
route just as well as the original eight headings.

``SummaryRoutingDecision`` keeps its existing shape; ``skeleton_should_change``
still tells the caller whether the skeleton JSON must be rewritten (only when
a new heading is genuinely needed). The optional LLM never routes on its own —
it may only pick from the deterministic shortlist (see
:mod:`src.llm_change_analyzer`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.section_candidate_scorer import (
    DEFAULT_SHORTLIST_SIZE,
    CandidateAssessment,
    SectionCandidate,
    assess_candidates,
    build_section_catalog,
    extract_change_signals,
    find_overview_section,
    rank_candidates,
)
from src.summary_skeleton_store import SummarySkeleton

UPDATE_EXISTING = "update_existing_section"
CREATE_NEW = "create_new_section"

# Only used when the skeleton has no eligible sections at all. Internal
# categories are deliberately NOT mapped to headings: a category is a routing
# bridge, never a user-facing heading.
_EMPTY_SKELETON_HEADING = "System Overview"


@dataclass
class SummaryRoutingDecision:
    """Where an update belongs in the summary."""

    decision: str  # UPDATE_EXISTING or CREATE_NEW
    target_section_id: Optional[str] = None
    target_heading: Optional[str] = None
    new_heading: Optional[str] = None
    reasoning: str = ""
    skeleton_should_change: bool = False
    # Debug/reporting extras (compact; never raw changed-line bodies).
    confidence: float = 0.0
    ambiguous: bool = False
    strength: str = ""
    candidates: list[dict] = field(default_factory=list)
    matched_signals: list[str] = field(default_factory=list)


def _decision_from_candidate(
    candidate: SectionCandidate,
    assessment: CandidateAssessment,
    reason: str,
) -> SummaryRoutingDecision:
    return SummaryRoutingDecision(
        decision=UPDATE_EXISTING,
        target_section_id=candidate.section_id,
        target_heading=candidate.heading,
        reasoning=reason,
        skeleton_should_change=False,
        confidence=assessment.confidence,
        ambiguous=assessment.ambiguous,
        strength=assessment.strength,
        candidates=[c.to_dict() for c in assessment.candidates],
        matched_signals=list(candidate.matched_signals),
    )


def build_routing_context(
    change_summary: str,
    changed_files: list[Any],
    skeleton: SummarySkeleton,
    file_details: Optional[list[Any]] = None,
    summary_text: Optional[str] = None,
    limit: int = DEFAULT_SHORTLIST_SIZE,
) -> tuple[CandidateAssessment, list]:
    """Score the real sections and return ``(assessment, catalog)``.

    ``file_details`` are schema-v2 entries; ``summary_text`` (when supplied)
    lets section *content* influence scoring.
    """
    from src.section_candidate_scorer import section_contents_from_markdown

    contents = section_contents_from_markdown(summary_text) if summary_text else None
    catalog = build_section_catalog(skeleton, contents)
    signals = extract_change_signals(change_summary, changed_files, file_details)
    candidates = rank_candidates(signals, catalog, limit)
    return assess_candidates(candidates, signals), catalog


def decide_from_assessment(
    assessment: CandidateAssessment,
    catalog: list,
) -> SummaryRoutingDecision:
    """Turn an already-computed assessment into a decision (no re-scoring).

    For a non-empty skeleton a section is **never** invented: internal
    categories are not headings, so a low-confidence change falls back to the
    semantically discovered overview-equivalent section, explicitly flagged
    ambiguous for review. ``CREATE_NEW`` remains only for the genuinely
    defensible case of a skeleton with no eligible sections.
    """
    top = assessment.top
    if top is not None and assessment.strength != "none":
        prefix = "Ambiguous match" if assessment.ambiguous else "Best match"
        reason = (
            f"{prefix}: {assessment.reason}; routing to existing section "
            f"{top.heading!r}."
        )
        return _decision_from_candidate(top, assessment, reason)

    overview = find_overview_section(catalog)
    if overview is not None:
        return SummaryRoutingDecision(
            decision=UPDATE_EXISTING,
            target_section_id=overview.section_id,
            target_heading=overview.heading,
            reasoning=(
                f"No section matched confidently ({assessment.reason}); "
                f"falling back to the overview-equivalent section "
                f"{overview.heading!r} for manual review. No section was "
                "created."
            ),
            skeleton_should_change=False,
            confidence=assessment.confidence,
            ambiguous=True,
            strength=assessment.strength,
            candidates=[c.to_dict() for c in assessment.candidates],
        )

    return SummaryRoutingDecision(
        decision=CREATE_NEW,
        new_heading=_EMPTY_SKELETON_HEADING,
        reasoning=(
            "Skeleton has no eligible sections; creating "
            f"{_EMPTY_SKELETON_HEADING!r}."
        ),
        skeleton_should_change=True,
        confidence=0.0,
        strength=assessment.strength,
    )


def route_change(
    change_summary: str,
    changed_files: list[Any],
    skeleton: SummarySkeleton,
    file_details: Optional[list[Any]] = None,
    summary_text: Optional[str] = None,
) -> SummaryRoutingDecision:
    """Pick the target section for this change, deterministically.

    Backward compatible: ``file_details`` (schema-v2 enrichment) and
    ``summary_text`` (section content) are optional, so schema-v1 callers and
    ``ChangedFile`` lists keep working.
    """
    assessment, catalog = build_routing_context(
        change_summary, changed_files, skeleton, file_details, summary_text
    )
    return decide_from_assessment(assessment, catalog)
