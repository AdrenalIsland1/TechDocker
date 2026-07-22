"""LLM-assisted change analysis: suggest where a change belongs.

Run as ``python3 -m src.llm_change_analyzer`` to preview the suggestion for
the latest change package without modifying any files.

The provider's output is **never trusted directly**: it must be strict JSON
matching :data:`SUGGESTION_JSON_SCHEMA`, the decision must be one of the two
allowed values, confidence must be within [0, 1], and an
``update_existing_section`` target must actually exist in the skeleton. Any
violation makes :func:`analyze_change` return ``None`` so callers fall back
to the rule-based router. The LLM never edits files.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from src.change_summary_generator import change_package_path
from src.git_change_detector import ChangedFile
from src.llm_provider import LLMProvider, get_llm_provider_from_env
from src.markdown_summary_parser import GENERIC_HEADINGS
from src.project_summary_generator import updated_summary_path
from src.section_candidate_scorer import (
    extract_change_signals,
    select_files_for_llm,
)
from src.summary_change_router import (
    CREATE_NEW,
    UPDATE_EXISTING,
    SummaryRoutingDecision,
    build_routing_context,
    route_change,
)
from src.summary_skeleton_builder import summary_skeleton_path
from src.summary_skeleton_store import (
    SummarySkeleton,
    find_section_by_id,
    load_summary_skeleton,
)

ALLOWED_DECISIONS = (UPDATE_EXISTING, CREATE_NEW)

SUGGESTION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "decision": {"enum": list(ALLOWED_DECISIONS)},
        "target_section_id": {"type": ["string", "null"]},
        "target_heading": {"type": ["string", "null"]},
        "requires_new_section": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "suggested_summary": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["decision", "confidence", "suggested_summary", "reasoning"],
}

SYSTEM_PROMPT = (
    "You are TechDocker's documentation routing assistant. You place code "
    "changes into an existing technical summary structure. Prefer existing "
    "sections; only propose a new heading when the change genuinely fits no "
    "existing section. Respond with strict JSON only — no prose, no markdown."
)

# New-section headings that convey nothing are rejected outright (the set is
# shared with base-summary validation; defined in markdown_summary_parser).
GENERIC_NEW_HEADINGS = GENERIC_HEADINGS

# Summaries that say nothing are rejected; the rule-based text is better.
_GENERIC_SUMMARY_RE = re.compile(
    r"^(the\s+)?(repo(sitory)?|code(base)?|files?|project)"
    r"(\s+structure)?\s+(was|were|has\s+been|have\s+been)\s+"
    r"(updated|changed|modified)\.?$",
    re.IGNORECASE,
)
MIN_SUMMARY_CHARS = 15


class SuggestionValidationError(ValueError):
    """The LLM output was not a valid, safe suggestion."""


@dataclass
class LLMChangeSuggestion:
    """A validated placement suggestion from the LLM."""

    decision: str
    target_section_id: Optional[str]
    target_heading: Optional[str]
    requires_new_section: bool
    confidence: float
    suggested_summary: str
    reasoning: str


def build_prompt(
    changed_files: list[Any],
    change_summary: str,
    skeleton: SummarySkeleton,
) -> str:
    """The routing prompt: changed files, summary, and skeleton sections."""
    file_lines = "\n".join(
        f"- {getattr(f, 'change_type', 'changed')}: {getattr(f, 'path', str(f))}"
        for f in changed_files
    ) or "- (no changed files)"

    section_lines = "\n".join(
        f"- section_id: {s.section_id} | heading: {s.heading} | path: {s.path}"
        for s in skeleton.sections
    ) or "- (skeleton is empty)"

    return f"""A code push changed these files:
{file_lines}

Generated change summary:
{change_summary}

The project summary document has these sections:
{section_lines}

Decide where a short documentation update about this change belongs.

Routing guidance:
- Strongly prefer an EXISTING section. Create a new section ONLY for a
  genuinely new major project area (e.g. a first-ever security layer),
  never for routine work.
- Test files or test changes -> Testing Strategy.
- CI / GitHub Actions / workflow YAML changes -> Deployment and CI.
- Configuration files -> Configuration.
- Cleanup, refactors, and changes to updater/router/pipeline/automation
  modules -> Automation Pipeline (or Core Modules for plain source).
- NEVER propose generic headings such as "Code Changes", "Updates",
  "Changes", or "Miscellaneous" — they will be rejected.

Quality guidance for suggested_summary:
- Name the specific modules/behaviour that changed and why it matters.
- One or two concrete sentences. Generic statements like "repository
  structure was updated" or "code was changed" will be rejected.

Examples of good outputs:

Changed: modified src/summary_change_router.py ->
{{"decision": "update_existing_section",
  "target_section_id": "<id of Automation Pipeline>",
  "target_heading": "Automation Pipeline",
  "requires_new_section": false, "confidence": 0.9,
  "suggested_summary": "The change router now validates section targets \
before routing, preventing updates from landing under missing headings.",
  "reasoning": "Router modules belong to the automation pipeline."}}

Changed: added tests/test_summary_updater.py ->
{{"decision": "update_existing_section",
  "target_section_id": "<id of Testing Strategy>",
  "target_heading": "Testing Strategy",
  "requires_new_section": false, "confidence": 0.9,
  "suggested_summary": "Added updater tests covering skeleton immutability \
and fallback routing when no before-commit is available.",
  "reasoning": "Test additions document the testing strategy."}}

Return STRICT JSON only, matching exactly:

{{
  "decision": "update_existing_section" | "create_new_section",
  "target_section_id": "<section_id from the list, or null>",
  "target_heading": "<heading text; for create_new_section the new heading>",
  "requires_new_section": true | false,
  "confidence": <number between 0 and 1>,
  "suggested_summary": "<one or two specific sentences about this change>",
  "reasoning": "<one sentence explaining the placement>"
}}"""


def _extract_json_object(text: str) -> dict:
    """Parse the first JSON object in the text (tolerates markdown fences)."""
    stripped = (text or "").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise SuggestionValidationError("LLM response contains no JSON object.")
    try:
        parsed = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as err:
        raise SuggestionValidationError(f"LLM response is not valid JSON: {err}")
    if not isinstance(parsed, dict):
        raise SuggestionValidationError("LLM JSON root must be an object.")
    return parsed


def parse_and_validate_suggestion(
    text: str, skeleton: SummarySkeleton
) -> LLMChangeSuggestion:
    """Validate raw LLM output into a safe suggestion (raises on any issue)."""
    data = _extract_json_object(text)

    decision = data.get("decision")
    if decision not in ALLOWED_DECISIONS:
        raise SuggestionValidationError(
            f"Invalid decision {decision!r}; allowed: {ALLOWED_DECISIONS}."
        )

    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise SuggestionValidationError("Confidence must be a number.")
    if not 0.0 <= float(confidence) <= 1.0:
        raise SuggestionValidationError(
            f"Confidence {confidence} is outside [0, 1]."
        )

    target_section_id = data.get("target_section_id")
    target_heading = data.get("target_heading")

    if decision == UPDATE_EXISTING:
        section = (
            find_section_by_id(skeleton, target_section_id)
            if target_section_id
            else None
        )
        if section is None:
            raise SuggestionValidationError(
                f"target_section_id {target_section_id!r} does not exist in "
                "the skeleton."
            )
        # Trust the skeleton for the heading, not the model.
        target_heading = section.heading
    else:
        if not target_heading or not str(target_heading).strip():
            raise SuggestionValidationError(
                "create_new_section requires a non-empty target_heading."
            )
        target_section_id = None
        target_heading = str(target_heading).strip()
        if target_heading.strip().lower().rstrip(".:") in GENERIC_NEW_HEADINGS:
            raise SuggestionValidationError(
                f"New heading {target_heading!r} is too generic; a new "
                "section must name a genuinely new project area."
            )

    suggested_summary = str(data.get("suggested_summary", "")).strip()
    if len(suggested_summary) < MIN_SUMMARY_CHARS:
        raise SuggestionValidationError(
            "suggested_summary is empty or too short to be useful."
        )
    if _GENERIC_SUMMARY_RE.match(suggested_summary):
        raise SuggestionValidationError(
            f"suggested_summary {suggested_summary!r} is too generic."
        )

    return LLMChangeSuggestion(
        decision=decision,
        target_section_id=target_section_id,
        target_heading=target_heading,
        requires_new_section=(decision == CREATE_NEW),
        confidence=float(confidence),
        suggested_summary=suggested_summary,
        reasoning=str(data.get("reasoning", "")).strip(),
    )


# ---------------------------------------------------------------------------
# Shortlist selection: the LLM may only pick from deterministic candidates
# ---------------------------------------------------------------------------
SELECT_EXISTING = "select_existing_section"
NO_SUITABLE_SECTION = "no_suitable_section"
SELECTION_DECISIONS = (SELECT_EXISTING, NO_SUITABLE_SECTION)

SELECTION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "decision": {"enum": list(SELECTION_DECISIONS)},
        "section_id": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string"},
    },
    "required": ["decision", "confidence", "reasoning"],
}

SELECTION_SYSTEM_PROMPT = (
    "You choose which existing documentation section a code change belongs to. "
    "You may ONLY pick a section_id from the supplied candidate list, or answer "
    "no_suitable_section. Never invent, rename, or create a section, never "
    "rewrite the summary, and never decide paragraph placement. Respond with "
    "strict JSON only."
)

MAX_REASONING_CHARS = 400
_MAX_CANDIDATE_EXCERPT = 300
MAX_PROMPT_FILES = 15
MAX_PROMPT_SYMBOLS = 15

# The response may contain ONLY these keys. Anything else — especially an
# invented targeting field such as ``new_heading`` — is rejected outright
# rather than silently ignored.
ALLOWED_SELECTION_KEYS = frozenset(
    {"decision", "section_id", "confidence", "reasoning"}
)


@dataclass
class LLMSectionSelection:
    """A validated shortlist selection."""

    decision: str
    section_id: Optional[str]
    heading: Optional[str]
    confidence: float
    reasoning: str


def build_selection_prompt(
    change_summary: str,
    candidates: list[Any],
    changed_paths: Optional[list[str]] = None,
    changed_symbols: Optional[list[str]] = None,
    additional_files_omitted: int = 0,
) -> str:
    """Prompt containing bounded change facts and ONLY the shortlisted ids."""
    lines = ["A code push produced this change summary:", change_summary or "(none)", ""]
    if changed_paths:
        lines.append("Changed files:")
        lines += [f"- {path}" for path in changed_paths[:MAX_PROMPT_FILES]]
        if additional_files_omitted > 0:
            lines.append(f'"additional_files_omitted": {additional_files_omitted}')
        lines.append("")
    if changed_symbols:
        lines.append("Changed symbols: " + ", ".join(changed_symbols[:MAX_PROMPT_SYMBOLS]))
        lines.append("")

    lines.append("Candidate sections (choose exactly one section_id, or none):")
    for candidate in candidates:
        data = candidate.to_dict() if hasattr(candidate, "to_dict") else dict(candidate)
        path = " > ".join(data.get("heading_path") or [data.get("heading", "")])
        excerpt = (data.get("content_excerpt") or "")[:_MAX_CANDIDATE_EXCERPT]
        lines.append(
            f"- section_id: {data['section_id']}\n"
            f"  heading: {data.get('heading', '')}\n"
            f"  path: {path}\n"
            f"  deterministic_score: {data.get('score')}\n"
            f"  matched_signals: {', '.join(data.get('matched_signals') or []) or 'none'}"
            + (f"\n  excerpt: {excerpt}" if excerpt else "")
        )

    allowed = ", ".join(
        (c.to_dict() if hasattr(c, "to_dict") else c)["section_id"] for c in candidates
    )
    lines += [
        "",
        f"Allowed section_id values: {allowed}",
        "",
        "Return STRICT JSON only:",
        "{",
        '  "decision": "select_existing_section" | "no_suitable_section",',
        '  "section_id": "<one allowed id, or null>",',
        '  "confidence": <number between 0 and 1>,',
        '  "reasoning": "<one short sentence>"',
        "}",
    ]
    return "\n".join(lines)


def parse_and_validate_selection(
    text: str, candidates: list[Any]
) -> LLMSectionSelection:
    """Validate a shortlist selection; raises on anything unsafe."""
    data = _extract_json_object(text)

    unexpected = sorted(set(data) - ALLOWED_SELECTION_KEYS)
    if unexpected:
        raise SuggestionValidationError(
            f"Response contains unexpected key(s) {unexpected}; only "
            f"{sorted(ALLOWED_SELECTION_KEYS)} are allowed. The model may not "
            "invent targeting fields."
        )

    allowed: dict[str, str] = {}
    for candidate in candidates:
        entry = candidate.to_dict() if hasattr(candidate, "to_dict") else dict(candidate)
        allowed[entry["section_id"]] = entry.get("heading", "")

    decision = data.get("decision")
    if decision not in SELECTION_DECISIONS:
        raise SuggestionValidationError(
            f"Invalid decision {decision!r}; allowed: {SELECTION_DECISIONS}."
        )

    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise SuggestionValidationError("Confidence must be a number.")
    if not 0.0 <= float(confidence) <= 1.0:
        raise SuggestionValidationError(f"Confidence {confidence} is outside [0, 1].")

    reasoning = str(data.get("reasoning", "")).strip()
    if not reasoning:
        raise SuggestionValidationError("Reasoning must not be empty.")
    reasoning = reasoning[:MAX_REASONING_CHARS]

    section_id = data.get("section_id")
    if decision == NO_SUITABLE_SECTION:
        return LLMSectionSelection(
            NO_SUITABLE_SECTION, None, None, float(confidence), reasoning
        )

    if not section_id or section_id not in allowed:
        raise SuggestionValidationError(
            f"section_id {section_id!r} is not one of the shortlisted "
            f"candidates: {sorted(allowed)}."
        )
    # The heading always comes from the candidate, never from the model.
    return LLMSectionSelection(
        SELECT_EXISTING, section_id, allowed[section_id], float(confidence), reasoning
    )


def select_section_with_llm(
    change_summary: str,
    candidates: list[Any],
    provider: Optional[LLMProvider] = None,
    changed_paths: Optional[list[str]] = None,
    changed_symbols: Optional[list[str]] = None,
    additional_files_omitted: int = 0,
) -> Optional[LLMSectionSelection]:
    """Ask the provider to pick one shortlisted section.

    Returns ``None`` for any unusable output (invalid JSON, bad decision,
    invented/out-of-shortlist id, bad confidence, empty reasoning, provider
    error) so callers fall back to deterministic routing.
    """
    if not candidates:
        return None
    provider = provider or get_llm_provider_from_env()
    prompt = build_selection_prompt(
        change_summary,
        candidates,
        changed_paths,
        changed_symbols,
        additional_files_omitted,
    )
    try:
        response = provider.generate(
            prompt,
            system_prompt=SELECTION_SYSTEM_PROMPT,
            json_schema=SELECTION_JSON_SCHEMA,
        )
        return parse_and_validate_selection(response.text, candidates)
    except SuggestionValidationError as err:
        print(f"[llm_change_analyzer] Rejecting LLM selection: {err}")
        return None


def selection_to_routing_decision(
    selection: LLMSectionSelection,
    candidates: list[Any],
) -> Optional[SummaryRoutingDecision]:
    """Convert a validated selection into a routing decision (or ``None``)."""
    if selection.decision != SELECT_EXISTING or not selection.section_id:
        return None
    return SummaryRoutingDecision(
        decision=UPDATE_EXISTING,
        target_section_id=selection.section_id,
        target_heading=selection.heading,
        reasoning=f"LLM selected from shortlist: {selection.reasoning}",
        skeleton_should_change=False,
        confidence=selection.confidence,
        candidates=[
            c.to_dict() if hasattr(c, "to_dict") else dict(c) for c in candidates
        ],
    )


def analyze_change(
    changed_files: list[Any],
    change_summary: str,
    skeleton: SummarySkeleton,
    provider: Optional[LLMProvider] = None,
) -> Optional[LLMChangeSuggestion]:
    """Ask the provider for a placement suggestion (legacy full-skeleton path).

    Returns ``None`` whenever the output is unusable (invalid JSON, bad
    decision, unknown section, out-of-range confidence, provider error), so
    callers fall back to the rule-based router.
    """
    provider = provider or get_llm_provider_from_env()
    prompt = build_prompt(changed_files, change_summary, skeleton)

    try:
        response = provider.generate(
            prompt, system_prompt=SYSTEM_PROMPT, json_schema=SUGGESTION_JSON_SCHEMA
        )
        return parse_and_validate_suggestion(response.text, skeleton)
    except SuggestionValidationError as err:
        print(f"[llm_change_analyzer] Falling back to rule-based routing: {err}")
        return None


def suggestion_to_routing_decision(
    suggestion: LLMChangeSuggestion,
) -> SummaryRoutingDecision:
    """Convert a validated suggestion into a routing decision."""
    return SummaryRoutingDecision(
        decision=suggestion.decision,
        target_section_id=suggestion.target_section_id,
        target_heading=(
            suggestion.target_heading if suggestion.decision == UPDATE_EXISTING else None
        ),
        new_heading=(
            suggestion.target_heading if suggestion.decision == CREATE_NEW else None
        ),
        reasoning=f"LLM suggestion: {suggestion.reasoning}",
        skeleton_should_change=(suggestion.decision == CREATE_NEW),
    )


def fallback_suggestion(
    changed_files: list[Any],
    change_summary: str,
    skeleton: SummarySkeleton,
) -> LLMChangeSuggestion:
    """Rule-based suggestion (used when the LLM output is unusable)."""
    decision = route_change(change_summary, changed_files, skeleton)
    heading = decision.target_heading or decision.new_heading
    return LLMChangeSuggestion(
        decision=decision.decision,
        target_section_id=decision.target_section_id,
        target_heading=heading,
        requires_new_section=decision.skeleton_should_change,
        confidence=1.0,  # rules are deterministic
        suggested_summary=change_summary,
        reasoning=decision.reasoning,
    )


def main() -> int:
    """Preview the suggestion for the latest change package. Read-only."""
    skeleton_file = summary_skeleton_path(".")
    if not skeleton_file.exists():
        print(
            "No base_skeleton.json found; run "
            "'python3 -m src.summary_skeleton_builder' first."
        )
        return 1
    skeleton = load_summary_skeleton(skeleton_file)

    package_file = change_package_path(".")
    changed_files: list[ChangedFile] = []
    change_summary = "No change package available."
    package: dict = {}
    if package_file.exists():
        package = json.loads(package_file.read_text(encoding="utf-8"))
        change_summary = package.get("generated_summary", change_summary)
        changed_files = [
            ChangedFile(
                path=entry.get("path", ""),
                change_type=entry.get("change_type", "unknown"),
                old_path=entry.get("old_path"),
            )
            for entry in package.get("changed_files", [])
        ]

    # Section content lets the deterministic scorer weigh real prose.
    summary_file = updated_summary_path(".")
    summary_markdown = (
        summary_file.read_text(encoding="utf-8") if summary_file.exists() else None
    )
    file_details = package.get("changed_files") if package_file.exists() else None

    assessment, _catalog = build_routing_context(
        change_summary,
        changed_files,
        skeleton,
        file_details=file_details,
        summary_text=summary_markdown,
    )
    candidates = assessment.candidates

    print("=" * 60)
    print("Deterministic shortlist (the only sections the LLM may choose from)")
    print("=" * 60)
    for candidate in candidates:
        print(
            f"  {candidate.rank}. {candidate.heading}  "
            f"[{candidate.section_id}]  score={candidate.score:.1f}"
        )
        print(f"     breakdown: {candidate.score_breakdown}")
    print(
        f"Deterministic verdict: {assessment.strength} "
        f"(confidence {assessment.confidence}, ambiguous={assessment.ambiguous})"
    )

    provider = get_llm_provider_from_env(os.environ)
    print(f"\nProvider: {getattr(provider, 'name', type(provider).__name__)}")

    signals = extract_change_signals(change_summary, changed_files, file_details)
    prompt_paths, omitted = select_files_for_llm(signals, candidates)
    selection = select_section_with_llm(
        change_summary,
        candidates,
        provider=provider,
        changed_paths=prompt_paths,
        changed_symbols=sorted(signals.symbols),
        additional_files_omitted=omitted,
    )

    print("=" * 60)
    if selection is None:
        print("LLM selection unavailable/invalid — deterministic result stands:")
        top = assessment.top
        print(f"  Target: {top.heading if top else '(none)'}")
    elif selection.decision == NO_SUITABLE_SECTION:
        print("LLM reported no suitable section — deterministic result stands:")
        print(f"  Reasoning: {selection.reasoning}")
    else:
        print("LLM selected from the shortlist:")
        print(f"  Section id: {selection.section_id}")
        print(f"  Heading:    {selection.heading}")
        print(f"  Confidence: {selection.confidence}")
        print(f"  Reasoning:  {selection.reasoning}")
    print(f"  Files sent: {len(prompt_paths)} (additional_files_omitted={omitted})")
    print("=" * 60)
    print("(read-only preview — no files were modified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
