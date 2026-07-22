"""Tests for deterministic section-candidate scoring over *actual* headings.

Every skeleton here deliberately uses headings unlike the original eight
("Quality and Tests", "CI/CD Review Flow", ...) to prove the router never
depends on a fixed heading vocabulary. Fully offline: mocked providers only,
no network, no real Ollama, no real artifacts.
"""

from __future__ import annotations

import json

import pytest

from src.git_change_detector import ChangedFile
from src.llm_change_analyzer import (
    NO_SUITABLE_SECTION,
    SELECT_EXISTING,
    LLMSectionSelection,
    SuggestionValidationError,
    build_selection_prompt,
    parse_and_validate_selection,
    select_section_with_llm,
    selection_to_routing_decision,
)
from src.llm_provider import LLMResponse
from src.section_candidate_scorer import (
    AUTOMATION_PIPELINE,
    CI_DEPLOYMENT,
    CONFIGURATION_DEPENDENCIES,
    MIN_CANDIDATE_SCORE,
    TESTING,
    build_section_catalog,
    extract_change_signals,
    find_overview_section,
    infer_categories,
    rank_candidates,
    assess_candidates,
)
from src.summary_change_router import (
    CREATE_NEW,
    UPDATE_EXISTING,
    build_routing_context,
    route_change,
)
from src.summary_skeleton_store import SummarySkeleton, append_section

# Headings deliberately unlike the legacy fixed vocabulary.
VARIABLE_HEADINGS = [
    "Purpose and Product",
    "Architecture",
    "Repository Automation",
    "Quality and Tests",
    "Dependencies and Environment",
    "CI/CD Review Flow",
    "Constraints and Risks",
]


def make_skeleton(headings=None, root="Project Technical Summary"):
    skeleton = SummarySkeleton(
        project_id="techdocker", source_summary_path="s.md", generated_at=""
    )
    top = append_section(skeleton, root, level=1)
    for heading in headings if headings is not None else VARIABLE_HEADINGS:
        append_section(skeleton, heading, level=2, parent_id=top.section_id)
    return skeleton


def summary_markdown(section_bodies: dict[str, str], root="Project Technical Summary"):
    parts = [f"# {root}", ""]
    for heading, body in section_bodies.items():
        parts += [f"## {heading}", "", body, ""]
    return "\n".join(parts)


def files(*paths, status="modified"):
    return [ChangedFile(path=path, change_type=status) for path in paths]


def route(paths, skeleton=None, summary_text=None, change_summary="change", **kwargs):
    return route_change(
        change_summary,
        files(*paths) if isinstance(paths, (list, tuple)) else paths,
        skeleton if skeleton is not None else make_skeleton(),
        summary_text=summary_text,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# routing to real, non-legacy headings
# ---------------------------------------------------------------------------
def test_test_only_changes_route_to_quality_and_tests():
    decision = route(["tests/test_router.py", "tests/test_updater.py"])
    assert decision.decision == UPDATE_EXISTING
    assert decision.target_heading == "Quality and Tests"


def test_workflow_changes_route_to_ci_cd_review_flow():
    decision = route([".github/workflows/documentation-update.yml"])
    assert decision.target_heading == "CI/CD Review Flow"


def test_requirements_route_to_dependencies_and_environment():
    assert route(["requirements.txt"]).target_heading == "Dependencies and Environment"
    assert route(["pyproject.toml"]).target_heading == "Dependencies and Environment"


def test_router_modules_route_to_repository_automation():
    decision = route(["src/summary_change_router.py", "src/summary_updater.py"])
    assert decision.target_heading == "Repository Automation"


def test_core_modules_route_to_architecture():
    decision = route(["src/payment_service.py"])
    assert decision.target_heading == "Architecture"


def test_no_legacy_heading_is_required_anywhere():
    # None of the legacy eight headings exist in this skeleton at all.
    legacy = {
        "System Overview", "Repository Structure", "Core Modules",
        "Automation Pipeline", "Testing Strategy", "Configuration",
        "Deployment and CI", "Known Limitations",
    }
    assert not legacy & set(VARIABLE_HEADINGS)
    for paths in (["tests/test_a.py"], [".github/workflows/ci.yml"], ["src/x.py"]):
        decision = route(paths)
        assert decision.decision == UPDATE_EXISTING
        assert decision.target_heading in VARIABLE_HEADINGS


def test_section_content_can_make_a_non_obvious_heading_win():
    skeleton = make_skeleton(["Architecture", "Data Plumbing"])
    markdown = summary_markdown(
        {
            "Architecture": "General notes about layering.",
            "Data Plumbing": (
                "The ingest path is implemented in src/ingest_pipeline.py "
                "and validated downstream."
            ),
        }
    )
    decision = route(["src/ingest_pipeline.py"], skeleton, summary_text=markdown)
    assert decision.target_heading == "Data Plumbing"


# ---------------------------------------------------------------------------
# signal strength
# ---------------------------------------------------------------------------
def test_explicit_module_mention_scores_strongly():
    skeleton = make_skeleton(["Architecture", "Storage Layer"])
    markdown = summary_markdown(
        {
            "Architecture": "High level design notes.",
            "Storage Layer": "Backed by src/storage_engine.py for persistence.",
        }
    )
    assessment, _ = build_routing_context(
        "change", files("src/storage_engine.py"), skeleton, summary_text=markdown
    )
    top = assessment.top
    assert top.heading == "Storage Layer"
    assert top.score_breakdown["explicit_path"] == 15.0


def test_explicit_symbol_mention_scores_strongly():
    skeleton = make_skeleton(["Architecture", "Reporting"])
    markdown = summary_markdown(
        {
            "Architecture": "Design notes.",
            "Reporting": "The ReportBuilder class renders the output.",
        }
    )
    details = [
        {
            "path": "src/misc_helper.py",
            "status": "modified",
            "what_changed": [
                {"summary": "Modified class ReportBuilder.", "symbols": ["ReportBuilder"]}
            ],
        }
    ]
    assessment, _ = build_routing_context(
        "change", [], skeleton, file_details=details, summary_text=markdown
    )
    assert assessment.top.heading == "Reporting"
    assert assessment.top.score_breakdown["explicit_symbol"] == 12.0


def test_hunk_summaries_and_changed_lines_affect_scoring():
    skeleton = make_skeleton(["Architecture", "Quality and Tests"])
    details = [
        {
            "path": "src/thing.py",
            "status": "modified",
            "what_changed": [
                {
                    "summary": "Modified function validate_pytest_fixture.",
                    "symbols": [],
                    "added_lines": [{"line_number": 1, "text": "pytest coverage hook"}],
                }
            ],
        }
    ]
    with_hunks, _ = build_routing_context("change", [], skeleton, file_details=details)
    without, _ = build_routing_context(
        "change", [], skeleton,
        file_details=[{"path": "src/thing.py", "status": "modified", "what_changed": []}],
    )
    quality_with = next(c.score for c in with_hunks.candidates if c.heading == "Quality and Tests")
    quality_without = next(
        (c.score for c in without.candidates if c.heading == "Quality and Tests"), 0.0
    )
    assert quality_with > quality_without


def test_repeated_tokens_do_not_inflate_scores():
    skeleton = make_skeleton(["Quality and Tests"])
    repeated = [
        {
            "path": "src/thing.py",
            "status": "modified",
            "what_changed": [
                {
                    "summary": "pytest pytest pytest validation validation",
                    "symbols": [],
                    "added_lines": [
                        {"line_number": i, "text": "pytest validation quality"}
                        for i in range(50)
                    ],
                }
            ],
        }
    ]
    once = [
        {
            "path": "src/thing.py",
            "status": "modified",
            "what_changed": [
                {"summary": "pytest validation", "symbols": [], "added_lines": []}
            ],
        }
    ]
    high, _ = build_routing_context("c", [], skeleton, file_details=repeated)
    low, _ = build_routing_context("c", [], skeleton, file_details=once)
    # Capped components mean repetition cannot run away.
    assert high.top.score <= low.top.score + 12.0


def test_tests_are_supporting_weight_when_source_also_changed():
    skeleton = make_skeleton(["Repository Automation", "Quality and Tests"])
    only_tests = extract_change_signals("c", files("tests/test_router.py"))
    mixed = extract_change_signals(
        "c", files("tests/test_router.py", "src/summary_change_router.py")
    )
    assert only_tests.category_weights[TESTING] == 1.0
    assert mixed.category_weights[TESTING] < 1.0
    # ...and the production section wins the mixed change.
    decision = route(
        ["tests/test_router.py", "src/summary_change_router.py"], skeleton
    )
    assert decision.target_heading == "Repository Automation"


def test_test_only_changes_still_win_the_testing_section():
    skeleton = make_skeleton(["Repository Automation", "Quality and Tests"])
    assert route(["tests/test_router.py"], skeleton).target_heading == "Quality and Tests"


# ---------------------------------------------------------------------------
# statuses, binary, renames, v1 compatibility
# ---------------------------------------------------------------------------
def test_binary_change_does_not_crash_and_uses_path_signals():
    details = [
        {
            "path": ".github/workflows/logo.png",
            "status": "added",
            "binary": True,
            "binary_note": "Binary file changed; textual hunks were not extracted.",
            "what_changed": [],
        }
    ]
    decision = route_change("change", [], make_skeleton(), file_details=details)
    assert decision.target_heading == "CI/CD Review Flow"


@pytest.mark.parametrize("status", ["added", "modified", "deleted", "renamed"])
def test_all_statuses_contribute_safely(status):
    decision = route(["tests/test_thing.py"], change_summary="c")
    assert decision.decision == UPDATE_EXISTING
    details = [{"path": "tests/test_thing.py", "status": status, "what_changed": []}]
    assert route_change("c", [], make_skeleton(), file_details=details).target_heading == (
        "Quality and Tests"
    )


def test_old_path_contributes_for_renames():
    details = [
        {
            "path": "src/renamed_module.py",
            "old_path": ".github/workflows/old_flow.yml",
            "status": "renamed",
            "what_changed": [],
        }
    ]
    signals = extract_change_signals("c", [], details)
    assert ".github/workflows/old_flow.yml" in signals.paths
    assert CI_DEPLOYMENT in signals.category_weights


def test_schema_v1_changed_file_input_still_works():
    decision = route_change(
        "summary", files("tests/test_v1.py"), make_skeleton(), file_details=None
    )
    assert decision.decision == UPDATE_EXISTING
    assert decision.target_heading == "Quality and Tests"


# ---------------------------------------------------------------------------
# determinism, ordering, breakdown
# ---------------------------------------------------------------------------
def test_shortlist_is_deterministic_and_limited_to_three():
    first = route(["src/summary_change_router.py"])
    second = route(["src/summary_change_router.py"])
    assert len(first.candidates) <= 3
    assert [c["section_id"] for c in first.candidates] == [
        c["section_id"] for c in second.candidates
    ]


def test_ties_break_on_skeleton_order():
    # Two sections with identical semantics: the earlier skeleton entry wins.
    skeleton = make_skeleton(["Alpha Notes", "Beta Notes"])
    assessment, _ = build_routing_context("change", files("README.md"), skeleton)
    scores = [c.score for c in assessment.candidates]
    assert scores[0] == scores[1]  # genuinely tied
    assert assessment.candidates[0].heading == "Alpha Notes"


def test_score_breakdown_sums_to_total():
    assessment, _ = build_routing_context(
        "change", files("tests/test_a.py"), make_skeleton()
    )
    for candidate in assessment.candidates:
        assert candidate.score == pytest.approx(sum(candidate.score_breakdown.values()))


# ---------------------------------------------------------------------------
# confidence, ambiguity, fallback
# ---------------------------------------------------------------------------
def test_weak_signal_change_is_low_confidence_or_unsuitable():
    assessment, _ = build_routing_context("change", files("Makefile"), make_skeleton())
    assert assessment.top is None or assessment.top.score < MIN_CANDIDATE_SCORE
    assert assessment.strength == "none"


def test_overview_fallback_is_found_semantically():
    # "Purpose and Product" is the overview-equivalent; no "System Overview".
    catalog = build_section_catalog(make_skeleton())
    overview = find_overview_section(catalog)
    assert overview.heading == "Purpose and Product"

    decision = route(["Makefile"])
    assert decision.decision == UPDATE_EXISTING
    assert decision.target_heading == "Purpose and Product"
    assert decision.ambiguous is True


def test_no_generic_section_is_invented():
    generic = {"code changes", "changes", "updates", "miscellaneous", "other", "general"}
    for paths in (["Makefile"], ["tests/test_a.py"], ["src/x.py"], ["weird.xyz"]):
        decision = route(paths)
        proposed = (decision.new_heading or decision.target_heading or "").lower()
        assert proposed not in generic


def test_document_root_is_not_a_candidate():
    catalog = build_section_catalog(make_skeleton())
    assert "Project Technical Summary" not in [entry.heading for entry in catalog]


def test_generated_update_sections_are_excluded():
    skeleton = make_skeleton(["Architecture"])
    append_section(skeleton, "Automated Change Update - 2026-01-01", level=3)
    catalog = build_section_catalog(skeleton)
    assert all(
        not entry.heading.startswith("Automated Change Update") for entry in catalog
    )


def test_categories_are_inferred_not_assumed_headings():
    assert TESTING in infer_categories("Quality and Tests")
    assert CI_DEPLOYMENT in infer_categories("CI/CD Review Flow")
    assert CONFIGURATION_DEPENDENCIES in infer_categories("Dependencies and Environment")
    assert AUTOMATION_PIPELINE in infer_categories("Repository Automation")


def test_candidate_context_is_bounded():
    long_body = "prose " * 5000
    skeleton = make_skeleton(["Architecture"])
    markdown = summary_markdown({"Architecture": long_body})
    catalog = build_section_catalog(
        skeleton,
        {entry.section_id: long_body for entry in build_section_catalog(skeleton)},
    )
    for entry in catalog:
        assert len(entry.content_excerpt) <= 600


# ---------------------------------------------------------------------------
# LLM shortlist selection (mocked providers only)
# ---------------------------------------------------------------------------
class FakeLLM:
    name = "fake"

    def __init__(self, text):
        self.text = text

    def generate(self, prompt, system_prompt=None, json_schema=None):
        self.prompt = prompt
        return LLMResponse(text=self.text, provider_name=self.name)


def shortlist():
    assessment, _ = build_routing_context(
        "change", files("src/summary_change_router.py"), make_skeleton()
    )
    return assessment.candidates


def selection_json(**overrides):
    candidates = overrides.pop("candidates", None) or shortlist()
    data = {
        "decision": SELECT_EXISTING,
        "section_id": candidates[0].section_id,
        "confidence": 0.86,
        "reasoning": "The changed router functions align with this section.",
    }
    data.update(overrides)
    return json.dumps(data)


def test_prompt_contains_only_shortlisted_ids():
    candidates = shortlist()
    prompt = build_selection_prompt("change summary", candidates)
    allowed = {c.section_id for c in candidates}
    for section_id in allowed:
        assert section_id in prompt
    # A real skeleton section outside the shortlist must not leak in.
    everything = {
        entry.section_id for entry in build_section_catalog(make_skeleton())
    }
    for outside in everything - allowed:
        assert outside not in prompt


def test_valid_selection_is_accepted():
    candidates = shortlist()
    selection = select_section_with_llm(
        "change", candidates, provider=FakeLLM(selection_json())
    )
    assert selection.decision == SELECT_EXISTING
    assert selection.section_id == candidates[0].section_id
    assert selection.heading == candidates[0].heading  # heading from candidate

    decision = selection_to_routing_decision(selection, candidates)
    assert decision.decision == UPDATE_EXISTING
    assert decision.target_section_id == candidates[0].section_id


def test_invented_section_id_is_rejected():
    text = selection_json(section_id="totally-made-up-section")
    assert select_section_with_llm("c", shortlist(), provider=FakeLLM(text)) is None


def test_real_section_outside_shortlist_is_rejected():
    candidates = shortlist()
    allowed = {c.section_id for c in candidates}
    outside = next(
        entry.section_id
        for entry in build_section_catalog(make_skeleton())
        if entry.section_id not in allowed
    )
    text = selection_json(section_id=outside)
    assert select_section_with_llm("c", candidates, provider=FakeLLM(text)) is None


def test_malformed_json_falls_back_safely():
    assert select_section_with_llm("c", shortlist(), provider=FakeLLM("not json")) is None


@pytest.mark.parametrize("bad", [1.4, -0.2, "high", None])
def test_out_of_range_confidence_rejected(bad):
    text = selection_json(confidence=bad)
    assert select_section_with_llm("c", shortlist(), provider=FakeLLM(text)) is None


def test_empty_reasoning_rejected():
    text = selection_json(reasoning="   ")
    assert select_section_with_llm("c", shortlist(), provider=FakeLLM(text)) is None


def test_no_suitable_section_is_valid_and_routes_to_none():
    text = json.dumps(
        {
            "decision": NO_SUITABLE_SECTION,
            "section_id": None,
            "confidence": 0.62,
            "reasoning": "None of the supplied sections describes this capability.",
        }
    )
    candidates = shortlist()
    selection = select_section_with_llm("c", candidates, provider=FakeLLM(text))
    assert selection.decision == NO_SUITABLE_SECTION
    assert selection_to_routing_decision(selection, candidates) is None


def test_empty_shortlist_never_calls_provider():
    class Exploding:
        name = "exploding"

        def generate(self, *args, **kwargs):
            raise AssertionError("provider must not be called with no candidates")

    assert select_section_with_llm("c", [], provider=Exploding()) is None


@pytest.mark.parametrize(
    "extra_key",
    ["new_heading", "target_heading", "new_section_id", "section", "replacement_section"],
)
def test_unexpected_targeting_keys_are_rejected(extra_key):
    candidates = shortlist()
    text = json.dumps(
        {
            "decision": SELECT_EXISTING,
            "section_id": candidates[0].section_id,
            extra_key: "Invented Section",
            "confidence": 0.9,
            "reasoning": "ok",
        }
    )
    with pytest.raises(SuggestionValidationError, match="unexpected key"):
        parse_and_validate_selection(text, candidates)
    # ...and the safe path returns None rather than accepting it.
    assert select_section_with_llm("c", candidates, provider=FakeLLM(text)) is None


def test_no_placement_or_patch_fields_are_produced():
    decision = route(["src/summary_change_router.py"])
    forbidden = {"block_id", "sentence_id", "patch", "placement", "offset"}
    assert not forbidden & set(vars(decision))


# ---------------------------------------------------------------------------
# generated-region and structural-inventory contamination (corrections 3 & 4)
# ---------------------------------------------------------------------------
def test_generated_update_region_contributes_no_routing_evidence():
    # "Constraints and Risks" once recorded a change to src/payment_router.py
    # inside a generated block. That history must not win a NEW change to it.
    skeleton = make_skeleton(["Architecture", "Constraints and Risks"])
    markdown = summary_markdown(
        {
            "Architecture": "The service layer implements payment processing.",
            "Constraints and Risks": (
                "Known deployment constraints.\n\n"
                "<!-- TECHDOCKER_UPDATE_START -->\n"
                "### Automated Change Update - 2026-01-01\n\n"
                "Changed files:\n"
                "- modified: src/payment_router.py\n\n"
                "Summary: reworked PaymentRouter dispatch.\n"
                "<!-- TECHDOCKER_UPDATE_END -->"
            ),
        }
    )
    assessment, catalog = build_routing_context(
        "change", files("src/payment_router.py"), skeleton, summary_text=markdown
    )
    risks = next(c for c in assessment.candidates if c.heading == "Constraints and Risks")
    assert "explicit_path" not in risks.score_breakdown
    assert assessment.top.heading != "Constraints and Risks"

    entry = next(e for e in catalog if e.heading == "Constraints and Risks")
    assert "payment_router" not in entry.direct_text
    assert "payment_router" not in entry.content_excerpt


def test_code_fence_file_tree_gives_no_direct_match():
    skeleton = make_skeleton(["Repository Layout", "Architecture"])
    markdown = summary_markdown(
        {
            "Repository Layout": "```text\nsrc/payment_router.py\nsrc/other.py\n```",
            "Architecture": "General design notes.",
        }
    )
    assessment, _ = build_routing_context(
        "change", files("src/payment_router.py"), skeleton, summary_text=markdown
    )
    layout = next(c for c in assessment.candidates if c.heading == "Repository Layout")
    assert "explicit_path" not in layout.score_breakdown


def test_file_inventory_table_gives_no_direct_match():
    skeleton = make_skeleton(["Module Index", "Architecture"])
    markdown = summary_markdown(
        {
            "Module Index": (
                "| module | purpose |\n| --- | --- |\n"
                "| src/payment_router.py | routing |\n"
            ),
            "Architecture": "General design notes.",
        }
    )
    assessment, _ = build_routing_context(
        "change", files("src/payment_router.py"), skeleton, summary_text=markdown
    )
    index = next(c for c in assessment.candidates if c.heading == "Module Index")
    assert "explicit_path" not in index.score_breakdown


def test_bare_inventory_bullets_give_no_direct_match_but_prose_does():
    skeleton = make_skeleton(["Module Index", "Payments"])
    markdown = summary_markdown(
        {
            "Module Index": "- `src/payment_router.py`\n- `src/other.py`",
            "Payments": (
                "Card capture is dispatched by src/payment_router.py before "
                "settlement runs."
            ),
        }
    )
    assessment, _ = build_routing_context(
        "change", files("src/payment_router.py"), skeleton, summary_text=markdown
    )
    index = next(c for c in assessment.candidates if c.heading == "Module Index")
    payments = next(c for c in assessment.candidates if c.heading == "Payments")
    assert "explicit_path" not in index.score_breakdown
    assert payments.score_breakdown["explicit_path"] == 15.0   # prose still strong
    assert assessment.top.heading == "Payments"


# ---------------------------------------------------------------------------
# all-file aggregation and bounded LLM payload (correction 5)
# ---------------------------------------------------------------------------
def test_relevant_file_after_position_25_still_routes():
    skeleton = make_skeleton(["Architecture", "CI/CD Review Flow"])
    # 40 uncategorized assets, then the only meaningful path at position 41.
    padding = [
        {"path": f"assets/blob_{i:03d}.dat", "status": "modified", "what_changed": []}
        for i in range(40)
    ]
    late = {
        "path": ".github/workflows/release.yml",
        "status": "modified",
        "what_changed": [],
    }
    decision = route_change("change", [], skeleton, file_details=padding + [late])
    assert decision.target_heading == "CI/CD Review Flow"

    signals = extract_change_signals("change", [], padding + [late])
    assert ".github/workflows/release.yml" in signals.paths  # not dropped
    assert len(signals.paths) == 41  # every file aggregated, no 25-file cutoff


def test_llm_payload_is_bounded_and_reports_omissions():
    from src.section_candidate_scorer import MAX_FILES_FOR_LLM, select_files_for_llm

    details = [
        {"path": f"src/module_{i:03d}.py", "status": "modified", "what_changed": []}
        for i in range(40)
    ]
    skeleton = make_skeleton(["Architecture"])
    assessment, _ = build_routing_context("change", [], skeleton, file_details=details)
    signals = extract_change_signals("change", [], details)

    paths, omitted = select_files_for_llm(signals, assessment.candidates)
    assert len(paths) == MAX_FILES_FOR_LLM
    assert omitted == 40 - MAX_FILES_FOR_LLM

    prompt = build_selection_prompt(
        "change", assessment.candidates, paths, [], additional_files_omitted=omitted
    )
    assert f'"additional_files_omitted": {omitted}' in prompt
    assert prompt.count("src/module_") <= MAX_FILES_FOR_LLM


# ---------------------------------------------------------------------------
# full-content keywords vs bounded excerpt (correction 6)
# ---------------------------------------------------------------------------
def test_prose_after_600_chars_still_scores_but_excerpt_stays_bounded():
    from src.section_candidate_scorer import MAX_CONTENT_EXCERPT_CHARS

    filler = "General background prose about the platform. " * 30  # > 600 chars
    late_prose = "Settlement is dispatched by src/payment_router.py at capture time."
    skeleton = make_skeleton(["Payments", "Architecture"])
    markdown = summary_markdown(
        {
            "Payments": f"{filler}\n\n{late_prose}",
            "Architecture": "General design notes.",
        }
    )
    assessment, catalog = build_routing_context(
        "change", files("src/payment_router.py"), skeleton, summary_text=markdown
    )
    entry = next(e for e in catalog if e.heading == "Payments")

    # The excerpt (LLM payload) is bounded...
    assert len(entry.content_excerpt) <= MAX_CONTENT_EXCERPT_CHARS
    assert "payment_router" not in entry.content_excerpt
    # ...but deterministic evidence saw the whole section.
    assert "payment_router" in entry.direct_text
    payments = next(c for c in assessment.candidates if c.heading == "Payments")
    assert payments.score_breakdown["explicit_path"] == 15.0
    assert assessment.top.heading == "Payments"


# ---------------------------------------------------------------------------
# tightened new-section fallback (correction 7)
# ---------------------------------------------------------------------------
def test_non_empty_variable_skeleton_never_gets_a_category_heading():
    # No testing-like section exists; a test-only change must not invent one.
    skeleton = make_skeleton(["Purpose and Product", "Architecture"])
    decision = route(["tests/test_thing.py"], skeleton)
    assert decision.decision == UPDATE_EXISTING
    assert decision.new_heading is None
    assert decision.skeleton_should_change is False
    assert decision.ambiguous is True
    # Internal category names never surface as headings.
    assert decision.target_heading in {"Purpose and Product", "Architecture"}


def test_internal_category_names_never_become_headings():
    skeleton = make_skeleton(["Purpose and Product"])
    for paths in (["requirements.txt"], ["src/summary_updater.py"], [".github/workflows/x.yml"]):
        decision = route(paths, skeleton)
        proposed = (decision.new_heading or "") .lower()
        assert proposed not in {
            "configuration_dependencies", "automation_pipeline", "ci_deployment",
            "testing", "architecture_modules",
        }


def test_empty_skeleton_still_creates_controlled_section():
    empty = SummarySkeleton(
        project_id="techdocker", source_summary_path="s.md", generated_at=""
    )
    decision = route_change("summary", [], empty)
    assert decision.decision == CREATE_NEW
    assert decision.new_heading == "System Overview"
    assert decision.skeleton_should_change is True


# ---------------------------------------------------------------------------
# routing confidence must agree with ambiguity (correction 3)
# ---------------------------------------------------------------------------
def test_exact_tie_is_ambiguous_and_not_high_confidence():
    from src.section_candidate_scorer import EXACT_TIE_CONFIDENCE_CAP

    # Two sections with identical semantics produce a genuine tie.
    skeleton = make_skeleton(["Alpha Notes", "Beta Notes"])
    assessment, _ = build_routing_context("change", files("README.md"), skeleton)

    assert assessment.candidates[0].score == assessment.candidates[1].score
    assert assessment.ambiguous is True
    assert assessment.confidence <= EXACT_TIE_CONFIDENCE_CAP
    # Deterministic tie-breaking by skeleton order is unchanged.
    assert assessment.candidates[0].heading == "Alpha Notes"


def test_high_scoring_exact_tie_cannot_report_high_confidence():
    from src.section_candidate_scorer import EXACT_TIE_CONFIDENCE_CAP

    # Mirrors the real preview: two strong sections tied at the same score.
    skeleton = make_skeleton(["Core Modules", "Automation Pipeline"])
    markdown = summary_markdown(
        {
            "Core Modules": "Modules implemented in src/summary_updater.py.",
            "Automation Pipeline": "Automation implemented in src/summary_updater.py.",
        }
    )
    assessment, _ = build_routing_context(
        "change", files("src/summary_updater.py"), skeleton, summary_text=markdown
    )
    top, second = assessment.candidates[0], assessment.candidates[1]
    assert top.score == second.score          # exact tie, high absolute score
    assert top.score >= 20
    assert assessment.ambiguous is True
    assert assessment.confidence <= EXACT_TIE_CONFIDENCE_CAP


def test_near_tie_inside_the_margin_is_penalized():
    from src.section_candidate_scorer import (
        AMBIGUITY_MARGIN,
        AMBIGUOUS_CONFIDENCE_CAP,
    )

    skeleton = make_skeleton(["Quality and Tests", "Repository Automation"])
    markdown = summary_markdown(
        {
            # Both plausible; a small margin separates them.
            "Quality and Tests": "Validation of pytest fixtures and coverage.",
            "Repository Automation": "Automation covers pytest validation too.",
        }
    )
    assessment, _ = build_routing_context(
        "change", files("tests/test_pipeline.py"), skeleton, summary_text=markdown
    )
    margin = assessment.candidates[0].score - assessment.candidates[1].score
    if 0 < margin < AMBIGUITY_MARGIN:
        assert assessment.ambiguous is True
        assert assessment.confidence <= AMBIGUOUS_CONFIDENCE_CAP


def test_strong_clear_match_retains_high_confidence():
    skeleton = make_skeleton(["Purpose and Product", "Payments"])
    markdown = summary_markdown(
        {
            "Purpose and Product": "General product overview prose.",
            "Payments": (
                "Settlement is dispatched by src/payment_router.py before "
                "capture completes."
            ),
        }
    )
    assessment, _ = build_routing_context(
        "change", files("src/payment_router.py"), skeleton, summary_text=markdown
    )
    assert assessment.ambiguous is False
    assert assessment.confidence > 0.7          # a clear winner stays confident
    assert assessment.strength == "strong"


def test_confidence_never_contradicts_ambiguity():
    from src.section_candidate_scorer import AMBIGUOUS_CONFIDENCE_CAP

    for headings, paths in (
        (["Alpha Notes", "Beta Notes"], ["README.md"]),
        (["Quality and Tests", "Repository Automation"], ["tests/test_a.py"]),
        (["Architecture", "Purpose and Product"], ["src/thing.py"]),
    ):
        assessment, _ = build_routing_context(
            "change", files(*paths), make_skeleton(headings)
        )
        if assessment.ambiguous:
            assert assessment.confidence <= AMBIGUOUS_CONFIDENCE_CAP
