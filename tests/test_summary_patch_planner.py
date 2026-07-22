"""Tests for the optional structured patch planner.

Real indexes and real placement assessments; providers are always mocked.
Nothing is applied, no network, no artifact writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.llm_provider import LLMResponse
from src.placement_candidate_scorer import (
    APPEND_TO_SECTION as PLACEMENT_APPEND,
    MANUAL_REVIEW_NEEDED as PLACEMENT_MANUAL,
    PlacementAssessment,
    PlacementCandidate,
    score_placement_candidates,
)
from src.summary_index_builder import build_summary_index
from src.summary_patch_planner import (
    APPEND_TO_SECTION,
    INSERT_AFTER_BLOCK,
    INSERT_AFTER_SENTENCE,
    MANUAL_REVIEW_NEEDED,
    MAX_NEW_TEXT_CHARS,
    MAX_PROMPT_CHARS,
    MAX_REASONING_CHARS,
    NO_CHANGE_NEEDED,
    PATCH_SCHEMA_VERSION,
    REPLACE_BLOCK,
    REPLACE_SENTENCE,
    STATUS_MANUAL_REVIEW,
    STATUS_NOT_INVOKED,
    STATUS_PLANNED,
    PatchPlanValidationError,
    allowed_operations,
    build_patch_prompt,
    main,
    parse_and_validate_patch,
    plan_summary_patch,
)

# A document whose prose is deliberately stale relative to the change below.
SUMMARY = """\
# Project Technical Summary

## Repository Automation

The pipeline opens a pull request for review. Routing is performed by \
route_change, which relies on a fixed list of heading names. Reports are \
rendered afterwards.

- Section routing lives in src/summary_change_router.py.

## Other Area

Unrelated prose that must never appear in the prompt.
"""

CHANGE_PACKAGE = {
    "schema_version": 2,
    "generated_summary": "1 file changed (modified src/summary_change_router.py)",
    "changed_files": [
        {
            "path": "src/summary_change_router.py",
            "old_path": None,
            "status": "modified",
            "additions": 40,
            "deletions": 12,
            "binary": False,
            "what_changed": [
                {
                    "hunk_header": "@@ -20,8 +20,12 @@ def route_change",
                    "summary": "Modified function route_change with 40 added and 12 removed lines.",
                    "symbols": ["route_change"],
                    "added_lines": [
                        {"line_number": 21, "text": "candidates = rank_candidates(signals, catalog)"}
                    ],
                    "removed_lines": [
                        {"line_number": 20, "text": "fixed list of heading names"}
                    ],
                }
            ],
        }
    ],
}


def build_context(markdown: str = SUMMARY, heading_fragment: str = "repository"):
    index = build_summary_index(markdown)
    section_id = next(
        s["section_id"] for s in index["sections"] if heading_fragment in s["section_id"]
    )
    assessment = score_placement_candidates(
        CHANGE_PACKAGE, index, section_id, source_markdown=markdown
    )
    section = next(s for s in index["sections"] if s["section_id"] == section_id)
    return index, assessment, section


INDEX, ASSESSMENT, SECTION = build_context()
SOURCE_SHA = INDEX["source"]["sha256"]


def candidate_of(assessment: PlacementAssessment, kind: str):
    for candidate in assessment.candidates:
        if candidate.candidate_type == kind:
            return candidate
    return None


class FakeProvider:
    """Records the prompt and returns canned text."""

    name = "fake"

    def __init__(self, text: str):
        self.text = text
        self.prompt: str | None = None
        self.calls = 0

    def generate(self, prompt, system_prompt=None, json_schema=None):
        self.prompt = prompt
        self.calls += 1
        return LLMResponse(text=self.text, provider_name=self.name)


class ExplodingProvider:
    name = "exploding"

    def generate(self, *args, **kwargs):
        raise RuntimeError("connection reset")


def response(**overrides) -> str:
    """The MINIMAL model-facing schema: only fields the model actually decides.

    Immutable fields (section_id, target_type, old_text, source hash, list
    marker, offsets) are Python-owned and deliberately absent — supplying any
    of them makes the response invalid (an unknown key).
    """
    candidate = candidate_of(ASSESSMENT, "sentence") or ASSESSMENT.top
    payload = {
        "schema_version": PATCH_SCHEMA_VERSION,
        "operation": REPLACE_SENTENCE,
        "target_id": candidate.candidate_id,
        "new_text": (
            "Routing is performed by route_change, which scores the actual "
            "sections of the summary instead of a fixed heading list."
        ),
        "confidence": 0.91,
        "reasoning": "route_change now scores actual sections.",
    }
    payload.update(overrides)
    for key in [k for k, v in overrides.items() if v is _REMOVE]:
        payload.pop(key, None)
    return json.dumps(payload)


_REMOVE = object()


def plan(text: str, assessment: PlacementAssessment = ASSESSMENT, **kwargs):
    provider = FakeProvider(text)
    result = plan_summary_patch(
        CHANGE_PACKAGE, assessment, INDEX, provider=provider, **kwargs
    )
    return result, provider


# ---------------------------------------------------------------------------
# valid operations
# ---------------------------------------------------------------------------
def test_valid_replace_sentence():
    # A valid response carrying NO immutable fields (no old_text, section_id,
    # target_type, or source hash) still produces a fully-populated plan.
    result, _ = plan(response())
    assert result.status == STATUS_PLANNED
    assert result.instruction.operation == REPLACE_SENTENCE
    assert result.is_mutation
    instruction = result.instruction
    candidate = candidate_of(ASSESSMENT, "sentence")
    # Every immutable field is derived by Python from the index, not the model.
    assert instruction.section_id == ASSESSMENT.section_id
    assert instruction.target_type == candidate.candidate_type == "sentence"
    assert instruction.old_text == candidate.text  # exact canonical index text
    assert instruction.expected_source_sha256 == SOURCE_SHA
    # The model did supply target_id and new_text.
    assert instruction.target_id == candidate.candidate_id
    assert "route_change" in instruction.new_text


def test_valid_insert_after_sentence():
    candidate = candidate_of(ASSESSMENT, "sentence")
    result, _ = plan(
        response(
            operation=INSERT_AFTER_SENTENCE,
            new_text="route_change now ranks the actual sections of the summary.",
        )
    )
    assert result.status == STATUS_PLANNED
    assert result.instruction.operation == INSERT_AFTER_SENTENCE
    assert result.instruction.target_id == candidate.candidate_id


def test_valid_replace_block_and_insert_after_block():
    block = candidate_of(ASSESSMENT, "block")
    assert block is not None
    for operation, new_text in (
        (REPLACE_BLOCK, "- Section routing lives in src/summary_change_router.py and scores actual sections."),
        (INSERT_AFTER_BLOCK, "- route_change ranks the actual sections of the summary."),
    ):
        result, _ = plan(
            response(operation=operation, target_id=block.candidate_id, new_text=new_text)
        )
        assert result.status == STATUS_PLANNED, result.reason
        assert result.instruction.operation == operation
        # Block granularity and exact old text are derived, not supplied.
        assert result.instruction.target_type == "block"
        assert result.instruction.old_text == block.text


def test_valid_append_to_section():
    result, _ = plan(
        response(
            operation=APPEND_TO_SECTION,
            target_id=None,
            new_text="Routing now records the matched signals for route_change.",
        )
    )
    assert result.status == STATUS_PLANNED
    # Python derives the null/empty target fields for a targetless append.
    assert result.instruction.target_id is None
    assert result.instruction.target_type is None
    assert result.instruction.old_text == ""


def test_valid_manual_review_needed():
    result, _ = plan(
        response(
            operation=MANUAL_REVIEW_NEEDED,
            target_id=None,
            new_text="",
            confidence=0.4,
            reasoning="Evidence is insufficient to place this change.",
        )
    )
    assert result.instruction.operation == MANUAL_REVIEW_NEEDED
    assert result.instruction.target_type is None
    assert result.instruction.old_text == ""
    assert not result.is_mutation


def test_high_confidence_supported_no_change_needed():
    candidate = candidate_of(ASSESSMENT, "sentence") or ASSESSMENT.top
    distinctive = next(w for w in candidate.text.split() if len(w) > 5)
    result, _ = plan(
        response(
            operation=NO_CHANGE_NEEDED,
            target_id=None,
            new_text="",
            confidence=0.95,
            reasoning=f"The existing text about {distinctive} already describes this.",
        )
    )
    assert result.status == STATUS_PLANNED
    assert result.instruction.operation == NO_CHANGE_NEEDED


def test_low_confidence_no_change_is_downgraded():
    result, _ = plan(
        response(
            operation=NO_CHANGE_NEEDED,
            target_id=None, new_text="",
            confidence=0.80,  # above the mutation bar, below the no-change bar
            reasoning="The routing description already covers this.",
        )
    )
    assert result.status == STATUS_MANUAL_REVIEW
    assert result.instruction.operation == MANUAL_REVIEW_NEEDED
    assert "below the required" in result.reason
    assert result.model_confidence == 0.80  # preserved as diagnostics


def test_unsupported_no_change_reasoning_is_downgraded():
    result, _ = plan(
        response(
            operation=NO_CHANGE_NEEDED,
            target_id=None, new_text="",
            confidence=0.97,
            reasoning="Nothing to do at all.",  # cites no candidate wording
        )
    )
    assert result.status == STATUS_MANUAL_REVIEW
    assert "does not cite" in result.reason


# ---------------------------------------------------------------------------
# candidate membership and compatibility
# ---------------------------------------------------------------------------
def test_target_outside_top_three_is_rejected():
    result, _ = plan(response(target_id="totally-invented-candidate-id"))
    assert result.status == STATUS_MANUAL_REVIEW
    assert "not one of the supplied candidates" in result.reason


def test_real_index_target_outside_candidates_is_rejected():
    # A genuine block id from the index that is not in the shortlist.
    shortlisted = {c.candidate_id for c in ASSESSMENT.candidates}
    other = next(
        block["block_id"]
        for section in INDEX["sections"]
        for block in section["blocks"]
        if block["block_id"] not in shortlisted
    )
    result, _ = plan(response(target_id=other))
    assert result.status == STATUS_MANUAL_REVIEW
    assert "not one of the supplied candidates" in result.reason


def test_target_from_another_section_is_rejected():
    # A candidate that IS in the shortlist but claims a different section is
    # rejected by the section-belonging guard (defends a corrupted assessment).
    good = candidate_of(ASSESSMENT, "sentence")
    foreign = PlacementCandidate(
        candidate_id=good.candidate_id,
        candidate_type="sentence",
        section_id="a-different-section",
        block_id=good.block_id,
        sentence_id=good.sentence_id,
        block_type=good.block_type,
        text=good.text,
        score=good.score,
    )
    tampered = PlacementAssessment(
        section_id=ASSESSMENT.section_id,
        candidates=[foreign],
        recommendation="use_existing_candidate",
        reasoning="tampered",
    )
    result, _ = plan(response(target_id=good.candidate_id), assessment=tampered)
    assert result.status == STATUS_MANUAL_REVIEW
    assert "does not belong to the selected section" in result.reason


def test_sentence_operation_targeting_block_is_rejected():
    block = candidate_of(ASSESSMENT, "block")
    result, _ = plan(
        response(operation=REPLACE_SENTENCE, target_id=block.candidate_id)
    )
    assert "cannot target a block" in result.reason


def test_block_operation_targeting_sentence_is_rejected():
    sentence = candidate_of(ASSESSMENT, "sentence")
    result, _ = plan(
        response(operation=REPLACE_BLOCK, target_id=sentence.candidate_id)
    )
    assert "cannot target a sentence" in result.reason


# ---------------------------------------------------------------------------
# the model owns NONE of the immutable fields; supplying any is rejected
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "forbidden,value",
    [
        ("section_id", "some-other-section"),
        ("target_type", "block"),
        ("old_text", "Text that does not match the index."),
        ("expected_source_sha256", "0" * 64),
        ("list_marker", "- "),
        ("source_start_offset", 0),
        ("source_end_offset", 999),
        ("start_line", 1),
        ("end_line", 9),
    ],
)
def test_model_cannot_override_immutable_field(forbidden, value):
    # None of these are model-supplied any more, so each is an unknown key and
    # the response is rejected — the model cannot redirect the patch through a
    # section, type, exact text, hash, marker, or offset it does not own.
    result, _ = plan(response(**{forbidden: value}))
    assert result.status == STATUS_MANUAL_REVIEW
    assert "Unexpected response key" in result.reason


def test_immutable_fields_are_derived_not_from_the_model():
    # The model targets a candidate; Python fills in everything immutable.
    candidate = candidate_of(ASSESSMENT, "sentence")
    result, _ = plan(response(target_id=candidate.candidate_id))
    instruction = result.instruction
    assert result.status == STATUS_PLANNED
    assert instruction.section_id == ASSESSMENT.section_id == SECTION["section_id"]
    assert instruction.target_type == candidate.candidate_type
    # The canonical old_text is exactly the indexed candidate text, byte for
    # byte — the model never sent it.
    assert instruction.old_text == candidate.text
    assert instruction.expected_source_sha256 == SOURCE_SHA


# ---------------------------------------------------------------------------
# response-shape validation
# ---------------------------------------------------------------------------
def test_unknown_response_key_is_rejected():
    payload = json.loads(response())
    payload["new_section"] = "Invented"
    result, _ = plan(json.dumps(payload))
    assert "Unexpected response key" in result.reason


def test_missing_response_key_is_rejected():
    payload = json.loads(response())
    del payload["reasoning"]
    result, _ = plan(json.dumps(payload))
    assert "Missing required key" in result.reason


def test_unsupported_operation_is_rejected():
    result, _ = plan(response(operation="delete_everything"))
    assert "Unsupported operation" in result.reason


def test_unsupported_schema_version_is_rejected():
    result, _ = plan(response(schema_version=99))
    assert "Unsupported patch schema_version" in result.reason


@pytest.mark.parametrize("bad", [1.4, -0.1, "high", None, True])
def test_invalid_confidence_is_rejected(bad):
    result, _ = plan(response(confidence=bad))
    assert result.status == STATUS_MANUAL_REVIEW


def test_low_confidence_mutation_is_downgraded():
    result, _ = plan(response(confidence=0.40))
    assert result.status == STATUS_MANUAL_REVIEW
    assert "below the minimum" in result.reason
    assert result.model_reasoning  # diagnostics preserved


def test_empty_and_overlong_reasoning_rejected():
    assert plan(response(reasoning="   "))[0].status == STATUS_MANUAL_REVIEW
    long_reasoning = "x" * (MAX_REASONING_CHARS + 1)
    assert plan(response(reasoning=long_reasoning))[0].status == STATUS_MANUAL_REVIEW


def test_malformed_json_returns_manual_review():
    result, _ = plan("not json at all")
    assert result.status == STATUS_MANUAL_REVIEW
    assert "not exactly one JSON object" in result.reason


def test_trailing_prose_is_rejected():
    result, _ = plan(response() + "\n\nHope that helps!")
    assert result.status == STATUS_MANUAL_REVIEW
    assert "not exactly one JSON object" in result.reason


def test_multiple_json_objects_rejected():
    result, _ = plan(response() + response())
    assert result.status == STATUS_MANUAL_REVIEW


def test_single_code_fence_is_tolerated_but_unterminated_is_not():
    fenced = "```json\n" + response() + "\n```"
    assert plan(fenced)[0].status == STATUS_PLANNED
    assert plan("```json\n" + response())[0].status == STATUS_MANUAL_REVIEW


# ---------------------------------------------------------------------------
# grounding and size
# ---------------------------------------------------------------------------
def test_empty_mutation_text_rejected():
    result, _ = plan(response(new_text="   "))
    assert "non-empty new_text" in result.reason


def test_identical_replacement_rejected():
    candidate = candidate_of(ASSESSMENT, "sentence")
    result, _ = plan(response(new_text=candidate.text))
    assert "identical" in result.reason


@pytest.mark.parametrize(
    "generic",
    [
        "The code was updated.",
        "Various improvements were made.",
        "The system was enhanced.",
        "Several changes were implemented.",
    ],
)
def test_generic_mutation_text_rejected(generic):
    result, _ = plan(response(new_text=generic))
    assert result.status == STATUS_MANUAL_REVIEW


def test_oversized_patch_rejected():
    huge = "route_change scoring. " * 200
    assert len(huge) > MAX_NEW_TEXT_CHARS
    result, _ = plan(response(new_text=huge))
    assert "exceeds" in result.reason


def test_excessive_replacement_growth_rejected():
    candidate = candidate_of(ASSESSMENT, "sentence")
    grown = "route_change " + ("scores the actual sections of the summary. " * 30)
    result, _ = plan(response(new_text=grown[:MAX_NEW_TEXT_CHARS]))
    assert result.status == STATUS_MANUAL_REVIEW
    assert "grows the text" in result.reason


@pytest.mark.parametrize(
    "text,fragment",
    [
        ("Routing now uses `quantum_dispatcher` for route_change.", "identifier"),
        ("Routing for route_change moved to src/invented_module.py.", "path"),
        ("route_change routing is now 40% faster in v9.9.9.", "version"),
        ("route_change routing now handles 9876 sections.", "numeric"),
    ],
)
def test_unsupported_new_claims_rejected(text, fragment):
    result, _ = plan(response(new_text=text))
    assert result.status == STATUS_MANUAL_REVIEW
    assert fragment in result.reason


def test_supported_symbol_and_module_accepted():
    result, _ = plan(
        response(
            new_text=(
                "Routing is performed by route_change in "
                "src/summary_change_router.py using candidate scoring."
            )
        )
    )
    assert result.status == STATUS_PLANNED


@pytest.mark.parametrize(
    "unsafe,fragment",
    [
        ("route_change <!-- TECHDOCKER_UPDATE_START --> notes", "marker"),
        ("## New Heading\n\nroute_change notes", "heading"),
        ("route_change ```python\nx=1\n```", "code fence"),
        ("route_change\n\n| a | b |\n| - | - |", "table"),
        ("route_change scoring\x00notes", "control"),
    ],
)
def test_unsafe_markdown_constructs_rejected(unsafe, fragment):
    result, _ = plan(response(new_text=unsafe))
    assert result.status == STATUS_MANUAL_REVIEW
    assert fragment in result.reason


def test_ungrounded_text_without_any_signal_rejected():
    result, _ = plan(response(new_text="Colours and shapes are pleasant today."))
    assert "cites no concrete evidence" in result.reason


# ---------------------------------------------------------------------------
# list handling
# ---------------------------------------------------------------------------
def test_list_item_replacement_must_preserve_marker():
    block = candidate_of(ASSESSMENT, "block")
    assert block.text.lstrip().startswith("-")

    ok, _ = plan(
        response(
            operation=REPLACE_BLOCK, target_id=block.candidate_id,
            new_text="- Section routing lives in src/summary_change_router.py and scores sections.",
        )
    )
    assert ok.status == STATUS_PLANNED
    # The list marker is derived from the indexed block, not supplied.
    assert ok.instruction.list_marker == "- "

    bad, _ = plan(
        response(
            operation=REPLACE_BLOCK, target_id=block.candidate_id,
            new_text="Section routing lives in src/summary_change_router.py now.",
        )
    )
    assert bad.status == STATUS_MANUAL_REVIEW
    assert "preserve the list marker" in bad.reason


# ---------------------------------------------------------------------------
# prompt content and budgets
# ---------------------------------------------------------------------------
def test_prompt_contains_only_the_selected_section():
    _, provider = plan(response())
    prompt = provider.prompt
    assert ASSESSMENT.section_id in prompt
    assert "Unrelated prose that must never appear" not in prompt
    assert "Other Area" not in prompt


def test_prompt_caps_candidates_at_three_and_stays_within_budget():
    _, provider = plan(response())
    prompt = provider.prompt
    assert prompt.count("\n- id: ") <= 3
    assert len(prompt) <= MAX_PROMPT_CHARS


def test_prompt_excludes_generated_regions():
    markdown = SUMMARY.replace(
        "## Other Area",
        "<!-- TECHDOCKER_UPDATE_START -->\n"
        "Historic note about route_change in src/summary_change_router.py.\n"
        "<!-- TECHDOCKER_UPDATE_END -->\n\n## Other Area",
    )
    index, assessment, section = build_context(markdown)
    prompt = build_patch_prompt(
        CHANGE_PACKAGE, assessment, index, section, allowed_operations(assessment)
    )
    assert "Historic note" not in prompt
    assert "TECHDOCKER_UPDATE_START" not in prompt


def test_prompt_reports_omission_counts_and_prefers_relevant_files():
    package = json.loads(json.dumps(CHANGE_PACKAGE))
    package["changed_files"] = [
        {"path": f"assets/blob_{i:03d}.dat", "status": "modified",
         "binary": True, "what_changed": []}
        for i in range(30)
    ] + package["changed_files"]

    prompt = build_patch_prompt(
        package, ASSESSMENT, INDEX, SECTION, allowed_operations(ASSESSMENT)
    )
    assert "omitted:" in prompt
    assert "file(s)" in prompt
    # The relevant source file outranks the 30 unrelated assets.
    assert "src/summary_change_router.py" in prompt
    assert prompt.index("src/summary_change_router.py") < prompt.index("assets/blob_")
    assert len(prompt) <= MAX_PROMPT_CHARS


def test_prompt_is_deterministic():
    first = build_patch_prompt(
        CHANGE_PACKAGE, ASSESSMENT, INDEX, SECTION, allowed_operations(ASSESSMENT)
    )
    second = build_patch_prompt(
        CHANGE_PACKAGE, ASSESSMENT, INDEX, SECTION, allowed_operations(ASSESSMENT)
    )
    assert first == second


def test_v1_package_and_binary_files_produce_bounded_facts():
    v1 = {
        "generated_summary": "2 files changed",
        "changed_files": [
            {"path": "src/summary_change_router.py", "change_type": "modified",
             "old_path": None},
            {"path": "assets/logo.png", "change_type": "added", "binary": True,
             "old_path": None, "what_changed": []},
        ],
    }
    prompt = build_patch_prompt(
        v1, ASSESSMENT, INDEX, SECTION, allowed_operations(ASSESSMENT)
    )
    assert "src/summary_change_router.py" in prompt
    assert len(prompt) <= MAX_PROMPT_CHARS


# ---------------------------------------------------------------------------
# allowed-operation gating and safe paths
# ---------------------------------------------------------------------------
def test_append_recommendation_restricts_operations():
    append_assessment = PlacementAssessment(
        section_id=ASSESSMENT.section_id,
        candidates=[],
        recommendation=PLACEMENT_APPEND,
        reasoning="nothing scored",
    )
    operations = allowed_operations(append_assessment)
    assert operations == [APPEND_TO_SECTION, NO_CHANGE_NEEDED, MANUAL_REVIEW_NEEDED]

    result, _ = plan(response(), assessment=append_assessment)
    assert result.status == STATUS_MANUAL_REVIEW
    assert "not allowed for this placement" in result.reason


def test_manual_review_placement_never_calls_the_model():
    unsafe = PlacementAssessment(
        section_id=ASSESSMENT.section_id,
        candidates=[],
        recommendation=PLACEMENT_MANUAL,
        reasoning="stale index",
    )
    provider = FakeProvider(response())
    result = plan_summary_patch(CHANGE_PACKAGE, unsafe, INDEX, provider=provider)
    assert provider.calls == 0
    assert result.instruction.operation == MANUAL_REVIEW_NEEDED


def test_missing_section_never_calls_the_model():
    orphan = PlacementAssessment(
        section_id="no-such-section", candidates=[],
        recommendation="use_existing_candidate", reasoning="x",
    )
    provider = FakeProvider(response())
    result = plan_summary_patch(CHANGE_PACKAGE, orphan, INDEX, provider=provider)
    assert provider.calls == 0
    assert "not present in the summary index" in result.reason


def test_no_provider_is_non_mutating():
    result = plan_summary_patch(CHANGE_PACKAGE, ASSESSMENT, INDEX)
    assert result.status == STATUS_NOT_INVOKED
    assert result.instruction.operation == MANUAL_REVIEW_NEEDED
    assert not result.is_mutation


def test_provider_exception_returns_manual_review():
    result = plan_summary_patch(
        CHANGE_PACKAGE, ASSESSMENT, INDEX, provider=ExplodingProvider()
    )
    assert result.status == STATUS_MANUAL_REVIEW
    assert "provider failed" in result.reason


def test_direct_validation_raises_for_unsafe_output():
    with pytest.raises(PatchPlanValidationError):
        parse_and_validate_patch(
            response(target_id="nope"), ASSESSMENT, SECTION, INDEX,
            allowed_operations(ASSESSMENT), CHANGE_PACKAGE,
        )


# ---------------------------------------------------------------------------
# offline guarantees
# ---------------------------------------------------------------------------
def test_no_patch_is_applied_and_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = SUMMARY
    result, _ = plan(response())
    assert result.status == STATUS_PLANNED
    assert SUMMARY == before          # planner never mutates Markdown
    assert list(tmp_path.iterdir()) == []  # and writes nothing


def test_module_never_instantiates_a_live_provider():
    source = Path("src/summary_patch_planner.py").read_text(encoding="utf-8")
    for forbidden in (
        "OllamaLLMProvider", "get_llm_provider_from_env", "urllib",
        "requests", "socket", "http.client",
    ):
        assert forbidden not in source, forbidden


def test_cli_is_read_only_and_contacts_no_provider(tmp_path, capsys):
    index_file = tmp_path / "index.json"
    package_file = tmp_path / "change.json"
    source_file = tmp_path / "summary.md"
    index_file.write_text(json.dumps(INDEX), encoding="utf-8")
    package_file.write_text(json.dumps(CHANGE_PACKAGE), encoding="utf-8")
    source_file.write_text(SUMMARY, encoding="utf-8")
    before = {p: p.read_bytes() for p in (index_file, package_file, source_file)}

    code = main(
        [
            "--change-package", str(package_file),
            "--index", str(index_file),
            "--section-id", ASSESSMENT.section_id,
            "--source", str(source_file),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["result"]["status"] == STATUS_NOT_INVOKED
    assert "no provider was contacted" in captured.err
    for path, content in before.items():
        assert path.read_bytes() == content


# ---------------------------------------------------------------------------
# Regression: the real qwen2.5-coder:7b output that wrongly passed
# ---------------------------------------------------------------------------
REAL_7B_NEW_TEXT = (
    "\n\nThe core modules of the project include essential components for "
    "handling technical documentation and change management. These modules "
    "are designed to streamline processes and ensure consistency across all "
    "documents."
)
REAL_7B_REASONING = (
    "The new text provides a brief overview of the core modules, which is "
    "relevant and informative for readers."
)

# The real v1 package: no hunks, deleted legacy modules.
V1_DELETION_PACKAGE = {
    "generated_summary": "10 file(s) changed (2 modified, 8 deleted)",
    "changed_files": [
        {"path": ".gitignore", "change_type": "modified", "old_path": None},
        {"path": "README.md", "change_type": "modified", "old_path": None},
        {"path": "src/change_router.py", "change_type": "deleted", "old_path": None},
        {"path": "src/skeleton_store.py", "change_type": "deleted", "old_path": None},
    ],
}


def v1_assessment_and_index():
    """Placement with zero candidates, mirroring the real bare-inventory case."""
    from src.placement_candidate_scorer import (
        APPEND_TO_SECTION as PLACEMENT_APPEND,
        PlacementAssessment,
    )

    index = build_summary_index(SUMMARY)
    section_id = next(
        s["section_id"] for s in index["sections"] if "repository" in s["section_id"]
    )
    assessment = PlacementAssessment(
        section_id=section_id,
        candidates=[],
        recommendation=PLACEMENT_APPEND,
        reasoning="no patchable prose",
    )
    return assessment, index


def plan_v1(new_text, reasoning="Legacy change_router.py was removed.", confidence=0.9):
    assessment, index = v1_assessment_and_index()
    # Minimal model-facing schema: append carries new_text only; Python owns
    # section_id, the empty old_text, and the source hash.
    payload = {
        "schema_version": PATCH_SCHEMA_VERSION,
        "operation": APPEND_TO_SECTION,
        "target_id": None,
        "new_text": new_text,
        "confidence": confidence,
        "reasoning": reasoning,
    }
    provider = FakeProvider(json.dumps(payload))
    return plan_summary_patch(
        V1_DELETION_PACKAGE, assessment, index, provider=provider
    )


def test_real_7b_generic_append_is_rejected():
    result = plan_v1(REAL_7B_NEW_TEXT, reasoning=REAL_7B_REASONING)
    assert result.status == STATUS_MANUAL_REVIEW
    assert result.instruction.operation == MANUAL_REVIEW_NEEDED
    assert not result.is_mutation


@pytest.mark.parametrize(
    "text",
    [
        "The change was applied to the project documentation.",
        "This module update improves the documentation for the project.",
        "The system documentation now covers the core component changes.",
    ],
)
def test_broad_vocabulary_alone_cannot_ground_text(text):
    result = plan_v1(text)
    assert result.status == STATUS_MANUAL_REVIEW
    assert "concrete evidence" in result.reason


def test_specific_changed_filename_grounds_a_factual_append():
    result = plan_v1(
        "The legacy modules change_router.py and skeleton_store.py were removed."
    )
    assert result.status == STATUS_PLANNED
    assert result.instruction.operation == APPEND_TO_SECTION


def test_specific_changed_symbol_grounds_a_mutation():
    result, _ = plan(
        response(
            new_text=(
                "Routing is performed by route_change, which scores the actual "
                "sections of the summary."
            ),
            reasoning="route_change now scores actual sections.",
        )
    )
    assert result.status == STATUS_PLANNED


def test_zero_candidate_append_requires_concrete_evidence():
    # Same operation, only the concreteness differs.
    assert plan_v1("Documentation for the project was updated.").status == (
        STATUS_MANUAL_REVIEW
    )
    assert plan_v1("The skeleton_store.py module was removed.").status == (
        STATUS_PLANNED
    )


# ---------------------------------------------------------------------------
# status consistency
# ---------------------------------------------------------------------------
def test_deleted_file_described_as_active_is_rejected():
    result = plan_v1(
        "The project includes change_router.py for routing documentation updates."
    )
    assert result.status == STATUS_MANUAL_REVIEW
    assert "deleted file" in result.reason


def test_deleted_file_described_as_removed_is_accepted():
    result = plan_v1("The legacy change_router.py module was removed.")
    assert result.status == STATUS_PLANNED


def test_added_file_described_as_removed_is_rejected():
    package_added = {
        "generated_summary": "1 file added",
        "changed_files": [
            {"path": "src/new_engine.py", "change_type": "added", "old_path": None}
        ],
    }
    assessment, index = v1_assessment_and_index()
    payload = {
        "schema_version": PATCH_SCHEMA_VERSION,
        "operation": APPEND_TO_SECTION,
        "target_id": None,
        "new_text": "The new_engine.py module was removed from the pipeline.",
        "confidence": 0.9,
        "reasoning": "new_engine.py changed.",
    }
    result = plan_summary_patch(
        package_added, assessment, index, provider=FakeProvider(json.dumps(payload))
    )
    assert result.status == STATUS_MANUAL_REVIEW
    assert "added file as removed" in result.reason


def test_renamed_old_path_presented_as_current_is_rejected():
    package_renamed = {
        "generated_summary": "1 file renamed",
        "changed_files": [
            {
                "path": "src/new_router.py",
                "old_path": "src/old_router.py",
                "status": "renamed",
                "what_changed": [],
            }
        ],
    }
    assessment, index = v1_assessment_and_index()

    def build(text):
        return json.dumps(
            {
                "schema_version": PATCH_SCHEMA_VERSION,
                "operation": APPEND_TO_SECTION,
                "target_id": None,
                "new_text": text,
                "confidence": 0.9,
                "reasoning": "old_router.py path changed.",
            }
        )

    bad = plan_summary_patch(
        package_renamed, assessment, index,
        provider=FakeProvider(build("Routing is handled by old_router.py today.")),
    )
    assert bad.status == STATUS_MANUAL_REVIEW
    assert "renamed old path" in bad.reason

    good = plan_summary_patch(
        package_renamed, assessment, index,
        provider=FakeProvider(build("old_router.py was renamed to new_router.py.")),
    )
    assert good.status == STATUS_PLANNED


# ---------------------------------------------------------------------------
# generic phrases and reasoning
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "phrase",
    [
        "streamline processes", "ensure consistency", "improve efficiency",
        "enhance functionality", "essential components", "seamless integration",
        "designed to ensure", "helps improve", "provides a robust",
    ],
)
def test_generic_benefit_phrases_are_rejected(phrase):
    result = plan_v1(f"The change_router.py module was removed to {phrase} here.")
    assert result.status == STATUS_MANUAL_REVIEW
    assert "generic phrase" in result.reason


def test_weak_reasoning_cannot_authorize_a_mutation():
    result = plan_v1(
        "The legacy change_router.py module was removed.",
        reasoning="This is relevant and informative for readers.",
    )
    assert result.status == STATUS_MANUAL_REVIEW
    assert "reasoning cites no concrete" in result.reason


def test_concrete_reasoning_is_accepted():
    result = plan_v1(
        "The legacy change_router.py module was removed.",
        reasoning="change_router.py was deleted in this push.",
    )
    assert result.status == STATUS_PLANNED


# ---------------------------------------------------------------------------
# whitespace normalization
# ---------------------------------------------------------------------------
def test_surrounding_whitespace_is_normalized():
    result = plan_v1("\n\n  The legacy change_router.py module was removed.  \n\n")
    assert result.status == STATUS_PLANNED
    new_text = result.instruction.new_text
    assert new_text == "The legacy change_router.py module was removed."
    assert not new_text.startswith(("\n", " "))
    assert not new_text.endswith(("\n", " "))


def test_normalized_text_has_no_leading_or_trailing_blank_lines():
    result, _ = plan(
        response(
            new_text=(
                "\n\nRouting is performed by route_change across actual "
                "sections.\n\n"
            ),
            reasoning="route_change now scores actual sections.",
        )
    )
    assert result.status == STATUS_PLANNED
    assert result.instruction.new_text.splitlines()[0].strip() != ""
    assert result.instruction.new_text == result.instruction.new_text.strip()
