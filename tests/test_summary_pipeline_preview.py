"""Tests for the read-only end-to-end pipeline preview.

Every LLM stage uses an injected fake provider — no localhost, no network, no
real Ollama. Nothing may be written, and no real artifact is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.llm_provider import LLMResponse
from src.summary_patch_planner import PATCH_SCHEMA_VERSION, STATUS_NOT_INVOKED
from src.summary_pipeline_preview import (
    EXIT_INVALID_INPUT,
    EXIT_LLM_UNAVAILABLE,
    EXIT_OK,
    EXIT_STALE_DATA,
    PREVIEW_SCHEMA_VERSION,
    SAFE_ROUTING_CONFIDENCE,
    STAGE_ALL,
    STAGE_NONE,
    STAGE_PATCH,
    STAGE_ROUTING,
    STRENGTH_LLM_RESOLVED,
    STRENGTH_MANUAL_OVERRIDE,
    PreviewError,
    main,
    run_preview,
)
from src.summary_skeleton_builder import build_and_save_summary_skeleton

SUMMARY = """\
# Project Technical Summary

## Repository Automation

The pipeline opens a pull request for review. Routing is performed by \
route_change, which relies on a fixed list of heading names. Reports follow.

- Section routing lives in src/summary_change_router.py.

## Quality and Tests

Tests run offline with deterministic fixtures.

<!-- TECHDOCKER_UPDATE_START -->
### Automated Change Update - 2026-01-01
Historic note about route_change in src/summary_change_router.py.
<!-- TECHDOCKER_UPDATE_END -->
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


@pytest.fixture
def repo(tmp_path):
    """A throwaway repo with real artifacts (never the project's own)."""
    summaries = tmp_path / "artifacts" / "summaries"
    packages = tmp_path / "artifacts" / "change_packages"
    summaries.mkdir(parents=True)
    packages.mkdir(parents=True)
    (summaries / "base_updated_summary.md").write_text(SUMMARY, encoding="utf-8")
    (summaries / "base_original_summary.md").write_text(SUMMARY, encoding="utf-8")
    (packages / "latest_change_summary.json").write_text(
        json.dumps(CHANGE_PACKAGE), encoding="utf-8"
    )
    build_and_save_summary_skeleton(tmp_path)
    return tmp_path


def input_files(repo: Path) -> list[Path]:
    return [
        repo / "artifacts" / "summaries" / "base_updated_summary.md",
        repo / "artifacts" / "skeletons" / "base_skeleton.json",
        repo / "artifacts" / "change_packages" / "latest_change_summary.json",
    ]


def snapshot(repo: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in input_files(repo)}


class FakeProvider:
    """Records prompts; returns a canned response per call."""

    name = "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt, system_prompt=None, json_schema=None):
        self.prompts.append(prompt)
        text = self.responses.pop(0) if self.responses else "{}"
        return LLMResponse(text=text, provider_name=self.name)


def factory(provider):
    return lambda env: provider


def routing_response(section_id: str) -> str:
    return json.dumps(
        {
            "decision": "select_existing_section",
            "section_id": section_id,
            "confidence": 0.88,
            "reasoning": "The changed router belongs to this section.",
        }
    )


def patch_response(report: dict, index_sha: str) -> str:
    candidate = report["placement"]["candidates"][0]
    return json.dumps(
        {
            "schema_version": PATCH_SCHEMA_VERSION,
            "operation": (
                "replace_sentence" if candidate["candidate_type"] == "sentence"
                else "replace_block"
            ),
            "section_id": report["routing"]["selected_section_id"],
            "target_id": candidate["candidate_id"],
            "target_type": candidate["candidate_type"],
            "old_text": candidate["text"],
            "new_text": (
                "Routing is performed by route_change, which scores the actual "
                "sections of the summary instead of a fixed heading list."
            ),
            "expected_source_sha256": index_sha,
            "confidence": 0.91,
            "reasoning": "route_change now scores actual sections.",
        }
    )


def index_sha_for(repo: Path) -> str:
    import hashlib

    text = (repo / "artifacts" / "summaries" / "base_updated_summary.md").read_text(
        encoding="utf-8"
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def test_deterministic_none_mode(repo):
    report = run_preview(repo, llm_stage=STAGE_NONE)

    assert report["preview_schema_version"] == PREVIEW_SCHEMA_VERSION
    assert report["routing"]["source"] == "deterministic"
    assert report["routing"]["selected_section_id"]
    assert report["placement"]["recommendation"]
    assert report["patch_planning"] == {"status": STATUS_NOT_INVOKED}
    assert report["source_safety"]["writes_performed"] is False


def test_none_mode_never_instantiates_a_provider(repo):
    def exploding_factory(env):
        raise AssertionError("no provider may be created in 'none' mode")

    report = run_preview(repo, llm_stage=STAGE_NONE, provider_factory=exploding_factory)
    assert report["routing"]["source"] == "deterministic"


def test_routing_stage_uses_llm_selection(repo):
    baseline = run_preview(repo, llm_stage=STAGE_NONE)
    section_id = baseline["routing"]["candidates"][0]["section_id"]
    provider = FakeProvider([routing_response(section_id)])

    report = run_preview(
        repo, llm_stage=STAGE_ROUTING, provider_factory=factory(provider),
        env={"TECHDOCKER_LLM_PROVIDER": "ollama"},
    )
    assert report["routing"]["source"] == "llm"
    assert report["routing"]["selected_section_id"] == section_id
    assert report["patch_planning"]["status"] == STATUS_NOT_INVOKED  # patch not called
    assert len(provider.prompts) == 1


def test_routing_prompt_contains_only_shortlisted_section_ids(repo):
    baseline = run_preview(repo, llm_stage=STAGE_NONE)
    shortlist = {c["section_id"] for c in baseline["routing"]["candidates"]}
    provider = FakeProvider([routing_response(sorted(shortlist)[0])])
    run_preview(
        repo, llm_stage=STAGE_ROUTING, provider_factory=factory(provider),
        env={"TECHDOCKER_LLM_PROVIDER": "ollama"},
    )
    prompt = provider.prompts[0]
    for section_id in shortlist:
        assert section_id in prompt
    assert prompt.count("section_id:") <= 3


def test_patch_stage_uses_deterministic_routing_then_plans(repo):
    baseline = run_preview(repo, llm_stage=STAGE_NONE)
    provider = FakeProvider([patch_response(baseline, index_sha_for(repo))])

    report = run_preview(
        repo, llm_stage=STAGE_PATCH, provider_factory=factory(provider),
        env={"TECHDOCKER_LLM_PROVIDER": "ollama"},
    )
    assert report["routing"]["source"] == "deterministic"  # routing LLM not used
    assert report["patch_planning"]["status"] == "planned"
    assert report["patch_planning"]["plan"]["operation"].startswith("replace_")
    assert len(provider.prompts) == 1


def test_all_stage_uses_both_models(repo):
    baseline = run_preview(repo, llm_stage=STAGE_NONE)
    section_id = baseline["routing"]["candidates"][0]["section_id"]
    provider = FakeProvider(
        [routing_response(section_id), patch_response(baseline, index_sha_for(repo))]
    )
    report = run_preview(
        repo, llm_stage=STAGE_ALL, provider_factory=factory(provider),
        env={"TECHDOCKER_LLM_PROVIDER": "ollama"},
    )
    assert report["routing"]["source"] == "llm"
    assert report["patch_planning"]["status"] == "planned"
    assert len(provider.prompts) == 2


def test_patch_prompt_contains_only_shortlisted_placement_ids(repo):
    baseline = run_preview(repo, llm_stage=STAGE_NONE)
    placement_ids = {c["candidate_id"] for c in baseline["placement"]["candidates"]}
    provider = FakeProvider([patch_response(baseline, index_sha_for(repo))])
    run_preview(
        repo, llm_stage=STAGE_PATCH, provider_factory=factory(provider),
        env={"TECHDOCKER_LLM_PROVIDER": "ollama"},
    )
    prompt = provider.prompts[0]
    for candidate_id in placement_ids:
        assert candidate_id in prompt
    assert prompt.count("\n- id: ") <= 3


# ---------------------------------------------------------------------------
# overrides and failures
# ---------------------------------------------------------------------------
def test_section_override_bypasses_routing_and_is_reported(repo):
    baseline = run_preview(repo, llm_stage=STAGE_NONE)
    alternatives = [
        c["section_id"] for c in baseline["routing"]["candidates"]
        if c["section_id"] != baseline["routing"]["selected_section_id"]
    ]
    override = alternatives[0]

    report = run_preview(repo, section_override=override, llm_stage=STAGE_NONE)
    assert report["routing"]["source"] == "manual_override"
    assert report["routing"]["selected_section_id"] == override
    assert report["placement"]["section_id"] == override
    assert any("overridden manually" in w for w in report["warnings"])


def test_invalid_section_override_is_rejected(repo):
    with pytest.raises(PreviewError) as error:
        run_preview(repo, section_override="no-such-section", llm_stage=STAGE_NONE)
    assert error.value.exit_code == EXIT_INVALID_INPUT


def test_llm_stage_without_configured_provider_is_reported(repo):
    with pytest.raises(PreviewError) as error:
        run_preview(repo, llm_stage=STAGE_ALL, env={})  # no TECHDOCKER_LLM_PROVIDER
    assert error.value.exit_code == EXIT_LLM_UNAVAILABLE
    # And nothing was written.
    assert run_preview(repo, llm_stage=STAGE_NONE)["source_safety"]["writes_performed"] is False


def test_malformed_llm_response_falls_back_safely(repo):
    baseline = run_preview(repo, llm_stage=STAGE_NONE)
    provider = FakeProvider(["not json at all", "not json either"])
    report = run_preview(
        repo, llm_stage=STAGE_ALL, provider_factory=factory(provider),
        env={"TECHDOCKER_LLM_PROVIDER": "ollama"},
    )
    assert report["routing"]["source"] == "deterministic"
    assert report["patch_planning"]["status"] == "manual_review"
    assert any("unavailable or invalid" in w for w in report["warnings"])


def test_low_confidence_patch_is_downgraded(repo):
    baseline = run_preview(repo, llm_stage=STAGE_NONE)
    low = json.loads(patch_response(baseline, index_sha_for(repo)))
    low["confidence"] = 0.30
    provider = FakeProvider([json.dumps(low)])
    report = run_preview(
        repo, llm_stage=STAGE_PATCH, provider_factory=factory(provider),
        env={"TECHDOCKER_LLM_PROVIDER": "ollama"},
    )
    assert report["patch_planning"]["status"] == "manual_review"
    assert report["patch_planning"]["plan"]["operation"] == "manual_review_needed"


@pytest.mark.parametrize(
    "missing", ["base_updated_summary.md", "base_skeleton.json", "latest_change_summary.json"]
)
def test_missing_inputs_are_reported(repo, missing):
    for path in input_files(repo):
        if path.name == missing:
            path.unlink()
    with pytest.raises(PreviewError) as error:
        run_preview(repo, llm_stage=STAGE_NONE)
    assert error.value.exit_code == EXIT_INVALID_INPUT


def test_malformed_change_package_is_reported(repo):
    package = repo / "artifacts" / "change_packages" / "latest_change_summary.json"
    package.write_text("{not json", encoding="utf-8")
    with pytest.raises(PreviewError) as error:
        run_preview(repo, llm_stage=STAGE_NONE)
    assert error.value.exit_code == EXIT_INVALID_INPUT


def test_unsupported_change_package_schema_is_reported(repo):
    package = repo / "artifacts" / "change_packages" / "latest_change_summary.json"
    payload = dict(CHANGE_PACKAGE, schema_version=99)
    package.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PreviewError) as error:
        run_preview(repo, llm_stage=STAGE_NONE)
    assert error.value.exit_code == EXIT_STALE_DATA


def test_stale_skeleton_section_is_reported_clearly(repo):
    # A skeleton section that no longer exists in the summary.
    skeleton_file = repo / "artifacts" / "skeletons" / "base_skeleton.json"
    data = json.loads(skeleton_file.read_text(encoding="utf-8"))
    data["sections"] = [
        {
            "section_id": "ghost-section", "heading": "Ghost", "level": 2,
            "parent_id": None, "path": "Ghost", "order": 1, "content_hash": None,
        }
    ]
    skeleton_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(PreviewError) as error:
        run_preview(repo, llm_stage=STAGE_NONE)
    assert error.value.exit_code == EXIT_STALE_DATA


# ---------------------------------------------------------------------------
# safety
# ---------------------------------------------------------------------------
def test_no_index_artifact_is_written(repo):
    run_preview(repo, llm_stage=STAGE_NONE)
    assert not (repo / "artifacts" / "skeletons" / "base_summary_index.json").exists()


def test_no_input_file_is_modified_and_hashes_match(repo):
    before = snapshot(repo)
    report = run_preview(repo, llm_stage=STAGE_NONE)

    for path, content in before.items():
        assert path.read_bytes() == content
    safety = report["source_safety"]
    assert safety["summary_sha256_before"] == safety["summary_sha256_after"]
    assert safety["skeleton_sha256_before"] == safety["skeleton_sha256_after"]
    assert (
        safety["change_package_sha256_before"]
        == safety["change_package_sha256_after"]
    )
    assert safety["writes_performed"] is False
    assert safety["index_written"] is False


def test_generated_blocks_are_absent_from_placement_candidates(repo):
    report = run_preview(repo, llm_stage=STAGE_NONE)
    serialized = json.dumps(report)
    assert "TECHDOCKER_UPDATE_START" not in serialized
    assert "Historic note" not in serialized


def test_output_is_bounded_and_excludes_the_whole_summary(repo):
    report = run_preview(repo, llm_stage=STAGE_NONE)
    serialized = json.dumps(report)
    assert len(report["routing"]["candidates"]) <= 3
    assert len(report["placement"]["candidates"]) <= 3
    # The full summary text never appears verbatim.
    assert SUMMARY not in serialized
    for candidate in report["placement"]["candidates"]:
        assert len(candidate["text"]) <= 700


def test_prompt_metadata_is_optional_and_has_no_prompt_text(repo):
    without = run_preview(repo, llm_stage=STAGE_NONE)
    assert "prompt_metadata" not in without

    with_metadata = run_preview(
        repo, llm_stage=STAGE_NONE, include_prompt_metadata=True
    )
    metadata = with_metadata["prompt_metadata"]
    assert metadata["patch_prompt_chars"] > 0
    assert metadata["candidates_included"] <= 3
    assert "files_omitted" in metadata
    assert not any("prompt" == key for key in metadata if key == "prompt")


def test_preview_never_calls_the_updater(monkeypatch, repo):
    from src import summary_updater

    def must_not_run(*args, **kwargs):
        raise AssertionError("run_update must never be called by the preview")

    monkeypatch.setattr(summary_updater, "run_update", must_not_run)
    report = run_preview(repo, llm_stage=STAGE_NONE)
    assert report["source_safety"]["writes_performed"] is False


def test_module_creates_no_provider_at_import_time():
    source = Path("src/summary_pipeline_preview.py").read_text(encoding="utf-8")
    # The provider import is deferred inside the factory, not at module scope.
    header = source.split("def default_provider_factory")[0]
    assert "get_llm_provider_from_env" not in header


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_prints_valid_json_to_stdout_and_diagnostics_to_stderr(repo, capsys):
    code = main(["--repo-path", str(repo), "--llm-stage", STAGE_NONE], env={})
    captured = capsys.readouterr()

    assert code == EXIT_OK
    payload = json.loads(captured.out)  # stdout is pure JSON
    assert payload["preview_schema_version"] == PREVIEW_SCHEMA_VERSION
    assert "read-only" in captured.err
    assert captured.err.strip() != ""


def test_cli_exit_codes(repo, capsys):
    assert main(["--repo-path", str(repo)], env={}) == EXIT_OK

    assert (
        main(["--repo-path", str(repo), "--section-id", "nope"], env={})
        == EXIT_INVALID_INPUT
    )
    assert (
        main(["--repo-path", str(repo), "--llm-stage", STAGE_ALL], env={})
        == EXIT_LLM_UNAVAILABLE
    )
    assert main(["--repo-path", str(repo / "missing")], env={}) == EXIT_INVALID_INPUT


def test_cli_writes_nothing(repo, capsys):
    before = snapshot(repo)
    main(["--repo-path", str(repo), "--include-prompt-metadata"], env={})
    capsys.readouterr()
    for path, content in before.items():
        assert path.read_bytes() == content
    assert not (repo / "artifacts" / "skeletons" / "base_summary_index.json").exists()


# ---------------------------------------------------------------------------
# routing confidence provenance (correction 6)
# ---------------------------------------------------------------------------
TIED_SUMMARY = """\
# Project Technical Summary

## Core Modules

Modules implemented in src/summary_change_router.py for the pipeline.

## Automation Pipeline

Automation implemented in src/summary_change_router.py for the pipeline.
"""


@pytest.fixture
def tied_repo(tmp_path):
    """A repo whose deterministic routing genuinely ties."""
    summaries = tmp_path / "artifacts" / "summaries"
    packages = tmp_path / "artifacts" / "change_packages"
    summaries.mkdir(parents=True)
    packages.mkdir(parents=True)
    (summaries / "base_updated_summary.md").write_text(TIED_SUMMARY, encoding="utf-8")
    (summaries / "base_original_summary.md").write_text(TIED_SUMMARY, encoding="utf-8")
    (packages / "latest_change_summary.json").write_text(
        json.dumps(CHANGE_PACKAGE), encoding="utf-8"
    )
    build_and_save_summary_skeleton(tmp_path)
    return tmp_path


def test_deterministic_routing_reports_its_own_confidence(repo):
    report = run_preview(repo, llm_stage=STAGE_NONE)
    routing = report["routing"]
    assert routing["source"] == "deterministic"
    assert routing["resolved_by_llm"] is False
    assert routing["confidence"] == routing["deterministic_confidence"]
    assert routing["llm_confidence"] is None


def test_llm_selection_reports_llm_confidence_and_keeps_determinism_visible(tied_repo):
    baseline = run_preview(tied_repo, llm_stage=STAGE_NONE)
    assert baseline["routing"]["ambiguous"] is True          # a genuine tie
    deterministic_confidence = baseline["routing"]["deterministic_confidence"]

    section_id = baseline["routing"]["candidates"][0]["section_id"]
    provider = FakeProvider([routing_response(section_id)])
    report = run_preview(
        tied_repo, llm_stage=STAGE_ROUTING, provider_factory=factory(provider),
        env={"TECHDOCKER_LLM_PROVIDER": "ollama"},
    )
    routing = report["routing"]

    assert routing["source"] == "llm"
    assert routing["resolved_by_llm"] is True
    # Headline confidence is the *validated LLM* confidence...
    assert routing["confidence"] == 0.88
    assert routing["llm_confidence"] == 0.88
    assert routing["llm_reasoning"]
    # ...while the deterministic tie remains visible and is not erased.
    assert routing["deterministic_confidence"] == deterministic_confidence
    assert routing["deterministic_ambiguous"] is True
    assert routing["confidence"] != routing["deterministic_confidence"]


def test_rejected_llm_selection_keeps_deterministic_confidence(tied_repo):
    provider = FakeProvider(["not json at all"])
    report = run_preview(
        tied_repo, llm_stage=STAGE_ROUTING, provider_factory=factory(provider),
        env={"TECHDOCKER_LLM_PROVIDER": "ollama"},
    )
    routing = report["routing"]
    assert routing["source"] == "deterministic"
    assert routing["resolved_by_llm"] is False
    assert routing["confidence"] == routing["deterministic_confidence"]
    assert routing["llm_confidence"] is None
    assert any("unavailable or invalid" in w for w in report["warnings"])


# ---------------------------------------------------------------------------
# patch-planning gate on unresolved routing (correction 7)
# ---------------------------------------------------------------------------
def test_unresolved_ambiguous_routing_skips_patch_planning(tied_repo):
    from src.summary_pipeline_preview import SAFE_ROUTING_CONFIDENCE

    baseline = run_preview(tied_repo, llm_stage=STAGE_NONE)
    assert baseline["routing"]["ambiguous"] is True
    assert baseline["routing"]["deterministic_confidence"] < SAFE_ROUTING_CONFIDENCE

    provider = FakeProvider([patch_response(baseline, index_sha_for(tied_repo))])
    report = run_preview(
        tied_repo, llm_stage=STAGE_PATCH, provider_factory=factory(provider),
        env={"TECHDOCKER_LLM_PROVIDER": "ollama"},
    )
    # The planner was never called and no executable mutation was produced.
    assert provider.prompts == []
    assert report["patch_planning"]["status"] == "manual_review"
    assert "not called" in report["patch_planning"]["reason"]
    assert any("ambiguous and unresolved" in w for w in report["warnings"])


def test_llm_resolved_routing_may_proceed_to_planning(tied_repo):
    baseline = run_preview(tied_repo, llm_stage=STAGE_NONE)
    section_id = baseline["routing"]["candidates"][0]["section_id"]
    provider = FakeProvider(
        [routing_response(section_id), patch_response(baseline, index_sha_for(tied_repo))]
    )
    report = run_preview(
        tied_repo, llm_stage=STAGE_ALL, provider_factory=factory(provider),
        env={"TECHDOCKER_LLM_PROVIDER": "ollama"},
    )
    assert report["routing"]["resolved_by_llm"] is True
    assert report["routing"]["deterministic_ambiguous"] is True  # tie still shown
    assert len(provider.prompts) == 2       # planner WAS called
    assert report["patch_planning"]["status"] in {"planned", "manual_review"}


def test_unambiguous_routing_is_not_gated(repo):
    baseline = run_preview(repo, llm_stage=STAGE_NONE)
    provider = FakeProvider([patch_response(baseline, index_sha_for(repo))])
    report = run_preview(
        repo, llm_stage=STAGE_PATCH, provider_factory=factory(provider),
        env={"TECHDOCKER_LLM_PROVIDER": "ollama"},
    )
    if not baseline["routing"]["ambiguous"]:
        assert provider.prompts  # planner reached
        assert report["patch_planning"]["status"] != STATUS_NOT_INVOKED


# ---------------------------------------------------------------------------
# final-route status semantics: ambiguous / strength / deterministic_strength
# ---------------------------------------------------------------------------
def routing_response_conf(section_id: str, confidence: float) -> str:
    return json.dumps(
        {
            "decision": "select_existing_section",
            "section_id": section_id,
            "confidence": confidence,
            "reasoning": "The changed router belongs to this section.",
        }
    )


def test_deterministic_route_exposes_its_own_strength(repo):
    routing = run_preview(repo, llm_stage=STAGE_NONE)["routing"]
    assert routing["source"] == "deterministic"
    # The headline strength/ambiguity ARE the deterministic assessment.
    assert routing["strength"] == routing["deterministic_strength"]
    assert routing["ambiguous"] == routing["deterministic_ambiguous"]
    assert routing["deterministic_strength"] in {
        "strong", "reasonable", "ambiguous", "none",
    }


def test_llm_resolution_marks_final_route_resolved(tied_repo):
    """A valid above-threshold pick makes the final route llm_resolved."""
    baseline = run_preview(tied_repo, llm_stage=STAGE_NONE)["routing"]
    assert baseline["ambiguous"] is True
    assert baseline["strength"] == "ambiguous"

    section_id = baseline["candidates"][0]["section_id"]
    provider = FakeProvider([routing_response_conf(section_id, 0.95)])
    routing = run_preview(
        tied_repo, llm_stage=STAGE_ROUTING, provider_factory=factory(provider),
        env={"TECHDOCKER_LLM_PROVIDER": "ollama"},
    )["routing"]

    # Final route: resolved, no longer ambiguous, strength llm_resolved.
    assert routing["source"] == "llm"
    assert routing["resolved_by_llm"] is True
    assert routing["ambiguous"] is False
    assert routing["strength"] == STRENGTH_LLM_RESOLVED
    assert routing["confidence"] == 0.95
    # Deterministic assessment preserved untouched for diagnostics.
    assert routing["deterministic_ambiguous"] is True
    assert routing["deterministic_strength"] == "ambiguous"
    assert routing["deterministic_confidence"] == baseline["deterministic_confidence"]


def test_below_threshold_llm_selection_does_not_resolve(tied_repo):
    """A valid but below-threshold pick must not resolve the deterministic tie."""
    baseline = run_preview(tied_repo, llm_stage=STAGE_NONE)["routing"]
    section_id = baseline["candidates"][0]["section_id"]
    # 0.50 is a valid confidence but below the 0.75 updater threshold.
    provider = FakeProvider([routing_response_conf(section_id, 0.50)])
    report = run_preview(
        tied_repo, llm_stage=STAGE_ROUTING, provider_factory=factory(provider),
        env={"TECHDOCKER_LLM_PROVIDER": "ollama"},
    )
    routing = report["routing"]

    assert routing["source"] == "deterministic"
    assert routing["resolved_by_llm"] is False
    # Final ambiguity/strength stay exactly deterministic.
    assert routing["ambiguous"] is True
    assert routing["strength"] == "ambiguous"
    assert routing["confidence"] == routing["deterministic_confidence"]
    assert routing["llm_confidence"] is None
    assert any("below the threshold" in w for w in report["warnings"])


def test_manual_override_reports_override_status_and_keeps_determinism(tied_repo):
    baseline = run_preview(tied_repo, llm_stage=STAGE_NONE)["routing"]
    override = [
        c["section_id"] for c in baseline["candidates"]
        if c["section_id"] != baseline["selected_section_id"]
    ][0]

    routing = run_preview(
        tied_repo, section_override=override, llm_stage=STAGE_NONE
    )["routing"]

    assert routing["source"] == "manual_override"
    assert routing["selected_section_id"] == override
    # An explicit human decision: never ambiguous, clearly labelled.
    assert routing["ambiguous"] is False
    assert routing["strength"] == STRENGTH_MANUAL_OVERRIDE
    assert routing["resolved_by_llm"] is False
    # The deterministic tie is preserved separately for diagnostics.
    assert routing["deterministic_ambiguous"] is True
    assert routing["deterministic_strength"] == "ambiguous"


# ---------------------------------------------------------------------------
# Rich, real-style read-only fixtures (variable headings, schema-v2 change)
#
# These are committed test *inputs* under tests/fixtures/. They exercise the
# full read-only pipeline with fake providers: deterministic routing to a
# variable section, placement onto a deliberately stale sentence, and a
# validated (never applied) patch plan. Nothing may be written.
# ---------------------------------------------------------------------------
FIXTURES = Path(__file__).parent / "fixtures" / "summary_pipeline_preview"
RICH_SUMMARY = FIXTURES / "rich_summary.md"
RICH_SKELETON = FIXTURES / "rich_skeleton.json"
RICH_CHANGE_PACKAGE = FIXTURES / "rich_change_package_v2.json"

# The two sentences the change makes stale, normalized (the source wraps them).
STALE_CHANGE_PACKAGE_SENTENCE = (
    "In `change_summary_generator.py`, `create_change_package` records only "
    "changed file paths in `latest_change_summary.json`."
)
STALE_FIXED_HEADING_SENTENCE = (
    "The section router in `section_candidate_scorer.py` uses a fixed list of "
    "expected heading names to choose where an update belongs."
)


def _normalize(text: str) -> str:
    return " ".join((text or "").split())


def rich_expected_section_id() -> str:
    """The variable 'Repository Automation' id, derived from the fixture."""
    data = json.loads(RICH_SKELETON.read_text(encoding="utf-8"))
    matches = [
        s["section_id"] for s in data["sections"]
        if s["heading"] == "Repository Automation"
    ]
    assert matches, "fixture skeleton must contain a 'Repository Automation' section"
    return matches[0]


def rich_index_sha() -> str:
    import hashlib

    return hashlib.sha256(
        RICH_SUMMARY.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()


def run_rich_preview(**kwargs):
    return run_preview(
        ".",
        change_package_path=str(RICH_CHANGE_PACKAGE),
        summary_path=str(RICH_SUMMARY),
        skeleton_path=str(RICH_SKELETON),
        **kwargs,
    )


def rich_patch_response(report: dict, index_sha: str) -> str:
    """A small, well-grounded replacement of the change-package stale sentence."""
    candidate = report["placement"]["candidates"][0]
    operation = (
        "replace_sentence" if candidate["candidate_type"] == "sentence"
        else "replace_block"
    )
    return json.dumps(
        {
            "schema_version": PATCH_SCHEMA_VERSION,
            "operation": operation,
            "section_id": report["routing"]["selected_section_id"],
            "target_id": candidate["candidate_id"],
            "target_type": candidate["candidate_type"],
            "old_text": candidate["text"],
            "new_text": (
                "In `change_summary_generator.py`, `create_change_package` now "
                "records hunk line ranges, additions, deletions, and symbols in "
                "`latest_change_summary.json`, not just changed file paths."
            ),
            "expected_source_sha256": index_sha,
            "confidence": 0.92,
            "reasoning": (
                "create_change_package now records hunk evidence in "
                "latest_change_summary.json, not only file paths."
            ),
        }
    )


def rich_snapshot() -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in (RICH_SUMMARY, RICH_SKELETON, RICH_CHANGE_PACKAGE)}


def test_rich_fixtures_exist_and_are_schema_v2():
    assert RICH_SUMMARY.exists() and RICH_SKELETON.exists()
    package = json.loads(RICH_CHANGE_PACKAGE.read_text(encoding="utf-8"))
    assert package["schema_version"] == 2
    paths = {entry["path"] for entry in package["changed_files"]}
    assert paths == {
        "src/change_summary_generator.py", "src/section_candidate_scorer.py",
    }
    symbols = {
        symbol
        for entry in package["changed_files"]
        for hunk in entry["what_changed"]
        for symbol in hunk["symbols"]
    }
    assert {"create_change_package", "build_section_catalog", "score_section"} <= symbols


def test_rich_deterministic_route_selects_repository_automation():
    routing = run_rich_preview(llm_stage=STAGE_NONE)["routing"]
    assert routing["source"] == "deterministic"
    assert routing["selected_section_id"] == rich_expected_section_id()
    assert routing["selected_heading"] == "Repository Automation"
    # A clear, unambiguous deterministic winner (so planning is never gated).
    assert routing["ambiguous"] is False
    assert routing["deterministic_confidence"] >= SAFE_ROUTING_CONFIDENCE


def test_rich_placement_shortlist_surfaces_both_stale_sentences():
    report = run_rich_preview(llm_stage=STAGE_NONE)
    placement = report["placement"]
    assert placement["section_id"] == rich_expected_section_id()

    texts = [_normalize(c["text"]) for c in placement["candidates"]]
    assert STALE_CHANGE_PACKAGE_SENTENCE in texts
    assert STALE_FIXED_HEADING_SENTENCE in texts
    # The top candidate is one of the deliberately stale sentences (prose, not
    # a bare file inventory entry).
    assert texts[0] in {STALE_CHANGE_PACKAGE_SENTENCE, STALE_FIXED_HEADING_SENTENCE}
    assert placement["candidates"][0]["candidate_type"] in {"sentence", "block"}


def test_rich_full_pipeline_plans_a_valid_patch_without_writing():
    baseline = run_rich_preview(llm_stage=STAGE_NONE)
    expected_section = rich_expected_section_id()
    assert baseline["routing"]["selected_section_id"] == expected_section

    # The fake shortlist LLM may select only the valid Repository Automation
    # candidate; the fake patch LLM returns a small grounded replacement.
    provider = FakeProvider(
        [
            routing_response_conf(expected_section, 0.9),
            rich_patch_response(baseline, rich_index_sha()),
        ]
    )
    before = rich_snapshot()
    report = run_rich_preview(
        llm_stage=STAGE_ALL, provider_factory=factory(provider),
        env={"TECHDOCKER_LLM_PROVIDER": "ollama"},
    )

    routing = report["routing"]
    assert routing["source"] == "llm"
    assert routing["resolved_by_llm"] is True
    assert routing["selected_section_id"] == expected_section
    assert routing["strength"] == STRENGTH_LLM_RESOLVED
    # The routing prompt only offered shortlisted sections, and the pick was one.
    assert expected_section in provider.prompts[0]

    # A concrete, validated plan targeting the stale change-package sentence.
    plan = report["patch_planning"]
    assert plan["status"] == "planned"
    assert plan["plan"]["operation"] in {"replace_sentence", "replace_block"}
    assert plan["plan"]["section_id"] == expected_section
    assert _normalize(plan["plan"]["old_text"]) == STALE_CHANGE_PACKAGE_SENTENCE
    assert "create_change_package" in plan["plan"]["new_text"]
    assert len(provider.prompts) == 2

    # Nothing was written: input hashes match and no index artifact appeared.
    safety = report["source_safety"]
    assert safety["writes_performed"] is False
    assert safety["index_written"] is False
    for path, content in before.items():
        assert path.read_bytes() == content
    assert not any(p.name.endswith("index.json") for p in FIXTURES.iterdir())


def test_rich_fixture_run_leaves_fixtures_byte_identical():
    before = rich_snapshot()
    run_rich_preview(llm_stage=STAGE_NONE)
    for path, content in before.items():
        assert path.read_bytes() == content
