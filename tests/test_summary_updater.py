"""Tests for the summary updater (no real git, no tokens, tmp dirs only)."""

from __future__ import annotations

import json

from src import summary_updater
from src.git_change_detector import ChangedFile, GitChangeSet
from src.project_summary_generator import (
    generate_original_summary,
    original_summary_path,
    updated_summary_path,
)
from src.summary_change_router import CREATE_NEW, UPDATE_EXISTING
from src.summary_skeleton_builder import (
    build_and_save_summary_skeleton,
    summary_skeleton_path,
)
from src.summary_updater import run_update

CI_ENV = {
    "GITHUB_REPOSITORY": "AdrenalIsland1/TechDocker",
    "GITHUB_REF_NAME": "main",
    "GITHUB_SHA": "def456",
    "GITHUB_ACTOR": "Vaibhav",
    "GITHUB_EVENT_BEFORE": "abc123",
}


def make_repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "README.md").write_text("# Demo\n\nDemo project.\n")
    (tmp_path / "src" / "core.py").write_text("def run():\n    return 1\n")
    (tmp_path / "tests" / "test_core.py").write_text("def test_run():\n    pass\n")
    return tmp_path


def mock_detector(monkeypatch, files):
    def fake_build_change_set(**kwargs):
        return GitChangeSet(
            repository=kwargs["repository"],
            branch=kwargs["branch"],
            before_sha=kwargs["before_sha"],
            after_sha=kwargs["after_sha"],
            changed_files=files,
        )

    monkeypatch.setattr(summary_updater, "build_change_set", fake_build_change_set)


def test_first_run_generates_all_artifacts(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    mock_detector(
        monkeypatch, [ChangedFile(path="src/core.py", change_type="modified")]
    )

    result = run_update(CI_ENV, repo_path=str(repo))

    assert result.original_generated is True
    assert result.original_summary.exists()
    assert result.updated_summary.exists()
    assert result.skeleton_path.exists()
    assert result.change_package_path.exists()
    assert result.skeleton_created is True


def test_original_summary_is_never_modified(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    generate_original_summary(repo)
    original = original_summary_path(repo)
    baseline_bytes = original.read_bytes()

    mock_detector(
        monkeypatch, [ChangedFile(path="src/core.py", change_type="modified")]
    )
    run_update(CI_ENV, repo_path=str(repo))
    run_update(CI_ENV, repo_path=str(repo))

    assert original.read_bytes() == baseline_bytes
    # ...while the updated summary did change.
    assert updated_summary_path(repo).read_bytes() != baseline_bytes


def test_update_lands_under_the_routed_section(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    mock_detector(
        monkeypatch,
        [ChangedFile(path="tests/test_core.py", change_type="modified")],
    )

    result = run_update(CI_ENV, repo_path=str(repo))

    assert result.decision.decision == UPDATE_EXISTING
    assert result.decision.target_heading == "Testing Strategy"

    text = updated_summary_path(repo).read_text(encoding="utf-8")
    testing_start = text.index("## Testing Strategy")
    next_section = text.index("## Configuration", testing_start)
    block_start = text.index("<!-- TECHDOCKER_UPDATE_START -->")
    assert testing_start < block_start < next_section
    assert "modified: tests/test_core.py" in text


def test_existing_section_update_does_not_touch_skeleton(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    generate_original_summary(repo)
    build_and_save_summary_skeleton(repo)
    skeleton_file = summary_skeleton_path(repo)
    skeleton_bytes = skeleton_file.read_bytes()

    mock_detector(
        monkeypatch,
        [ChangedFile(path="tests/test_core.py", change_type="modified")],
    )
    result = run_update(CI_ENV, repo_path=str(repo))

    assert result.decision.skeleton_should_change is False
    assert result.skeleton_updated is False
    assert skeleton_file.read_bytes() == skeleton_bytes  # byte-identical


def test_new_section_update_extends_skeleton(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    # A summary with no headings at all leaves the skeleton with no eligible
    # sections — the only case where creating a section is defensible.
    original = original_summary_path(repo)
    original.parent.mkdir(parents=True, exist_ok=True)
    content = "Prose without any headings at all.\n"
    original.write_text(content, encoding="utf-8")
    updated_summary_path(repo).write_text(content, encoding="utf-8")
    build_and_save_summary_skeleton(repo)

    mock_detector(
        monkeypatch,
        [ChangedFile(path="tests/test_core.py", change_type="added")],
    )
    result = run_update(CI_ENV, repo_path=str(repo))

    assert result.decision.decision == CREATE_NEW
    assert result.decision.new_heading == "System Overview"
    assert result.decision.skeleton_should_change is True
    assert result.skeleton_updated is True

    text = updated_summary_path(repo).read_text(encoding="utf-8")
    assert "## System Overview" in text
    # Skeleton was extended with the new section...
    data = json.loads(summary_skeleton_path(repo).read_text(encoding="utf-8"))
    assert "System Overview" in [s["heading"] for s in data["sections"]]
    # ...but remains based on the ORIGINAL baseline summary.
    assert "base_original_summary.md" in data["source_summary_path"]
    # The transient update-block heading did not become a section.
    assert not any(
        s["heading"].startswith("Automated Change Update") for s in data["sections"]
    )
    # The original summary stayed untouched.
    assert original.read_text(encoding="utf-8") == content


def test_change_package_is_written_with_push_metadata(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    mock_detector(
        monkeypatch,
        [ChangedFile(path="src/core.py", change_type="modified")],
    )

    result = run_update(CI_ENV, repo_path=str(repo))

    package = json.loads(result.change_package_path.read_text(encoding="utf-8"))
    assert package["repository"] == "TechDocker"
    assert package["branch"] == "main"
    assert package["actor"] == "Vaibhav"
    assert package["before_sha"] == "abc123"
    assert package["after_sha"] == "def456"
    assert package["changed_files"][0]["path"] == "src/core.py"
    assert "generated_summary" in package


def test_valid_llm_resolution_clears_the_ambiguity_warning(tmp_path, monkeypatch):
    """A validated above-threshold LLM pick resolves a deterministic tie.

    The ambiguity warning must reflect the *final* route: once the LLM has
    resolved an ambiguous deterministic decision, the updater must not keep
    warning that routing was ambiguous.
    """
    from src.llm_change_analyzer import SELECT_EXISTING, LLMSectionSelection
    from src.summary_change_router import decide_from_assessment as real_decide

    repo = make_repo(tmp_path)
    mock_detector(
        monkeypatch,
        [ChangedFile(path="tests/test_core.py", change_type="modified")],
    )

    # Force the deterministic decision to look like a genuine tie.
    def ambiguous_decide(assessment, catalog):
        decision = real_decide(assessment, catalog)
        decision.ambiguous = True
        return decision

    monkeypatch.setattr(summary_updater, "decide_from_assessment", ambiguous_decide)

    # The LLM validly selects a real shortlisted section, above threshold.
    def fake_select(summary_text, candidates, **kwargs):
        top = candidates[0]
        return LLMSectionSelection(
            SELECT_EXISTING, top.section_id, top.heading, 0.95, "Belongs here."
        )

    monkeypatch.setattr(summary_updater, "select_section_with_llm", fake_select)
    monkeypatch.setattr(
        summary_updater, "get_llm_provider_from_env", lambda env=None: object()
    )

    result = run_update(
        dict(CI_ENV, TECHDOCKER_LLM_PROVIDER="ollama"), repo_path=str(repo)
    )

    assert result.routing_source == "llm"
    assert result.decision.ambiguous is False  # the tie is resolved
    assert not any("ambiguous" in warning.lower() for warning in result.warnings)


def test_unresolved_tie_still_warns_about_ambiguity(tmp_path, monkeypatch):
    """Without a valid LLM resolution, an ambiguous tie must still warn."""
    from src.summary_change_router import decide_from_assessment as real_decide

    repo = make_repo(tmp_path)
    mock_detector(
        monkeypatch,
        [ChangedFile(path="tests/test_core.py", change_type="modified")],
    )

    def ambiguous_decide(assessment, catalog):
        decision = real_decide(assessment, catalog)
        decision.ambiguous = True
        return decision

    monkeypatch.setattr(summary_updater, "decide_from_assessment", ambiguous_decide)

    # No LLM configured: the deterministic (ambiguous) decision stands.
    result = run_update(CI_ENV, repo_path=str(repo))

    assert result.routing_source == "rule_based"
    assert result.decision.ambiguous is True
    assert any("ambiguous" in warning.lower() for warning in result.warnings)


def test_missing_before_sha_still_updates_safely(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)

    def must_not_be_called(**kwargs):
        raise AssertionError("git diff must not run without a before SHA")

    monkeypatch.setattr(summary_updater, "build_change_set", must_not_be_called)

    env = dict(CI_ENV, GITHUB_EVENT_BEFORE="0" * 40)
    result = run_update(env, repo_path=str(repo))

    assert result.changed_files == []
    assert any("missing or all zeroes" in w for w in result.warnings)
    text = updated_summary_path(repo).read_text(encoding="utf-8")
    assert "(no changed files were available for this run)" in text
