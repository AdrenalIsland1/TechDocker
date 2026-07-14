"""Tests for context collection, summary generation, skeleton, and routing."""

from __future__ import annotations

import json

import pytest

from src.change_summary_generator import create_change_package
from src.git_change_detector import ChangedFile
from src.project_summary_generator import (
    SUMMARY_HEADINGS,
    CopilotSummaryProvider,
    generate_original_summary,
    original_summary_path,
    updated_summary_path,
)
from src.repo_context_collector import collect_repo_context
from src.summary_change_router import CREATE_NEW, UPDATE_EXISTING, route_change
from src.summary_skeleton_builder import (
    build_and_save_summary_skeleton,
    build_summary_skeleton,
)
from src.summary_skeleton_store import (
    SummarySkeleton,
    append_section,
    find_best_section_by_keywords,
    find_section_by_heading,
    find_section_by_id,
    load_summary_skeleton,
    save_summary_skeleton,
    update_section_metadata,
)


def make_repo(tmp_path):
    """A small fake repository with safe and unsafe content."""
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / "artifacts" / "summaries").mkdir(parents=True)

    (tmp_path / "README.md").write_text(
        "# Demo\n\nA demo project for context collection.\n", encoding="utf-8"
    )
    (tmp_path / "src" / "core.py").write_text("def run():\n    return 1\n")
    (tmp_path / "src" / "summary_updater.py").write_text("# updater\n")
    (tmp_path / "tests" / "test_core.py").write_text("def test_run():\n    pass\n")
    (tmp_path / "requirements.txt").write_text("pytest\n")

    # Must all be excluded:
    (tmp_path / ".env").write_text("API_KEY=nope\n")
    (tmp_path / "secret_token.txt").write_text("nope\n")
    (tmp_path / "report.docx").write_bytes(b"PK\x03\x04fake")
    (tmp_path / "data.csv").write_text("a,b\n1,2\n")
    (tmp_path / ".git" / "config").write_text("[core]\n")
    (tmp_path / ".venv" / "lib" / "big.py").write_text("x = 1\n")
    (tmp_path / "image.bin").write_bytes(b"\x00\x01\x02binary")
    (tmp_path / "artifacts" / "summaries" / "old.md").write_text("old\n")
    return tmp_path


# ---------------------------------------------------------------------------
# repo_context_collector
# ---------------------------------------------------------------------------
def test_collector_includes_safe_files_and_excludes_unsafe(tmp_path):
    repo = make_repo(tmp_path)
    context = collect_repo_context(repo)

    assert "src/core.py" in context.files
    assert "tests/test_core.py" in context.files
    assert "README.md" in context.files
    assert "requirements.txt" in context.files

    joined = " ".join(context.file_tree) + " ".join(context.files)
    for forbidden in (".env", "secret_token", ".docx", ".csv", ".git/",
                      ".venv", "artifacts", "image.bin"):
        assert forbidden not in joined

    assert context.project_name == repo.name
    assert context.total_files == len(context.file_tree)


def test_collector_truncates_large_files(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "src" / "huge.py").write_text("# line\n" * 5000)
    context = collect_repo_context(repo)

    assert "src/huge.py" in context.truncated_files
    assert len(context.files["src/huge.py"]) <= 6_000


# ---------------------------------------------------------------------------
# project_summary_generator
# ---------------------------------------------------------------------------
def test_generator_writes_summary_with_stable_headings(tmp_path):
    repo = make_repo(tmp_path)
    path = generate_original_summary(repo)

    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Project Technical Summary")
    for heading in SUMMARY_HEADINGS:
        assert f"## {heading}" in text

    # Updated summary initialized as an identical copy.
    updated = updated_summary_path(repo)
    assert updated.exists()
    assert updated.read_text(encoding="utf-8") == text


def test_generator_does_not_overwrite_without_force(tmp_path):
    repo = make_repo(tmp_path)
    path = generate_original_summary(repo)
    path.write_text("# Custom baseline\n", encoding="utf-8")

    generate_original_summary(repo)  # no force -> untouched
    assert path.read_text(encoding="utf-8") == "# Custom baseline\n"

    generate_original_summary(repo, force=True)
    assert "# Project Technical Summary" in path.read_text(encoding="utf-8")


def test_copilot_provider_is_optional_placeholder(monkeypatch):
    monkeypatch.delenv("TECHDOCKER_COPILOT_TOKEN", raising=False)
    provider = CopilotSummaryProvider()
    assert provider.is_available() is False
    with pytest.raises(NotImplementedError):
        provider.generate_summary(None)


# ---------------------------------------------------------------------------
# summary skeleton builder + store
# ---------------------------------------------------------------------------
def test_skeleton_built_from_generated_summary(tmp_path):
    repo = make_repo(tmp_path)
    generate_original_summary(repo)

    skeleton, path = build_and_save_summary_skeleton(repo)

    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["project_id"] == "techdocker"
    # The skeleton is based on the ORIGINAL baseline summary.
    assert "base_original_summary.md" in data["source_summary_path"]

    headings = [section["heading"] for section in data["sections"]]
    for heading in SUMMARY_HEADINGS:
        assert heading in headings
    overview = find_section_by_heading(skeleton, "System Overview")
    assert overview.parent_id == "project-technical-summary"
    # Headings inside the fenced tree block were not parsed as sections.
    assert all("# " not in section["heading"] for section in data["sections"])


def test_skeleton_builder_skips_update_block_headings(tmp_path):
    summary = tmp_path / "s.md"
    summary.write_text(
        "# T\n\n## Core Modules\n\n"
        "### Automated Change Update - 2026-01-01\n\nblock\n\n"
        "## Testing Strategy\n",
        encoding="utf-8",
    )
    skeleton = build_summary_skeleton(summary)
    headings = [section.heading for section in skeleton.sections]
    assert "Core Modules" in headings and "Testing Strategy" in headings
    assert not any(h.startswith("Automated Change Update") for h in headings)


def test_store_roundtrip_find_append_update(tmp_path):
    skeleton = SummarySkeleton(
        project_id="techdocker", source_summary_path="s.md", generated_at="now"
    )
    append_section(skeleton, "System Overview", level=2)
    append_section(skeleton, "Core Modules", level=2)

    path = tmp_path / "skeleton.json"
    save_summary_skeleton(skeleton, path)
    loaded = load_summary_skeleton(path)

    assert find_section_by_id(loaded, "system-overview") is not None
    assert find_section_by_heading(loaded, "core modules").order == 2
    assert (
        find_best_section_by_keywords(loaded, ["modules"]).heading == "Core Modules"
    )

    updated = update_section_metadata(
        loaded, "core-modules", content_hash="abc123"
    )
    assert updated.content_hash == "abc123"
    with pytest.raises(AttributeError):
        update_section_metadata(loaded, "core-modules", nonsense=1)


# ---------------------------------------------------------------------------
# summary_change_router
# ---------------------------------------------------------------------------
def make_skeleton(headings):
    skeleton = SummarySkeleton(
        project_id="techdocker", source_summary_path="s.md", generated_at="now"
    )
    for heading in headings:
        append_section(skeleton, heading, level=2)
    return skeleton


FULL = make_skeleton(
    ["System Overview", "Repository Structure", "Core Modules",
     "Automation Pipeline", "Testing Strategy", "Configuration",
     "Deployment and CI", "Known Limitations"]
)


@pytest.mark.parametrize(
    "path,expected_heading",
    [
        ("tests/test_router.py", "Testing Strategy"),
        (".github/workflows/ci.yml", "Deployment and CI"),
        ("config/projects.json", "Configuration"),
        ("README.md", "System Overview"),
        ("src/summary_updater.py", "Automation Pipeline"),
        ("src/plainmodule.py", "Core Modules"),
        ("Makefile", "System Overview"),  # fallback
    ],
)
def test_router_rules(path, expected_heading):
    decision = route_change(
        "summary", [ChangedFile(path=path, change_type="modified")], FULL
    )
    assert decision.decision == UPDATE_EXISTING
    assert decision.target_heading == expected_heading
    assert decision.skeleton_should_change is False


def test_router_creates_section_when_none_exists():
    small = make_skeleton(["System Overview"])
    decision = route_change(
        "summary",
        [ChangedFile(path="tests/test_x.py", change_type="added")],
        small,
    )
    assert decision.decision == CREATE_NEW
    assert decision.new_heading == "Testing Strategy"
    assert decision.skeleton_should_change is True


def test_router_empty_skeleton_creates_fallback():
    empty = SummarySkeleton(
        project_id="techdocker", source_summary_path="s.md", generated_at="now"
    )
    decision = route_change("summary", [], empty)
    assert decision.decision == CREATE_NEW
    assert decision.new_heading == "System Overview"
    assert decision.skeleton_should_change is True


# ---------------------------------------------------------------------------
# change_summary_generator
# ---------------------------------------------------------------------------
def test_change_package_written_with_all_fields(tmp_path):
    files = [
        ChangedFile(path="src/example.py", change_type="modified"),
        ChangedFile(path="old.py", change_type="renamed", old_path="older.py"),
    ]
    package, path = create_change_package(
        repository="TechDocker",
        branch="main",
        actor="Vaibhav",
        before_sha="abc123",
        after_sha="def456",
        changed_files=files,
        repo_path=tmp_path,
    )

    assert path.name == "latest_change_summary.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    for key in ("repository", "branch", "actor", "before_sha", "after_sha",
                "changed_files", "generated_summary", "generated_at"):
        assert key in stored
    assert stored["changed_files"][0]["path"] == "src/example.py"
    assert "2 file(s) changed" in stored["generated_summary"]
    assert "modified src/example.py" in stored["generated_summary"]
