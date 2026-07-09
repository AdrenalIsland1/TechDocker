"""Tests for the push-triggered automation demo runner."""

from __future__ import annotations

from src import automation_demo
from src.automation_demo import (
    build_demo_summary_from_env,
    extract_repository_name,
    format_summary,
    is_missing_sha,
)
from src.git_change_detector import ChangedFile, GitChangeSet
from src.project_resolver import ProjectConfig

TECHDOCKER_PROJECT = ProjectConfig(
    repository_name="TechDocker",
    project_id="techdocker",
    production_branch="main",
    document_name="TechDocker Master Technical Document.docx",
    document_location="sharepoint-placeholder",
)

CI_ENV = {
    "GITHUB_REPOSITORY": "AdrenalIsland1/TechDocker",
    "GITHUB_REF_NAME": "main",
    "GITHUB_SHA": "def456",
    "GITHUB_ACTOR": "Vaibhav",
    "GITHUB_EVENT_BEFORE": "abc123",
}


def mock_resolver(monkeypatch, project=TECHDOCKER_PROJECT):
    monkeypatch.setattr(
        automation_demo, "resolve_project", lambda repository_name: project
    )


def mock_detector(monkeypatch, files):
    calls = []

    def fake_build_change_set(**kwargs):
        calls.append(kwargs)
        return GitChangeSet(
            repository=kwargs["repository"],
            branch=kwargs["branch"],
            before_sha=kwargs["before_sha"],
            after_sha=kwargs["after_sha"],
            changed_files=files,
        )

    monkeypatch.setattr(automation_demo, "build_change_set", fake_build_change_set)
    return calls


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def test_extract_repository_name_from_owner_slash_name():
    assert extract_repository_name("AdrenalIsland1/TechDocker") == "TechDocker"
    assert extract_repository_name("TechDocker") == "TechDocker"
    assert extract_repository_name("") == "TechDocker"  # local default


def test_all_zero_before_sha_is_treated_as_missing():
    assert is_missing_sha("0" * 40) is True
    assert is_missing_sha("") is True
    assert is_missing_sha(None) is True
    assert is_missing_sha("abc123") is False


# ---------------------------------------------------------------------------
# summary building
# ---------------------------------------------------------------------------
def test_missing_before_sha_does_not_crash(monkeypatch):
    mock_resolver(monkeypatch)
    calls = mock_detector(monkeypatch, [ChangedFile("x.py", "modified")])

    env = dict(CI_ENV, GITHUB_EVENT_BEFORE="0" * 40)
    summary = build_demo_summary_from_env(env)

    assert summary.changed_files == []
    assert summary.before_sha is None
    assert calls == []  # git diff never attempted
    assert any("missing or all zeroes" in warning for warning in summary.warnings)
    # Project mapping is still resolved and printed.
    assert summary.project is TECHDOCKER_PROJECT


def test_summary_resolves_project_metadata(monkeypatch):
    mock_resolver(monkeypatch)
    mock_detector(monkeypatch, [])

    summary = build_demo_summary_from_env(CI_ENV)

    assert summary.repository == "TechDocker"
    assert summary.branch == "main"
    assert summary.actor == "Vaibhav"
    assert summary.before_sha == "abc123"
    assert summary.after_sha == "def456"
    assert summary.project.project_id == "techdocker"
    assert summary.project.document_name == (
        "TechDocker Master Technical Document.docx"
    )


def test_changed_files_included_when_detector_returns_files(monkeypatch):
    mock_resolver(monkeypatch)
    files = [
        ChangedFile(path="src/example.py", change_type="modified"),
        ChangedFile(path="tests/test_example.py", change_type="added"),
    ]
    calls = mock_detector(monkeypatch, files)

    summary = build_demo_summary_from_env(CI_ENV)

    assert summary.changed_files == files
    assert calls[0]["before_sha"] == "abc123"
    assert calls[0]["after_sha"] == "def456"


def test_unconfigured_repository_warns_but_does_not_crash(monkeypatch):
    def raise_key_error(repository_name):
        raise KeyError(f"Repository {repository_name!r} is not configured")

    monkeypatch.setattr(automation_demo, "resolve_project", raise_key_error)
    mock_detector(monkeypatch, [])

    summary = build_demo_summary_from_env(CI_ENV)

    assert summary.project is None
    assert any("not configured" in warning for warning in summary.warnings)


# ---------------------------------------------------------------------------
# output formatting
# ---------------------------------------------------------------------------
def test_output_contains_project_id_and_document_name(monkeypatch):
    mock_resolver(monkeypatch)
    mock_detector(
        monkeypatch,
        [ChangedFile(path="src/example.py", change_type="modified")],
    )

    output = format_summary(build_demo_summary_from_env(CI_ENV))

    assert "TechDocker Automation Demo" in output
    assert "techdocker" in output
    assert "TechDocker Master Technical Document.docx" in output
    assert "sharepoint-placeholder" in output
    assert "- modified: src/example.py" in output
    assert "Actor:      Vaibhav" in output
