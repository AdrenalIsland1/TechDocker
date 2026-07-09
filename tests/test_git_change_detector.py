"""Tests for git changed-file detection and name-status parsing."""

from __future__ import annotations

from types import SimpleNamespace

from src import git_change_detector
from src.git_change_detector import (
    ChangedFile,
    build_change_set,
    detect_changed_files,
    parse_name_status_output,
)


# ---------------------------------------------------------------------------
# parse_name_status_output
# ---------------------------------------------------------------------------
def test_parse_modified_file():
    (changed,) = parse_name_status_output("M\tsrc/api/health.py")
    assert changed.path == "src/api/health.py"
    assert changed.change_type == "modified"
    assert changed.old_path is None


def test_parse_added_file():
    (changed,) = parse_name_status_output("A\tsrc/new_file.py")
    assert changed.path == "src/new_file.py"
    assert changed.change_type == "added"


def test_parse_deleted_file():
    (changed,) = parse_name_status_output("D\tsrc/old_file.py")
    assert changed.path == "src/old_file.py"
    assert changed.change_type == "deleted"


def test_parse_renamed_file():
    (changed,) = parse_name_status_output("R100\tsrc/old_name.py\tsrc/new_name.py")
    assert changed.path == "src/new_name.py"
    assert changed.change_type == "renamed"
    assert changed.old_path == "src/old_name.py"


def test_parse_copied_file():
    (changed,) = parse_name_status_output("C100\tsrc/source.py\tsrc/copied.py")
    assert changed.path == "src/copied.py"
    assert changed.change_type == "copied"
    assert changed.old_path == "src/source.py"


def test_parse_unknown_status_keeps_path():
    (changed,) = parse_name_status_output("X\tsrc/weird.py")
    assert changed.path == "src/weird.py"
    assert changed.change_type == "unknown"


def test_blank_lines_are_ignored():
    output = "\nM\tsrc/api/health.py\n\n\nA\tsrc/new_file.py\n\n"
    changed = parse_name_status_output(output)
    assert [item.path for item in changed] == [
        "src/api/health.py",
        "src/new_file.py",
    ]


def test_malformed_lines_do_not_crash():
    output = "M\nR100\tsrc/only_old.py\n\t\nM\tsrc/ok.py"
    changed = parse_name_status_output(output)
    # The status-only line and the empty-path line are dropped; the one-path
    # rename keeps its single path; the valid line parses normally.
    assert [(item.path, item.change_type) for item in changed] == [
        ("src/only_old.py", "renamed"),
        ("src/ok.py", "modified"),
    ]


# ---------------------------------------------------------------------------
# build_change_set
# ---------------------------------------------------------------------------
def test_build_change_set_returns_metadata_and_files(monkeypatch):
    fake_files = [ChangedFile(path="src/api/health.py", change_type="modified")]

    def fake_detect(before_sha, after_sha, repo_path):
        assert (before_sha, after_sha, repo_path) == ("abc123", "def456", "/repo")
        return fake_files

    monkeypatch.setattr(git_change_detector, "detect_changed_files", fake_detect)

    change_set = build_change_set(
        repository="TechDocker",
        branch="main",
        before_sha="abc123",
        after_sha="def456",
        commit_message="Fix health endpoint",
        author="Vaibhav",
        repo_path="/repo",
    )

    assert change_set.repository == "TechDocker"
    assert change_set.branch == "main"
    assert change_set.before_sha == "abc123"
    assert change_set.after_sha == "def456"
    assert change_set.commit_message == "Fix health endpoint"
    assert change_set.author == "Vaibhav"
    assert change_set.changed_files == fake_files


# ---------------------------------------------------------------------------
# detect_changed_files
# ---------------------------------------------------------------------------
def test_detect_changed_files_calls_git_correctly(monkeypatch):
    recorded = {}

    def fake_run(args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return SimpleNamespace(stdout="M\tsrc/api/health.py\n", stderr="", returncode=0)

    monkeypatch.setattr(git_change_detector.subprocess, "run", fake_run)

    changed = detect_changed_files("abc123", "def456", repo_path="/some/repo")

    assert recorded["args"] == ["git", "diff", "--name-status", "abc123", "def456"]
    assert recorded["kwargs"]["cwd"] == "/some/repo"
    assert recorded["kwargs"]["capture_output"] is True
    assert recorded["kwargs"]["text"] is True
    assert recorded["kwargs"]["check"] is True

    (item,) = changed
    assert item.path == "src/api/health.py"
    assert item.change_type == "modified"
