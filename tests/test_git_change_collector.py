"""Tests for collect_file_changes against temporary git repositories.

Offline: creates throwaway git repos in tmp_path. No network, no LLM, no
access to the real repository.
"""

from __future__ import annotations

import subprocess

import pytest

from src.change_summary_generator import SCHEMA_VERSION, create_change_package
from src.git_change_detector import ChangedFile, collect_file_changes


def git(repo, *args, **kwargs):
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True, **kwargs
    )


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "Test")
    return tmp_path


def commit_all(repo, message):
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD").stdout.strip()


def by_path(changes):
    return {c.path: c for c in changes}


# ---------------------------------------------------------------------------
# file statuses and metadata
# ---------------------------------------------------------------------------
def test_modified_text_file_metadata_and_hunks(repo):
    (repo / "mod.py").write_text("def f():\n    return 1\n")
    before = commit_all(repo, "base")
    (repo / "mod.py").write_text("def f():\n    return 2\n    print('extra')\n")
    after = commit_all(repo, "change")

    change = by_path(collect_file_changes(before, after, repo))["mod.py"]
    assert change.status == "modified"
    assert change.binary is False
    assert change.additions == 2 and change.deletions == 1
    assert len(change.what_changed) == 1
    hunk = change.what_changed[0]
    assert hunk["change_type"] == "modified"
    assert "f" in hunk["symbols"]
    assert any(line["text"] == "    return 2" for line in hunk["added_lines"])
    assert any(line["text"] == "    return 1" for line in hunk["removed_lines"])


def test_added_text_file(repo):
    (repo / "keep.txt").write_text("x\n")
    before = commit_all(repo, "base")
    (repo / "new_module.py").write_text("def brand_new():\n    return 42\n")
    after = commit_all(repo, "add")

    change = by_path(collect_file_changes(before, after, repo))["new_module.py"]
    assert change.status == "added"
    assert change.deletions == 0
    assert change.what_changed[0]["old_end_line"] is None
    assert change.what_changed[0]["change_type"] == "added"
    assert "brand_new" in change.what_changed[0]["symbols"]
    assert "Added" in change.what_changed[0]["summary"]


def test_deleted_text_file_symbols_from_pre_change(repo):
    (repo / "gone.py").write_text("class LegacyParser:\n    def m(self):\n        return 1\n")
    before = commit_all(repo, "base")
    (repo / "gone.py").unlink()
    after = commit_all(repo, "delete")

    change = by_path(collect_file_changes(before, after, repo))["gone.py"]
    assert change.status == "deleted"
    hunk = change.what_changed[0]
    assert hunk["change_type"] == "deleted"
    assert hunk["new_end_line"] is None
    # Symbol comes from pre-change content.
    assert any("LegacyParser" in s for s in hunk["symbols"])


def test_pure_rename_has_no_hunks(repo):
    (repo / "orig.py").write_text("aaa\nbbb\nccc\nddd\neee\n")
    before = commit_all(repo, "base")
    git(repo, "mv", "orig.py", "renamed.py")
    after = commit_all(repo, "rename")

    change = by_path(collect_file_changes(before, after, repo))["renamed.py"]
    assert change.status == "renamed"
    assert change.old_path == "orig.py"
    assert change.what_changed == []


def test_rename_with_modification(repo):
    (repo / "orig.py").write_text("aaa\nbbb\nccc\nddd\neee\nfff\nggg\nhhh\n")
    before = commit_all(repo, "base")
    git(repo, "mv", "orig.py", "moved.py")
    (repo / "moved.py").write_text(
        "aaa\nbbb\nccc\nddd\neee\nfff\nggg\nhhh\nADDED\n"
    )
    after = commit_all(repo, "rename+mod")

    change = by_path(collect_file_changes(before, after, repo))["moved.py"]
    assert change.status == "renamed"
    assert change.old_path == "orig.py"
    assert change.what_changed  # has hunks
    assert any(
        line["text"] == "ADDED"
        for hunk in change.what_changed
        for line in hunk["added_lines"]
    )


def test_binary_file_is_marked_and_not_decoded(repo):
    (repo / "keep.txt").write_text("x\n")
    before = commit_all(repo, "base")
    (repo / "image.bin").write_bytes(bytes(range(256)) * 4)
    after = commit_all(repo, "add binary")

    change = by_path(collect_file_changes(before, after, repo))["image.bin"]
    assert change.binary is True
    assert change.additions is None and change.deletions is None
    assert change.what_changed == []
    assert change.binary_note and "Binary" in change.binary_note


def test_path_with_spaces(repo):
    (repo / "a file.py").write_text("def spaced():\n    return 1\n")
    before = commit_all(repo, "base")
    (repo / "a file.py").write_text("def spaced():\n    return 2\n")
    after = commit_all(repo, "change")

    changes = by_path(collect_file_changes(before, after, repo))
    assert "a file.py" in changes
    assert changes["a file.py"].what_changed


def test_syntax_error_in_python_does_not_crash(repo):
    (repo / "keep.txt").write_text("x\n")
    before = commit_all(repo, "base")
    # A newly added file whose only (post-change) content is unparseable:
    # symbol detection must degrade to an empty list, not raise.
    (repo / "broken.py").write_text("def ok(:\n    return 1\n    return 2\n")
    after = commit_all(repo, "add broken")

    change = by_path(collect_file_changes(before, after, repo))["broken.py"]
    assert change.status == "added"
    assert change.what_changed  # hunks still produced (no crash)
    assert change.what_changed[0]["symbols"] == []


def test_partially_edited_python_uses_parseable_side(repo):
    # When the post-change side is broken but the pre-change side parses,
    # symbols are still recovered from the good side (best-effort).
    (repo / "svc.py").write_text("def handler():\n    return 1\n")
    before = commit_all(repo, "base")
    (repo / "svc.py").write_text("def handler(:\n    return 1\n    return 2\n")
    after = commit_all(repo, "break")

    change = by_path(collect_file_changes(before, after, repo))["svc.py"]
    assert change.what_changed  # no crash
    assert "handler" in change.what_changed[0]["symbols"]


def test_non_python_file_has_no_python_symbols(repo):
    (repo / "notes.md").write_text("# Title\n\nold text\n")
    before = commit_all(repo, "base")
    (repo / "notes.md").write_text("# Title\n\nnew text\nmore\n")
    after = commit_all(repo, "change")

    change = by_path(collect_file_changes(before, after, repo))["notes.md"]
    assert all(hunk["symbols"] == [] for hunk in change.what_changed)


def test_collection_is_deterministic(repo):
    (repo / "a.py").write_text("def f():\n    return 1\n")
    before = commit_all(repo, "base")
    (repo / "a.py").write_text("def f():\n    return 2\n")
    after = commit_all(repo, "change")

    first = [c.to_dict() for c in collect_file_changes(before, after, repo)]
    second = [c.to_dict() for c in collect_file_changes(before, after, repo)]
    assert first == second


# ---------------------------------------------------------------------------
# JSON serialization and backward compatibility
# ---------------------------------------------------------------------------
def test_what_changed_serializes_into_package(repo, tmp_path):
    (repo / "svc.py").write_text("def handler():\n    return 1\n")
    before = commit_all(repo, "base")
    (repo / "svc.py").write_text("def handler():\n    return 2\n    log()\n")
    after = commit_all(repo, "change")

    details = [c.to_dict() for c in collect_file_changes(before, after, repo)]
    package, path = create_change_package(
        repository="TechDocker",
        branch="main",
        actor="tester",
        before_sha=before,
        after_sha=after,
        changed_files=[ChangedFile(path="svc.py", change_type="modified")],
        repo_path=tmp_path,
        file_details=details,
    )

    assert package["schema_version"] == SCHEMA_VERSION
    entry = package["changed_files"][0]
    # v2 fields present...
    assert entry["status"] == "modified"
    assert "what_changed" in entry and entry["what_changed"]
    assert "additions" in entry and "binary" in entry
    # ...and v1 field preserved for old readers.
    assert entry["change_type"] == "modified"


def test_plus_prefixed_source_line_round_trips_through_git(repo):
    # A real source line "++value" is emitted by git as "+++value"; it must be
    # captured as added content, not discarded as a file header.
    (repo / "code.txt").write_text("base\n")
    before = commit_all(repo, "base")
    (repo / "code.txt").write_text("base\n++value\n")
    after = commit_all(repo, "add ++ line")

    change = by_path(collect_file_changes(before, after, repo))["code.txt"]
    added_texts = [
        line["text"] for hunk in change.what_changed for line in hunk["added_lines"]
    ]
    assert "++value" in added_texts


def test_long_line_truncation_surfaces_in_json(repo):
    (repo / "keep.txt").write_text("x\n")
    before = commit_all(repo, "base")
    (repo / "big.txt").write_text("y" * 5000 + "\n")  # exceeds 2000-char cap
    after = commit_all(repo, "add long line")

    change = by_path(collect_file_changes(before, after, repo))["big.txt"]
    entry = change.to_dict()
    hunk = entry["what_changed"][0]
    assert hunk["hunk_text_truncated"] is True
    line = hunk["added_lines"][0]
    assert line["text_truncated"] is True
    assert len(line["text"]) == 2000  # DEFAULT_MAX_LINE_CHARS


def test_ordinary_lines_omit_text_truncated_key(repo):
    (repo / "svc.py").write_text("def f():\n    return 1\n")
    before = commit_all(repo, "base")
    (repo / "svc.py").write_text("def f():\n    return 2\n")
    after = commit_all(repo, "change")

    change = by_path(collect_file_changes(before, after, repo))["svc.py"]
    for hunk in change.to_dict()["what_changed"]:
        for line in [*hunk["added_lines"], *hunk["removed_lines"]]:
            assert "text_truncated" not in line  # stable {line_number, text}


def test_package_without_details_keeps_v1_shape(tmp_path):
    package, _ = create_change_package(
        repository="TechDocker",
        branch="main",
        actor="tester",
        before_sha="abc",
        after_sha="def",
        changed_files=[ChangedFile(path="x.py", change_type="modified")],
        repo_path=tmp_path,
    )
    entry = package["changed_files"][0]
    assert entry == {"path": "x.py", "change_type": "modified", "old_path": None}
    assert package["schema_version"] == SCHEMA_VERSION
