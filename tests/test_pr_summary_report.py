"""Tests for the PR body helper (no network, no token, no gh CLI)."""

from __future__ import annotations

import json

from src.change_summary_generator import change_package_path
from src.pr_summary_report import build_pr_body, load_change_package

PACKAGE = {
    "repository": "TechDocker",
    "branch": "main",
    "actor": "Vaibhav",
    "before_sha": "abc123",
    "after_sha": "def456",
    "changed_files": [
        {"path": "src/core.py", "change_type": "modified", "old_path": None},
        {"path": "src/new_name.py", "change_type": "renamed", "old_path": "src/old.py"},
    ],
    "generated_summary": "2 file(s) changed (1 modified, 1 renamed): ...",
    "generated_at": "2026-07-15 00:00:00 UTC",
}


def test_pr_body_includes_changed_files():
    body = build_pr_body(PACKAGE, env={})
    assert "- modified: `src/core.py`" in body
    assert "- renamed: `src/old.py` -> `src/new_name.py`" in body
    assert "2 file(s) changed" in body  # generated summary section


def test_pr_body_includes_actor_branch_and_sha():
    body = build_pr_body(PACKAGE, env={})
    assert "`def456`" in body
    assert "**Actor:** Vaibhav" in body
    assert "**Branch:** main" in body
    # Reviewer guidance and baseline note are always present.
    assert "base_original_summary.md" in body
    assert "base_updated_summary.md" in body


def test_missing_package_is_handled_gracefully(tmp_path):
    assert load_change_package(tmp_path) is None

    body = build_pr_body(None, env={"GITHUB_SHA": "fff000", "GITHUB_ACTOR": "ci"})
    assert "change details unavailable" in body
    assert "`fff000`" in body  # env fallback for the SHA
    assert "**Actor:** ci" in body


def test_corrupt_package_is_handled_gracefully(tmp_path):
    path = change_package_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert load_change_package(tmp_path) is None


def test_load_change_package_roundtrip(tmp_path):
    path = change_package_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(PACKAGE), encoding="utf-8")
    loaded = load_change_package(tmp_path)
    assert loaded["after_sha"] == "def456"
