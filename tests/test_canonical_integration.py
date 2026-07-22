"""Integration tests: the canonical baseline resolver and the updater/skeleton.

No real git, no network, no provider calls: the change SHA is set to all-zeroes
so the updater skips git entirely, and every repository is a fresh ``tmp_path``.
"""

from __future__ import annotations

import json

import pytest

from src import canonical_document, summary_updater
from src.baseline_initializer import initialize_baseline
from src.summary_skeleton_builder import default_summary_source
from src.summary_updater import run_update


@pytest.fixture(autouse=True)
def _no_git_remote(monkeypatch):
    """Guarantee no git subprocess: the remote-name reader is neutralized."""
    monkeypatch.setattr(
        canonical_document, "_git_remote_repo_name", lambda repo_path: None
    )

VALID_DOC = """\
# Widget Technical Summary

## Product Purpose

Widget assembles configurable widgets from parts so teams ship consistent
products quickly across many different environments.

## Repository Automation

The pipeline detects changed files and proposes documentation updates through a
pull request for human review before anything merges.

## Quality and Tests

The suite runs offline with deterministic fixtures and never contacts a network
service or an external model during a run.

## Deployment Flow

Continuous integration opens a pull request and never commits generated files
directly to the main branch.
"""

# A canonical marker phrase that must survive untouched.
CANONICAL_MARKER = "Widget assembles configurable widgets from parts"


def summaries(tmp_path):
    directory = tmp_path / "artifacts" / "summaries"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def canonical(tmp_path, name="Widget"):
    return summaries(tmp_path) / f"{name}_TechnicalDocument.md"


def write_canonical(tmp_path, name="Widget"):
    doc = canonical(tmp_path, name)
    doc.write_text(VALID_DOC, encoding="utf-8")
    return doc


# A CI-style env whose before-SHA is all zeroes, so the updater treats the
# changed-file list as empty and never shells out to git.
def ci_env(name="Widget"):
    return {
        "GITHUB_REPOSITORY": f"owner/{name}",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_REF_NAME": "main",
        "GITHUB_ACTOR": "tester",
        "GITHUB_EVENT_BEFORE": "0" * 40,
    }


def make_repo(tmp_path, name="Widget"):
    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text(f"# {name}\n\n{name} builds widgets.\n")
    for module in ("engine", "service"):
        (tmp_path / "src" / f"{module}.py").write_text("def f():\n    return 1\n")
    return tmp_path


# ---------------------------------------------------------------------------
# resolver: skeleton source prefers the canonical document
# ---------------------------------------------------------------------------
def test_default_summary_source_prefers_canonical(tmp_path):
    make_repo(tmp_path, "Widget")
    # Directory name drives the local fallback; name the canonical file to match.
    doc = canonical(tmp_path, tmp_path.name)
    doc.write_text(VALID_DOC, encoding="utf-8")
    assert default_summary_source(tmp_path) == doc


def test_default_summary_source_falls_back_to_legacy_original(tmp_path):
    legacy = summaries(tmp_path) / "base_original_summary.md"
    legacy.write_text(VALID_DOC, encoding="utf-8")
    assert default_summary_source(tmp_path) == legacy


# ---------------------------------------------------------------------------
# updater: an existing canonical document is used and never regenerated
# ---------------------------------------------------------------------------
def test_update_uses_canonical_as_skeleton_source(tmp_path):
    make_repo(tmp_path)
    write_canonical(tmp_path)

    result = run_update(ci_env(), repo_path=str(tmp_path))

    data = json.loads(
        (tmp_path / "artifacts" / "skeletons" / "base_skeleton.json").read_text()
    )
    assert data["source_summary_path"].endswith("Widget_TechnicalDocument.md")
    # The routed section came from the canonical document's headings.
    assert result.decision is not None


def test_update_never_modifies_the_canonical_document(tmp_path):
    make_repo(tmp_path)
    doc = write_canonical(tmp_path)
    before = doc.read_bytes()

    run_update(ci_env(), repo_path=str(tmp_path))

    assert doc.read_bytes() == before  # canonical document is immutable here


def test_update_never_creates_legacy_original_when_canonical_present(tmp_path):
    make_repo(tmp_path)
    write_canonical(tmp_path)
    run_update(ci_env(), repo_path=str(tmp_path))
    assert not (summaries(tmp_path) / "base_original_summary.md").exists()


def test_update_does_not_call_the_base_summary_provider(tmp_path, monkeypatch):
    make_repo(tmp_path)
    write_canonical(tmp_path)

    def must_not_run(*args, **kwargs):
        raise AssertionError("no baseline provider may run when a canonical exists")

    monkeypatch.setattr(summary_updater, "generate_original_summary", must_not_run)

    # Must complete without invoking the (Copilot/deterministic) generator.
    result = run_update(ci_env(), repo_path=str(tmp_path))
    assert result.original_generated is False


def test_update_seeds_reviewable_copy_from_canonical(tmp_path):
    make_repo(tmp_path)
    write_canonical(tmp_path)
    assert not (summaries(tmp_path) / "base_updated_summary.md").exists()

    run_update(ci_env(), repo_path=str(tmp_path))

    reviewable = (summaries(tmp_path) / "base_updated_summary.md").read_text(
        encoding="utf-8"
    )
    # The reviewable copy was seeded from the canonical document.
    assert CANONICAL_MARKER in reviewable


# ---------------------------------------------------------------------------
# legacy path still works (no canonical document present)
# ---------------------------------------------------------------------------
def test_legacy_path_generates_base_original_when_no_canonical(tmp_path):
    make_repo(tmp_path)
    # No canonical document -> the legacy deterministic baseline is generated.
    result = run_update(ci_env(), repo_path=str(tmp_path))
    assert result.original_generated is True
    assert (summaries(tmp_path) / "base_original_summary.md").exists()
    data = json.loads(
        (tmp_path / "artifacts" / "skeletons" / "base_skeleton.json").read_text()
    )
    assert data["source_summary_path"].endswith("base_original_summary.md")


# ---------------------------------------------------------------------------
# initialization and incremental updates are mutually exclusive
# ---------------------------------------------------------------------------
def init_env(name="Widget"):
    return {
        "GITHUB_REPOSITORY": f"owner/{name}",
        "TECHDOCKER_ENABLE_CANONICAL_INITIALIZATION": "true",
        "TECHDOCKER_BASE_SUMMARY_PROVIDER": "deterministic",
    }


def test_initialization_does_not_insert_an_incremental_block(tmp_path):
    make_repo(tmp_path)
    initialize_baseline(tmp_path, init_env())

    reviewable = summaries(tmp_path) / "base_updated_summary.md"
    canonical_doc = canonical(tmp_path)
    # Initialization seeds the reviewable copy byte-identically and inserts NO
    # update block, and writes no change package.
    assert reviewable.read_bytes() == canonical_doc.read_bytes()
    assert "TECHDOCKER_UPDATE_START" not in reviewable.read_text(encoding="utf-8")
    assert not (tmp_path / "artifacts" / "change_packages").exists()


def test_incremental_update_after_initialization_does_not_regenerate(
    tmp_path, monkeypatch
):
    make_repo(tmp_path)
    initialize_baseline(tmp_path, init_env())
    canonical_before = canonical(tmp_path).read_bytes()

    def must_not_run(*args, **kwargs):
        raise AssertionError("baseline provider must not run on a later push")

    monkeypatch.setattr(summary_updater, "generate_original_summary", must_not_run)

    run_update(ci_env(), repo_path=str(tmp_path))
    # The accepted canonical baseline is never rebuilt by a later push.
    assert canonical(tmp_path).read_bytes() == canonical_before


# ---------------------------------------------------------------------------
# Problem 4: the reviewable copy is never a baseline / skeleton source
# ---------------------------------------------------------------------------
REVIEWABLE_WITH_BLOCK = (
    VALID_DOC
    + "\n<!-- TECHDOCKER_UPDATE_START -->\n"
    "### Automated Change Update - 2026-01-01\n"
    "Generated block mentioning engine.py.\n"
    "<!-- TECHDOCKER_UPDATE_END -->\n"
)


def test_reviewable_only_state_is_not_a_baseline(tmp_path):
    from src.summary_skeleton_builder import (
        build_and_save_summary_skeleton,
        default_summary_source,
    )

    make_repo(tmp_path)
    (summaries(tmp_path) / "base_updated_summary.md").write_text(
        REVIEWABLE_WITH_BLOCK, encoding="utf-8"
    )
    # The skeleton source must NOT be the reviewable copy; with no canonical or
    # legacy baseline, resolution points at the (non-existent) canonical path so
    # building fails loudly rather than ingesting generated update blocks.
    source = default_summary_source(tmp_path)
    assert source.name.endswith("_TechnicalDocument.md")
    assert source.name != "base_updated_summary.md"
    with pytest.raises(FileNotFoundError):
        build_and_save_summary_skeleton(tmp_path)


def test_generated_update_blocks_cannot_become_skeleton_sections(tmp_path):
    make_repo(tmp_path)
    # Only the reviewable copy exists, and it already carries a generated block.
    (summaries(tmp_path) / "base_updated_summary.md").write_text(
        REVIEWABLE_WITH_BLOCK, encoding="utf-8"
    )

    run_update(ci_env(), repo_path=str(tmp_path))

    # A clean legacy base_original was generated and used as the skeleton source
    # — never the reviewable copy — so no generated block entered the skeleton.
    assert (summaries(tmp_path) / "base_original_summary.md").exists()
    data = json.loads(
        (tmp_path / "artifacts" / "skeletons" / "base_skeleton.json").read_text()
    )
    assert data["source_summary_path"].endswith("base_original_summary.md")
    assert not any(
        "Automated Change Update" in section["heading"] for section in data["sections"]
    )
