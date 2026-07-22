"""Tests for the repository-name and canonical-document resolver.

No network, no git required (the remote reader is injected), and nothing is
written except a temporary GitHub Actions output file in the CLI test.
"""

from __future__ import annotations

import json

import pytest

from src.canonical_document import (
    BASELINE_CANONICAL,
    BASELINE_LEGACY_ORIGINAL,
    BASELINE_NONE,
    STATUS_EXISTING_INVALID,
    STATUS_EXISTING_VALID,
    STATUS_MISSING,
    TECHNICAL_DOCUMENT_SUFFIX,
    CanonicalDocumentError,
    canonical_document_filename,
    canonical_document_path,
    check_canonical_document,
    main,
    resolve_canonical_baseline,
    resolve_repository_name,
    sanitize_repository_name,
    validate_canonical_document,
)

_NO_REMOTE = lambda repo_path: None  # noqa: E731 - test seam

VALID_DOC = """\
# Widget Technical Summary

## Product Purpose

Widget assembles configurable widgets from parts so teams ship consistent
products quickly across environments.

## Repository Automation

The pipeline detects changed files and proposes documentation updates through
a pull request for human review.

## Quality and Tests

The suite runs offline with deterministic fixtures and never contacts a network
service or external model during a run.

## Deployment Flow

Continuous integration opens a pull request and never commits generated files
directly to the main branch.
"""


# ---------------------------------------------------------------------------
# repository name resolution
# ---------------------------------------------------------------------------
def test_github_repository_owner_repo_is_split():
    assert resolve_repository_name(
        {"GITHUB_REPOSITORY": "AdrenalIsland1/TechDocker"}, ".", remote_reader=_NO_REMOTE
    ) == "TechDocker"


def test_github_repository_bare_name():
    assert resolve_repository_name(
        {"GITHUB_REPOSITORY": "ProfitPulse"}, ".", remote_reader=_NO_REMOTE
    ) == "ProfitPulse"


def test_git_remote_used_when_no_github_repository():
    name = resolve_repository_name(
        {}, ".", remote_reader=lambda repo_path: "RemoteRepo"
    )
    assert name == "RemoteRepo"


def test_directory_name_is_the_local_fallback(tmp_path):
    repo = tmp_path / "LocalWidget"
    repo.mkdir()
    assert resolve_repository_name({}, repo, remote_reader=_NO_REMOTE) == "LocalWidget"


# ---------------------------------------------------------------------------
# sanitization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("TechDocker", "TechDocker"),
        ("ProfitPulse", "ProfitPulse"),
        ("my-repo", "my-repo"),        # hyphen preserved
        ("my.repo", "my.repo"),        # dot preserved
        ("weird name!", "weird_name"), # unsafe chars -> underscore
        (".hidden", "hidden"),         # leading dot stripped
        ("repo.", "repo"),             # trailing dot stripped
    ],
)
def test_sanitize_preserves_sensible_names(raw, expected):
    assert sanitize_repository_name(raw) == expected


@pytest.mark.parametrize(
    "unsafe", ["", "   ", ".", "..", "a/b", "a\\b", "../etc", "x\x00y", "/"]
)
def test_sanitize_rejects_unsafe_names(unsafe):
    with pytest.raises(CanonicalDocumentError):
        sanitize_repository_name(unsafe)


# ---------------------------------------------------------------------------
# canonical path
# ---------------------------------------------------------------------------
def test_canonical_filename_uses_correct_suffix():
    filename = canonical_document_filename({"GITHUB_REPOSITORY": "a/ProfitPulse"}, ".")
    assert filename == "ProfitPulse_TechnicalDocument.md"
    assert filename.endswith(TECHNICAL_DOCUMENT_SUFFIX)


def test_canonical_path_is_deterministic_and_under_summaries(tmp_path):
    path = canonical_document_path(tmp_path, {"GITHUB_REPOSITORY": "a/TechDocker"})
    assert path == tmp_path / "artifacts" / "summaries" / "TechDocker_TechnicalDocument.md"


def test_hyphenated_and_dotted_names_produce_valid_paths(tmp_path):
    for name in ("my-repo", "my.repo.core"):
        path = canonical_document_path(tmp_path, {"GITHUB_REPOSITORY": f"o/{name}"})
        assert path.name == f"{name}{TECHNICAL_DOCUMENT_SUFFIX}"
        assert path.parent == tmp_path / "artifacts" / "summaries"


@pytest.mark.parametrize("bad_name", ["../etc", "..", "a/b", "..\\evil"])
def test_path_traversal_name_is_rejected(tmp_path, bad_name):
    # A traversal-shaped repository name is rejected when building the path.
    with pytest.raises(CanonicalDocumentError):
        canonical_document_path(tmp_path, repository_name=bad_name)


def test_owner_repo_traversal_reduces_to_safe_last_component(tmp_path):
    # "owner/repo" is split on "/", so a traversal-looking value collapses to
    # its final component ("etc"), which is itself a safe filename.
    path = canonical_document_path(tmp_path, {"GITHUB_REPOSITORY": "o/../../etc"})
    assert path.name == "etc_TechnicalDocument.md"
    assert path.parent == tmp_path / "artifacts" / "summaries"


# ---------------------------------------------------------------------------
# document validation (never a bare exists())
# ---------------------------------------------------------------------------
def _summaries(tmp_path):
    directory = tmp_path / "artifacts" / "summaries"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def test_valid_document_passes(tmp_path):
    doc = _summaries(tmp_path) / "Widget_TechnicalDocument.md"
    doc.write_text(VALID_DOC, encoding="utf-8")
    assert validate_canonical_document(doc).ok


def test_missing_document_is_not_valid(tmp_path):
    doc = _summaries(tmp_path) / "Widget_TechnicalDocument.md"
    result = validate_canonical_document(doc)
    assert not result.ok


def test_directory_at_path_is_rejected(tmp_path):
    doc = _summaries(tmp_path) / "Widget_TechnicalDocument.md"
    doc.mkdir()
    result = validate_canonical_document(doc)
    assert not result.ok and any("regular file" in p for p in result.problems)


def test_symlink_is_rejected(tmp_path):
    real = _summaries(tmp_path) / "real.md"
    real.write_text(VALID_DOC, encoding="utf-8")
    link = _summaries(tmp_path) / "Widget_TechnicalDocument.md"
    link.symlink_to(real)
    result = validate_canonical_document(link)
    assert not result.ok and any("symlink" in p for p in result.problems)


@pytest.mark.parametrize(
    "bad",
    [
        "not markdown, one line, no headings",             # no H1 / too short
        "# Only A Title\n\nJust one paragraph, no sections.",  # too few H2
        "```\n# Fenced Whole Doc\n\n## A\n\ncontent\n```",     # whole-doc fence
    ],
)
def test_invalid_documents_fail(tmp_path, bad):
    doc = _summaries(tmp_path) / "Widget_TechnicalDocument.md"
    doc.write_text(bad, encoding="utf-8")
    assert not validate_canonical_document(doc).ok


# ---------------------------------------------------------------------------
# check_canonical_document status
# ---------------------------------------------------------------------------
def test_check_reports_missing(tmp_path):
    check = check_canonical_document(
        tmp_path, {"GITHUB_REPOSITORY": "o/Widget"}, remote_reader=_NO_REMOTE
    )
    assert check.status == STATUS_MISSING
    assert check.exists is False
    assert check.path.name == "Widget_TechnicalDocument.md"


def test_check_reports_existing_valid(tmp_path):
    (_summaries(tmp_path) / "Widget_TechnicalDocument.md").write_text(
        VALID_DOC, encoding="utf-8"
    )
    check = check_canonical_document(
        tmp_path, {"GITHUB_REPOSITORY": "o/Widget"}, remote_reader=_NO_REMOTE
    )
    assert check.status == STATUS_EXISTING_VALID
    assert check.is_valid


def test_check_reports_existing_invalid(tmp_path):
    (_summaries(tmp_path) / "Widget_TechnicalDocument.md").write_text(
        "garbage one line", encoding="utf-8"
    )
    check = check_canonical_document(
        tmp_path, {"GITHUB_REPOSITORY": "o/Widget"}, remote_reader=_NO_REMOTE
    )
    assert check.status == STATUS_EXISTING_INVALID
    assert check.exists is True
    assert check.problems


# ---------------------------------------------------------------------------
# resolve_canonical_baseline preference order
# ---------------------------------------------------------------------------
def test_baseline_prefers_canonical(tmp_path):
    summaries = _summaries(tmp_path)
    (summaries / "Widget_TechnicalDocument.md").write_text(VALID_DOC, encoding="utf-8")
    (summaries / "base_original_summary.md").write_text(VALID_DOC, encoding="utf-8")
    (summaries / "base_updated_summary.md").write_text(VALID_DOC, encoding="utf-8")
    resolved = resolve_canonical_baseline(
        tmp_path, {"GITHUB_REPOSITORY": "o/Widget"}, remote_reader=_NO_REMOTE
    )
    assert resolved.kind == BASELINE_CANONICAL
    assert resolved.path.name == "Widget_TechnicalDocument.md"


def test_baseline_falls_back_to_legacy_original(tmp_path):
    summaries = _summaries(tmp_path)
    (summaries / "base_original_summary.md").write_text(VALID_DOC, encoding="utf-8")
    resolved = resolve_canonical_baseline(
        tmp_path, {"GITHUB_REPOSITORY": "o/Widget"}, remote_reader=_NO_REMOTE
    )
    assert resolved.kind == BASELINE_LEGACY_ORIGINAL
    assert resolved.path.name == "base_original_summary.md"


def test_baseline_never_falls_back_to_updated(tmp_path):
    # base_updated_summary.md is the reviewable output, NEVER a baseline. When
    # only it exists, resolution reports no valid baseline (Problem 4).
    (_summaries(tmp_path) / "base_updated_summary.md").write_text(
        VALID_DOC, encoding="utf-8"
    )
    resolved = resolve_canonical_baseline(
        tmp_path, {"GITHUB_REPOSITORY": "o/Widget"}, remote_reader=_NO_REMOTE
    )
    assert resolved.kind == BASELINE_NONE
    assert resolved.exists is False


def test_baseline_none_when_nothing_exists(tmp_path):
    resolved = resolve_canonical_baseline(
        tmp_path, {"GITHUB_REPOSITORY": "o/Widget"}, remote_reader=_NO_REMOTE
    )
    assert resolved.kind == BASELINE_NONE
    assert resolved.exists is False


# ---------------------------------------------------------------------------
# read-only CLI
# ---------------------------------------------------------------------------
def test_cli_emits_json_and_writes_nothing(tmp_path, capsys):
    code = main(["--repo-path", str(tmp_path)], env={"GITHUB_REPOSITORY": "o/Widget"})
    out = capsys.readouterr()
    assert code == 0
    payload = json.loads(out.out)  # stdout is pure JSON
    assert payload["status"] == STATUS_MISSING
    assert payload["filename"] == "Widget_TechnicalDocument.md"
    # Nothing was written.
    assert not (tmp_path / "artifacts").exists()


def test_cli_writes_github_outputs_but_not_in_preview(tmp_path):
    output_file = tmp_path / "gh_output.txt"
    env = {"GITHUB_REPOSITORY": "o/Widget", "GITHUB_OUTPUT": str(output_file)}

    assert main(["--repo-path", str(tmp_path)], env=env) == 0
    written = output_file.read_text(encoding="utf-8")
    assert "status=missing" in written
    assert "filename=Widget_TechnicalDocument.md" in written

    # --preview must emit no GitHub outputs (writes nothing at all).
    output_file.unlink()
    assert main(["--repo-path", str(tmp_path), "--preview"], env=env) == 0
    assert not output_file.exists()
