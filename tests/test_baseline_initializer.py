"""Tests for baseline initialization — gated, non-fallback, and transactional.

Providers are always mocked/deterministic — no real Copilot, no network. Every
repository is a fresh ``tmp_path``; the real repository artifacts are never
touched.
"""

from __future__ import annotations

import json
import os

import pytest

from src.baseline_initializer import (
    ACTION_INCREMENTAL_CANONICAL,
    ACTION_INCREMENTAL_LEGACY,
    ACTION_INITIALIZATION_PENDING,
    ACTION_INITIALIZE,
    ACTION_MANUAL_REVIEW,
    ALLOW_PROVIDER_FALLBACK_ENV_VAR,
    ENABLE_INITIALIZATION_ENV_VAR,
    STATUS_GENERATION_FAILED,
    STATUS_INITIALIZATION_PENDING,
    InitializationCommitError,
    _resolve_initialization_provider,
    initialize_baseline,
    install_initialization_outputs,
    main,
    resolve_push_plan,
)
from src.canonical_document import (
    STATUS_EXISTING_INVALID,
    STATUS_EXISTING_VALID,
    STATUS_GENERATED_COPILOT,
    STATUS_GENERATED_DETERMINISTIC,
)
from src.copilot_summary_provider import BASE_SUMMARY_PROVIDER_ENV_VAR, CopilotCliSummaryProvider, CopilotRunResult
from src.project_summary_generator import LocalDeterministicSummaryProvider

VALID_MD = (
    "# Widget Technical Summary\n\n"
    "## Purpose\n\nWidget assembles configurable widgets from parts and modules "
    "so teams ship consistent products across environments quickly and safely.\n\n"
    "## Structure\n\nThe src directory holds the engine and service modules that "
    "coordinate building and validating each widget through the pipeline.\n\n"
    "## Testing\n\nThe suite runs entirely offline with deterministic fixtures "
    "and never contacts a network service or external model during a run.\n\n"
    "## Deployment\n\nContinuous integration proposes documentation updates only "
    "through pull requests and never commits generated files to main.\n"
)


def make_repo(tmp_path, name="Widget"):
    """A repository whose deterministic summary passes validation."""
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "README.md").write_text(
        f"# {name}\n\n{name} assembles configurable widgets from parts and reports "
        "results deterministically for reviewers.\n",
        encoding="utf-8",
    )
    for module in ("engine", "service", "models", "ledger"):
        (tmp_path / "src" / f"{module}.py").write_text(
            f"def {module}():\n    return 1\n", encoding="utf-8"
        )
    (tmp_path / "tests" / "test_engine.py").write_text(
        "def test_engine():\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    return tmp_path


def env_for(name="Widget", **extra):
    return {"GITHUB_REPOSITORY": f"owner/{name}", **extra}


def enabled_env(name="Widget", provider="deterministic", **extra):
    return env_for(
        name,
        **{
            ENABLE_INITIALIZATION_ENV_VAR: "true",
            BASE_SUMMARY_PROVIDER_ENV_VAR: provider,
            **extra,
        },
    )


def canonical(tmp_path, name="Widget"):
    return tmp_path / "artifacts" / "summaries" / f"{name}_TechnicalDocument.md"


def skeleton(tmp_path):
    return tmp_path / "artifacts" / "skeletons" / "base_skeleton.json"


def updated(tmp_path):
    return tmp_path / "artifacts" / "summaries" / "base_updated_summary.md"


def original(tmp_path):
    return tmp_path / "artifacts" / "summaries" / "base_original_summary.md"


def copilot_provider(returncode=0, stdout=VALID_MD, stderr="", exc=None):
    def _run(command, timeout):
        if exc is not None:
            raise exc
        return CopilotRunResult(returncode=returncode, stdout=stdout, stderr=stderr)
    return CopilotCliSummaryProvider(runner=_run)


class ExplodingProvider:
    """A SummaryProvider that must never be called."""

    name = "exploding"

    def generate_summary(self, context):
        raise AssertionError("provider must not be called")


def no_temp_or_backup(tmp_path):
    for directory in (
        tmp_path / "artifacts" / "summaries",
        tmp_path / "artifacts" / "skeletons",
    ):
        if directory.exists():
            for entry in directory.iterdir():
                assert ".tmp-" not in entry.name, entry
                assert ".bak-" not in entry.name, entry


# ---------------------------------------------------------------------------
# PROBLEM 1: production gate — no silent deterministic canonical on a push
# ---------------------------------------------------------------------------
def test_missing_without_enable_is_pending_and_writes_nothing(tmp_path):
    make_repo(tmp_path)
    result = initialize_baseline(tmp_path, env_for())  # not enabled
    assert result.status == STATUS_INITIALIZATION_PENDING
    assert result.wrote_files is False
    assert not (tmp_path / "artifacts").exists()  # nothing created


def test_enabled_but_empty_provider_is_pending(tmp_path):
    make_repo(tmp_path)
    result = initialize_baseline(
        tmp_path,
        env_for(**{ENABLE_INITIALIZATION_ENV_VAR: "true", BASE_SUMMARY_PROVIDER_ENV_VAR: ""}),
    )
    assert result.status == STATUS_INITIALIZATION_PENDING
    assert result.wrote_files is False


def test_enabled_explicit_deterministic_writes(tmp_path):
    make_repo(tmp_path)
    result = initialize_baseline(tmp_path, enabled_env(provider="deterministic"))
    assert result.status == STATUS_GENERATED_DETERMINISTIC
    assert result.wrote_files is True
    assert canonical(tmp_path).exists()
    assert skeleton(tmp_path).exists()
    assert updated(tmp_path).exists()
    # base_updated is initialized byte-identically from the canonical document.
    assert updated(tmp_path).read_bytes() == canonical(tmp_path).read_bytes()
    data = json.loads(skeleton(tmp_path).read_text(encoding="utf-8"))
    assert data["source_summary_path"].endswith("Widget_TechnicalDocument.md")
    assert not original(tmp_path).exists()  # legacy never created
    assert not (tmp_path / "artifacts" / "change_packages").exists()  # no incremental


def test_injected_provider_is_explicit_permission(tmp_path):
    make_repo(tmp_path)
    # A directly-injected provider (tests/manual previews) counts as explicit
    # permission even without the enable flag.
    result = initialize_baseline(
        tmp_path, env_for(), provider=LocalDeterministicSummaryProvider()
    )
    assert result.status == STATUS_GENERATED_DETERMINISTIC
    assert result.wrote_files is True


@pytest.mark.parametrize(
    "env,provider_kind,ok",
    [
        ({}, None, False),  # not enabled
        ({ENABLE_INITIALIZATION_ENV_VAR: "true"}, None, False),  # no provider
        ({ENABLE_INITIALIZATION_ENV_VAR: "true", BASE_SUMMARY_PROVIDER_ENV_VAR: ""}, None, False),
        ({ENABLE_INITIALIZATION_ENV_VAR: "true", BASE_SUMMARY_PROVIDER_ENV_VAR: "deterministic"}, "deterministic", True),
        ({ENABLE_INITIALIZATION_ENV_VAR: "true", BASE_SUMMARY_PROVIDER_ENV_VAR: "copilot-cli"}, "copilot", True),
        ({ENABLE_INITIALIZATION_ENV_VAR: "true", BASE_SUMMARY_PROVIDER_ENV_VAR: "made-up"}, None, False),
    ],
)
def test_initialization_provider_gate(env, provider_kind, ok):
    provider, kind, pending = _resolve_initialization_provider(env, None)
    assert (provider is not None) is ok
    assert kind == provider_kind
    assert (pending is None) is ok


# ---------------------------------------------------------------------------
# existing valid / invalid: provider never invoked
# ---------------------------------------------------------------------------
def test_existing_valid_makes_no_provider_call_and_no_write(tmp_path):
    make_repo(tmp_path)
    initialize_baseline(tmp_path, enabled_env())  # create it
    before = canonical(tmp_path).read_bytes()

    result = initialize_baseline(tmp_path, enabled_env(), provider=ExplodingProvider())
    assert result.status == STATUS_EXISTING_VALID
    assert result.wrote_files is False
    assert canonical(tmp_path).read_bytes() == before


def test_existing_invalid_never_invokes_provider_and_is_not_overwritten(tmp_path):
    make_repo(tmp_path)
    doc = canonical(tmp_path)
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("garbage, one invalid line", encoding="utf-8")

    result = initialize_baseline(tmp_path, enabled_env(), provider=ExplodingProvider())
    assert result.status == STATUS_EXISTING_INVALID
    assert result.wrote_files is False
    assert doc.read_text(encoding="utf-8") == "garbage, one invalid line"


# ---------------------------------------------------------------------------
# PROBLEM 2: explicit non-deterministic provider failure does not fall back
# ---------------------------------------------------------------------------
def test_copilot_success_is_generated_copilot(tmp_path):
    make_repo(tmp_path)
    result = initialize_baseline(tmp_path, env_for(), provider=copilot_provider())
    assert result.status == STATUS_GENERATED_COPILOT
    assert result.provider_used == "copilot"
    assert "configurable widgets" in canonical(tmp_path).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "provider_factory",
    [
        lambda: copilot_provider(exc=FileNotFoundError("copilot")),
        lambda: copilot_provider(returncode=2, stdout="", stderr="boom"),
        lambda: copilot_provider(stdout="   "),
        lambda: copilot_provider(stdout="short invalid markdown"),
    ],
)
def test_copilot_failure_does_not_fall_back_by_default(tmp_path, provider_factory):
    make_repo(tmp_path)
    result = initialize_baseline(
        tmp_path, env_for(), provider=provider_factory()  # allow_fallback defaults False
    )
    assert result.status == STATUS_GENERATION_FAILED
    assert result.wrote_files is False
    assert not (tmp_path / "artifacts").exists()  # no silent deterministic baseline


def test_copilot_failure_falls_back_only_with_explicit_flag(tmp_path):
    make_repo(tmp_path)
    result = initialize_baseline(
        tmp_path,
        env_for(),
        provider=copilot_provider(exc=FileNotFoundError("copilot")),
        allow_fallback=True,
    )
    assert result.status == STATUS_GENERATED_DETERMINISTIC
    assert result.provider_used == "deterministic"
    assert canonical(tmp_path).exists()


def test_fallback_flag_is_read_from_env(tmp_path):
    make_repo(tmp_path)
    result = initialize_baseline(
        tmp_path,
        env_for(**{ALLOW_PROVIDER_FALLBACK_ENV_VAR: "true"}),
        provider=copilot_provider(exc=FileNotFoundError("copilot")),
    )
    assert result.status == STATUS_GENERATED_DETERMINISTIC
    assert canonical(tmp_path).exists()


def test_explicit_deterministic_invalid_output_fails_cleanly(tmp_path):
    # A tiny repo whose deterministic template is too thin to validate.
    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text("# Tiny\n\nTiny.\n")
    (tmp_path / "src" / "only.py").write_text("x = 1\n")
    result = initialize_baseline(tmp_path, enabled_env("Tiny", provider="deterministic"))
    assert result.status == STATUS_GENERATION_FAILED
    assert result.wrote_files is False
    assert not (tmp_path / "artifacts" / "summaries").exists()


# ---------------------------------------------------------------------------
# preview writes nothing
# ---------------------------------------------------------------------------
def test_preview_writes_nothing(tmp_path):
    make_repo(tmp_path)
    result = initialize_baseline(tmp_path, enabled_env(), preview=True)
    assert result.wrote_files is False
    assert result.proposed_paths
    assert result.content_metadata["h2_count"] >= 4
    assert not (tmp_path / "artifacts").exists()


# ---------------------------------------------------------------------------
# PROBLEM 3: rollback-safe transactional install
# ---------------------------------------------------------------------------
def fail_on_call(n, real=os.replace):
    """A replace that fails on its ``n``-th invocation (install order:
    canonical=1, updated=2, skeleton=3)."""
    state = {"i": 0}

    def _replace(src, dst):
        state["i"] += 1
        if state["i"] == n:
            raise OSError(f"injected replace failure #{n}")
        return real(src, dst)

    return _replace


def install_paths(tmp_path):
    return (
        canonical(tmp_path),
        updated(tmp_path),
        skeleton(tmp_path),
    )


def do_install(tmp_path, *, replace=os.replace, summary=VALID_MD):
    c, u, s = install_paths(tmp_path)
    return install_initialization_outputs(
        c, u, s, summary, "artifacts/summaries/Widget_TechnicalDocument.md",
        replace=replace,
    )


def test_successful_install_writes_all_three_no_leftovers(tmp_path):
    make_repo(tmp_path)
    created = do_install(tmp_path)
    c, u, s = install_paths(tmp_path)
    assert c.exists() and u.exists() and s.exists()
    assert set(created) == {c, s, u}
    no_temp_or_backup(tmp_path)


def test_failure_before_any_replacement_changes_nothing(tmp_path, monkeypatch):
    make_repo(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("skeleton build failed")

    monkeypatch.setattr("src.baseline_initializer.build_summary_skeleton", boom)
    c, u, s = install_paths(tmp_path)
    with pytest.raises(RuntimeError):
        do_install(tmp_path)
    assert not c.exists() and not u.exists() and not s.exists()
    no_temp_or_backup(tmp_path)


def test_failure_replacing_canonical_leaves_everything_unchanged(tmp_path):
    make_repo(tmp_path)
    c, u, s = install_paths(tmp_path)
    with pytest.raises(InitializationCommitError):
        do_install(tmp_path, replace=fail_on_call(1))
    assert not c.exists() and not u.exists() and not s.exists()
    no_temp_or_backup(tmp_path)


def test_failure_replacing_updated_rolls_back_new_canonical(tmp_path):
    make_repo(tmp_path)
    c, u, s = install_paths(tmp_path)
    # None pre-exist -> canonical (just installed) must be removed on rollback.
    with pytest.raises(InitializationCommitError):
        do_install(tmp_path, replace=fail_on_call(2))
    assert not c.exists() and not u.exists() and not s.exists()
    no_temp_or_backup(tmp_path)


def test_failure_replacing_skeleton_restores_preexisting(tmp_path):
    make_repo(tmp_path)
    c, u, s = install_paths(tmp_path)
    # Pre-existing updated + skeleton with known bytes must be restored exactly.
    c.parent.mkdir(parents=True, exist_ok=True)
    s.parent.mkdir(parents=True, exist_ok=True)
    u.write_text("ORIGINAL UPDATED", encoding="utf-8")
    s.write_text("ORIGINAL SKELETON", encoding="utf-8")

    with pytest.raises(InitializationCommitError):
        do_install(tmp_path, replace=fail_on_call(3))

    # Canonical did not pre-exist -> removed; updated pre-existed -> restored;
    # skeleton never replaced -> untouched.
    assert not c.exists()
    assert u.read_text(encoding="utf-8") == "ORIGINAL UPDATED"
    assert s.read_text(encoding="utf-8") == "ORIGINAL SKELETON"
    no_temp_or_backup(tmp_path)


def test_rollback_restores_all_three_preexisting(tmp_path):
    make_repo(tmp_path)
    c, u, s = install_paths(tmp_path)
    c.parent.mkdir(parents=True, exist_ok=True)
    s.parent.mkdir(parents=True, exist_ok=True)
    c.write_text("ORIGINAL CANONICAL", encoding="utf-8")
    u.write_text("ORIGINAL UPDATED", encoding="utf-8")
    s.write_text("ORIGINAL SKELETON", encoding="utf-8")

    with pytest.raises(InitializationCommitError):
        do_install(tmp_path, replace=fail_on_call(3))

    # Every pre-existing destination is restored byte-for-byte.
    assert c.read_text(encoding="utf-8") == "ORIGINAL CANONICAL"
    assert u.read_text(encoding="utf-8") == "ORIGINAL UPDATED"
    assert s.read_text(encoding="utf-8") == "ORIGINAL SKELETON"
    no_temp_or_backup(tmp_path)


def test_initialize_baseline_threads_replace_and_rolls_back(tmp_path):
    make_repo(tmp_path)
    result = initialize_baseline(
        tmp_path, enabled_env(), replace=fail_on_call(2)
    )
    # A commit failure surfaces as generation failure with nothing installed.
    assert result.status in (STATUS_GENERATION_FAILED,)
    assert result.wrote_files is False
    assert not canonical(tmp_path).exists()
    assert not updated(tmp_path).exists()
    assert not skeleton(tmp_path).exists()
    no_temp_or_backup(tmp_path)


# ---------------------------------------------------------------------------
# read-only push plan
# ---------------------------------------------------------------------------
def test_plan_missing_not_enabled_no_legacy_is_pending(tmp_path):
    make_repo(tmp_path)
    plan = resolve_push_plan(tmp_path, env_for())
    assert plan.action == ACTION_INITIALIZATION_PENDING
    assert plan.run_updater is False and plan.run_initializer is False


def test_plan_missing_with_valid_legacy_is_incremental_legacy(tmp_path):
    make_repo(tmp_path)
    original(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    original(tmp_path).write_text(VALID_MD, encoding="utf-8")
    plan = resolve_push_plan(tmp_path, env_for())
    assert plan.action == ACTION_INCREMENTAL_LEGACY
    assert plan.legacy_baseline_valid is True
    assert plan.run_updater is True


def test_plan_missing_enabled_provider_is_initialize(tmp_path):
    make_repo(tmp_path)
    plan = resolve_push_plan(tmp_path, enabled_env())
    assert plan.action == ACTION_INITIALIZE
    assert plan.run_initializer is True


def test_plan_existing_valid_is_incremental(tmp_path):
    make_repo(tmp_path)
    initialize_baseline(tmp_path, enabled_env())
    plan = resolve_push_plan(tmp_path, enabled_env())
    assert plan.action == ACTION_INCREMENTAL_CANONICAL
    assert plan.run_updater is True and plan.run_initializer is False


def test_plan_invalid_canonical_is_manual_review(tmp_path):
    make_repo(tmp_path)
    doc = canonical(tmp_path)
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("garbage", encoding="utf-8")
    plan = resolve_push_plan(tmp_path, enabled_env())
    assert plan.action == ACTION_MANUAL_REVIEW
    assert plan.run_updater is False and plan.run_initializer is False


def test_plan_writes_nothing(tmp_path):
    make_repo(tmp_path)
    resolve_push_plan(tmp_path, enabled_env())
    assert not (tmp_path / "artifacts").exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_plan_is_readonly_and_emits_action(tmp_path, capsys):
    make_repo(tmp_path)
    output = tmp_path / "gh_output.txt"
    code = main(
        ["--repo-path", str(tmp_path), "--plan"],
        env=env_for(**{"GITHUB_OUTPUT": str(output)}),
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["action"] == ACTION_INITIALIZATION_PENDING
    written = output.read_text(encoding="utf-8")
    assert "action=initialization_pending" in written
    assert "run_initializer=false" in written
    assert not (tmp_path / "artifacts").exists()


def test_cli_initializes_with_explicit_env(tmp_path, capsys):
    make_repo(tmp_path)
    code = main(["--repo-path", str(tmp_path)], env=enabled_env())
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == STATUS_GENERATED_DETERMINISTIC
    assert payload["baseline_initialization"] is True
    assert canonical(tmp_path).exists()


def test_cli_preview_writes_nothing(tmp_path, capsys):
    make_repo(tmp_path)
    code = main(["--repo-path", str(tmp_path), "--preview"], env=enabled_env())
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["wrote_files"] is False
    assert not (tmp_path / "artifacts").exists()
