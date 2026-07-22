"""Static checks on the documentation-update GitHub Actions workflow.

Text-based (the CI environment has no YAML parser dependency): they assert the
safety-relevant structure — a read-only routing plan, no auto-selected provider,
gated/optional Copilot, mutually-exclusive branches, no hardcoded secrets,
PR-only flow, and the retained loop guards.
"""

from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(".github/workflows/documentation-update.yml")


def workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def non_comment_lines() -> list[str]:
    return [
        line for line in workflow_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


# ---------------------------------------------------------------------------
# triggers and loop guards retained
# ---------------------------------------------------------------------------
def test_push_and_workflow_dispatch_triggers():
    text = workflow_text()
    assert "workflow_dispatch:" in text
    assert "branches:" in text and "- main" in text


def test_artifact_loop_guards_retained():
    text = workflow_text()
    assert "paths-ignore:" in text
    assert "artifacts/**" in text
    assert "github.actor != 'github-actions[bot]'" in text


# ---------------------------------------------------------------------------
# least privilege: no active copilot-requests, no hardcoded tokens
# ---------------------------------------------------------------------------
def test_permissions_are_least_privilege():
    text = workflow_text()
    assert "contents: write" in text
    assert "pull-requests: write" in text
    active = [line for line in non_comment_lines() if "copilot-requests" in line]
    assert active == []


def test_no_hardcoded_personal_access_token():
    text = workflow_text()
    assert "ghp_" not in text
    assert "github_pat_" not in text
    assert "github.token" in text
    assert "secrets." not in text or "secrets.GITHUB_TOKEN" in text


# ---------------------------------------------------------------------------
# read-only plan decides routing; no provider is auto-selected
# ---------------------------------------------------------------------------
def test_read_only_plan_step_drives_routing():
    text = workflow_text()
    assert "src.baseline_initializer --plan" in text
    # Branches key off the plan's action / run_* outputs.
    assert "steps.plan.outputs.action" in text
    assert "steps.plan.outputs.run_updater == 'true'" in text
    assert "steps.plan.outputs.run_initializer == 'true'" in text


def test_no_provider_is_auto_selected():
    text = workflow_text()
    # The provider is only ever the repository variable — never a literal
    # 'deterministic' or 'copilot-cli' default forced by the workflow.
    assert "vars.TECHDOCKER_BASE_SUMMARY_PROVIDER" in text
    # No expression coerces an empty provider to a concrete default.
    assert "'copilot-cli' || ''" not in text
    assert "|| 'deterministic'" not in text


def test_initialization_requires_explicit_enable_and_provider():
    text = workflow_text()
    assert "vars.TECHDOCKER_ENABLE_CANONICAL_INITIALIZATION" in text
    assert "TECHDOCKER_BASE_SUMMARY_PROVIDER: ${{ vars.TECHDOCKER_BASE_SUMMARY_PROVIDER }}" in text


def test_model_is_configurable_not_hardcoded():
    text = workflow_text()
    assert "vars.TECHDOCKER_COPILOT_MODEL" in text
    for hardcoded in ("gpt-4", "gpt-5", "claude-", "o1-", "gemini-"):
        assert hardcoded not in text


# ---------------------------------------------------------------------------
# Copilot is optional, gated by explicit provider + initialize action + non-fork
# ---------------------------------------------------------------------------
def test_copilot_install_requires_explicit_provider_and_initialize_action():
    text = workflow_text()
    assert "npm install -g @github/copilot" in text
    assert "steps.plan.outputs.action == 'initialize_baseline'" in text
    assert "vars.TECHDOCKER_BASE_SUMMARY_PROVIDER == 'copilot-cli'" in text
    assert "github.event.repository.fork == false" in text


def test_copilot_route_documented_as_not_production_ready():
    text = workflow_text().lower()
    assert "not production-ready" in text or "not production ready" in text


# ---------------------------------------------------------------------------
# mutually-exclusive branches for every document state
# ---------------------------------------------------------------------------
def test_initializer_and_updater_are_mutually_exclusive():
    text = workflow_text()
    assert "src.baseline_initializer\n" in text or "src.baseline_initializer" in text
    assert "src.summary_updater" in text
    # The initializer runs only on run_initializer; the updater only on
    # run_updater. The plan guarantees exactly one is true.
    assert "if: steps.plan.outputs.run_initializer == 'true'" in text
    assert "if: steps.plan.outputs.run_updater == 'true'" in text


def test_pending_and_invalid_states_fail_visibly_without_writing():
    text = workflow_text()
    assert "steps.plan.outputs.action == 'initialization_pending'" in text
    assert "steps.plan.outputs.action == 'manual_review'" in text
    # Both surface a visible failure.
    assert text.count("exit 1") >= 2


def test_deferred_legacy_notice_present():
    text = workflow_text()
    assert "steps.plan.outputs.action == 'incremental_update_legacy'" in text
    assert "deferred" in text.lower()


# ---------------------------------------------------------------------------
# PR type from ACTUAL initialization output; PR-only; dynamic doc staged
# ---------------------------------------------------------------------------
def test_pr_type_derived_from_actual_initialization_output():
    text = workflow_text()
    # BASELINE_INIT comes from the initializer's real output, not the status.
    assert "steps.init.outputs.baseline_initialized == 'true'" in text
    assert "BASELINE_INIT" in text


def test_dynamic_canonical_document_is_staged():
    text = workflow_text()
    assert "steps.plan.outputs.canonical_path" in text
    assert "steps.plan.outputs.canonical_filename" in text


def test_never_commits_directly_to_main():
    text = workflow_text()
    assert "gh pr create" in text
    assert 'git push origin "$BRANCH"' in text
    assert "git push origin main" not in text
    assert "git push origin HEAD:main" not in text


def test_pr_titles_distinguish_baseline_and_incremental():
    text = workflow_text().lower()
    assert "baseline" in text
    assert "incremental" in text
