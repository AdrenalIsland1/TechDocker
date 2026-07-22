"""Tests for the optional Copilot CLI baseline provider.

Every invocation uses an injected fake runner — no real ``copilot`` executable,
no subprocess, no network. Command construction and all failure modes are
exercised, and diagnostics are checked for secret redaction.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.copilot_summary_provider import (
    DEFAULT_ALLOW_ALL_TOOLS_FLAG,
    CopilotCliSummaryProvider,
    CopilotConfig,
    CopilotEmptyOutputError,
    CopilotExecutionError,
    CopilotRunResult,
    CopilotTimeoutError,
    CopilotUnavailableError,
    is_copilot_cli_selected,
)
from src.repo_context_collector import RepoContext

VALID_MD = (
    "# Widget Technical Summary\n\n"
    "## Purpose\n\nWidget assembles configurable widgets from parts and modules "
    "so teams can ship consistent products across environments quickly.\n\n"
    "## Structure\n\nThe src directory holds the engine and service modules that "
    "coordinate building and validating each widget in the pipeline.\n\n"
    "## Testing\n\nThe suite runs entirely offline with deterministic fixtures "
    "and never contacts a network service or model during a run.\n\n"
    "## Deployment\n\nContinuous integration proposes documentation updates only "
    "through pull requests and never commits directly to the main branch.\n"
)


def context() -> RepoContext:
    return RepoContext(
        root=".", project_name="Widget", file_tree=["src/engine.py"],
        files={"README.md": "# Widget\n\nBuilds widgets."}, total_files=1,
    )


def runner_returning(returncode=0, stdout="", stderr=""):
    def _run(command, timeout):
        return CopilotRunResult(returncode=returncode, stdout=stdout, stderr=stderr)
    return _run


# ---------------------------------------------------------------------------
# selection + configuration
# ---------------------------------------------------------------------------
def test_is_copilot_cli_selected():
    assert is_copilot_cli_selected({"TECHDOCKER_BASE_SUMMARY_PROVIDER": "copilot-cli"})
    assert not is_copilot_cli_selected({})
    assert not is_copilot_cli_selected(
        {"TECHDOCKER_BASE_SUMMARY_PROVIDER": "deterministic"}
    )


def test_config_from_env():
    config = CopilotConfig.from_env(
        {
            "TECHDOCKER_COPILOT_EXECUTABLE": "copilot",
            "TECHDOCKER_COPILOT_MODEL": "some-model",
            "TECHDOCKER_COPILOT_TIMEOUT": "45",
        }
    )
    assert config.executable == "copilot"
    assert config.model == "some-model"
    assert config.timeout_seconds == 45.0
    assert config.allow_all_tools is False  # off by default


def test_command_is_narrow_by_default():
    config = CopilotConfig(executable="copilot", model="m")
    command = config.build_command("PROMPT")
    assert command[0] == "copilot"
    assert command[1] == "-p"
    assert "PROMPT" in command
    assert "--model" in command and "m" in command
    # No broad-permission flag unless explicitly enabled.
    assert DEFAULT_ALLOW_ALL_TOOLS_FLAG not in command


def test_command_adds_permissive_flag_only_when_explicitly_enabled():
    config = CopilotConfig(allow_all_tools=True)
    assert DEFAULT_ALLOW_ALL_TOOLS_FLAG in config.build_command("P")
    custom = CopilotConfig(allow_all_tools=True, allow_all_tools_flag="--yolo")
    assert "--yolo" in custom.build_command("P")


def test_allow_all_tools_opt_in_from_env():
    config = CopilotConfig.from_env(
        {"TECHDOCKER_COPILOT_ALLOW_ALL_TOOLS": "true"}
    )
    assert config.allow_all_tools is True


# ---------------------------------------------------------------------------
# success + failure modes
# ---------------------------------------------------------------------------
def test_success_returns_markdown():
    provider = CopilotCliSummaryProvider(runner=runner_returning(0, VALID_MD))
    out = provider.generate_summary(context())
    assert out.startswith("# Widget Technical Summary")
    assert out.endswith("\n")


def test_outer_fence_is_stripped():
    fenced = "```markdown\n" + VALID_MD + "```\n"
    provider = CopilotCliSummaryProvider(runner=runner_returning(0, fenced))
    out = provider.generate_summary(context())
    assert not out.lstrip().startswith("```")


def test_missing_executable_raises_unavailable():
    def _missing(command, timeout):
        raise FileNotFoundError("copilot")
    provider = CopilotCliSummaryProvider(runner=_missing)
    with pytest.raises(CopilotUnavailableError):
        provider.generate_summary(context())


def test_timeout_raises_timeout_error():
    def _timeout(command, timeout):
        raise subprocess.TimeoutExpired(command, timeout)
    provider = CopilotCliSummaryProvider(runner=_timeout)
    with pytest.raises(CopilotTimeoutError):
        provider.generate_summary(context())


def test_nonzero_exit_raises_execution_error():
    provider = CopilotCliSummaryProvider(
        runner=runner_returning(2, "", "some internal failure")
    )
    with pytest.raises(CopilotExecutionError):
        provider.generate_summary(context())


def test_auth_failure_raises_unavailable():
    provider = CopilotCliSummaryProvider(
        runner=runner_returning(1, "", "error: not logged in; run gh auth login")
    )
    with pytest.raises(CopilotUnavailableError):
        provider.generate_summary(context())


def test_empty_output_raises_empty_error():
    provider = CopilotCliSummaryProvider(runner=runner_returning(0, "   \n  "))
    with pytest.raises(CopilotEmptyOutputError):
        provider.generate_summary(context())


def test_diagnostics_redact_secrets():
    leaky = "auth error token=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345 failed"
    provider = CopilotCliSummaryProvider(runner=runner_returning(3, "", leaky))
    with pytest.raises(CopilotExecutionError) as error:
        provider.generate_summary(context())
    message = str(error.value)
    assert "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in message
    assert "[redacted]" in message


# ---------------------------------------------------------------------------
# static safety: no shell, no real executable required
# ---------------------------------------------------------------------------
def test_module_never_uses_shell_true():
    source = Path("src/copilot_summary_provider.py").read_text(encoding="utf-8")
    assert "shell=True" not in source
