"""Optional GitHub Copilot CLI provider for baseline summary generation.

Opt-in only: selected when ``TECHDOCKER_BASE_SUMMARY_PROVIDER=copilot-cli``.
It implements the same :class:`~src.project_summary_generator.SummaryProvider`
interface as the deterministic and Ollama providers — ``generate_summary``
turns a bounded :class:`RepoContext` into Markdown — so callers stay provider-
agnostic.

Design and safety:

* The Copilot CLI is invoked as an **argument list** via ``subprocess`` — never
  through a shell — with stdout and stderr captured separately.
* Python supplies the bounded repository context (the existing base-summary
  prompt) and **owns all writes**; Copilot returns Markdown only and is given
  no shell/write tools by default. If the installed CLI needs a permissive flag
  (e.g. ``--allow-all-tools``/``--yolo``) to run non-interactively, that must be
  enabled explicitly through configuration — it is never added silently.
* Every failure mode — missing executable, authentication failure, timeout,
  non-zero exit, empty output — raises a **typed** error so the initializer can
  choose the deterministic fallback (non-strict) or fail cleanly (strict).
* Diagnostics are redacted and truncated; tokens and environment secrets are
  never echoed.

Command construction and process execution are injectable so tests never
require a real ``copilot`` executable.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

from src.project_summary_generator import (
    BASE_SUMMARY_SYSTEM_PROMPT,
    _strip_outer_fences,
    build_base_summary_prompt,
)
from src.repo_context_collector import RepoContext

# Selection / configuration environment variables.
BASE_SUMMARY_PROVIDER_ENV_VAR = "TECHDOCKER_BASE_SUMMARY_PROVIDER"
COPILOT_CLI_PROVIDER_NAME = "copilot-cli"

COPILOT_EXECUTABLE_ENV_VAR = "TECHDOCKER_COPILOT_EXECUTABLE"
COPILOT_MODEL_ENV_VAR = "TECHDOCKER_COPILOT_MODEL"
COPILOT_TIMEOUT_ENV_VAR = "TECHDOCKER_COPILOT_TIMEOUT"
# Explicit, documented opt-in for a broad-permission flag. Off by default:
# Python owns all writes, so Copilot needs no shell/write tools.
COPILOT_ALLOW_ALL_TOOLS_ENV_VAR = "TECHDOCKER_COPILOT_ALLOW_ALL_TOOLS"
COPILOT_ALLOW_ALL_TOOLS_FLAG_ENV_VAR = "TECHDOCKER_COPILOT_ALLOW_ALL_TOOLS_FLAG"

DEFAULT_COPILOT_EXECUTABLE = "copilot"
DEFAULT_COPILOT_TIMEOUT_SECONDS = 120.0
DEFAULT_ALLOW_ALL_TOOLS_FLAG = "--allow-all-tools"

_MAX_DIAGNOSTIC_CHARS = 500
_TRUTHY = {"true", "1", "yes", "on"}

# Redact anything that looks like a credential from CLI diagnostics.
_SECRET_PATTERNS = (
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)\b(token|secret|password|authorization)\b\s*[:=]\s*\S+"),
)

_AUTH_FAILURE_MARKERS = (
    "not logged in", "unauthorized", "authentication", "authorization",
    "auth failed", "401", "403", "please sign in", "login required",
    "gh auth login", "no valid credentials",
)


# ---------------------------------------------------------------------------
# Typed failures
# ---------------------------------------------------------------------------
class CopilotProviderError(RuntimeError):
    """Base class for all Copilot CLI provider failures."""


class CopilotUnavailableError(CopilotProviderError):
    """The CLI is missing, not executable, or authentication failed."""


class CopilotTimeoutError(CopilotProviderError):
    """The CLI did not finish within the configured timeout."""


class CopilotExecutionError(CopilotProviderError):
    """The CLI ran but exited non-zero for a non-auth reason."""


class CopilotEmptyOutputError(CopilotProviderError):
    """The CLI succeeded but produced no usable Markdown."""


# ---------------------------------------------------------------------------
# Diagnostics redaction
# ---------------------------------------------------------------------------
def _sanitize_diagnostic(text: str) -> str:
    """Bounded, credential-redacted snippet of CLI stderr for error messages."""
    snippet = (text or "").strip()
    for pattern in _SECRET_PATTERNS:
        snippet = pattern.sub("[redacted]", snippet)
    if len(snippet) > _MAX_DIAGNOSTIC_CHARS:
        snippet = snippet[:_MAX_DIAGNOSTIC_CHARS] + "… [truncated]"
    return snippet


def _looks_like_auth_failure(stderr: str) -> bool:
    lowered = (stderr or "").lower()
    return any(marker in lowered for marker in _AUTH_FAILURE_MARKERS)


# ---------------------------------------------------------------------------
# Process seam (injectable)
# ---------------------------------------------------------------------------
@dataclass
class CopilotRunResult:
    """The captured outcome of one CLI invocation."""

    returncode: int
    stdout: str
    stderr: str


# A runner takes (command argv, timeout seconds) and returns a result. It may
# raise FileNotFoundError (missing exe), subprocess.TimeoutExpired, or OSError.
CopilotRunner = Callable[[list[str], float], CopilotRunResult]


def _default_runner(command: list[str], timeout_seconds: float) -> CopilotRunResult:
    """Run the CLI with an argument list (never a shell), capturing streams."""
    completed = subprocess.run(  # noqa: S603 - argument list, no shell
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    return CopilotRunResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class CopilotConfig:
    """Copilot CLI invocation configuration, resolved from the environment."""

    executable: str = DEFAULT_COPILOT_EXECUTABLE
    model: Optional[str] = None
    timeout_seconds: float = DEFAULT_COPILOT_TIMEOUT_SECONDS
    allow_all_tools: bool = False
    allow_all_tools_flag: str = DEFAULT_ALLOW_ALL_TOOLS_FLAG
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "CopilotConfig":
        import os

        env = env if env is not None else os.environ
        try:
            timeout = float(
                env.get(COPILOT_TIMEOUT_ENV_VAR, "") or DEFAULT_COPILOT_TIMEOUT_SECONDS
            )
        except (TypeError, ValueError):
            timeout = DEFAULT_COPILOT_TIMEOUT_SECONDS
        return cls(
            executable=(env.get(COPILOT_EXECUTABLE_ENV_VAR, "").strip()
                        or DEFAULT_COPILOT_EXECUTABLE),
            model=(env.get(COPILOT_MODEL_ENV_VAR, "").strip() or None),
            timeout_seconds=timeout,
            allow_all_tools=(
                env.get(COPILOT_ALLOW_ALL_TOOLS_ENV_VAR, "").strip().lower() in _TRUTHY
            ),
            allow_all_tools_flag=(
                env.get(COPILOT_ALLOW_ALL_TOOLS_FLAG_ENV_VAR, "").strip()
                or DEFAULT_ALLOW_ALL_TOOLS_FLAG
            ),
        )

    def build_command(self, prompt: str) -> list[str]:
        """Argument list for a single non-interactive prompt.

        Uses the narrow, non-interactive ``-p`` mode. No tool-granting flag is
        added unless ``allow_all_tools`` is explicitly enabled — Python owns all
        writes, so Copilot is not asked to touch the filesystem.
        """
        command = [self.executable, "-p", prompt]
        if self.model:
            command += ["--model", self.model]
        if self.allow_all_tools:
            command.append(self.allow_all_tools_flag)
        command += list(self.extra_args)
        return command


def is_copilot_cli_selected(env: Optional[Mapping[str, str]] = None) -> bool:
    """True when ``TECHDOCKER_BASE_SUMMARY_PROVIDER=copilot-cli`` is set."""
    import os

    env = env if env is not None else os.environ
    return (env.get(BASE_SUMMARY_PROVIDER_ENV_VAR, "").strip().lower()
            == COPILOT_CLI_PROVIDER_NAME)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------
class CopilotCliSummaryProvider:
    """Generate the baseline summary Markdown via the Copilot CLI.

    Produces Markdown or raises a typed :class:`CopilotProviderError`. It never
    writes repository files and never validates content — the caller (the
    baseline initializer) validates and atomically writes accepted output.
    """

    name = COPILOT_CLI_PROVIDER_NAME

    def __init__(
        self,
        config: Optional[CopilotConfig] = None,
        env: Optional[Mapping[str, str]] = None,
        runner: Optional[CopilotRunner] = None,
        prompt_builder: Callable[[RepoContext], str] = build_base_summary_prompt,
        system_prompt: Optional[str] = BASE_SUMMARY_SYSTEM_PROMPT,
    ) -> None:
        self.config = config or CopilotConfig.from_env(env)
        self.runner = runner or _default_runner
        self.prompt_builder = prompt_builder
        self.system_prompt = system_prompt

    def generate_summary(self, context: RepoContext) -> str:
        prompt = self.prompt_builder(context)
        full_prompt = (
            f"{self.system_prompt}\n\n{prompt}" if self.system_prompt else prompt
        )
        command = self.config.build_command(full_prompt)

        try:
            result = self.runner(command, self.config.timeout_seconds)
        except FileNotFoundError as error:
            raise CopilotUnavailableError(
                f"Copilot CLI executable {self.config.executable!r} was not found "
                "on PATH."
            ) from error
        except subprocess.TimeoutExpired as error:
            raise CopilotTimeoutError(
                f"Copilot CLI timed out after {self.config.timeout_seconds:g}s."
            ) from error
        except OSError as error:
            raise CopilotUnavailableError(
                f"Copilot CLI could not be executed: {error}."
            ) from error

        if result.returncode != 0:
            snippet = _sanitize_diagnostic(result.stderr)
            if _looks_like_auth_failure(result.stderr):
                raise CopilotUnavailableError(
                    f"Copilot CLI authentication failed (exit {result.returncode}): "
                    f"{snippet}"
                )
            raise CopilotExecutionError(
                f"Copilot CLI exited with status {result.returncode}: {snippet}"
            )

        text = _strip_outer_fences(result.stdout)
        if not text.strip():
            raise CopilotEmptyOutputError(
                "Copilot CLI returned empty output; no Markdown was produced."
            )
        return text if text.endswith("\n") else text + "\n"
