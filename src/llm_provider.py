"""LLM provider abstraction for TechDocker (Phase 1: local/dev only).

The active pipeline stays deterministic: :class:`DeterministicLLMProvider` is
always available, needs no network or tokens, and is the default. Ollama is
an *optional* local provider for development; when it is unreachable the
call transparently falls back to the deterministic provider — unless strict
mode (``TECHDOCKER_LLM_STRICT=true``) is enabled, in which case a clear error
is raised instead.

Environment variables:

- ``TECHDOCKER_LLM_PROVIDER``: ``deterministic`` (default) or ``ollama``
- ``TECHDOCKER_OLLAMA_MODEL``: default ``qwen2.5-coder:3b``
- ``TECHDOCKER_OLLAMA_HOST``: default ``http://localhost:11434``
- ``TECHDOCKER_LLM_STRICT``: ``true`` to fail instead of falling back

Only the Python standard library is used (``urllib.request``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol
from urllib import error as urllib_error
from urllib import request as urllib_request

DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:3b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

PROVIDER_ENV_VAR = "TECHDOCKER_LLM_PROVIDER"
MODEL_ENV_VAR = "TECHDOCKER_OLLAMA_MODEL"
HOST_ENV_VAR = "TECHDOCKER_OLLAMA_HOST"
STRICT_ENV_VAR = "TECHDOCKER_LLM_STRICT"
TIMEOUT_ENV_VAR = "TECHDOCKER_OLLAMA_TIMEOUT"
TEMPERATURE_ENV_VAR = "TECHDOCKER_OLLAMA_TEMPERATURE"

# Local 7b models can take minutes to load and generate a full summary.
DEFAULT_TIMEOUT_SECONDS = 180.0

# Low temperature: structured technical output should be near-deterministic.
DEFAULT_TEMPERATURE = 0.1
_TEMPERATURE_BOUNDS = (0.0, 2.0)


class LLMUnavailableError(RuntimeError):
    """Raised in strict mode when the configured LLM cannot be reached."""


class LLMOutputValidationError(RuntimeError):
    """Raised in strict mode when LLM output fails validation.

    Distinct from :class:`LLMUnavailableError`: the provider responded, but
    what it produced was malformed or ungrounded.
    """


@dataclass
class LLMMessage:
    """One message of a conversation (reserved for future chat use)."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    """The result of one generation call."""

    text: str
    provider_name: str
    model: Optional[str] = None
    fallback_used: bool = False


@dataclass
class LLMProviderConfig:
    """Configuration resolved from the environment."""

    provider: str = "deterministic"
    model: str = DEFAULT_OLLAMA_MODEL
    host: str = DEFAULT_OLLAMA_HOST
    strict: bool = False
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    temperature: float = DEFAULT_TEMPERATURE

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "LLMProviderConfig":
        env = env if env is not None else os.environ
        try:
            timeout = float(env.get(TIMEOUT_ENV_VAR, "") or DEFAULT_TIMEOUT_SECONDS)
        except ValueError:
            timeout = DEFAULT_TIMEOUT_SECONDS
        try:
            temperature = float(
                env.get(TEMPERATURE_ENV_VAR, "") or DEFAULT_TEMPERATURE
            )
        except ValueError:
            temperature = DEFAULT_TEMPERATURE
        low, high = _TEMPERATURE_BOUNDS
        temperature = min(max(temperature, low), high)
        return cls(
            provider=env.get(PROVIDER_ENV_VAR, "deterministic").strip().lower()
            or "deterministic",
            model=env.get(MODEL_ENV_VAR, "").strip() or DEFAULT_OLLAMA_MODEL,
            host=(env.get(HOST_ENV_VAR, "").strip() or DEFAULT_OLLAMA_HOST).rstrip("/"),
            strict=env.get(STRICT_ENV_VAR, "").strip().lower() in ("true", "1", "yes"),
            timeout_seconds=timeout,
            temperature=temperature,
        )


class LLMProvider(Protocol):
    """Interface every provider implements."""

    name: str

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_schema: Optional[dict] = None,
    ) -> LLMResponse:
        """Generate a completion for the prompt."""
        ...


class DeterministicLLMProvider:
    """Always-available, test-safe provider with predictable output.

    It does not attempt to imitate a real model: callers that require valid
    structured suggestions will fail validation and use their rule-based
    fallback, which is exactly the intended CI behaviour.
    """

    name = "deterministic"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_schema: Optional[dict] = None,
    ) -> LLMResponse:
        payload = {
            "provider": self.name,
            "note": "Deterministic placeholder output; no LLM was called.",
            "prompt_chars": len(prompt or ""),
        }
        return LLMResponse(
            text=json.dumps(payload),
            provider_name=self.name,
            model=None,
        )


class OllamaLLMProvider:
    """Optional local provider using Ollama's HTTP API.

    Connection failures fall back to :class:`DeterministicLLMProvider`
    unless ``config.strict`` is set, in which case
    :class:`LLMUnavailableError` is raised with a clear message.
    """

    name = "ollama"

    def __init__(
        self,
        config: Optional[LLMProviderConfig] = None,
        fallback: Optional[LLMProvider] = None,
    ) -> None:
        self.config = config or LLMProviderConfig(provider="ollama")
        self.fallback = fallback or DeterministicLLMProvider()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_schema: Optional[dict] = None,
    ) -> LLMResponse:
        payload: dict = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            # Ollama defaults to a small context window (4096); long prompts
            # would be silently truncated and the model would lose its
            # instructions. 8192 fits our bounded prompts comfortably.
            "options": {
                "num_ctx": 8192,
                "temperature": self.config.temperature,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt
        if json_schema is not None:
            # Ollama constrains output to valid JSON with format=json.
            payload["format"] = "json"

        request = urllib_request.Request(
            f"{self.config.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib_request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
            return LLMResponse(
                text=body.get("response", ""),
                provider_name=self.name,
                model=self.config.model,
            )
        except (urllib_error.URLError, OSError, TimeoutError, ValueError) as err:
            if self.config.strict:
                raise LLMUnavailableError(
                    f"Ollama is unavailable at {self.config.host} "
                    f"(model {self.config.model!r}): {err}. "
                    "Strict mode is enabled (TECHDOCKER_LLM_STRICT), so no "
                    "fallback was attempted. Start Ollama with e.g. "
                    f"'ollama run {self.config.model}' or unset strict mode."
                ) from err
            fallback_response = self.fallback.generate(
                prompt, system_prompt=system_prompt, json_schema=json_schema
            )
            fallback_response.fallback_used = True
            fallback_response.provider_name = (
                f"{self.fallback.name} (fallback: ollama unavailable)"
            )
            return fallback_response


def get_llm_provider_from_env(
    env: Mapping[str, str] | None = None,
) -> LLMProvider:
    """Select the provider from ``TECHDOCKER_LLM_PROVIDER``.

    ``deterministic`` or unset -> :class:`DeterministicLLMProvider`;
    ``ollama`` -> :class:`OllamaLLMProvider` (deterministic fallback unless
    strict mode is enabled).
    """
    config = LLMProviderConfig.from_env(env)
    if config.provider == "ollama":
        return OllamaLLMProvider(config)
    return DeterministicLLMProvider()
