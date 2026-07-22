"""Tests for the optional LLM layer: providers, analyzer validation, updater.

No network calls, no Ollama required, no tokens. HTTP is mocked at the
urllib level; env vars via monkeypatch.
"""

from __future__ import annotations

import io
import json
from urllib import error as urllib_error

import pytest

from src import llm_provider as llm_provider_module
from src import summary_updater
from src.git_change_detector import ChangedFile, GitChangeSet
from src.llm_change_analyzer import (
    SELECT_EXISTING,
    LLMChangeSuggestion,
    LLMSectionSelection,
    analyze_change,
    fallback_suggestion,
    parse_and_validate_suggestion,
)
from src.llm_provider import (
    DeterministicLLMProvider,
    LLMProviderConfig,
    LLMResponse,
    LLMUnavailableError,
    OllamaLLMProvider,
    get_llm_provider_from_env,
)
from src.project_summary_generator import (
    generate_original_summary,
    original_summary_path,
)
from src.summary_change_router import CREATE_NEW, UPDATE_EXISTING
from src.summary_skeleton_store import SummarySkeleton, append_section
from src.summary_updater import run_update

CI_ENV = {
    "GITHUB_REPOSITORY": "AdrenalIsland1/TechDocker",
    "GITHUB_REF_NAME": "main",
    "GITHUB_SHA": "def456",
    "GITHUB_ACTOR": "Vaibhav",
    "GITHUB_EVENT_BEFORE": "abc123",
}


def make_skeleton():
    skeleton = SummarySkeleton(
        project_id="techdocker", source_summary_path="s.md", generated_at="now"
    )
    for heading in ("System Overview", "Core Modules", "Testing Strategy"):
        append_section(skeleton, heading, level=2)
    return skeleton


class FakeProvider:
    """Test provider returning a canned response."""

    name = "fake"

    def __init__(self, text):
        self.text = text

    def generate(self, prompt, system_prompt=None, json_schema=None):
        return LLMResponse(text=self.text, provider_name=self.name)


def valid_suggestion_json(**overrides):
    data = {
        "decision": UPDATE_EXISTING,
        "target_section_id": "core-modules",
        "target_heading": "Core Modules",
        "requires_new_section": False,
        "confidence": 0.9,
        "suggested_summary": "Refactored the core module.",
        "reasoning": "Change touches core source files.",
    }
    data.update(overrides)
    return json.dumps(data)


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------
def test_deterministic_provider_works():
    response = DeterministicLLMProvider().generate("hello", json_schema={})
    assert response.provider_name == "deterministic"
    assert response.fallback_used is False
    assert json.loads(response.text)["provider"] == "deterministic"


def test_env_selection_defaults_to_deterministic():
    assert isinstance(get_llm_provider_from_env({}), DeterministicLLMProvider)
    provider = get_llm_provider_from_env({"TECHDOCKER_LLM_PROVIDER": "ollama"})
    assert isinstance(provider, OllamaLLMProvider)


def test_ollama_falls_back_to_deterministic_when_unavailable(monkeypatch):
    def refuse(*args, **kwargs):
        raise urllib_error.URLError("connection refused")

    monkeypatch.setattr(llm_provider_module.urllib_request, "urlopen", refuse)

    provider = OllamaLLMProvider(LLMProviderConfig(provider="ollama"))
    response = provider.generate("prompt", json_schema={})

    assert response.fallback_used is True
    assert "fallback" in response.provider_name
    assert json.loads(response.text)["provider"] == "deterministic"


def test_strict_mode_raises_clear_error_when_ollama_unavailable(monkeypatch):
    def refuse(*args, **kwargs):
        raise urllib_error.URLError("connection refused")

    monkeypatch.setattr(llm_provider_module.urllib_request, "urlopen", refuse)

    provider = OllamaLLMProvider(
        LLMProviderConfig(provider="ollama", strict=True)
    )
    with pytest.raises(LLMUnavailableError, match="Ollama is unavailable"):
        provider.generate("prompt")


def test_ollama_success_parses_response(monkeypatch):
    body = json.dumps({"response": valid_suggestion_json()}).encode()

    class FakeHTTPResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        llm_provider_module.urllib_request,
        "urlopen",
        lambda *a, **k: FakeHTTPResponse(body),
    )

    provider = OllamaLLMProvider(LLMProviderConfig(provider="ollama"))
    response = provider.generate("prompt", json_schema={})
    assert response.provider_name == "ollama"
    assert json.loads(response.text)["decision"] == UPDATE_EXISTING


# ---------------------------------------------------------------------------
# analyzer validation
# ---------------------------------------------------------------------------
def test_valid_llm_json_with_existing_section_is_accepted():
    suggestion = analyze_change(
        [ChangedFile("src/core.py", "modified")],
        "core changed",
        make_skeleton(),
        provider=FakeProvider(valid_suggestion_json()),
    )
    assert suggestion is not None
    assert suggestion.decision == UPDATE_EXISTING
    assert suggestion.target_section_id == "core-modules"
    assert suggestion.target_heading == "Core Modules"  # from skeleton
    assert suggestion.confidence == 0.9


def test_invalid_llm_json_returns_none():
    suggestion = analyze_change(
        [], "summary", make_skeleton(), provider=FakeProvider("not json at all")
    )
    assert suggestion is None


def test_invalid_target_section_id_is_rejected():
    text = valid_suggestion_json(target_section_id="does-not-exist")
    suggestion = analyze_change(
        [], "summary", make_skeleton(), provider=FakeProvider(text)
    )
    assert suggestion is None


def test_out_of_range_confidence_is_rejected():
    for bad in (1.5, -0.1, "high", None):
        text = valid_suggestion_json(confidence=bad)
        assert (
            analyze_change([], "s", make_skeleton(), provider=FakeProvider(text))
            is None
        )


def test_invalid_decision_is_rejected():
    text = valid_suggestion_json(decision="delete_everything")
    assert (
        analyze_change([], "s", make_skeleton(), provider=FakeProvider(text)) is None
    )


def test_create_new_requires_heading_and_ignores_section_id():
    text = valid_suggestion_json(
        decision=CREATE_NEW, target_heading="Security", target_section_id="x"
    )
    suggestion = parse_and_validate_suggestion(text, make_skeleton())
    assert suggestion.requires_new_section is True
    assert suggestion.target_section_id is None
    assert suggestion.target_heading == "Security"


def test_fallback_suggestion_uses_rule_router():
    suggestion = fallback_suggestion(
        [ChangedFile("tests/test_x.py", "modified")], "summary", make_skeleton()
    )
    assert suggestion.decision == UPDATE_EXISTING
    assert suggestion.target_heading == "Testing Strategy"
    assert suggestion.confidence == 1.0


# ---------------------------------------------------------------------------
# updater integration
# ---------------------------------------------------------------------------
def make_repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "README.md").write_text("# Demo\n\nDemo project.\n")
    (tmp_path / "src" / "core.py").write_text("def run():\n    return 1\n")
    (tmp_path / "tests" / "test_core.py").write_text("def test_run():\n    pass\n")
    return tmp_path


def mock_detector(monkeypatch, files):
    def fake_build_change_set(**kwargs):
        return GitChangeSet(
            repository=kwargs["repository"],
            branch=kwargs["branch"],
            before_sha=kwargs["before_sha"],
            after_sha=kwargs["after_sha"],
            changed_files=files,
        )

    monkeypatch.setattr(summary_updater, "build_change_set", fake_build_change_set)


def llm_selection(confidence=0.9, section_id=None, heading=None):
    """A validated shortlist selection (the updater's only LLM entry point)."""
    return LLMSectionSelection(
        decision=SELECT_EXISTING,
        section_id=section_id or "project-technical-summary-core-modules",
        heading=heading or "Core Modules",
        confidence=confidence,
        reasoning="Core source changed.",
    )


def capture_selection(monkeypatch, result):
    """Patch the shortlist selector and record the candidates it received."""
    seen: dict = {}

    def fake_select(change_summary, candidates, **kwargs):
        seen["candidates"] = candidates
        seen["kwargs"] = kwargs
        return result

    monkeypatch.setattr(summary_updater, "select_section_with_llm", fake_select)
    return seen


def forbid_legacy_analyzer(monkeypatch):
    """The unrestricted legacy analyzer must never run in normal routing."""
    def must_not_be_called(*args, **kwargs):
        raise AssertionError("legacy analyze_change must not be used")

    monkeypatch.setattr(
        summary_updater.llm_change_analyzer_module
        if hasattr(summary_updater, "llm_change_analyzer_module")
        else summary_updater,
        "analyze_change",
        must_not_be_called,
        raising=False,
    )


def test_updater_without_llm_env_never_calls_selector(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    mock_detector(monkeypatch, [ChangedFile("src/core.py", "modified")])

    def must_not_be_called(*args, **kwargs):
        raise AssertionError("the LLM selector must not run without LLM env")

    monkeypatch.setattr(
        summary_updater, "select_section_with_llm", must_not_be_called
    )
    forbid_legacy_analyzer(monkeypatch)

    result = run_update(CI_ENV, repo_path=str(repo))  # no TECHDOCKER_LLM_* vars
    assert result.routing_source == "rule_based"


def test_updater_uses_shortlist_selector_not_legacy_analyzer(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    generate_original_summary(repo)
    mock_detector(monkeypatch, [ChangedFile("src/core.py", "modified")])
    seen = capture_selection(monkeypatch, llm_selection(0.9))
    forbid_legacy_analyzer(monkeypatch)

    env = dict(CI_ENV, TECHDOCKER_LLM_PROVIDER="ollama")
    result = run_update(env, repo_path=str(repo))

    assert result.routing_source == "llm"
    assert result.llm_confidence == 0.9
    assert result.decision.target_heading == "Core Modules"
    # The selector received the deterministic shortlist (<= 3 real candidates).
    candidates = seen["candidates"]
    assert 0 < len(candidates) <= 3
    assert all(hasattr(c, "section_id") for c in candidates)
    # Bounded prompt facts, including omission metadata.
    assert "additional_files_omitted" in seen["kwargs"]


def test_updater_low_confidence_falls_back_to_deterministic(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    generate_original_summary(repo)
    mock_detector(monkeypatch, [ChangedFile("tests/test_core.py", "modified")])
    capture_selection(monkeypatch, llm_selection(0.4))
    forbid_legacy_analyzer(monkeypatch)

    env = dict(CI_ENV, TECHDOCKER_LLM_PROVIDER="ollama")
    result = run_update(env, repo_path=str(repo))

    assert result.routing_source == "rule_based"
    assert result.decision.target_heading == "Testing Strategy"  # deterministic won
    assert any("below the threshold" in w for w in result.warnings)


def test_updater_invalid_llm_result_falls_back_to_deterministic(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    generate_original_summary(repo)
    mock_detector(monkeypatch, [ChangedFile("tests/test_core.py", "modified")])
    capture_selection(monkeypatch, None)
    forbid_legacy_analyzer(monkeypatch)

    env = dict(CI_ENV, TECHDOCKER_LLM_PROVIDER="ollama")
    result = run_update(env, repo_path=str(repo))

    assert result.routing_source == "rule_based"
    assert result.decision.target_heading == "Testing Strategy"
    assert any("unavailable or invalid" in w for w in result.warnings)


def test_updater_with_llm_still_never_modifies_original(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    generate_original_summary(repo)
    baseline = original_summary_path(repo).read_bytes()

    mock_detector(monkeypatch, [ChangedFile("src/core.py", "modified")])
    capture_selection(monkeypatch, llm_selection(0.95))

    env = dict(CI_ENV, TECHDOCKER_LLM_PROVIDER="ollama")
    run_update(env, repo_path=str(repo))

    assert original_summary_path(repo).read_bytes() == baseline


def test_github_actions_path_runs_without_ollama(tmp_path, monkeypatch):
    """CI has no TECHDOCKER_LLM_* env: the full run must succeed untouched."""
    repo = make_repo(tmp_path)
    mock_detector(monkeypatch, [ChangedFile("src/core.py", "modified")])
    for var in (
        "TECHDOCKER_LLM_PROVIDER",
        "TECHDOCKER_OLLAMA_MODEL",
        "TECHDOCKER_OLLAMA_HOST",
        "TECHDOCKER_LLM_STRICT",
    ):
        monkeypatch.delenv(var, raising=False)

    result = run_update(CI_ENV, repo_path=str(repo))

    assert result.routing_source == "rule_based"
    assert result.updated_summary.exists()
    assert result.skeleton_path.exists()
