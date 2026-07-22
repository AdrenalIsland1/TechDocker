"""Tests for the LLM quality layer: base-summary providers and validation.

No network, no Ollama, no tokens — providers are faked or mocked.
"""

from __future__ import annotations

import json
import sys

import pytest

from src.llm_change_analyzer import LLMChangeSuggestion, analyze_change, build_prompt
from src.llm_provider import (
    LLMOutputValidationError,
    LLMResponse,
    LLMUnavailableError,
)
from src.project_summary_generator import (
    SUMMARY_HEADINGS,
    LLMSummaryProvider,
    LocalDeterministicSummaryProvider,
    build_base_summary_prompt,
    extract_project_description,
    generate_original_summary,
    get_summary_provider_for_env,
    original_summary_path,
    prioritized_context_files,
    summary_has_required_headings,
    validate_llm_summary,
)
from src.project_summary_generator import main as generator_main
from src.repo_context_collector import collect_repo_context
from src.summary_change_router import CREATE_NEW, UPDATE_EXISTING
from src.summary_skeleton_store import SummarySkeleton, append_section

VALID_LLM_SUMMARY = (
    "# Project Technical Summary\n\n"
    + "\n\n".join(f"## {h}\n\nSpecific {h} details from the LLM." for h in SUMMARY_HEADINGS)
    + "\n"
)


class FakeLLM:
    name = "fake"

    def __init__(self, text, fallback_used=False):
        self.text = text
        self.fallback_used = fallback_used

    def generate(self, prompt, system_prompt=None, json_schema=None):
        return LLMResponse(
            text=self.text, provider_name=self.name, fallback_used=self.fallback_used
        )


def make_repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text("# Demo\n\nDemo project.\n")
    (tmp_path / "src" / "core.py").write_text("def run():\n    return 1\n")
    return tmp_path


def make_skeleton():
    skeleton = SummarySkeleton(
        project_id="techdocker", source_summary_path="s.md", generated_at="now"
    )
    for heading in ("System Overview", "Core Modules", "Automation Pipeline",
                    "Testing Strategy"):
        append_section(skeleton, heading, level=2)
    return skeleton


def suggestion_json(**overrides):
    data = {
        "decision": UPDATE_EXISTING,
        "target_section_id": "core-modules",
        "target_heading": "Core Modules",
        "requires_new_section": False,
        "confidence": 0.9,
        "suggested_summary": "Refactored the run-aggregation helper for clarity.",
        "reasoning": "Core source changed.",
    }
    data.update(overrides)
    return json.dumps(data)


# ---------------------------------------------------------------------------
# base summary providers
# ---------------------------------------------------------------------------
def test_deterministic_base_summary_still_works(tmp_path):
    repo = make_repo(tmp_path)
    path = generate_original_summary(repo)  # default provider
    text = path.read_text(encoding="utf-8")
    assert summary_has_required_headings(text)
    assert "deterministic local provider" in text


def test_env_selects_summary_provider():
    assert isinstance(
        get_summary_provider_for_env({}), LocalDeterministicSummaryProvider
    )
    assert isinstance(
        get_summary_provider_for_env({"TECHDOCKER_LLM_PROVIDER": "ollama"}),
        LLMSummaryProvider,
    )


def test_llm_base_summary_with_mocked_provider(tmp_path):
    repo = make_repo(tmp_path)
    provider = LLMSummaryProvider(env={}, llm=FakeLLM(VALID_LLM_SUMMARY))

    path = generate_original_summary(repo, provider=provider)

    text = path.read_text(encoding="utf-8")
    assert "Specific System Overview details from the LLM." in text
    assert summary_has_required_headings(text)


def test_llm_base_summary_falls_back_when_headings_missing(tmp_path):
    repo = make_repo(tmp_path)
    provider = LLMSummaryProvider(env={}, llm=FakeLLM("## Wrong structure only"))

    context = collect_repo_context(repo)
    text = provider.generate_summary(context)

    assert summary_has_required_headings(text)
    assert "deterministic local provider" in text  # fallback content


def test_llm_base_summary_falls_back_when_provider_fell_back(tmp_path):
    repo = make_repo(tmp_path)
    provider = LLMSummaryProvider(
        env={}, llm=FakeLLM('{"provider": "deterministic"}', fallback_used=True)
    )
    text = provider.generate_summary(collect_repo_context(repo))
    assert summary_has_required_headings(text)


def test_llm_base_summary_strict_mode_raises_validation_error(tmp_path):
    repo = make_repo(tmp_path)
    provider = LLMSummaryProvider(
        env={"TECHDOCKER_LLM_STRICT": "true"},
        llm=FakeLLM("## Wrong structure only"),
    )
    with pytest.raises(LLMOutputValidationError, match="failed validation"):
        provider.generate_summary(collect_repo_context(repo))


def test_force_still_required_to_overwrite(tmp_path):
    repo = make_repo(tmp_path)
    path = generate_original_summary(repo)
    path.write_text("# Custom baseline\n", encoding="utf-8")

    generate_original_summary(
        repo, provider=LLMSummaryProvider(env={}, llm=FakeLLM(VALID_LLM_SUMMARY))
    )
    assert path.read_text(encoding="utf-8") == "# Custom baseline\n"

    generate_original_summary(
        repo,
        force=True,
        provider=LLMSummaryProvider(env={}, llm=FakeLLM(VALID_LLM_SUMMARY)),
    )
    assert "Specific System Overview details" in path.read_text(encoding="utf-8")


def test_preview_does_not_write_files(tmp_path, monkeypatch, capsys):
    repo = make_repo(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["project_summary_generator", "--preview", "--repo-path", str(repo)]
    )
    monkeypatch.delenv("TECHDOCKER_LLM_PROVIDER", raising=False)

    assert generator_main() == 0

    captured = capsys.readouterr()
    # stdout carries only the previewed Markdown; diagnostics go to stderr.
    assert "# Project Technical Summary" in captured.out
    assert "No files were written" not in captured.out
    assert "No files were written" in captured.err
    assert not original_summary_path(repo).exists()
    assert not (repo / "artifacts").exists()


# ---------------------------------------------------------------------------
# quality validation in the analyzer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "heading", ["Code Changes", "Updates", "Changes", "Miscellaneous", "misc"]
)
def test_generic_new_headings_are_rejected(heading):
    text = suggestion_json(
        decision=CREATE_NEW, target_heading=heading, target_section_id=None
    )
    result = analyze_change([], "summary", make_skeleton(), provider=FakeLLM(text))
    assert result is None


def test_specific_new_heading_is_still_accepted():
    text = suggestion_json(
        decision=CREATE_NEW, target_heading="Security Model", target_section_id=None
    )
    result = analyze_change([], "summary", make_skeleton(), provider=FakeLLM(text))
    assert result is not None
    assert result.target_heading == "Security Model"


@pytest.mark.parametrize(
    "summary",
    ["", "short", "Repository structure was updated.", "the code was changed"],
)
def test_generic_or_empty_summaries_are_rejected(summary):
    text = suggestion_json(suggested_summary=summary)
    result = analyze_change([], "summary", make_skeleton(), provider=FakeLLM(text))
    assert result is None


def test_prompt_contains_routing_and_quality_guidance():
    prompt = build_prompt([], "summary", make_skeleton())
    assert "Testing Strategy" in prompt
    assert "Deployment and CI" in prompt
    assert "Automation Pipeline" in prompt
    assert '"Code Changes"' in prompt  # anti-generic instruction
    assert "Examples of good outputs" in prompt  # few-shot block


# ---------------------------------------------------------------------------
# deterministic description grounding (no hardcoded product claims)
# ---------------------------------------------------------------------------
def test_deterministic_overview_uses_readme_description(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "README.md").write_text(
        "# Alpha\n\nAlpha is a routing service for widget telemetry.\n"
    )
    text = LocalDeterministicSummaryProvider().generate_summary(
        collect_repo_context(repo)
    )
    assert "Alpha is a routing service for widget telemetry." in text
    assert "DOCX technical documents" not in text  # nothing hardcoded


def test_deterministic_overview_neutral_fallback_without_readme(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "README.md").unlink()
    text = LocalDeterministicSummaryProvider().generate_summary(
        collect_repo_context(repo)
    )
    assert "No project description was found" in text


def test_extract_description_skips_headings_badges_and_fences(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "README.md").write_text(
        "# Title\n\n[![badge](x)](y)\n\n```bash\nnot this\n```\n\n"
        "The real description line.\nSecond line of it.\n\nNext paragraph.\n"
    )
    description = extract_project_description(collect_repo_context(repo))
    assert description == "The real description line. Second line of it."


# ---------------------------------------------------------------------------
# variable-heading validation
# ---------------------------------------------------------------------------
def make_summary(
    headings,
    content=(
        "This section documents real observed behaviour of the repository "
        "in enough detail to be a useful technical reference for reviewers."
    ),
    title="# Demo Technical Summary",
):
    parts = [title, ""]
    for heading in headings:
        parts += [f"## {heading}", "", content, ""]
    return "\n".join(parts)


VARIABLE_HEADINGS = [
    "Architecture", "Quality and Tests", "CI/CD Review Flow",
    "Repository Automation", "Dependencies and Environment",
]


def test_variable_meaningful_headings_are_accepted():
    result = validate_llm_summary(make_summary(VARIABLE_HEADINGS))
    assert result.ok, result.problems


def test_missing_h1_is_rejected():
    text = make_summary(VARIABLE_HEADINGS, title="No title here, just prose.")
    result = validate_llm_summary(text)
    assert not result.ok
    assert any("H1" in p for p in result.problems)


def test_too_few_h2_sections_rejected():
    result = validate_llm_summary(make_summary(["Architecture", "Testing"]))
    assert not result.ok
    assert any("sections" in p for p in result.problems)


def test_duplicate_normalized_headings_rejected():
    result = validate_llm_summary(
        make_summary(["Architecture", "architecture:", "Tests", "CI", "Config"])
    )
    assert not result.ok
    assert any("duplicate" in p for p in result.problems)


def test_generic_headings_rejected_in_summary():
    result = validate_llm_summary(
        make_summary(["Architecture", "Code Changes", "Tests", "CI", "Config"])
    )
    assert not result.ok
    assert any("generic" in p.lower() for p in result.problems)


def test_empty_sections_rejected():
    text = make_summary(VARIABLE_HEADINGS[:4]) + "\n## Hollow Section\n\n"
    result = validate_llm_summary(text)
    assert not result.ok
    assert any("Hollow Section" in p for p in result.problems)


def test_whole_document_fence_is_stripped_then_validated():
    fenced = "```markdown\n" + make_summary(VARIABLE_HEADINGS) + "\n```"
    result = validate_llm_summary(fenced)
    assert result.ok, result.problems


def test_empty_summary_rejected():
    result = validate_llm_summary("")
    assert not result.ok
    assert result.problems == ["summary is empty"]


# ---------------------------------------------------------------------------
# conservative grounding checks
# ---------------------------------------------------------------------------
def test_unsupported_ml_nlp_claims_rejected(tmp_path):
    repo = make_repo(tmp_path)  # contains no ML/NLP evidence
    context = collect_repo_context(repo)
    text = make_summary(
        VARIABLE_HEADINGS,
        content="The system uses NLP and machine learning to score headings.",
    )
    result = validate_llm_summary(text, context)
    assert not result.ok
    assert any("machine learning" in p for p in result.problems)
    assert any("'nlp'" in p for p in result.problems)


def test_supported_technology_claims_accepted(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "requirements.txt").write_text("scikit-learn\nspacy\n")
    context = collect_repo_context(repo)
    text = make_summary(
        VARIABLE_HEADINGS,
        content="The system uses NLP and machine learning to score headings.",
    )
    result = validate_llm_summary(text, context)
    assert result.ok, result.problems


def _make_repo_with_circular_mentions(tmp_path):
    """Repo that mentions ML/NLP only in grounding machinery and tests.

    Mirrors TechDocker summarizing itself: the validator/prompt module and the
    quality test file contain the phrases, but no dependency or import proves
    the project actually uses those technologies.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "requirements.txt").write_text("pytest\npandas\npython-docx\n")
    (tmp_path / "README.md").write_text(
        "# Demo\n\nDemo is a deterministic documentation pipeline.\n"
    )
    # Grounding machinery: mentions the phrases to reason ABOUT them.
    (tmp_path / "src" / "project_summary_generator.py").write_text(
        'CLAIMS = {"machine learning": (...), "nlp": (...)}\n'
        '# reject unsupported "machine learning" / "natural language processing"\n'
    )
    # Test fixtures: mocked summaries claiming the technologies.
    (tmp_path / "tests" / "test_llm_quality.py").write_text(
        'BAD = "The system uses machine learning and NLP."\n'
    )
    # Real, ML-free implementation.
    (tmp_path / "src" / "core.py").write_text("def run():\n    return 1\n")
    return tmp_path


def test_circular_mentions_do_not_ground_ml_nlp_claims(tmp_path):
    # Regression: the phrases exist only in the validator module and the test
    # file (both excluded from evidence), with no real dependency/import.
    repo = _make_repo_with_circular_mentions(tmp_path)
    context = collect_repo_context(repo)

    # Sanity: the phrases really are present in the collected context...
    corpus = "\n".join(context.files.values()).lower()
    assert "machine learning" in corpus and "nlp" in corpus
    # ...but they come only from excluded files, so a claim is still rejected.
    text = make_summary(
        VARIABLE_HEADINGS,
        content="The project uses machine learning and NLP internally.",
    )
    result = validate_llm_summary(text, context)
    assert not result.ok
    assert any("machine learning" in p for p in result.problems)
    assert any("'nlp'" in p for p in result.problems)


GROUNDED_ML_CONTENT = (
    "Gradient boosting with machine learning ranks documents; this section "
    "describes the observed behaviour in enough detail to be a real reference."
)


def test_dependency_evidence_accepts_ml_claim(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "requirements.txt").write_text("pytest\nxgboost>=1.7\n")
    context = collect_repo_context(repo)
    text = make_summary(VARIABLE_HEADINGS, content=GROUNDED_ML_CONTENT)
    assert validate_llm_summary(text, context).ok


def test_import_evidence_accepts_ml_claim(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "src" / "model.py").write_text("import torch\n\nMODEL = None\n")
    context = collect_repo_context(repo)
    text = make_summary(
        VARIABLE_HEADINGS,
        content=(
            "A neural network trained with machine learning powers ranking; "
            "this section describes it in enough detail to be a real reference."
        ),
    )
    result = validate_llm_summary(text, context)
    assert result.ok, result.problems


def test_import_in_test_file_is_not_evidence(tmp_path):
    # An `import torch` living in a test fixture must not ground a claim.
    repo = make_repo(tmp_path)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_model.py").write_text("import torch\n")
    context = collect_repo_context(repo)
    text = make_summary(
        VARIABLE_HEADINGS, content="The project uses machine learning throughout."
    )
    assert not validate_llm_summary(text, context).ok


def test_affirmative_readme_grounds_claim(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "README.md").write_text(
        "# Demo\n\nDemo uses machine learning to rank documents.\n"
    )
    context = collect_repo_context(repo)
    text = make_summary(VARIABLE_HEADINGS, content=GROUNDED_ML_CONTENT)
    assert validate_llm_summary(text, context).ok


def test_negative_documentation_is_not_evidence(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "README.md").write_text(
        "# Demo\n\nThis project does not use machine learning.\n"
    )
    context = collect_repo_context(repo)
    text = make_summary(
        VARIABLE_HEADINGS, content="The project uses machine learning to decide."
    )
    result = validate_llm_summary(text, context)
    assert not result.ok
    assert any("machine learning" in p for p in result.problems)


def test_llm_api_use_alone_is_not_ml_evidence(tmp_path):
    # Calling an LLM/Ollama HTTP API must not count as the repo implementing ML.
    repo = make_repo(tmp_path)
    (repo / "src" / "provider.py").write_text(
        "import urllib.request\n\n# calls the Ollama HTTP API\n"
    )
    context = collect_repo_context(repo)
    text = make_summary(
        VARIABLE_HEADINGS, content="TechDocker implements machine learning models."
    )
    assert not validate_llm_summary(text, context).ok


def test_ungrounded_llm_output_falls_back_non_strict(tmp_path):
    repo = make_repo(tmp_path)
    bad = make_summary(
        VARIABLE_HEADINGS, content="Uses machine learning for everything."
    )
    provider = LLMSummaryProvider(env={}, llm=FakeLLM(bad))
    text = provider.generate_summary(collect_repo_context(repo))
    assert "deterministic local provider" in text  # fell back


def test_ungrounded_llm_output_raises_in_strict_mode(tmp_path):
    repo = make_repo(tmp_path)
    bad = make_summary(
        VARIABLE_HEADINGS, content="Uses machine learning for everything."
    )
    provider = LLMSummaryProvider(
        env={"TECHDOCKER_LLM_STRICT": "true"}, llm=FakeLLM(bad)
    )
    with pytest.raises(LLMOutputValidationError, match="machine learning"):
        provider.generate_summary(collect_repo_context(repo))


# ---------------------------------------------------------------------------
# context prioritization and prompt grounding instructions
# ---------------------------------------------------------------------------
def make_context_with(files):
    from src.repo_context_collector import RepoContext

    context = RepoContext(root="/x", project_name="demo")
    context.files = dict(files)
    context.file_tree = list(files)
    context.total_files = len(files)
    return context


SCRAMBLED_FILES = {
    "src/docx_parser.py": "legacy docx parsing",
    "tests/test_core.py": "tests",
    ".github/workflows/ci.yml": "workflow",
    "src/summary_updater.py": "orchestration",
    "requirements.txt": "deps",
    "README.md": "Readme text",
    "config/projects.json": "{}",
}


def test_prompt_context_ordering_is_deterministic():
    context = make_context_with(SCRAMBLED_FILES)
    first = [path for path, _ in prioritized_context_files(context)]
    second = [path for path, _ in prioritized_context_files(context)]
    assert first == second


def test_authoritative_files_precede_legacy_heavy_context():
    context = make_context_with(SCRAMBLED_FILES)
    order = [path for path, _ in prioritized_context_files(context)]
    assert order == [
        "README.md",
        "requirements.txt",
        ".github/workflows/ci.yml",
        "config/projects.json",
        "src/summary_updater.py",
        "src/docx_parser.py",
        "tests/test_core.py",
    ]
    # And the prompt embeds them in that order.
    prompt = build_base_summary_prompt(context)
    assert prompt.index("--- README.md ---") < prompt.index(
        "--- src/summary_updater.py ---"
    ) < prompt.index("--- src/docx_parser.py ---")


def test_prompt_forbids_unsupported_ml_claims_and_invention():
    prompt = build_base_summary_prompt(make_context_with(SCRAMBLED_FILES))
    assert "ONLY evidence" in prompt
    assert '"machine learning"' in prompt
    assert "Do not invent" in prompt
    assert "legacy" in prompt
    assert "Code Changes" in prompt  # anti-generic heading instruction


# ---------------------------------------------------------------------------
# Ollama temperature configuration (mocked, no network)
# ---------------------------------------------------------------------------
def test_ollama_temperature_defaults_to_low():
    from src.llm_provider import LLMProviderConfig

    assert LLMProviderConfig.from_env({}).temperature == 0.1


def test_ollama_temperature_configurable_and_bounded():
    from src.llm_provider import LLMProviderConfig

    env = {"TECHDOCKER_OLLAMA_TEMPERATURE": "0.7"}
    assert LLMProviderConfig.from_env(env).temperature == 0.7
    assert LLMProviderConfig.from_env(
        {"TECHDOCKER_OLLAMA_TEMPERATURE": "99"}
    ).temperature == 2.0  # clamped


def test_ollama_malformed_temperature_uses_default():
    from src.llm_provider import LLMProviderConfig

    env = {"TECHDOCKER_OLLAMA_TEMPERATURE": "hot"}
    assert LLMProviderConfig.from_env(env).temperature == 0.1


def test_ollama_request_includes_temperature(monkeypatch):
    import io
    import json as json_module

    from src import llm_provider as llm_provider_module
    from src.llm_provider import LLMProviderConfig, OllamaLLMProvider

    recorded = {}

    class FakeHTTPResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=None):
        recorded["payload"] = json_module.loads(request.data.decode("utf-8"))
        recorded["timeout"] = timeout
        return FakeHTTPResponse(json_module.dumps({"response": "ok"}).encode())

    monkeypatch.setattr(
        llm_provider_module.urllib_request, "urlopen", fake_urlopen
    )

    provider = OllamaLLMProvider(
        LLMProviderConfig(provider="ollama", temperature=0.3)
    )
    provider.generate("prompt")

    assert recorded["payload"]["options"]["temperature"] == 0.3
    assert recorded["payload"]["options"]["num_ctx"] == 8192
    assert recorded["timeout"] == 180.0
