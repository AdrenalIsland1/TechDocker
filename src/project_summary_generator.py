"""Generate the base technical project summary from repository context.

Run as ``python3 -m src.project_summary_generator`` (add ``--force`` to
regenerate an existing baseline).

Produces ``artifacts/summaries/base_original_summary.md`` — the immutable
baseline of the new pipeline — with deterministic Markdown headings, and
initializes ``base_updated_summary.md`` as a copy when it does not exist yet.
The baseline is never modified during normal update runs; only ``--force``
(or deleting the file) regenerates it.

Providers are pluggable: :class:`LocalDeterministicSummaryProvider` needs no
network or tokens and is the default; :class:`CopilotSummaryProvider` is a
placeholder for the future Copilot/LLM integration and is never required.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Protocol

from src.llm_provider import (
    LLMOutputValidationError,
    LLMProvider,
    LLMProviderConfig,
    LLMUnavailableError,
    get_llm_provider_from_env,
)
from src.markdown_summary_parser import (
    GENERIC_HEADINGS,
    normalize_heading,
    parse_markdown_sections,
)
from src.repo_context_collector import RepoContext, collect_repo_context

SUMMARIES_DIRECTORY = Path("artifacts") / "summaries"
ORIGINAL_SUMMARY_NAME = "base_original_summary.md"
UPDATED_SUMMARY_NAME = "base_updated_summary.md"

# The stable heading set every generated summary must contain.
SUMMARY_HEADINGS = [
    "System Overview",
    "Repository Structure",
    "Core Modules",
    "Automation Pipeline",
    "Testing Strategy",
    "Configuration",
    "Deployment and CI",
    "Known Limitations",
]

COPILOT_TOKEN_ENV_VAR = "TECHDOCKER_COPILOT_TOKEN"


def original_summary_path(repo_path: str | Path = ".") -> Path:
    return Path(repo_path) / SUMMARIES_DIRECTORY / ORIGINAL_SUMMARY_NAME


def updated_summary_path(repo_path: str | Path = ".") -> Path:
    return Path(repo_path) / SUMMARIES_DIRECTORY / UPDATED_SUMMARY_NAME


class SummaryProvider(Protocol):
    """Interface every summary provider implements."""

    def generate_summary(self, context: RepoContext) -> str:
        """Return the full project summary as Markdown."""
        ...


def extract_project_description(context: RepoContext) -> Optional[str]:
    """First prose paragraph of the repository's README, or ``None``.

    Skips headings, badges, and fenced blocks; joins the first run of
    consecutive prose lines; caps the result at ~400 characters. The
    description therefore always comes from the target repository itself —
    nothing project-specific is hardcoded.
    """
    readme_paths = sorted(
        (path for path in context.files if Path(path).name.lower().startswith("readme")),
        key=lambda path: (len(Path(path).parts), path),
    )
    for path in readme_paths:
        in_fence = False
        paragraph: list[str] = []
        for line in context.files[path].splitlines():
            stripped = line.strip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_fence = not in_fence
                continue
            if in_fence or stripped.startswith("#") or stripped.startswith("[!["):
                continue
            if stripped:
                paragraph.append(stripped)
            elif paragraph:
                break
        if paragraph:
            description = " ".join(paragraph)
            return description[:400].rstrip()
    return None


# ---------------------------------------------------------------------------
# Context prioritization for prompts
# ---------------------------------------------------------------------------
_MANIFEST_NAMES = {
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "package.json", "pytest.ini", "cargo.toml", "go.mod",
}

# File-name fragments that usually mark entry points / orchestration modules.
_ORCHESTRATION_KEYWORDS = (
    "main", "cli", "app", "updater", "generator", "router", "builder",
    "provider", "collector", "detector", "report", "automation", "pipeline",
    "store", "resolver",
)


def _priority_group(path: str) -> int:
    """Smaller group = more authoritative context for the summary prompt."""
    name = Path(path).name.lower()
    lowered = path.lower()
    if name.startswith("readme"):
        return 0
    if name in _MANIFEST_NAMES:
        return 1
    if lowered.startswith(".github/workflows"):
        return 2
    if lowered.startswith("config/") or name.endswith((".toml", ".ini", ".cfg", ".json")):
        return 3
    if lowered.startswith("src/") and any(k in name for k in _ORCHESTRATION_KEYWORDS):
        return 4
    if lowered.startswith("src/"):
        return 5
    if lowered.startswith("tests/") or lowered.startswith("docs/") or name.endswith(".md"):
        return 6
    return 7


def prioritized_context_files(context: RepoContext) -> list[tuple[str, str]]:
    """Context files ordered README -> manifests -> CI -> config ->
    orchestration source -> other source -> tests/docs.

    Deterministic: sorted by (priority group, path). This keeps authoritative
    files inside the bounded prompt budget even when low-priority modules
    (e.g. a large legacy subsystem) outnumber them.
    """
    return sorted(
        context.files.items(), key=lambda item: (_priority_group(item[0]), item[0])
    )


class LocalDeterministicSummaryProvider:
    """Token-free provider: builds the summary directly from the context.

    Deliberately deterministic (no timestamps in the body) so tests and
    demo runs are stable and repeatable. The overview description is taken
    from the target repository's README — never hardcoded.
    """

    def generate_summary(self, context: RepoContext) -> str:
        tree = context.file_tree
        python_sources = [p for p in tree if p.startswith("src/") and p.endswith(".py")]
        test_files = [p for p in tree if p.startswith("tests/") and p.endswith(".py")]
        workflows = [p for p in tree if p.startswith(".github/workflows/")]
        configs = [
            p
            for p in tree
            if p.endswith((".json", ".toml", ".ini", ".cfg"))
            or p in ("requirements.txt",)
        ]
        automation_modules = [
            p
            for p in python_sources
            if any(k in p for k in ("automation", "updater", "router", "detector"))
        ]

        description = extract_project_description(context) or (
            "No project description was found in the repository README; "
            "this overview is structural only."
        )

        def bullets(paths: list[str], empty: str) -> str:
            if not paths:
                return empty
            return "\n".join(f"- `{p}`" for p in paths)

        tree_block = "\n".join(tree[:60])

        return f"""# Project Technical Summary

_Generated by TechDocker (deterministic local provider)._

## System Overview

{context.project_name} contains {context.total_files} tracked files.
{description}

## Repository Structure

```text
{tree_block}
```

## Core Modules

{bullets(python_sources, "No Python modules under src/ were found.")}

## Automation Pipeline

{bullets(automation_modules, "No automation-related modules were detected.")}

## Testing Strategy

{len(test_files)} test file(s) under tests/:

{bullets(test_files, "No test files were found.")}

## Configuration

{bullets(configs, "No configuration files were detected.")}

## Deployment and CI

{bullets(workflows, "No CI workflow configuration was detected.")}

## Known Limitations

- This summary was generated by the deterministic local provider, not an LLM;
  section prose is structural rather than semantic.
- LLM-based summary generation is an optional provider behind the same
  interface.
- Only repository content was used as evidence; no external systems or
  documents were accessed.
"""


BASE_SUMMARY_SYSTEM_PROMPT = (
    "You are TechDocker's technical writer. You produce accurate, specific "
    "Markdown summaries of a software repository from its file tree and "
    "source excerpts. Never invent features, never claim access to official "
    "or external documents — the repository content is your only source. "
    "Output Markdown only, no code fences around the whole document."
)


def summary_has_required_headings(text: str) -> bool:
    """True when the text carries the title and all eight stable headings.

    Only describes the *deterministic* provider's fixed template. LLM output
    is validated structurally by :func:`validate_llm_summary` instead — LLM
    summaries may use any repository-specific headings.
    """
    if "# Project Technical Summary" not in (text or ""):
        return False
    return all(f"## {heading}" in text for heading in SUMMARY_HEADINGS)


# ---------------------------------------------------------------------------
# LLM output validation (structure + conservative grounding)
# ---------------------------------------------------------------------------
MIN_H2_SECTIONS = 4
MAX_H2_SECTIONS = 12
MIN_SECTION_CONTENT_CHARS = 20
MIN_SUMMARY_TOTAL_CHARS = 400

# --- Conservative technology-claim grounding ---------------------------------
#
# High-risk capability claims (ML / NLP / deep learning) are only accepted when
# the repository shows STRONG positive evidence: a recognized library in a
# dependency manifest, an actual import of one, or an affirmative statement in
# an authoritative README. Crucially, the claim *phrase itself* is never
# evidence — otherwise TechDocker summarizing its own validator, prompt, or
# tests (which all contain the strings "machine learning", "NLP", ...) would
# fabricate circular evidence for a claim it does not implement.
#
# Each family maps to (import module names, dependency-manifest tokens).
_ML_LIBS = (
    {"sklearn", "xgboost", "lightgbm", "tensorflow", "keras", "torch"},
    {"scikit-learn", "sklearn", "xgboost", "lightgbm", "tensorflow", "keras",
     "torch", "pytorch"},
)
_NLP_LIBS = (
    {"spacy", "nltk", "transformers", "sentence_transformers"},
    {"spacy", "nltk", "transformers", "sentence-transformers",
     "sentence_transformers"},
)
_DL_LIBS = (
    {"tensorflow", "keras", "torch"},
    {"tensorflow", "keras", "torch", "pytorch"},
)
_TECH_CLAIM_FAMILIES: dict[str, tuple[set[str], set[str]]] = {
    "machine learning": _ML_LIBS,
    "natural language processing": _NLP_LIBS,
    "nlp": _NLP_LIBS,
    "neural network": _DL_LIBS,
    "deep learning": _DL_LIBS,
}

# Dependency manifests whose contents count as strong dependency evidence.
_MANIFEST_FILENAMES = {
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "pipfile", "pipfile.lock", "package.json",
}

# Negation cues: a README sentence containing any of these does not
# affirmatively support a technology claim (e.g. "does not use machine learning").
_NEGATION_CUES = (
    "not", "no", "without", "never", "n't", "cannot", "neither", "nor",
    "avoid", "instead of", "rather than",
)

_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)


@dataclass
class SummaryValidationResult:
    """Outcome of validating an LLM-generated summary, with reasons."""

    ok: bool
    problems: list[str] = field(default_factory=list)


def _is_evidence_excluded(path: str) -> bool:
    """True for files that must NOT count as technology evidence.

    Excludes tests and fixtures, generated artifacts, and the grounding
    machinery itself (this module and the change-analyzer prompt/claim map) —
    all of which mention the claim phrases without proving the target project
    uses those technologies.
    """
    lowered = path.replace("\\", "/").lower()
    parts = lowered.split("/")
    name = parts[-1]
    if any(part in ("tests", "artifacts", "__pycache__", ".venv") for part in parts):
        return True
    if name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py":
        return True
    # The validator/prompt/claim-map modules describe these technologies in
    # order to reason about them; that is not evidence of use.
    if lowered in (
        "src/project_summary_generator.py",
        "src/llm_change_analyzer.py",
    ):
        return True
    return False


def _imported_top_modules(content: str) -> set[str]:
    """Top-level module names imported by a Python source file."""
    return {match.lower() for match in _IMPORT_RE.findall(content or "")}


def _readme_affirms(claim: str, readme_text: str) -> bool:
    """True when an authoritative README affirmatively states the claim.

    The claim phrase must appear in a sentence with no negation cue, so
    "This project does not use machine learning." is never affirmative.
    """
    low = (readme_text or "").lower()
    for match in re.finditer(rf"\b{re.escape(claim)}\b", low):
        sentence_start = (
            max(low.rfind(".", 0, match.start()), low.rfind("\n", 0, match.start()))
            + 1
        )
        ends = [pos for pos in (low.find(".", match.end()), low.find("\n", match.end())) if pos != -1]
        sentence_end = min(ends) if ends else len(low)
        sentence = low[sentence_start:sentence_end]
        if not any(cue in sentence for cue in _NEGATION_CUES):
            return True
    return False


def _has_technology_evidence(claim: str, context: RepoContext) -> bool:
    """Strong, non-circular evidence that the repository uses ``claim``.

    Positive evidence: a recognized library in a dependency manifest, an
    actual import of one, or an affirmative README statement. Tests, generated
    artifacts, and the grounding machinery are excluded, and the claim phrase
    on its own is never treated as evidence.
    """
    import_libs, manifest_tokens = _TECH_CLAIM_FAMILIES[claim]

    for path, content in context.files.items():
        if _is_evidence_excluded(path):
            continue
        name = Path(path).name.lower()
        lowered = content.lower()

        if name in _MANIFEST_FILENAMES:
            for token in manifest_tokens:
                if re.search(rf"(?<![\w-]){re.escape(token)}(?![\w-])", lowered):
                    return True

        if name.endswith(".py") and _imported_top_modules(content) & import_libs:
            return True

        if name.startswith("readme") and _readme_affirms(claim, content):
            return True

    return False


def validate_llm_summary(
    text: str, context: Optional[RepoContext] = None
) -> SummaryValidationResult:
    """Structurally validate an LLM summary with variable headings.

    Requirements: exactly one H1 title; 4-12 substantive, unique,
    non-generic H2 sections, each with meaningful content; no
    document-wrapping code fence; not an empty/placeholder document; and no
    unsupported high-risk technology claims (checked against the collected
    repository evidence when ``context`` is provided).
    """
    problems: list[str] = []
    cleaned = _strip_outer_fences(text)

    if not cleaned.strip():
        return SummaryValidationResult(False, ["summary is empty"])
    if cleaned.lstrip().startswith("```"):
        problems.append("document is still wrapped in a code fence")
    if len(cleaned) < MIN_SUMMARY_TOTAL_CHARS:
        problems.append(
            f"summary is too short ({len(cleaned)} chars) to be a real "
            "technical summary"
        )

    sections = parse_markdown_sections(cleaned)
    h1_sections = [s for s in sections if s.level == 1]
    h2_sections = [s for s in sections if s.level == 2]

    if len(h1_sections) != 1:
        problems.append(
            f"expected exactly one H1 title, found {len(h1_sections)}"
        )
    if not MIN_H2_SECTIONS <= len(h2_sections) <= MAX_H2_SECTIONS:
        problems.append(
            f"expected {MIN_H2_SECTIONS}-{MAX_H2_SECTIONS} '##' sections, "
            f"found {len(h2_sections)}"
        )

    normalized = [normalize_heading(s.heading) for s in h2_sections]
    duplicates = {h for h in normalized if normalized.count(h) > 1}
    if duplicates:
        problems.append(f"duplicate section headings: {sorted(duplicates)}")

    generic = [
        s.heading for s in h2_sections
        if normalize_heading(s.heading) in GENERIC_HEADINGS
    ]
    if generic:
        problems.append(f"generic section headings: {generic}")

    empty_sections = [
        s.heading for s in h2_sections
        if len((s.content or "").strip()) < MIN_SECTION_CONTENT_CHARS
    ]
    if empty_sections:
        problems.append(
            f"sections without meaningful content: {empty_sections}"
        )

    if context is not None:
        summary_lower = cleaned.lower()
        for claim in _TECH_CLAIM_FAMILIES:
            if re.search(rf"\b{re.escape(claim)}\b", summary_lower) and not (
                _has_technology_evidence(claim, context)
            ):
                problems.append(
                    f"unsupported technology claim {claim!r}: no dependency, "
                    "import, or affirmative README evidence in the repository"
                )

    return SummaryValidationResult(ok=not problems, problems=problems)


def _strip_outer_fences(text: str) -> str:
    """Remove a single markdown fence wrapping the whole document, if any."""
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def build_base_summary_prompt(context: RepoContext) -> str:
    """Prompt for LLM base-summary generation from repository context.

    Deterministic and bounded (fits the 8192-token Ollama window with room
    for output). Authoritative files come first via
    :func:`prioritized_context_files`; nothing repository-specific is
    hardcoded — all facts must come from the supplied context.
    """
    tree_block = "\n".join(context.file_tree[:120])

    excerpts: list[str] = []
    budget = 9_000
    for path, content in prioritized_context_files(context):
        snippet = content[:800]
        entry = f"--- {path} ---\n{snippet}"
        if budget - len(entry) < 0:
            break
        budget -= len(entry)
        excerpts.append(entry)
    excerpt_block = "\n\n".join(excerpts)

    return f"""Write the baseline technical summary of the repository
"{context.project_name}" ({context.total_files} tracked files).

Output format:
- Exactly ONE H1 title line naming the project (e.g. "# {context.project_name}
  Technical Summary"), then 4 to 10 "##" sections.
- Choose section headings that fit THIS repository (for example
  "Architecture", "CI/CD Review Flow", "Quality and Tests", "Repository
  Automation", "Dependencies and Environment"). Do not follow a generic
  template, and never use vague headings such as "Code Changes", "Updates",
  "Changes", or "Miscellaneous".
- Write a rich normal technical summary: paragraphs and bullet lists as
  appropriate, not only short bullets. No code fence around the document.

Grounding rules — these override everything else:
- The repository context below is your ONLY evidence. Do not invent
  technologies, behaviours, integrations, metrics, or capabilities.
- Do not describe deterministic parsing, scoring, or heuristics as
  "machine learning", "NLP", or "AI" unless the context contains explicit
  evidence (imports, dependencies, or documentation saying so).
- Distinguish ACTIVE functionality from anything the context labels as
  legacy, backup, deprecated, experimental, or historical. When the README
  states the current direction and marks older components as legacy,
  follow the README.
- Name real files, modules, commands, artifacts, and workflows only as
  they appear in the context below.

File tree:
{tree_block}

Repository context (most authoritative files first):
{excerpt_block}
"""


class LLMSummaryProvider:
    """Generate the base summary with the configured LLM provider.

    Falls back to :class:`LocalDeterministicSummaryProvider` whenever the LLM
    is unavailable or its output does not carry the required stable headings
    — unless strict mode is enabled, in which case a clear error is raised.
    """

    def __init__(
        self,
        env: Mapping[str, str] | None = None,
        llm: Optional[LLMProvider] = None,
    ) -> None:
        self.env = env
        self.llm = llm
        self.config = LLMProviderConfig.from_env(env)

    def generate_summary(self, context: RepoContext) -> str:
        llm = self.llm or get_llm_provider_from_env(self.env)
        # Strict-mode connection failures raise inside generate().
        response = llm.generate(
            build_base_summary_prompt(context),
            system_prompt=BASE_SUMMARY_SYSTEM_PROMPT,
        )
        text = _strip_outer_fences(response.text)

        if response.fallback_used:
            # The underlying LLM was unreachable; the fallback text is a
            # placeholder, not a summary. (Strict mode raised already.)
            print(
                "[project_summary_generator] the LLM provider fell back "
                "(unavailable); using the deterministic provider.",
                file=sys.stderr,
            )
            return LocalDeterministicSummaryProvider().generate_summary(context)

        result = validate_llm_summary(text, context)
        if result.ok:
            return text if text.endswith("\n") else text + "\n"

        reasons = "; ".join(result.problems)
        if self.config.strict:
            raise LLMOutputValidationError(
                f"LLM base summary failed validation: {reasons}. Strict mode "
                "(TECHDOCKER_LLM_STRICT) is enabled, so no deterministic "
                "fallback was used."
            )
        print(
            f"[project_summary_generator] LLM summary rejected ({reasons}); "
            "using the deterministic provider.",
            file=sys.stderr,
        )
        return LocalDeterministicSummaryProvider().generate_summary(context)


class CopilotSummaryProvider:
    """Placeholder for future Copilot/LLM summary generation.

    Reads its token from the ``TECHDOCKER_COPILOT_TOKEN`` environment variable
    when present. Never required by tests or local demos.
    """

    def __init__(self, token: str | None = None) -> None:
        self.token = token or os.environ.get(COPILOT_TOKEN_ENV_VAR)

    def is_available(self) -> bool:
        return bool(self.token)

    def generate_summary(self, context: RepoContext) -> str:
        raise NotImplementedError(
            "Copilot summary generation is a future phase; use "
            "LocalDeterministicSummaryProvider for now."
        )


def get_default_provider() -> SummaryProvider:
    """The provider used when none is passed: always the deterministic one."""
    return LocalDeterministicSummaryProvider()


def get_summary_provider_for_env(
    env: Mapping[str, str] | None = None,
) -> SummaryProvider:
    """Deterministic by default; the LLM provider only when explicitly
    enabled via ``TECHDOCKER_LLM_PROVIDER=ollama``."""
    config = LLMProviderConfig.from_env(env)
    if config.provider == "ollama":
        return LLMSummaryProvider(env=env)
    return LocalDeterministicSummaryProvider()


def generate_original_summary(
    repo_path: str | Path = ".",
    force: bool = False,
    provider: SummaryProvider | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Write the baseline summary (and initialize the updated copy).

    Refuses to overwrite an existing ``base_original_summary.md`` unless
    ``force`` is set. Always ensures ``base_updated_summary.md`` exists,
    initializing it as a copy of the baseline when missing.
    """
    original_path = original_summary_path(repo_path)
    updated_path = updated_summary_path(repo_path)

    if original_path.exists() and not force:
        if not updated_path.exists():
            shutil.copyfile(original_path, updated_path)
        return original_path

    context = collect_repo_context(repo_path)
    summary = (provider or get_summary_provider_for_env(env)).generate_summary(
        context
    )

    original_path.parent.mkdir(parents=True, exist_ok=True)
    original_path.write_text(summary, encoding="utf-8")

    if not updated_path.exists():
        shutil.copyfile(original_path, updated_path)

    return original_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the baseline TechDocker project summary."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if base_original_summary.md already exists.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print the generated summary to stdout without writing any files.",
    )
    parser.add_argument("--repo-path", default=".", help="Repository root.")
    arguments = parser.parse_args()

    if arguments.preview:
        context = collect_repo_context(arguments.repo_path)
        provider = get_summary_provider_for_env(os.environ)
        # stdout carries only the previewed Markdown; diagnostics go to
        # stderr so the output can be piped/saved as a document.
        print(provider.generate_summary(context))
        print(
            "[preview] No files were written. Use --force to overwrite "
            "base_original_summary.md.",
            file=sys.stderr,
        )
        return 0

    existed = original_summary_path(arguments.repo_path).exists()
    path = generate_original_summary(arguments.repo_path, force=arguments.force)

    if existed and not arguments.force:
        print(f"Baseline already exists (use --force to regenerate): {path}")
    else:
        print(f"Baseline summary written: {path}")
    print(f"Updated summary: {updated_summary_path(arguments.repo_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
