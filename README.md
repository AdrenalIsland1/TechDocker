# TechDocker

TechDocker is a GitHub project-summary automation pipeline: it detects Git code changes, produces structured change packages, routes each change to the relevant section of a reviewable Markdown project summary, and opens a pull request for developer approval — with LLM assistance kept optional behind a deterministic default.

## Purpose

TechDocker keeps a technical project summary in sync with the code without
anyone editing documentation by hand, and without any change reaching `main`
unreviewed. On every push it:

- detects the changed files between the before/after commits (`git diff`),
- packages them into a structured change summary
  (`artifacts/change_packages/latest_change_summary.json`),
- routes the change to the right section of the summary using a stored
  skeleton of its headings,
- inserts a clearly marked update block into the reviewable summary
  (`artifacts/summaries/base_updated_summary.md`),
- opens a pull request so a developer can approve, edit, or reject the
  suggestion.

The permanent baseline (`artifacts/summaries/base_original_summary.md`) is
never modified by update runs. LLM support (local Ollama today, vLLM on
company servers later) is strictly optional: CI and the default pipeline are
fully deterministic.

The repository also contains a DOCX parser and heading scorer from the
project's earlier direction — these are **legacy/backup capabilities**, not
the active product (see "Legacy DOCX Pipeline" below).

## Installation

```bash
git clone <your-repo-url>
cd TechDocker
python3 -m venv .venv
```

### Activate the virtual environment

```bash
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### Install requirements

```bash
pip install -r requirements.txt
```

## Usage

### Preview or generate the project summary

```bash
python3 -m src.project_summary_generator --preview   # print, write nothing
python3 -m src.project_summary_generator             # write baseline once
python3 -m src.project_summary_generator --force     # regenerate baseline
```

### Run the summary updater locally

```bash
python3 -m src.summary_updater
```

### Run the legacy DOCX analyzer (inactive backup tooling)

```bash
python3 -m src.analyze_headings path/to/document.docx
python3 -m src.analyze_headings path/to/document.docx --output analysis.csv
```

The legacy analyzer prints per-paragraph heading scores and can export every
extracted formatting feature to CSV; see
[docs/heading-detection-scoring.md](docs/heading-detection-scoring.md).

### Run the tests

```bash
pytest -v
```

## GitHub Project Summary Pipeline

Official project documents cannot be accessed due to security restrictions,
so the active pipeline works from the GitHub repository itself. The
repository's own context (file tree, source, tests, docs, config — never
secrets, binaries, or official documents) is summarized into three core
artifacts:

- `artifacts/summaries/base_original_summary.md` — the baseline technical
  summary, generated once. **Never modified** during normal update runs;
  regenerate only with `python3 -m src.project_summary_generator --force`.
- `artifacts/skeletons/base_skeleton.json` — the routing structure, built
  from **`base_original_summary.md`** and its deterministic Markdown headings
  (`# Project Technical Summary`, `## System Overview`, `## Core Modules`, …).
  It changes **only** when a new heading/subheading is needed, in which case
  the new section is appended — it is never rebuilt from the reviewable copy.
- `artifacts/summaries/base_updated_summary.md` — the reviewable copy.
  Starts identical to the baseline; every push's change block is inserted
  here under the routed section. Reviewers diff this against the original.

On every push to `main`, the workflow
([.github/workflows/documentation-update.yml](.github/workflows/documentation-update.yml))
runs the tests, then `python3 -m src.summary_updater`, which:

1. detects the changed files with `git diff`,
2. writes `artifacts/change_packages/latest_change_summary.json` (files,
   SHAs, actor, branch, generated change summary),
3. routes the change against `base_skeleton.json` (rule-based today,
   LLM-ready interface),
4. inserts a marked `<!-- TECHDOCKER_UPDATE_START/END -->` block into
   `base_updated_summary.md` under the routed section,
5. extends the skeleton only when a new section had to be created,
6. **proposes** the result as a Pull Request — nothing is committed
   directly to `main`.

### Review flow

```text
push to main
  → workflow generates the summary suggestion
  → bot pushes a branch (techdocker/summary-update-<run id>)
  → bot opens a PR: "Suggested project summary update"
  → developer reviews: merge as-is, edit base_updated_summary.md
    in the PR branch, or close the PR to reject
  → no summary change reaches main without review
```

When the updater produces no artifact changes the workflow prints
"No summary changes to propose" and exits successfully. The PR body
(`python3 -m src.pr_summary_report`) lists the source commit, actor, branch,
and changed files, and reminds reviewers that `base_original_summary.md`
remains the unchanged baseline. GitHub PRs are the current
confirm/edit/reject mechanism; email/Teams notifications are future work.

Summary generation sits behind a provider interface: the current
`LocalDeterministicSummaryProvider` needs no network or tokens (safe for
tests and demos); Copilot/LLM generation is a future provider behind the
same interface.

## Optional Local LLM Provider

The default pipeline is fully deterministic — CI and the GitHub Actions
workflow need no LLM, no Ollama, and no tokens. For local development an
optional [Ollama](https://ollama.com) provider can generate richer change
summaries and suggest routing decisions. Recommended first model:
`qwen2.5-coder:3b`.

```bash
ollama run qwen2.5-coder:3b

# read-only preview of the placement suggestion for the latest change:
TECHDOCKER_LLM_PROVIDER=ollama TECHDOCKER_OLLAMA_MODEL=qwen2.5-coder:3b \
  python3 -m src.llm_change_analyzer

# full updater run with LLM-assisted routing:
TECHDOCKER_LLM_PROVIDER=ollama TECHDOCKER_OLLAMA_MODEL=qwen2.5-coder:3b \
  python3 -m src.summary_updater
```

Safety properties:

- If Ollama is unavailable, the pipeline falls back to deterministic mode
  (set `TECHDOCKER_LLM_STRICT=true` to fail loudly instead).
- LLM output is validated and **never directly trusted**: it must be strict
  JSON, the decision must be one of the two allowed values, confidence must
  be within [0, 1], and the target section must exist in the skeleton —
  anything else falls back to the rule-based router. The LLM never edits
  files.
- Suggestions below `TECHDOCKER_LLM_MIN_CONFIDENCE` (default 0.75) are
  discarded in favour of the rule-based router.
- Production plan: vLLM on company servers later, behind the same provider
  interface.

## LLM Quality Layer

The deterministic provider remains the default so CI is always safe — the
GitHub Actions workflow needs no model, no tokens, no network. Locally,
Ollama can produce a much better baseline summary and per-push change
summaries; the current local test model is `qwen2.5-coder:7b`.

```bash
# preview the LLM baseline without writing anything:
TECHDOCKER_LLM_PROVIDER=ollama TECHDOCKER_OLLAMA_MODEL=qwen2.5-coder:7b \
  python3 -m src.project_summary_generator --preview

# regenerate the baseline with the LLM (only --force overwrites it):
TECHDOCKER_LLM_PROVIDER=ollama TECHDOCKER_OLLAMA_MODEL=qwen2.5-coder:7b \
  python3 -m src.project_summary_generator --force

# preview the routing suggestion for the latest change:
TECHDOCKER_LLM_PROVIDER=ollama TECHDOCKER_OLLAMA_MODEL=qwen2.5-coder:7b \
  python3 -m src.llm_change_analyzer
```

Every LLM output is validated before use: the baseline must carry the eight
stable headings (otherwise the deterministic summary is used), and routing
suggestions are rejected when they propose generic headings ("Code Changes",
"Updates", …), generic or empty summaries, unknown section ids, or
out-of-range confidence — rejection always falls back to the rule-based
router. The PR review flow remains the final approval layer regardless of
which provider produced the suggestion. Production plan: vLLM on company
servers behind the same provider interface.

## Legacy DOCX Pipeline

The core DOCX parser remains in the repository as previous work and backup
(`docx_parser.py`, `docx_feature_extractor.py`, `docx_heading_analyzer.py`,
`docx_numbering.py`, `format_pattern_detector.py`, `heading_scorer.py`,
`analyze_headings.py`, plus their tests and sample documents), but it is
**no longer the active production path** after the security-driven direction
change — the active workflow does not read or update DOCX files.

The old DOCX automation demo (DOCX updater, DOCX skeleton builder/store, and
DOCX change router) has been **removed from the code base entirely**; the
GitHub Project Summary Pipeline above replaces it. The standalone analyzer
still works for inspection:
`python3 -m src.analyze_headings path/to/document.docx`.

## Project status

- Active pipeline: GitHub push → change detection → summary routing →
  `base_updated_summary.md` review artifact (Markdown-based, no DOCX).
- Deterministic summary provider in place; Copilot/LLM provider is a stub.
- Legacy DOCX feature extraction, heading scoring, and CSV inspection remain
  available but inactive.
- All tests are passing.

## Roadmap

1. Replace the deterministic summary provider with Copilot/LLM generation.
2. Replace rule-based routing with LLM routing over the change summary.
3. Richer change summaries generated from actual diffs, not just file lists.
4. Review/approval flow comparing `base_updated_summary.md` against the
   baseline before merging documentation changes.

## Documentation

See [docs/heading-detection-scoring.md](docs/heading-detection-scoring.md) for
the full scoring rules, thresholds, combination-rule system, and limitations.
