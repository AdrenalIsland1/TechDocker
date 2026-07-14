# TechDocker

TechDocker analyzes DOCX technical documents, extracts paragraph formatting and list metadata, and scores likely headings to build a reliable document structure.

## Purpose

Many real-world Word documents do not use official Heading styles. TechDocker
reads a `.docx` file (read-only — the source document is never modified),
extracts every paragraph's visible formatting, and produces:

- a transparent 0–100 heading score per paragraph, with the exact signals
  that added or subtracted points,
- a classification: `heading`, `probable_heading`, or `normal_content`,
- Word list metadata (bullet/numbered, nesting level, reconstructed markers
  such as `1.` or `•`) resolved from the document's numbering XML,
- manual line-break segments within single paragraphs,
- a structured heading hierarchy, a compact terminal table, and a full CSV
  report for inspection.

Official Word Heading styles always win (score 100). For everything else, an
explainable formatting heuristic scores signals such as bold, relative font
size, numbering prefixes, colon endings, spacing, and repeated formatting
patterns, with hard rules for `Note:`/`Link:` prefixes and strong signal
combinations.

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

### Run the DOCX analyzer

```bash
python3 -m src.analyze_headings path/to/document.docx
```

This prints a compact table with each paragraph's text, style, score,
classification, predicted heading level, detection method, and scoring
signals.

### Export the full analysis to CSV

```bash
python3 -m src.analyze_headings path/to/document.docx --output analysis.csv
```

The CSV contains every extracted feature per paragraph: formatting (font
size/family/color, bold, italic, underline, spacing, alignment), text shape
(word/sentence counts, `is_title_case`, `is_all_caps`), list metadata,
line-break segments, and the scoring outcome.

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
5. rebuilds the skeleton only when a new section had to be created,
6. commits the four artifacts back as `github-actions[bot]` with `[skip ci]`.

Summary generation sits behind a provider interface: the current
`LocalDeterministicSummaryProvider` needs no network or tokens (safe for
tests and demos); Copilot/LLM generation is a future provider behind the
same interface.

## Legacy DOCX Pipeline

The DOCX parser, heading scorer, analyzer, and DOCX skeleton/updater modules
remain in the repository as previous work and backup, but they are **no
longer the active production path** after the security-driven direction
change — the active workflow does not read or update DOCX files anymore.
The standalone analyzer still works for inspection:
`python3 -m src.analyze_headings path/to/document.docx`.

## Persistent Document Skeleton Architecture (legacy DOCX pipeline)

The full feature-based DOCX parse is expensive, so it runs **once**:

```bash
python3 -m src.document_skeleton_builder
```

This parses the configured document and stores its heading structure as JSON
at `artifacts/skeletons/techdocker_skeleton.json` — one entry per section with
a stable slug id, level, parent, path, order, and content hash.

On subsequent pushes the updater avoids full parsing entirely:

1. `git diff` produces the changed files,
2. a change summary is built from them,
3. the stored skeleton JSON is loaded (and built first if missing),
4. `src/change_router.py` decides whether the update belongs to an existing
   section or needs a new heading (simple keyword rules today — an LLM will
   replace this in a future phase),
5. the DOCX is updated under the routed heading (found by scanning paragraph
   text, not by re-parsing),
6. the skeleton JSON is rewritten **only** when a new section was created —
   structure changes are the only thing that invalidates it.

If the routed heading cannot be found in the DOCX, the update is appended to
the end with a warning.

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
