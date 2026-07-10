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

## Working GitHub-to-DOCX Demo

A push to `main` runs the GitHub Actions workflow
([.github/workflows/documentation-update.yml](.github/workflows/documentation-update.yml)),
which:

1. runs the full test suite,
2. detects the files changed by the push (`git diff` between the before/after
   commits),
3. resolves the configured project document from `config/projects.json`,
4. appends a marked "Automated Documentation Update" section (timestamp,
   commit metadata, changed-file list) to `samples/techdocker_test1.docx`,
5. commits the updated DOCX back to the repository as `github-actions[bot]`.

The workflow guards against infinite loops twice: pushes that only touch the
demo DOCX are ignored (`paths-ignore`), and runs triggered by the bot user are
skipped. The demo uses a local sample DOCX instead of SharePoint — SharePoint
retrieval/upload and LLM-based placement are future phases. The updater also
runs locally with `python3 -m src.demo_docx_updater`, falling back to
`HEAD~1..HEAD`.

## Persistent Document Skeleton Architecture

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

- DOCX feature extraction is complete.
- Heading scoring (official styles + formatting heuristic) is implemented.
- CSV inspection reporting is working.
- All DOCX tests are passing.
- GitHub push-triggered automation now updates and commits the demo DOCX.

## Roadmap

1. GitHub-triggered automation: analyze changed documents automatically on push.
2. Feed the detected structure to an LLM for controlled document updates.
3. Additional strong combination rules tuned on more real documents.
4. Write-back support for controlled DOCX editing (analysis stays read-only).

## Documentation

See [docs/heading-detection-scoring.md](docs/heading-detection-scoring.md) for
the full scoring rules, thresholds, combination-rule system, and limitations.
