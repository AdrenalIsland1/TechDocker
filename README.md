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

## Project status

- DOCX feature extraction is complete.
- Heading scoring (official styles + formatting heuristic) is implemented.
- CSV inspection reporting is working.
- All DOCX tests are passing.
- GitHub-triggered automation is planned next.

## Roadmap

1. GitHub-triggered automation: analyze changed documents automatically on push.
2. Feed the detected structure to an LLM for controlled document updates.
3. Additional strong combination rules tuned on more real documents.
4. Write-back support for controlled DOCX editing (analysis stays read-only).

## Documentation

See [docs/heading-detection-scoring.md](docs/heading-detection-scoring.md) for
the full scoring rules, thresholds, combination-rule system, and limitations.
