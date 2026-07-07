# Heading Detection & Scoring (DOCX)

This document explains how the DOCX parser decides whether a paragraph is a
heading, especially when the author did **not** use official Word heading
styles. The goal is a *transparent, explainable* score: every point added or
removed is recorded as a signal so a reviewer can see exactly why a paragraph
was classified the way it was.

The original DOCX file is only ever **read**. Nothing here modifies the source
document.

## Two detection methods

### Method 1 — Official Word heading styles (always trusted)

If a paragraph uses a built-in `Heading 1` … `Heading 9` style it is:

- classified as `heading`,
- given a score of **100**,
- assigned its level directly from the Word style,
- tagged with `detection_method = "official_word_heading_style"`.

Official styles always take priority over the heuristic. The Word `Title` style
is used to set the document title (never treated as a heading).

**Why 100?** An official heading style is an explicit authoring decision. It is
the strongest possible signal of intent, so it bypasses scoring entirely.

### Method 2 — Formatting heuristic (for everything else)

Many real documents fake headings with bold/large text instead of styles. For
those paragraphs we compute a score from 0–100 out of visible formatting and
text features. This is why heuristic detection is needed at all: without it, a
document with zero heading styles would have no detectable structure.

## Body-format detection (dynamic)

The "normal" body formatting is detected **per document** — never hard-coded.
The body vote is taken over non-empty `Normal`-style paragraphs **plus Word
list items** (list items are body content by definition, and in list-heavy
documents they are most of the ordinary text), taking the most common:

- font size,
- font family,
- font colour (here `None`/automatic counts as a real value, since default
  black is the ordinary body colour).

Two refinements keep the vote honest:

- Mostly-bold/underlined paragraphs are excluded from the vote whenever
  plainer candidates exist — a document whose only `Normal` paragraphs are
  bold pseudo-headings must not have "bold 14pt" detected as its body format.
- When neither runs nor the style chain define a font size/family, the
  document-wide defaults from `w:docDefaults` in `styles.xml` are used as the
  final fallback, so plain default-formatted paragraphs resolve to the real
  base font instead of "unknown".

If there are too few candidate paragraphs we fall back to all non-heading
paragraphs. "Clearly larger than body" is then defined relative to the detected
body size (see `larger_font_min_delta`).

## Scoring rules

Weights live in `HEADING_SCORE_WEIGHTS` and thresholds in `SCORING_CONFIG`
(both in `src/heading_scorer.py`) so they can be tuned without touching logic.

### Positive signals

| Signal | Points |
| --- | --- |
| Textual numbering prefix (`1`, `1.2`, `3.4.1`) | +25 |
| Mostly bold (≥75% of visible runs) | +15 |
| Font clearly larger than body (≥ body + `larger_font_min_delta`, default 1pt) | +20 |
| Font family different from body | +5 |
| Mostly underlined (≥75% of visible runs) | +5 |
| Fewer than 10 words | +10 |
| Ends with colon / colon-hyphen | +10 |
| Extra spacing before | +10 |
| Extra spacing after | +5 |
| Repeated **non-body** formatting pattern | +20 |
| Title Case or ALL CAPS | +5 |
| Does not end with a full stop | +5 |

### Negative signals

| Signal | Points |
| --- | --- |
| Real Word bullet / numbered-list item (`w:numPr`) | −20 |
| More than 15 words | −20 |
| Multiple sentences | −20 |
| Ends with a full stop | −10 |
| Same formatting as normal body text | −15 |
| Mostly sentence-style content | −10 |
| Coloured text | −20 |

The final score is always clamped to **0–100**.

### Coloured-text penalty

Colour is read from the dominant visible run colour. A paragraph counts as
coloured when that colour is an explicit non-black RGB value or a visible
theme/accent colour. Automatic colour, inherited default, and pure black
(`000000`) are **not** penalised — ordinary black body text must never be
flagged. The detected colour value is stored in the analysis output.

## Hard-negative rules: `Note:` and `Link:`

A paragraph that *begins* with `Note` or `Link` immediately followed by a colon
or dash is never a heading. All of these are forced to normal content:

```
Note:   Note:-   Note -   Note-
Link:   Link:-   Link -   Link-
```

The match is case-insensitive and allows optional whitespace before the
punctuation. It only applies at the **start** of a paragraph — a sentence like
*"This paragraph contains a link to another service."* is unaffected, and
*"Notebook configuration"* is not a Note line.

For these paragraphs the parser forces:

- `score = 0`,
- `classification = "normal_content"`,
- `detection_method = "hard_negative_rule"`,
- a signal such as `hard_negative_prefix:note` / `hard_negative_prefix:link`.

No positive rule or combination can override this.

## Strong combination rule

Combination rules live in `COMBINATION_RULES` and are easy to extend. The first
hard-coded one is:

> **numbering prefix + mostly bold + colon/colon-hyphen ending** → score raised
> to **at least 90**, with signal `combination:numbering+bold+colon:min_90`.

Combinations raise the score to a floor; they never lower it and never override
official heading styles or the Note/Link rule.

**Adding a future combination:** append a `CombinationRule(name, predicate,
min_score, signal)` to `COMBINATION_RULES`. The `predicate` is a function of the
paragraph features returning `True`/`False`. No other code changes are needed.

## Classification thresholds

| Score | Classification |
| --- | --- |
| 80–100 | `heading` |
| 60–79 | `probable_heading` |
| below 60 | `normal_content` |

In the parser output:

- official headings and heuristic headings (score ≥ 80 **with an inferable
  level**) go into the confirmed `headings` hierarchy;
- `probable_headings` (60–79, or strong headings whose level can't be inferred)
  are recorded in review metadata and are **never** silently confirmed;
- everything else stays as content under the active heading.

## Numbering: textual prefix vs. real Word list item

These are two independent facts, stored and scored separately:

- **Textual numbering prefix** — e.g. `2.1 API Configuration`. Detected at the
  start of the text only (mid-sentence numbers and 4-digit years are ignored).
  Earns **+25**.
- **Real Word list item** — bullet/numbered list detected via the underlying
  `w:numPr` XML (direct paragraph numbering or a list style). Earns **−20**.

A paragraph could in theory have both; both are recorded. Note that a Word
list's visible marker (`1.`, `•`) lives in the numbering XML, **not** in the
paragraph text — so a Word-generated `1. Improve running` never triggers the
textual-prefix bonus, while a manually typed `2.1 API Configuration:` never
triggers the list-item penalty.

## Word list metadata & marker reconstruction

For every list paragraph, `src/docx_numbering.py` resolves the chain
`w:numPr → w:numId → w:num → w:abstractNum → w:lvl` against
`word/numbering.xml` and exposes:

- `list_type` — `numbered`, `bullet`, or `unknown`,
- `list_level` — the nesting level (`w:ilvl`),
- `numbering_id` — the `w:numId` instance,
- `numbering_format` — `decimal`, `bullet`, `lowerLetter`, …,
- `list_marker` — the reconstructed visible marker (`1.`, `2.`, `a.`, `•`),
- `display_text` — marker + paragraph text (`1. Improve running`).

Markers for `decimal`, `lowerLetter`/`upperLetter` and `lowerRoman`/
`upperRoman` lists are reconstructed by counting items **in document order**,
honouring each level's `w:start` and `w:lvlText` template, and resetting deeper
counters whenever a shallower level advances (multi-level markers like `1.2.`
work). Symbol-font bullet glyphs are displayed as `•`.

When a numbering definition cannot be resolved (missing `numId`, undefined
level, exotic format), the marker is **`null`** and a warning is added — markers
are never invented.

## Manual line breaks inside one paragraph

A Word paragraph containing manual line breaks (`w:br`, Shift+Enter) is kept as
**one** paragraph in the hierarchy — segments are never promoted to separate
paragraphs or headings. The full text (with internal `\n`) is preserved, and
each non-empty line is additionally exposed as:

- `segments` — the list of non-empty lines,
- `segment_count`,
- `has_internal_line_breaks`.

These appear in the structured parser output for the affected paragraphs and in
the CSV (`segments` serialized as a `" | "`-separated string), so a downstream
LLM can address each line individually.

## Predicted heading level

For official headings the Word level is used. For heuristic headings the level
is inferred in order:

1. **Numbering depth** — `1` → 1, `1.2` → 2, `1.2.3` → 3.
2. **Relative font-size groups** in the same document — the largest heading font
   is level 1, the next distinct size level 2, and so on.
3. Repeated formatting groups.

Levels are capped to 1–9. Global assumptions ("18pt is always Heading 1") are
never used. If no reliable level can be inferred, `predicted_level` is `null`,
the paragraph is flagged for review, and it is not inserted into the confirmed
hierarchy.

## Repeated-format detection

A formatting signature is built from rounded font size, font family, bold,
underline, dominant colour, spacing before/after, and alignment. A pattern
counts as repeated when it appears at least twice **and** differs meaningfully
from the detected body formatting. Ordinary body paragraphs therefore never earn
the repeated bonus just because body text repeats.

## Running the analysis command

Inspect any DOCX with:

```bash
python3 -m src.analyze_headings path/to/document.docx
```

Write the full per-paragraph feature table to CSV:

```bash
python3 -m src.analyze_headings path/to/document.docx --output heading-analysis.csv
```

The terminal prints a compact table (text, style, score, classification,
predicted level, detection method, signals). The CSV additionally includes word
/ sentence counts, numbering prefix, list-item flag, font size/family/colour,
body font size/family, alignment, spacing, and the repeated-pattern flag.

## Current limitations & review requirements

- Spacing and some inherited formatting are only read as *direct* formatting;
  values inherited purely from a style may register as absent. The style base
  chain is walked for size/family/bold/underline/colour, but not for spacing.
- "Sentence-style" and title-case detection are heuristics and can misjudge
  unusual text (abbreviations, decimals, acronyms).
- Numbering detection intentionally ignores 4-digit numbers, so a genuine
  heading numbered above 99 would not get the numbering bonus.
- List-marker reconstruction does not honour `w:lvlOverride` / `w:lvlRestart`
  overrides, `w:numStyleLink` indirection, or exotic formats (`ordinalText`,
  `chicago`, …) — those markers come back as `null` with a warning rather than
  a guess. Numbering is counted per `numId` in document order, which matches
  ordinary lists but not documents that deliberately continue one numbering
  instance across differently-styled regions.
- Probable headings (60–79) **must** be reviewed by a human before they are
  allowed to drive automated document updates.
- The parser does not claim to handle every possible DOCX layout; it is a
  best-effort structural analyser, not a full Word renderer.
