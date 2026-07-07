"""Transparent, explainable heading scoring for DOCX paragraphs.

This module holds all the *decision* logic for the formatting heuristic:

* the scoring weights and thresholds (kept in dictionaries so they are easy
  to tune later),
* pure text helpers (numbering-prefix extraction, sentence counting, colon and
  full-stop detection, title-case / all-caps detection),
* the Note/Link hard-negative rule,
* the extensible combination-rule system,
* score clamping, classification and predicted-level helpers.

It deliberately does **not** import python-docx. It works on a lightweight
``features`` object (see :class:`src.docx_feature_extractor.ParagraphFeatures`)
by reading attributes, which keeps this module dependency-free and easy to
unit-test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
#
# All scoring weights live here so they can be adjusted without touching the
# scoring logic below. Positive values push a paragraph towards "heading";
# negative values push it towards "normal content".
HEADING_SCORE_WEIGHTS: dict[str, int] = {
    # positive signals
    "numbering_prefix": 25,
    "bold": 15,
    "larger_font": 20,
    "different_font_family": 5,
    "underlined": 5,
    "few_words": 10,
    "colon_ending": 10,
    "spacing_before": 10,
    "spacing_after": 5,
    "repeated_pattern": 20,
    "title_or_caps": 5,
    "no_full_stop": 5,
    # negative signals
    "word_list_item": -20,
    "many_words": -20,
    "multiple_sentences": -20,
    "full_stop_ending": -10,
    "body_formatting": -15,
    "sentence_style": -10,
    "colored_text": -20,
}

# Tunable thresholds. Kept separate from the weights so the two concerns do not
# get tangled together.
SCORING_CONFIG: dict[str, float] = {
    # "clearly larger than body" means at least this many points above the
    # dynamically-detected body font size.
    "larger_font_min_delta": 1.0,
    # fewer than this many words earns the "few_words" bonus.
    "few_words_max": 10,
    # more than this many words earns the "many_words" penalty.
    "many_words_min": 15,
    # classification thresholds
    "heading_min": 80,
    "probable_min": 60,
    # font-size equality tolerance when deciding "same as body".
    "body_size_tolerance": 0.5,
}

# Colon-style endings that count for the "colon_ending" bonus. Includes the
# hyphen and en/em-dash variants requested in the spec.
COLON_ENDINGS: tuple[str, ...] = (":", ":-", ":–", ":—")

# Small words ignored when deciding whether text is Title Case.
_TITLE_CASE_STOPWORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "in", "nor", "of", "on",
    "or", "the", "to", "up", "via", "with",
}

# Numbering prefix such as "1", "1.", "1)", "1:", "1 -", "1.2", "3.4.1".
# Each numeric segment is limited to 1-2 digits so that ordinary numbers like
# years ("1999 was ...") are not mistaken for section numbering. Anchored at the
# start so mid-sentence numbers are ignored.
_NUMBERING_PREFIX_RE = re.compile(
    r"^\s*(\d{1,2}(?:\.\d{1,2})*)\s*[.):\-–—]?\s+(?=\S)"
)

# Note/Link hard-negative rule. Matches "Note"/"Link" at the start followed by
# optional whitespace and a colon or dash (with optional trailing dash), e.g.
# "Note:", "Note:-", "Note -", "Note-", "Link:". Case-insensitive.
_HARD_NEGATIVE_RE = re.compile(
    r"^\s*(note|link)\s*(?::[-–—]?|[-–—])",
    re.IGNORECASE,
)

# Sentence boundary detector (used after masking decimals/numbering).
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]+(?:\s|$)")
_DECIMAL_RE = re.compile(r"\d{1,2}(?:\.\d{1,2})+")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ScoringResult:
    """Outcome of scoring a single paragraph."""

    score: int
    classification: str
    detection_method: str
    signals: list[str] = field(default_factory=list)
    predicted_level: Optional[int] = None


# ---------------------------------------------------------------------------
# Pure text helpers
# ---------------------------------------------------------------------------
def extract_numbering_prefix(text: str) -> Optional[str]:
    """Return the textual section-number prefix (e.g. ``"2.1"``) or ``None``.

    Only matches numbering at the very start of the paragraph. Numbers in the
    middle of ordinary text are ignored.
    """
    match = _NUMBERING_PREFIX_RE.match(text or "")
    if match is None:
        return None
    return match.group(1)


def numbering_depth(prefix: Optional[str]) -> Optional[int]:
    """Return the nesting depth implied by a numbering prefix.

    ``"1"`` -> 1, ``"1.2"`` -> 2, ``"1.2.3"`` -> 3.
    """
    if not prefix:
        return None
    return prefix.count(".") + 1


def count_sentences(text: str) -> int:
    """Count sentence-like segments in ``text``.

    Decimal numbers and section numbering (``2.1``) are masked first so their
    dots are not counted as sentence boundaries. A non-empty paragraph always
    counts as at least one sentence.
    """
    stripped = (text or "").strip()
    if not stripped:
        return 0

    # Strip a leading section-number prefix (e.g. "1." or "2.1") so its dot is
    # not mistaken for a sentence boundary, then mask any remaining decimals.
    without_prefix = _NUMBERING_PREFIX_RE.sub("", stripped)
    masked = _DECIMAL_RE.sub(" ", without_prefix or stripped)
    segments = [
        segment
        for segment in _SENTENCE_BOUNDARY_RE.split(masked)
        if segment.strip()
    ]
    return max(len(segments), 1)


def has_colon_ending(text: str) -> bool:
    """Return ``True`` when the text ends with ``:``/``:-``/``:–``/``:—``."""
    return (text or "").rstrip().endswith(COLON_ENDINGS)


def has_full_stop_ending(text: str) -> bool:
    """Return ``True`` when the text ends with a full stop (and not a colon)."""
    stripped = (text or "").rstrip()
    if not stripped:
        return False
    if has_colon_ending(stripped):
        return False
    return stripped.endswith(".")


def is_title_case(text: str) -> bool:
    """Return ``True`` when every significant word starts with a capital.

    Small connecting words (``the``, ``of``, ``and`` ...) are ignored so that
    natural title casing is recognised.
    """
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", text or "")
    if not words:
        return False
    significant = [w for w in words if w.lower() not in _TITLE_CASE_STOPWORDS]
    if not significant:
        return False
    return all(word[0].isupper() for word in significant)


def is_all_caps(text: str) -> bool:
    """Return ``True`` when the text contains letters and they are all caps."""
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return False
    return all(c.isupper() for c in letters)


def hard_negative_prefix(text: str) -> Optional[str]:
    """Return ``"note"``/``"link"`` if the text is a Note/Link line, else None.

    Only triggers when Note/Link begins the paragraph and is immediately
    followed (ignoring whitespace) by a colon or dash.
    """
    match = _HARD_NEGATIVE_RE.match(text or "")
    if match is None:
        return None
    return match.group(1).lower()


# ---------------------------------------------------------------------------
# Body-formatting comparison helpers (operate on a features object)
# ---------------------------------------------------------------------------
def _larger_than_body(features: Any, config: dict[str, float]) -> bool:
    if features.font_size is None or features.body_font_size is None:
        return False
    return features.font_size >= features.body_font_size + config["larger_font_min_delta"]


def _different_font_family(features: Any) -> bool:
    return bool(
        features.font_family
        and features.body_font_family
        and features.font_family != features.body_font_family
    )


def _extra_spacing_before(features: Any) -> bool:
    return features.spacing_before is not None and features.spacing_before > 0


def _extra_spacing_after(features: Any) -> bool:
    return features.spacing_after is not None and features.spacing_after > 0


def is_body_formatting(features: Any, config: dict[str, float] = SCORING_CONFIG) -> bool:
    """Return ``True`` when a paragraph's formatting matches the body text.

    Body text is: not bold, not underlined, not coloured, the same font family
    as the detected body, and within ``body_size_tolerance`` points of the body
    font size.
    """
    if features.is_bold or features.is_underlined or features.is_colored:
        return False
    if (
        features.font_family
        and features.body_font_family
        and features.font_family != features.body_font_family
    ):
        return False
    if (
        features.font_size is not None
        and features.body_font_size is not None
        and abs(features.font_size - features.body_font_size) > config["body_size_tolerance"]
    ):
        return False
    return True


def is_sentence_style(features: Any, body_formatting: bool) -> bool:
    """Heuristic for "reads like a body sentence, not a heading".

    Counts independent signals and returns ``True`` when at least two agree:

    * more than ten words,
    * ends with sentence punctuation (full stop),
    * more than one sentence,
    * uses ordinary body formatting.

    Using a 2-of-4 vote avoids a single property dominating the decision and
    keeps the penalty from double-counting the same evidence.
    
    Basically contains the scoring rules, point values, thresholds, classifications, and some text-based checks.
    """
    signals = 0
    if features.word_count > 10:
        signals += 1
    if features.ends_with_full_stop:
        signals += 1
    if features.sentence_count > 1:
        signals += 1
    if body_formatting:
        signals += 1
    return signals >= 2


# ---------------------------------------------------------------------------
# Combination rules (extensible)
# ---------------------------------------------------------------------------
@dataclass
class CombinationRule:
    """A hard-coded strong combination that raises the score to a floor.

    Add new rules by appending to :data:`COMBINATION_RULES`.
    """

    name: str
    predicate: Callable[[Any], bool]
    min_score: int
    signal: str


COMBINATION_RULES: list[CombinationRule] = [
    CombinationRule(
        name="numbering+bold+colon",
        predicate=lambda f: bool(
            f.numbering_prefix is not None and f.is_bold and f.ends_with_colon
        ),
        min_score=90,
        signal="combination:numbering+bold+colon:min_90",
    ),
]


def apply_combination_rules(
    features: Any,
    score: int,
    signals: list[str],
    rules: list[CombinationRule] = COMBINATION_RULES,
) -> int:
    """Raise ``score`` to any matching combination-rule floor.

    Returns the (possibly increased) score and appends a signal for every rule
    that fired. Never lowers the score.
    """
    for rule in rules:
        if rule.predicate(features):
            if score < rule.min_score:
                score = rule.min_score
            signals.append(rule.signal)
    return score


# ---------------------------------------------------------------------------
# Score, clamp, classify, level
# ---------------------------------------------------------------------------
def clamp_score(score: int) -> int:
    """Clamp a raw score into the inclusive 0-100 range."""
    return max(0, min(100, score))


def classify(score: int, config: dict[str, float] = SCORING_CONFIG) -> str:
    """Map a clamped score to a classification label."""
    if score >= config["heading_min"]:
        return "heading"
    if score >= config["probable_min"]:
        return "probable_heading"
    return "normal_content"


def _add(
    condition: bool,
    key: str,
    signals: list[str],
    weights: dict[str, int],
) -> int:
    """Apply one weighted rule, recording a readable signal when it fires."""
    if not condition:
        return 0
    weight = weights[key]
    sign = "+" if weight >= 0 else ""
    signals.append(f"{key}:{sign}{weight}")
    return weight


def score_paragraph(
    features: Any,
    weights: dict[str, int] = HEADING_SCORE_WEIGHTS,
    config: dict[str, float] = SCORING_CONFIG,
) -> ScoringResult:
    """Score a single paragraph and return a fully explained result.

    Priority order (highest first):

    1. Official Word heading style -> score 100.
    2. Note/Link hard-negative rule -> forced score 0, normal content.
    3. Formatting heuristic -> transparent 0-100 score with signals.
    """
    # 1. Official Word heading styles always win.
    if features.is_official_heading:
        return ScoringResult(
            score=100,
            classification="heading",
            detection_method="official_word_heading_style",
            signals=["official_word_heading_style"],
            predicted_level=features.official_heading_level,
        )

    # 2. Note/Link hard-negative rule overrides every positive rule.
    negative_prefix = hard_negative_prefix(features.text)
    if negative_prefix is not None:
        return ScoringResult(
            score=0,
            classification="normal_content",
            detection_method="hard_negative_rule",
            signals=[f"hard_negative_prefix:{negative_prefix}"],
            predicted_level=None,
        )

    # 3. Formatting heuristic.
    signals: list[str] = []
    score = 0

    # Positive signals.
    score += _add(features.numbering_prefix is not None, "numbering_prefix", signals, weights)
    score += _add(features.is_bold, "bold", signals, weights)
    score += _add(_larger_than_body(features, config), "larger_font", signals, weights)
    score += _add(_different_font_family(features), "different_font_family", signals, weights)
    score += _add(features.is_underlined, "underlined", signals, weights)
    score += _add(features.word_count < config["few_words_max"], "few_words", signals, weights)
    score += _add(features.ends_with_colon, "colon_ending", signals, weights)
    score += _add(_extra_spacing_before(features), "spacing_before", signals, weights)
    score += _add(_extra_spacing_after(features), "spacing_after", signals, weights)
    score += _add(features.repeated_formatting_pattern, "repeated_pattern", signals, weights)
    score += _add(
        features.is_title_case or features.is_all_caps, "title_or_caps", signals, weights
    )
    score += _add(not features.ends_with_full_stop, "no_full_stop", signals, weights)

    # Negative signals.
    body_formatting = is_body_formatting(features, config)
    score += _add(features.is_word_list_item, "word_list_item", signals, weights)
    score += _add(features.word_count > config["many_words_min"], "many_words", signals, weights)
    score += _add(features.sentence_count > 1, "multiple_sentences", signals, weights)
    score += _add(features.ends_with_full_stop, "full_stop_ending", signals, weights)
    score += _add(body_formatting, "body_formatting", signals, weights)
    score += _add(is_sentence_style(features, body_formatting), "sentence_style", signals, weights)
    score += _add(features.is_colored, "colored_text", signals, weights)

    # Strong combination rules (may raise to a floor, never override 1 or 2).
    score = apply_combination_rules(features, score, signals)

    score = clamp_score(score)
    classification = classify(score, config)
    predicted_level = numbering_depth(features.numbering_prefix)
    if predicted_level is not None:
        predicted_level = min(max(predicted_level, 1), 9)

    return ScoringResult(
        score=score,
        classification=classification,
        detection_method="formatting_heuristic",
        signals=signals,
        predicted_level=predicted_level,
    )
