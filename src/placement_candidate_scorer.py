"""Deterministic placement scoring *inside* one already-selected section.

Phase 2B picked the section; this module answers the next question — which
existing paragraph, list item, or sentence in that section is most relevant to
the code change — and returns an explainable top-three shortlist.

It is pure and offline: no LLM provider is imported, nothing is written, and
no Markdown is mutated. Callers that need a fresh index must call
``ensure_summary_index()`` themselves *before* scoring; this module never
rebuilds an index behind the caller's back and never returns offsets from an
index it knows to be stale.

Only candidate *locations* are produced here. Patch wording, patch operations,
and summary edits belong to later phases.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from src.change_package_reader import normalize_hunk
from src.section_candidate_scorer import (
    MAX_HUNK_TOKEN_BUDGET,
    MAX_HUNKS_PER_FILE,
    MAX_LINE_CHARS,
    MAX_LINES_PER_HUNK,
    infer_categories,
)
from src.summary_index_builder import (
    SUPPORTED_SCHEMA_VERSIONS,
    extract_keywords,
    is_structural_inventory,
)

# ---------------------------------------------------------------------------
# Weights and caps (centralized, inspectable, testable)
# ---------------------------------------------------------------------------
PLACEMENT_WEIGHTS: dict[str, float] = {
    "symbol_match": 16.0,        # an exact changed symbol appears in the text
    "module_match": 14.0,        # an exact module/file stem appears
    "path_match": 14.0,          # an exact changed path appears
    "removed_evidence": 5.0,     # per removed-line term (stale-text signal)
    "hunk_overlap": 4.0,         # per hunk-summary / hunk-header term
    "added_evidence": 3.0,       # per added-line term (extension signal)
    "category_overlap": 6.0,     # topical agreement with the change
    "summary_overlap": 2.0,      # per generated-summary term (supporting)
    "keyword_overlap": 1.0,      # per general shared keyword (weak)
}

# Every component is capped so a repeated token cannot inflate a score.
PLACEMENT_CAPS: dict[str, int] = {
    "symbol_match": 2,
    "module_match": 2,
    "path_match": 2,
    "removed_evidence": 3,
    "hunk_overlap": 3,
    "added_evidence": 3,
    "category_overlap": 1,
    "summary_overlap": 3,
    "keyword_overlap": 4,
}

MIN_CANDIDATE_SCORE = 8.0    # below this: append rather than edit
STRONG_SCORE = 22.0          # at/above this (unambiguous): strong placement
AMBIGUITY_MARGIN = 4.0       # first-vs-second gap under this => ambiguous
DEFAULT_TOP_K = 3

# Broad components (category / generated-summary / generic keyword overlap)
# describe topic, not location. Editing existing prose requires at least one
# of these *specific* components, which tie the change to that exact text.
SPECIFIC_COMPONENTS = frozenset(
    {"symbol_match", "module_match", "path_match", "removed_evidence",
     "hunk_overlap", "added_evidence"}
)

# Matched signals are for explanation, not exhaustive listing.
MAX_MATCHED_SIGNALS = 12

# Bounded neighbour context for the future prompt.
MAX_CONTEXT_EXCERPT_CHARS = 200
MAX_PARENT_TEXT_CHARS = 400

# Recommendations.
USE_EXISTING_CANDIDATE = "use_existing_candidate"
APPEND_TO_SECTION = "append_to_section"
MANUAL_REVIEW_NEEDED = "manual_review_needed"
NO_CHANGE_NEEDED = "no_change_needed"

# Granularity explanations.
_REASON_SENTENCE = "Most matched evidence occurs in one sentence."
_REASON_BLOCK = (
    "Relevant symbols and change terms are distributed across the block."
)

# Block types that may ever be edited (mirrors the index's own vocabulary).
_PATCHABLE_BLOCK_TYPES = frozenset(
    {"paragraph", "unordered_list_item", "ordered_list_item"}
)

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


class PlacementIndexError(RuntimeError):
    """The supplied index is unusable (schema, staleness, or structure)."""


# ---------------------------------------------------------------------------
# Change signals, with added/removed evidence kept separate
# ---------------------------------------------------------------------------
@dataclass
class PlacementSignals:
    """Change evidence, separated by how it should be interpreted.

    Unlike section routing (which merges diff text into one bag), placement
    distinguishes *removed* terms — which point at text that may now be stale
    — from *added* terms, which point at where new information belongs.
    """

    paths: set[str] = field(default_factory=set)
    modules: set[str] = field(default_factory=set)  # file stems
    symbols: set[str] = field(default_factory=set)
    removed_tokens: set[str] = field(default_factory=set)
    added_tokens: set[str] = field(default_factory=set)
    hunk_tokens: set[str] = field(default_factory=set)
    summary_tokens: set[str] = field(default_factory=set)
    categories: set[str] = field(default_factory=set)
    statuses: set[str] = field(default_factory=set)

    def meaningful_signals(self) -> set[str]:
        """Signals a candidate could plausibly match (for confidence)."""
        return (
            {p.lower() for p in self.paths}
            | {m.lower() for m in self.modules}
            | {s.lower() for s in self.symbols}
            | self.removed_tokens
            | self.added_tokens
        )


def _iter_changed_files(change_package: Mapping[str, Any]) -> list[dict]:
    """Normalize schema-v2 and older/v1 change packages to a common shape."""
    entries = change_package.get("changed_files") or []
    normalized: list[dict] = []
    for entry in entries:
        if isinstance(entry, dict):
            normalized.append(
                {
                    "path": entry.get("path") or "",
                    "old_path": entry.get("old_path"),
                    "status": entry.get("status") or entry.get("change_type") or "",
                    "binary": bool(entry.get("binary", False)),
                    "what_changed": entry.get("what_changed") or [],
                }
            )
        elif isinstance(entry, str):
            normalized.append(
                {"path": entry, "old_path": None, "status": "", "binary": False,
                 "what_changed": []}
            )
        else:  # ChangedFile-like object
            normalized.append(
                {
                    "path": getattr(entry, "path", "") or "",
                    "old_path": getattr(entry, "old_path", None),
                    "status": getattr(entry, "change_type", "") or "",
                    "binary": False,
                    "what_changed": [],
                }
            )
    return normalized


def extract_placement_signals(
    change_package: Mapping[str, Any]
) -> PlacementSignals:
    """Deterministic, bounded evidence for placement scoring.

    Every changed file contributes its path/stem/status; only diff *text* is
    budget-limited, reusing the existing schema-v2 truncation constants.
    """
    signals = PlacementSignals()

    for entry in _iter_changed_files(change_package):
        if entry["status"]:
            signals.statuses.add(str(entry["status"]).lower())

        for path in (entry["path"], entry.get("old_path")):
            if not path:
                continue
            signals.paths.add(path)
            stem = Path(path).stem
            if stem:
                signals.modules.add(stem)

        for hunk in (entry.get("what_changed") or [])[:MAX_HUNKS_PER_FILE]:
            if not isinstance(hunk, dict):
                continue
            # One normalization point handles v2 per-line arrays and v3 blocks.
            normalized = normalize_hunk(hunk)
            for symbol in normalized.symbols:
                signals.symbols.add(str(symbol))

            if len(signals.hunk_tokens) >= MAX_HUNK_TOKEN_BUDGET:
                continue

            if normalized.summary:
                signals.hunk_tokens.update(extract_keywords(normalized.summary))
            if normalized.hunk_header:
                signals.hunk_tokens.update(extract_keywords(normalized.hunk_header))

            for text in normalized.removed_lines[:MAX_LINES_PER_HUNK]:
                if text:
                    signals.removed_tokens.update(
                        extract_keywords(text[:MAX_LINE_CHARS])
                    )
            for text in normalized.added_lines[:MAX_LINES_PER_HUNK]:
                if text:
                    signals.added_tokens.update(
                        extract_keywords(text[:MAX_LINE_CHARS])
                    )

    signals.summary_tokens.update(
        extract_keywords(change_package.get("generated_summary") or "")
    )
    signals.categories = infer_categories(
        " ".join(sorted(signals.paths | signals.modules))
    )
    return signals


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------
@dataclass
class PlacementCandidate:
    """One scored, addressable location inside the selected section."""

    candidate_id: str
    candidate_type: str  # "sentence" | "block"
    section_id: str
    block_id: str
    sentence_id: Optional[str]
    block_type: str
    text: str
    score: float
    rank: int = 0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    matched_signals: list[str] = field(default_factory=list)
    source_start_offset: int = 0
    source_end_offset: int = 0
    start_line: int = 0
    end_line: int = 0
    block_order: int = 0
    sentence_order: int = 0
    granularity_reason: str = ""
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "candidate_type": self.candidate_type,
            "section_id": self.section_id,
            "block_id": self.block_id,
            "sentence_id": self.sentence_id,
            "block_type": self.block_type,
            "text": self.text,
            "score": round(self.score, 3),
            "rank": self.rank,
            "score_breakdown": {
                key: round(value, 3) for key, value in self.score_breakdown.items()
            },
            "matched_signals": list(self.matched_signals),
            "source_start_offset": self.source_start_offset,
            "source_end_offset": self.source_end_offset,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "granularity_reason": self.granularity_reason,
            "context": self.context,
        }


@dataclass
class PlacementAssessment:
    """Deterministic verdict for one section."""

    section_id: str
    candidates: list[PlacementCandidate] = field(default_factory=list)
    confidence: float = 0.0
    ambiguous: bool = False
    recommendation: str = MANUAL_REVIEW_NEEDED
    reasoning: str = ""

    @property
    def top(self) -> Optional[PlacementCandidate]:
        return self.candidates[0] if self.candidates else None

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "candidates": [c.to_dict() for c in self.candidates],
            "confidence": round(self.confidence, 3),
            "ambiguous": self.ambiguous,
            "recommendation": self.recommendation,
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# Index validation
# ---------------------------------------------------------------------------
def _find_section(summary_index: Mapping[str, Any], section_id: str) -> dict:
    if summary_index.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise PlacementIndexError(
            f"Unsupported summary-index schema version "
            f"{summary_index.get('schema_version')!r}; supported: "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}."
        )
    sections = summary_index.get("sections")
    if not isinstance(sections, list):
        raise PlacementIndexError("Summary index has no 'sections' list.")
    for section in sections:
        if isinstance(section, dict) and section.get("section_id") == section_id:
            return section
    raise PlacementIndexError(
        f"Section {section_id!r} is not present in the summary index; "
        "refusing to score a different section."
    )


def _verify_source_hash(
    summary_index: Mapping[str, Any], source_markdown: str
) -> None:
    stored = (summary_index.get("source") or {}).get("sha256")
    actual = hashlib.sha256(source_markdown.encode("utf-8")).hexdigest()
    if stored != actual:
        raise PlacementIndexError(
            "Summary index is stale: its stored source SHA-256 does not match "
            "the supplied Markdown. Rebuild the index (ensure_summary_index) "
            "before scoring; offsets from a stale index are unsafe."
        )


def _eligible_blocks(section: Mapping[str, Any]) -> list[dict]:
    """Patchable, non-generated prose blocks only.

    Code blocks, tables, HTML comments, generated update regions, and anything
    flagged ``patchable: false`` are excluded — even when they contain an exact
    changed symbol or path.
    """
    if section.get("generated"):
        return []
    eligible: list[dict] = []
    for block in section.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        if block.get("generated"):
            continue
        if not block.get("patchable"):
            continue
        if block.get("block_type") not in _PATCHABLE_BLOCK_TYPES:
            continue
        # Bare file/module inventory is never a patch target. Newer indexes
        # carry ``structural_inventory``; older ones are re-checked here so a
        # pre-existing index needs no migration.
        if block.get("structural_inventory") or is_structural_inventory(
            block.get("text", "")
        ):
            continue
        eligible.append(block)
    return eligible


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _capped(component: str, count: int) -> float:
    return min(count, PLACEMENT_CAPS[component]) * PLACEMENT_WEIGHTS[component]


def _word_set(text: str) -> set[str]:
    """Whole-word tokens, so ``route`` never matches inside ``router``."""
    return {word.lower() for word in _WORD_RE.findall(text or "")}


def _score_text(
    text: str, signals: PlacementSignals
) -> tuple[dict[str, float], list[str]]:
    """Score one piece of indexed text; returns (breakdown, matched signals)."""
    breakdown: dict[str, float] = {}
    matched: list[str] = []
    lowered = (text or "").lower()
    words = _word_set(text)
    keywords = set(extract_keywords(text))

    # Direct evidence uses exact whole-token / substring-path matching.
    symbol_hits = sorted(
        {s for s in signals.symbols if s and _symbol_in(s, words, lowered)}
    )
    if symbol_hits:
        breakdown["symbol_match"] = _capped("symbol_match", len(symbol_hits))
        matched.extend(symbol_hits[: PLACEMENT_CAPS["symbol_match"]])

    path_hits = sorted({p for p in signals.paths if p and p.lower() in lowered})
    if path_hits:
        breakdown["path_match"] = _capped("path_match", len(path_hits))
        matched.extend(path_hits[: PLACEMENT_CAPS["path_match"]])

    # A module stem only counts on its own when the full path did not match,
    # so one file cannot be paid twice for the same mention.
    module_hits = sorted(
        {
            m for m in signals.modules
            if m and m.lower() in words and not any(m.lower() in p.lower() for p in path_hits)
        }
    )
    if module_hits:
        breakdown["module_match"] = _capped("module_match", len(module_hits))
        matched.extend(module_hits[: PLACEMENT_CAPS["module_match"]])

    removed_hits = sorted(signals.removed_tokens & keywords)
    if removed_hits:
        breakdown["removed_evidence"] = _capped("removed_evidence", len(removed_hits))
        matched.extend(removed_hits[: PLACEMENT_CAPS["removed_evidence"]])

    hunk_hits = sorted(signals.hunk_tokens & keywords)
    if hunk_hits:
        breakdown["hunk_overlap"] = _capped("hunk_overlap", len(hunk_hits))
        matched.extend(hunk_hits[: PLACEMENT_CAPS["hunk_overlap"]])

    added_hits = sorted(signals.added_tokens & keywords)
    if added_hits:
        breakdown["added_evidence"] = _capped("added_evidence", len(added_hits))
        matched.extend(added_hits[: PLACEMENT_CAPS["added_evidence"]])

    shared_categories = sorted(signals.categories & infer_categories(text or ""))
    if shared_categories:
        breakdown["category_overlap"] = _capped("category_overlap", 1)
        matched.extend(shared_categories[: PLACEMENT_CAPS["category_overlap"]])

    summary_hits = sorted(signals.summary_tokens & keywords)
    if summary_hits:
        breakdown["summary_overlap"] = _capped("summary_overlap", len(summary_hits))
        matched.extend(summary_hits[: PLACEMENT_CAPS["summary_overlap"]])

    general = sorted(
        (signals.added_tokens | signals.removed_tokens | signals.hunk_tokens)
        & keywords
    )
    if general:
        breakdown["keyword_overlap"] = _capped("keyword_overlap", len(general))
        matched.extend(general[: PLACEMENT_CAPS["keyword_overlap"]])

    # Every scoring component contributes an explanation; bounded and stable.
    seen: set[str] = set()
    unique_matched = [m for m in matched if not (m in seen or seen.add(m))]
    return breakdown, unique_matched[:MAX_MATCHED_SIGNALS]


def _symbol_in(symbol: str, words: set[str], lowered: str) -> bool:
    """Exact symbol match; a dotted name matches its final component too."""
    lower_symbol = symbol.lower()
    if lower_symbol in words:
        return True
    if "." in lower_symbol:
        return lower_symbol in lowered or lower_symbol.rsplit(".", 1)[-1] in words
    return False


def _direct_component(candidate: PlacementCandidate) -> float:
    return (
        candidate.score_breakdown.get("symbol_match", 0.0)
        + candidate.score_breakdown.get("module_match", 0.0)
        + candidate.score_breakdown.get("path_match", 0.0)
    )


def _bounded(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit]


def _build_context(
    section: Mapping[str, Any],
    blocks: list[dict],
    block_index: int,
    parent_text: Optional[str],
) -> dict:
    """Bounded neighbour context for a future prompt (never a whole section)."""
    context: dict = {
        "section_heading": section.get("heading", ""),
        "section_heading_path": list(section.get("heading_path") or []),
    }
    if block_index > 0:
        previous = blocks[block_index - 1]
        context["previous_block_id"] = previous.get("block_id")
        context["previous_excerpt"] = _bounded(
            previous.get("text", ""), MAX_CONTEXT_EXCERPT_CHARS
        )
    if block_index < len(blocks) - 1:
        following = blocks[block_index + 1]
        context["next_block_id"] = following.get("block_id")
        context["next_excerpt"] = _bounded(
            following.get("text", ""), MAX_CONTEXT_EXCERPT_CHARS
        )
    if parent_text is not None:
        context["parent_block_text"] = _bounded(parent_text, MAX_PARENT_TEXT_CHARS)
    return context


def score_placement_candidates(
    change_package: Mapping[str, Any],
    summary_index: Mapping[str, Any],
    section_id: str,
    *,
    source_markdown: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
) -> PlacementAssessment:
    """Rank existing locations in ``section_id`` against a change.

    Raises :class:`PlacementIndexError` for an unsupported schema, a stale
    source hash, or a missing section — the caller decides whether to rebuild
    the index; this function never does so implicitly.
    """
    if source_markdown is not None:
        _verify_source_hash(summary_index, source_markdown)
    section = _find_section(summary_index, section_id)

    signals = extract_placement_signals(change_package)
    blocks = _eligible_blocks(section)

    if not blocks:
        return PlacementAssessment(
            section_id=section_id,
            confidence=0.0,
            recommendation=APPEND_TO_SECTION,
            reasoning=(
                "The selected section has no patchable prose blocks; new "
                "content must be appended rather than edited."
            ),
        )

    all_candidates: list[PlacementCandidate] = []
    block_lookup: dict[str, PlacementCandidate] = {}
    sentences_by_block: dict[str, list[PlacementCandidate]] = {}

    for block_index, block in enumerate(blocks):
        block_breakdown, block_matched = _score_text(block.get("text", ""), signals)
        block_candidate = PlacementCandidate(
            candidate_id=block["block_id"],
            candidate_type="block",
            section_id=section_id,
            block_id=block["block_id"],
            sentence_id=None,
            block_type=block.get("block_type", ""),
            text=block.get("text", ""),
            score=sum(block_breakdown.values()),
            score_breakdown=block_breakdown,
            matched_signals=block_matched,
            source_start_offset=block.get("source_start_offset", 0),
            source_end_offset=block.get("source_end_offset", 0),
            start_line=block.get("start_line", 0),
            end_line=block.get("end_line", 0),
            block_order=block_index,
        )
        block_lookup[block["block_id"]] = block_candidate
        all_candidates.append(block_candidate)

        sentence_candidates: list[PlacementCandidate] = []
        for sentence_index, sentence in enumerate(block.get("sentences") or []):
            if not isinstance(sentence, dict):
                continue
            breakdown, matched = _score_text(sentence.get("text", ""), signals)
            sentence_candidates.append(
                PlacementCandidate(
                    candidate_id=sentence["sentence_id"],
                    candidate_type="sentence",
                    section_id=section_id,
                    block_id=block["block_id"],
                    sentence_id=sentence["sentence_id"],
                    block_type=block.get("block_type", ""),
                    text=sentence.get("text", ""),
                    score=sum(breakdown.values()),
                    score_breakdown=breakdown,
                    matched_signals=matched,
                    source_start_offset=sentence.get("source_start_offset", 0),
                    source_end_offset=sentence.get("source_end_offset", 0),
                    start_line=sentence.get("start_line", 0),
                    end_line=sentence.get("end_line", 0),
                    block_order=block_index,
                    sentence_order=sentence_index,
                )
            )
        sentences_by_block[block["block_id"]] = sentence_candidates
        all_candidates.extend(sentence_candidates)

    selected = _select_granularity(block_lookup, sentences_by_block)
    ranked = _rank(selected)
    shortlist = ranked[:top_k]

    for rank, candidate in enumerate(shortlist, start=1):
        candidate.rank = rank
        parent_text = (
            block_lookup[candidate.block_id].text
            if candidate.candidate_type == "sentence"
            else None
        )
        candidate.context = _build_context(
            section, blocks, candidate.block_order, parent_text
        )

    return _assess(section_id, shortlist, signals)


def _select_granularity(
    block_lookup: dict[str, PlacementCandidate],
    sentences_by_block: dict[str, list[PlacementCandidate]],
) -> list[PlacementCandidate]:
    """Choose exactly one representative per block: its best sentence, or itself.

    Granularity follows where the *direct* evidence (symbols, modules, paths)
    sits. A block always accumulates at least its sentences' weak keyword
    points, so raw totals cannot decide this; direct evidence can:

    * a block with one sentence is already a compact unit -> block;
    * one sentence holding all of the block's direct evidence -> sentence
      (concentrated);
    * direct evidence split across sentences -> block (distributed).

    Returning one representative per block also guarantees the shortlist never
    holds both a block and its own child sentence, nor two sentences from one
    block.
    """
    representatives: list[PlacementCandidate] = []
    for block_id, block_candidate in block_lookup.items():
        sentences = sentences_by_block.get(block_id) or []
        best_sentence = None
        if sentences:
            best_sentence = min(
                sentences,
                key=lambda c: (
                    -_direct_component(c), -c.score, c.sentence_order, c.candidate_id
                ),
            )

        block_direct = _direct_component(block_candidate)
        sentence_direct = _direct_component(best_sentence) if best_sentence else 0.0

        concentrated = (
            best_sentence is not None
            and len(sentences) > 1
            and (
                (sentence_direct > 0 and sentence_direct >= block_direct)
                or (block_direct == 0 and best_sentence.score >= block_candidate.score)
            )
        )

        if concentrated:
            best_sentence.granularity_reason = _REASON_SENTENCE
            representatives.append(best_sentence)
        else:
            block_candidate.granularity_reason = (
                _REASON_BLOCK if block_candidate.score > 0 else "No matched evidence."
            )
            representatives.append(block_candidate)
    return representatives


def _rank(candidates: list[PlacementCandidate]) -> list[PlacementCandidate]:
    """Deterministic ordering: score, direct evidence, block/sentence order, id."""
    return sorted(
        candidates,
        key=lambda c: (
            -c.score,
            -_direct_component(c),
            c.block_order,
            c.sentence_order,
            c.candidate_id,
        ),
    )


def _assess(
    section_id: str,
    shortlist: list[PlacementCandidate],
    signals: PlacementSignals,
) -> PlacementAssessment:
    """Confidence, ambiguity, and recommendation from the evidence."""
    scoring = [c for c in shortlist if c.score > 0]
    if not scoring or shortlist[0].score < MIN_CANDIDATE_SCORE:
        return PlacementAssessment(
            section_id=section_id,
            candidates=shortlist,
            confidence=0.0,
            ambiguous=False,
            recommendation=APPEND_TO_SECTION,
            reasoning=(
                "No existing location scored above the minimum "
                f"({MIN_CANDIDATE_SCORE}); new content should be appended to "
                "the section rather than edited in place."
            ),
        )

    # Broad topical overlap alone must never authorize editing existing prose:
    # it says the section is related, not that *this* text is now wrong.
    if not any(
        set(candidate.score_breakdown) & SPECIFIC_COMPONENTS
        for candidate in shortlist
    ):
        return PlacementAssessment(
            section_id=section_id,
            candidates=shortlist,
            confidence=0.0,
            ambiguous=False,
            recommendation=APPEND_TO_SECTION,
            reasoning=(
                "No safe existing target was found: candidates matched only "
                "broad signals (category/summary/keyword overlap) with no "
                "changed symbol, module, path, or diff-line evidence. New "
                "content should be appended rather than replacing existing text."
            ),
        )

    top = shortlist[0]
    if not (set(top.score_breakdown) & SPECIFIC_COMPONENTS):
        # A weaker-but-specific candidate is a safer target than a broad leader.
        specific = [
            candidate for candidate in shortlist
            if set(candidate.score_breakdown) & SPECIFIC_COMPONENTS
        ]
        shortlist = specific + [c for c in shortlist if c not in specific]
        for rank, candidate in enumerate(shortlist, start=1):
            candidate.rank = rank
        top = shortlist[0]
    runner_up = shortlist[1].score if len(shortlist) > 1 else 0.0
    margin = top.score - runner_up
    ambiguous = len(shortlist) > 1 and margin < AMBIGUITY_MARGIN

    meaningful = signals.meaningful_signals()
    matched_fraction = (
        len({m.lower() for m in top.matched_signals} & meaningful) / len(meaningful)
        if meaningful
        else 0.0
    )

    confidence = min(top.score / 40.0, 0.75)
    confidence += min(margin / 25.0, 0.12)
    if _direct_component(top) > 0:
        confidence += 0.1
    confidence += min(matched_fraction * 0.1, 0.1)
    confidence = round(max(0.0, min(confidence, 0.99)), 3)

    if ambiguous:
        reasoning = (
            f"Top candidates are within {AMBIGUITY_MARGIN} points "
            f"({top.score:.1f} vs {runner_up:.1f}); review the chosen location."
        )
    elif top.score >= STRONG_SCORE:
        reasoning = (
            f"Strong placement at {top.score:.1f} points: "
            f"{', '.join(top.matched_signals[:3]) or 'matched change terms'}."
        )
    else:
        reasoning = f"Leading placement at {top.score:.1f} points."

    return PlacementAssessment(
        section_id=section_id,
        candidates=shortlist,
        confidence=confidence,
        ambiguous=ambiguous,
        recommendation=USE_EXISTING_CANDIDATE,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Read-only CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    """Print a placement assessment as JSON. Writes nothing, calls no LLM."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Score placement candidates inside one summary section."
    )
    parser.add_argument("--change-package", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--section-id", required=True)
    parser.add_argument("--source", default=None, help="Markdown for hash checking.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    arguments = parser.parse_args(argv)

    try:
        change_package = json.loads(
            Path(arguments.change_package).read_text(encoding="utf-8")
        )
        summary_index = json.loads(Path(arguments.index).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"[placement] could not read input: {error}", file=sys.stderr)
        return 2

    source_markdown = None
    if arguments.source:
        source_markdown = Path(arguments.source).read_text(encoding="utf-8")

    try:
        assessment = score_placement_candidates(
            change_package,
            summary_index,
            arguments.section_id,
            source_markdown=source_markdown,
            top_k=arguments.top_k,
        )
    except PlacementIndexError as error:
        print(f"[placement] {error}", file=sys.stderr)
        return 3

    print(json.dumps(assessment.to_dict(), indent=2, ensure_ascii=False))
    print("[placement] read-only preview; no files were written.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
