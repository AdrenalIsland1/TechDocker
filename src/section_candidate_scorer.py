"""Deterministic, explainable scoring of the *actual* summary sections.

The router must never assume a fixed heading vocabulary: future summaries may
use "Quality and Tests", "CI/CD Review Flow", or "Dependencies and
Environment" instead of the original eight headings. This module therefore

1. extracts bounded signals from a change (schema-v2 aware, v1 tolerant),
2. builds a catalog from whatever sections the skeleton actually contains,
3. scores every eligible section with named, capped weights,
4. returns a deterministic top-N shortlist with a score breakdown.

Semantic categories are only an internal bridge between change signals and
real headings — they are never treated as required or suggested headings.
No embeddings, no model scores, no network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from src.markdown_summary_parser import parse_markdown_sections
from src.summary_index_builder import extract_keywords
from src.summary_skeleton_store import SummarySkeleton

# ---------------------------------------------------------------------------
# Internal semantic categories (a bridge, never a heading vocabulary)
# ---------------------------------------------------------------------------
TESTING = "testing"
CI_DEPLOYMENT = "ci_deployment"
CONFIGURATION_DEPENDENCIES = "configuration_dependencies"
AUTOMATION_PIPELINE = "automation_pipeline"
ARCHITECTURE_MODULES = "architecture_modules"
REPOSITORY_STRUCTURE = "repository_structure"
OVERVIEW = "overview"
LIMITATIONS_RISKS = "limitations_risks"
DOCUMENTATION = "documentation"

# Aliases are matched against section headings/paths/content to infer which
# category a *real* section covers. Extend this map; do not turn it into a
# list of expected headings.
CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    TESTING: (
        "tests", "test", "testing", "test suite", "test coverage", "validation",
        "quality", "qa", "pytest", "unit tests", "integration tests", "coverage",
    ),
    CI_DEPLOYMENT: (
        "ci", "cd", "ci/cd", "deployment", "deploy", "workflow", "workflows",
        "github actions", "actions", "build", "release", "pipeline runner",
        "automation runner", "continuous integration",
    ),
    CONFIGURATION_DEPENDENCIES: (
        "configuration", "config", "settings", "environment", "dependencies",
        "requirements", "packages", "pyproject", "setup", "install",
    ),
    AUTOMATION_PIPELINE: (
        "automation", "pipeline", "routing", "router", "updater", "generator",
        "change detection", "change package", "summary update", "orchestration",
        "pr workflow", "detector",
    ),
    ARCHITECTURE_MODULES: (
        "architecture", "modules", "module", "components", "component",
        "implementation", "source code", "core", "providers", "parsers",
        "analyzers", "services", "internals", "design",
    ),
    REPOSITORY_STRUCTURE: (
        "repository", "repo", "structure", "layout", "file tree", "organization",
        "directories", "tree",
    ),
    OVERVIEW: (
        "overview", "purpose", "introduction", "intro", "system", "project",
        "product", "summary", "about",
    ),
    LIMITATIONS_RISKS: (
        "limitations", "limitation", "constraints", "risks", "known issues",
        "security", "restrictions", "deprecated", "legacy", "caveats",
    ),
    DOCUMENTATION: ("readme", "documentation", "docs", "guides", "usage", "manual"),
}

# ---------------------------------------------------------------------------
# Scoring weights and caps (centralized and explainable)
# ---------------------------------------------------------------------------
SCORE_WEIGHTS: dict[str, float] = {
    "explicit_path": 15.0,     # a changed path/module named in section content
    "explicit_symbol": 12.0,   # a changed symbol named in heading/content
    "category": 10.0,          # internal category agreement
    "heading_overlap": 6.0,    # per unique heading token
    "path_overlap": 4.0,       # per unique heading-path token
    "content_overlap": 2.0,    # per unique section-content keyword
    "hunk_overlap": 1.5,       # per unique hunk-summary / changed-line token
    "summary_overlap": 1.0,    # per unique generated-summary token
    "extension": 1.0,          # file extension alone (weak)
}

# Maximum number of times each component may contribute, so one repeated
# token can never inflate a score without bound.
SCORE_CAPS: dict[str, int] = {
    "explicit_path": 2,
    "explicit_symbol": 2,
    "category": 2,
    "heading_overlap": 3,
    "path_overlap": 3,
    "content_overlap": 5,
    "hunk_overlap": 4,
    "summary_overlap": 3,
    "extension": 2,
}

# A change to tests alongside production source is supporting evidence only.
TEST_SUPPORTING_WEIGHT = 0.35

# Categories implied by diff content (hunk summaries, changed lines, symbols)
# are weaker evidence than the changed paths themselves.
DIFF_CATEGORY_WEIGHT = 0.5

# Decision thresholds.
MIN_CANDIDATE_SCORE = 6.0     # below this, no section is considered suitable
STRONG_SCORE = 25.0           # at/above this (and unambiguous) => strong match
AMBIGUITY_MARGIN = 3.0        # first-vs-second gap under this => ambiguous
DEFAULT_SHORTLIST_SIZE = 3

# Confidence must agree with ambiguity: a tie cannot be reported confidently
# however high the absolute score is.
EXACT_TIE_CONFIDENCE_CAP = 0.55
AMBIGUOUS_CONFIDENCE_CAP = 0.70
_TIE_EPSILON = 1e-6

# Deterministic scoring sees EVERY changed file (a relevant path must never be
# dropped for appearing late); only hunk *text* is budget-limited.
MAX_HUNKS_PER_FILE = 5
MAX_LINES_PER_HUNK = 10
MAX_LINE_CHARS = 200
MAX_HUNK_TOKEN_BUDGET = 2000
MAX_SIGNAL_TOKENS = 4000

# Bounded excerpt sent to the LLM (deterministic keyword scoring is NOT
# limited to this excerpt — see ``CatalogEntry.content_keywords``).
MAX_CONTENT_EXCERPT_CHARS = 600

# Files included in the LLM prompt (the most informative, not the first N).
MAX_FILES_FOR_LLM = 15

_STRENGTH_STRONG = "strong"
_STRENGTH_REASONABLE = "reasonable"
_STRENGTH_AMBIGUOUS = "ambiguous"
_STRENGTH_NONE = "none"

_TEST_NAME_RE = re.compile(r"(^|/)(tests?)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$")
_AUTOMATION_NAME_RE = re.compile(
    r"(router|updater|generator|detector|orchestrat|pipeline|automation|workflow)"
)
_MANIFEST_NAMES = {
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", "pipfile",
    "package.json", "pytest.ini", "tox.ini", ".gitignore", "environment.yml",
}
_CI_FILENAMES = {"jenkinsfile", "dockerfile", ".travis.yml", "azure-pipelines.yml"}
_RISK_TOKENS = ("security", "legacy", "deprecat", "vulnerab", "restrict")


# ---------------------------------------------------------------------------
# Change signals
# ---------------------------------------------------------------------------
@dataclass
class ChangeSignals:
    """Bounded, deterministic evidence extracted from one change."""

    paths: list[str] = field(default_factory=list)
    path_tokens: set[str] = field(default_factory=set)
    symbols: set[str] = field(default_factory=set)
    symbol_tokens: set[str] = field(default_factory=set)
    hunk_tokens: set[str] = field(default_factory=set)
    summary_tokens: set[str] = field(default_factory=set)
    extensions: set[str] = field(default_factory=set)
    category_weights: dict[str, float] = field(default_factory=dict)

    def has_evidence(self) -> bool:
        return bool(
            self.paths or self.symbols or self.hunk_tokens or self.summary_tokens
        )


def _normalize_files(
    changed_files: Iterable[Any], file_details: Optional[Iterable[Any]]
) -> list[dict]:
    """Accept schema-v2 detail dicts, ChangedFile objects, dicts, or strings."""
    source = file_details if file_details else changed_files
    normalized: list[dict] = []
    # Every changed file is normalized: deterministic scoring must never miss
    # a relevant path because it appears late in a large change.
    for entry in list(source or []):
        if isinstance(entry, dict):
            normalized.append(
                {
                    "path": entry.get("path", "") or "",
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
        else:  # ChangedFile-like object (schema v1)
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


def _categories_for_path(path: str) -> set[str]:
    """Primary category for one path; ordered so the most specific wins."""
    lowered = (path or "").lower()
    if not lowered:
        return set()
    name = lowered.rsplit("/", 1)[-1]

    categories: set[str] = set()
    if any(token in lowered for token in _RISK_TOKENS):
        categories.add(LIMITATIONS_RISKS)

    if _TEST_NAME_RE.search(lowered):
        return categories | {TESTING}
    if lowered.startswith(".github/") or ".github/workflows" in lowered or name in _CI_FILENAMES:
        return categories | {CI_DEPLOYMENT}
    if name in _MANIFEST_NAMES or lowered.startswith("config/") or "/config/" in lowered:
        return categories | {CONFIGURATION_DEPENDENCIES}
    if name.startswith("readme") or lowered.startswith("docs/") or lowered.endswith(".md"):
        return categories | {DOCUMENTATION, OVERVIEW}
    if name.endswith((".json", ".toml", ".ini", ".cfg", ".yml", ".yaml", ".env")):
        return categories | {CONFIGURATION_DEPENDENCIES}
    if _AUTOMATION_NAME_RE.search(name):
        return categories | {AUTOMATION_PIPELINE}
    if lowered.endswith((".py", ".js", ".ts", ".go", ".java", ".rb", ".rs")):
        return categories | {ARCHITECTURE_MODULES}
    return categories


def extract_change_signals(
    change_summary: str,
    changed_files: Iterable[Any],
    file_details: Optional[Iterable[Any]] = None,
) -> ChangeSignals:
    """Collect bounded routing evidence from a change package or file list.

    Works with schema-v2 ``what_changed`` details when present and degrades to
    plain paths for schema-v1 packages / ``ChangedFile`` objects.
    """
    signals = ChangeSignals()
    files = _normalize_files(changed_files, file_details)

    categories: set[str] = set()
    has_production_source = False
    has_tests = False

    for entry in files:
        for path in (entry["path"], entry.get("old_path")):
            if not path:
                continue
            signals.paths.append(path)
            signals.path_tokens.update(extract_keywords(path))
            suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
            if suffix:
                signals.extensions.add(suffix)

        path_categories = _categories_for_path(entry["path"])
        if entry.get("old_path"):
            path_categories |= _categories_for_path(entry["old_path"])
        categories |= path_categories
        if TESTING in path_categories:
            has_tests = True
        elif path_categories & {ARCHITECTURE_MODULES, AUTOMATION_PIPELINE}:
            has_production_source = True

        # Hunk *text* is budget-limited (paths/categories above never are).
        for hunk in (entry.get("what_changed") or [])[:MAX_HUNKS_PER_FILE]:
            if not isinstance(hunk, dict):
                continue
            for symbol in hunk.get("symbols") or []:
                signals.symbols.add(str(symbol))
                signals.symbol_tokens.update(extract_keywords(str(symbol)))
            if len(signals.hunk_tokens) >= MAX_HUNK_TOKEN_BUDGET:
                continue
            summary_text = hunk.get("summary") or ""
            if summary_text:
                signals.hunk_tokens.update(extract_keywords(summary_text))
            for key in ("added_lines", "removed_lines"):
                for line in (hunk.get(key) or [])[:MAX_LINES_PER_HUNK]:
                    text = (line or {}).get("text", "") if isinstance(line, dict) else ""
                    if text:
                        signals.hunk_tokens.update(extract_keywords(text[:MAX_LINE_CHARS]))

    signals.summary_tokens.update(extract_keywords(change_summary or ""))

    # Tests accompanying production source are supporting evidence only.
    weights = {category: 1.0 for category in categories}
    if has_tests and has_production_source and TESTING in weights:
        weights[TESTING] = TEST_SUPPORTING_WEIGHT

    # Diff content (hunk summaries, changed lines, symbols) can also imply a
    # category — e.g. a hunk mentioning pytest — but only as supporting
    # evidence. The generated summary is excluded: it is boilerplate that
    # merely restates file names.
    diff_text = " ".join(sorted(signals.hunk_tokens | signals.symbol_tokens))
    for category in infer_categories(diff_text):
        weights.setdefault(category, DIFF_CATEGORY_WEIGHT)

    signals.category_weights = weights
    return signals


# ---------------------------------------------------------------------------
# Section catalog (built purely from the real skeleton)
# ---------------------------------------------------------------------------
@dataclass
class CatalogEntry:
    """One real, eligible skeleton section normalized for scoring."""

    section_id: str
    heading: str
    heading_path: list[str]
    level: int
    order: int
    content_excerpt: str  # bounded; the LLM sees only this
    direct_text: str  # prose only; eligible for explicit path/symbol matches
    heading_tokens: set[str]
    path_tokens: set[str]
    content_keywords: set[str]  # from the COMPLETE eligible content
    semantic_categories: set[str]


def infer_categories(text: str) -> set[str]:
    """Internal categories implied by a piece of section text."""
    lowered = (text or "").lower()
    tokens = set(extract_keywords(text))
    categories: set[str] = set()
    for category, aliases in CATEGORY_ALIASES.items():
        for alias in aliases:
            if " " in alias or "/" in alias:
                if alias in lowered:
                    categories.add(category)
                    break
            elif alias in tokens:
                categories.add(category)
                break
    return categories


def section_contents_from_markdown(markdown_text: str) -> dict[str, str]:
    """Map section_id -> content using the existing Markdown parser.

    Reuses :func:`parse_markdown_sections` so section ids match the skeleton;
    no skeleton parsing is duplicated here.
    """
    return {
        section.section_id: section.content
        for section in parse_markdown_sections(markdown_text or "")
    }


def _is_generated_heading(heading: str) -> bool:
    return heading.strip().lower().startswith("automated change update")


_GENERATED_START = "TECHDOCKER_UPDATE_START"
_GENERATED_END = "TECHDOCKER_UPDATE_END"
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_PATHLIKE_RE = re.compile(
    r"^[\w./\\-]+(?:\.(?:py|js|ts|go|java|rb|rs|md|txt|json|toml|ini|cfg|ya?ml))$"
    r"|^[\w./\\-]+/[\w./\\-]*$"
)


def strip_generated_regions(content: str) -> str:
    """Drop TechDocker-generated update regions (and their markers).

    Historical audit blocks list the paths/symbols of *past* changes. Letting
    them score would make a section win simply because it once recorded a
    change to the same file.
    """
    kept: list[str] = []
    in_generated = False
    for line in (content or "").splitlines():
        if _GENERATED_START in line:
            in_generated = True
            continue
        if _GENERATED_END in line:
            in_generated = False
            continue
        if not in_generated:
            kept.append(line)
    return "\n".join(kept)


def strip_code_fences(content: str) -> str:
    """Drop fenced blocks from section content before scoring.

    A fenced block is opaque inventory (typically a full file-tree dump), not
    a semantic mention: without this, a "Repository Structure" section listing
    every file would claim an explicit-path match for *every* change.
    """
    kept: list[str] = []
    in_fence = False
    for line in (content or "").splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if not in_fence:
            kept.append(line)
    return "\n".join(kept)


def _is_inventory_item(line: str) -> bool:
    """True for a list item that is a bare path/module entry, not prose.

    ``- `src/foo.py``` is inventory; ``- The router in src/foo.py picks a
    section.`` is prose and keeps full direct-match weight.
    """
    match = _LIST_ITEM_RE.match(line)
    if match is None:
        return False
    body = match.group(1).replace("`", "").strip()
    if not body:
        return False
    words = body.split()
    if len(words) > 3:
        return False
    return any(_PATHLIKE_RE.match(word.rstrip(":,;.")) for word in words)


def strip_structural_inventory(content: str) -> str:
    """Drop Markdown table rows and bare path/module list items.

    Structural inventories enumerate everything, so they must not supply the
    strongest direct path/symbol evidence; they still contribute ordinary
    low-weight content keywords via :func:`keyword_content`.
    """
    return "\n".join(
        line
        for line in (content or "").splitlines()
        if not line.lstrip().startswith("|") and not _is_inventory_item(line)
    )


def keyword_content(content: str) -> str:
    """Content eligible for general keyword scoring.

    Generated regions, HTML comments, and fenced blocks are removed; tables
    and inventories remain as ordinary low-weight signal.
    """
    text = strip_generated_regions(content or "")
    text = _HTML_COMMENT_RE.sub(" ", text)
    return strip_code_fences(text)


def direct_match_content(content: str) -> str:
    """Prose eligible for the strongest explicit path/symbol matching."""
    return strip_structural_inventory(keyword_content(content))


def build_section_catalog(
    skeleton: SummarySkeleton,
    section_contents: Optional[dict[str, str]] = None,
) -> list[CatalogEntry]:
    """Normalize the skeleton's real sections into scoring candidates.

    Excludes the document root/title (not a meaningful update target) and any
    generated update-only section. Skeleton order is preserved for
    deterministic tie-breaking. No sections are invented.
    """
    contents = section_contents or {}
    catalog: list[CatalogEntry] = []
    has_deeper_sections = any(section.level > 1 for section in skeleton.sections)

    for section in skeleton.sections:
        if _is_generated_heading(section.heading):
            continue
        if section.level <= 1 and has_deeper_sections:
            continue  # document title is not an update target
        raw = contents.get(section.section_id) or ""
        # Keyword scoring inspects the COMPLETE eligible content, so relevant
        # prose late in a long section still counts; only the LLM excerpt and
        # the unique keyword set are bounded.
        keyword_text = keyword_content(raw)
        direct_text = direct_match_content(raw)
        heading_path = [part.strip() for part in section.path.split(">")]
        catalog.append(
            CatalogEntry(
                section_id=section.section_id,
                heading=section.heading,
                heading_path=heading_path,
                level=section.level,
                order=section.order,
                content_excerpt=keyword_text[:MAX_CONTENT_EXCERPT_CHARS],
                direct_text=direct_text,
                heading_tokens=set(extract_keywords(section.heading)),
                path_tokens=set(extract_keywords(section.path)),
                content_keywords=set(
                    extract_keywords(keyword_text, limit=MAX_SIGNAL_TOKENS)
                ),
                # Categories come from prose only: a historical generated block
                # or a file inventory must not define what a section is about.
                semantic_categories=infer_categories(
                    f"{section.path} {direct_text}"
                ),
            )
        )
    return catalog


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@dataclass
class SectionCandidate:
    """A scored, real section with an explainable breakdown."""

    section_id: str
    heading: str
    heading_path: list[str]
    score: float
    rank: int = 0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    matched_signals: list[str] = field(default_factory=list)
    order: int = 0

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "heading": self.heading,
            "heading_path": list(self.heading_path),
            "score": round(self.score, 3),
            "rank": self.rank,
            "score_breakdown": {k: round(v, 3) for k, v in self.score_breakdown.items()},
            "matched_signals": list(self.matched_signals),
        }


def _capped(component: str, count: int) -> float:
    return min(count, SCORE_CAPS[component]) * SCORE_WEIGHTS[component]


def score_section(signals: ChangeSignals, entry: CatalogEntry) -> SectionCandidate:
    """Score one real section; the breakdown always sums to the total."""
    breakdown: dict[str, float] = {}
    matched: list[str] = []
    # Direct matches come from prose only: generated regions, fenced trees,
    # HTML comments, tables, and bare inventory lists are excluded.
    haystack = f"{entry.heading}\n{entry.direct_text}".lower()

    explicit_paths = sorted({p for p in signals.paths if p and p.lower() in haystack})
    if explicit_paths:
        breakdown["explicit_path"] = _capped("explicit_path", len(explicit_paths))
        matched.extend(explicit_paths[: SCORE_CAPS["explicit_path"]])

    explicit_symbols = sorted(
        {s for s in signals.symbols if s and s.lower() in haystack}
    )
    if explicit_symbols:
        breakdown["explicit_symbol"] = _capped("explicit_symbol", len(explicit_symbols))
        matched.extend(explicit_symbols[: SCORE_CAPS["explicit_symbol"]])

    shared_categories = sorted(
        set(signals.category_weights) & entry.semantic_categories
    )
    if shared_categories:
        capped = shared_categories[: SCORE_CAPS["category"]]
        breakdown["category"] = sum(
            SCORE_WEIGHTS["category"] * signals.category_weights.get(name, 1.0)
            for name in capped
        )
        matched.extend(capped)

    change_tokens = (
        signals.path_tokens | signals.symbol_tokens | signals.hunk_tokens
    )

    heading_hits = sorted(entry.heading_tokens & change_tokens)
    if heading_hits:
        breakdown["heading_overlap"] = _capped("heading_overlap", len(heading_hits))
        matched.extend(heading_hits[: SCORE_CAPS["heading_overlap"]])

    path_hits = sorted((entry.path_tokens - entry.heading_tokens) & change_tokens)
    if path_hits:
        breakdown["path_overlap"] = _capped("path_overlap", len(path_hits))

    content_hits = sorted(entry.content_keywords & change_tokens)
    if content_hits:
        breakdown["content_overlap"] = _capped("content_overlap", len(content_hits))
        matched.extend(content_hits[: SCORE_CAPS["content_overlap"]])

    hunk_hits = sorted(
        (entry.heading_tokens | entry.content_keywords) & signals.hunk_tokens
    )
    if hunk_hits:
        breakdown["hunk_overlap"] = _capped("hunk_overlap", len(hunk_hits))

    summary_hits = sorted(
        (entry.heading_tokens | entry.content_keywords) & signals.summary_tokens
    )
    if summary_hits:
        breakdown["summary_overlap"] = _capped("summary_overlap", len(summary_hits))

    extension_hits = sorted(
        (entry.heading_tokens | entry.content_keywords) & signals.extensions
    )
    if extension_hits:
        breakdown["extension"] = _capped("extension", len(extension_hits))

    total = sum(breakdown.values())
    # Deduplicate matched signals while preserving deterministic order.
    seen: set[str] = set()
    unique_matched = [m for m in matched if not (m in seen or seen.add(m))]
    return SectionCandidate(
        section_id=entry.section_id,
        heading=entry.heading,
        heading_path=entry.heading_path,
        score=total,
        score_breakdown=breakdown,
        matched_signals=unique_matched,
        order=entry.order,
    )


def _direct_component(candidate: SectionCandidate) -> float:
    return candidate.score_breakdown.get("explicit_path", 0.0) + (
        candidate.score_breakdown.get("explicit_symbol", 0.0)
    )


def rank_candidates(
    signals: ChangeSignals,
    catalog: list[CatalogEntry],
    limit: int = DEFAULT_SHORTLIST_SIZE,
) -> list[SectionCandidate]:
    """Score every eligible section and return the deterministic top ``limit``.

    Tie-breaking: total score, then stronger direct (path/symbol) evidence,
    then skeleton order, then section id.
    """
    scored = [score_section(signals, entry) for entry in catalog]
    scored.sort(
        key=lambda c: (-c.score, -_direct_component(c), c.order, c.section_id)
    )
    shortlist = scored[:limit]
    for rank, candidate in enumerate(shortlist, start=1):
        candidate.rank = rank
    return shortlist


# ---------------------------------------------------------------------------
# Confidence / ambiguity
# ---------------------------------------------------------------------------
@dataclass
class CandidateAssessment:
    """Deterministic verdict over a shortlist."""

    candidates: list[SectionCandidate]
    confidence: float
    strength: str  # strong | reasonable | ambiguous | none
    ambiguous: bool
    reason: str

    @property
    def top(self) -> Optional[SectionCandidate]:
        return self.candidates[0] if self.candidates else None


def assess_candidates(
    candidates: list[SectionCandidate], signals: ChangeSignals
) -> CandidateAssessment:
    """Derive confidence from evidence rather than a hardcoded constant."""
    if not candidates or candidates[0].score < MIN_CANDIDATE_SCORE:
        return CandidateAssessment(
            candidates=candidates,
            confidence=0.0,
            strength=_STRENGTH_NONE,
            ambiguous=False,
            reason=(
                "no section scored above the minimum "
                f"({MIN_CANDIDATE_SCORE})"
            ),
        )

    top = candidates[0]
    runner_up = candidates[1].score if len(candidates) > 1 else 0.0
    margin = top.score - runner_up
    ambiguous = len(candidates) > 1 and margin < AMBIGUITY_MARGIN

    confidence = min(top.score / 40.0, 0.9)
    confidence += min(margin / 20.0, 0.15)
    if _direct_component(top) > 0:
        confidence += 0.1
    # An exact tie is a coin flip between real sections; a near tie is close to
    # one. Cap confidence so it can never contradict `ambiguous`.
    if ambiguous:
        cap = (
            EXACT_TIE_CONFIDENCE_CAP if margin <= _TIE_EPSILON
            else AMBIGUOUS_CONFIDENCE_CAP
        )
        confidence = min(confidence, cap)
    confidence = round(max(0.0, min(confidence, 0.99)), 3)

    if ambiguous:
        strength = _STRENGTH_AMBIGUOUS
        reason = (
            f"top two candidates are within {AMBIGUITY_MARGIN} points "
            f"({top.score:.1f} vs {runner_up:.1f})"
        )
    elif top.score >= STRONG_SCORE:
        strength = _STRENGTH_STRONG
        reason = f"clear leader at {top.score:.1f} points"
    else:
        strength = _STRENGTH_REASONABLE
        reason = f"leading candidate at {top.score:.1f} points"

    return CandidateAssessment(
        candidates=candidates,
        confidence=confidence,
        strength=strength,
        ambiguous=ambiguous,
        reason=reason,
    )


def select_files_for_llm(
    signals: ChangeSignals,
    candidates: list[SectionCandidate],
    limit: int = MAX_FILES_FOR_LLM,
) -> tuple[list[str], int]:
    """Most informative changed paths for the prompt, plus the omitted count.

    Files are ranked by overlap with the shortlisted candidates' matched
    signals rather than taken in arrival order, so a large change still shows
    the LLM the paths that actually drove the shortlist.
    """
    unique_paths: list[str] = []
    seen: set[str] = set()
    for path in signals.paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)

    relevant: set[str] = set()
    for candidate in candidates:
        relevant.update(token.lower() for token in candidate.matched_signals)

    def relevance(path: str) -> tuple[int, int, str]:
        tokens = set(extract_keywords(path))
        direct = 1 if path.lower() in relevant else 0
        return (-direct, -len(tokens & relevant), path)

    ranked = sorted(unique_paths, key=relevance)
    return ranked[:limit], max(len(unique_paths) - limit, 0)


def find_overview_section(catalog: list[CatalogEntry]) -> Optional[CatalogEntry]:
    """Semantically locate an overview-equivalent section.

    Found from the real headings/content via the ``overview`` category — never
    by assuming a section named "System Overview" exists.
    """
    for entry in catalog:
        if OVERVIEW in entry.semantic_categories:
            return entry
    return catalog[0] if catalog else None
