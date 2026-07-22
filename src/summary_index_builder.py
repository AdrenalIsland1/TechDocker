"""Build the paragraph/sentence placement index for the reviewable summary.

Two complementary artifacts drive placement:

* ``artifacts/skeletons/base_skeleton.json`` answers *"which section?"*
* ``artifacts/skeletons/base_summary_index.json`` (this module) answers
  *"where inside the selected section?"* by indexing every block and sentence
  with exact source offsets, content hashes, and deterministic ids.

The index describes ``base_updated_summary.md`` — the document future patches
modify — not the permanent baseline. Staleness is detected by comparing the
stored SHA-256 of the whole source document (never modification time).

Everything here is deterministic and offline: no timestamps are stored, no
LLM/provider is imported, and identical Markdown always produces byte-identical
JSON. This phase only builds and maintains the index; section scoring,
placement scoring, patch planning, and workflow integration come later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.markdown_summary_parser import content_hash, slugify, unique_id
from src.project_summary_generator import (
    original_summary_path,
    updated_summary_path,
)
from src.summary_skeleton_builder import SKELETON_DIRECTORY, summary_skeleton_path
from src.summary_skeleton_store import SummarySkeleton, load_summary_skeleton

SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = frozenset({1})

SUMMARY_INDEX_NAME = "base_summary_index.json"

# Marker comments written by summary_updater around generated audit blocks.
UPDATE_MARKER_START = "TECHDOCKER_UPDATE_START"
UPDATE_MARKER_END = "TECHDOCKER_UPDATE_END"

# Block vocabulary.
BLOCK_PARAGRAPH = "paragraph"
BLOCK_UNORDERED_LIST_ITEM = "unordered_list_item"
BLOCK_ORDERED_LIST_ITEM = "ordered_list_item"
BLOCK_BLOCKQUOTE = "blockquote"
BLOCK_CODE = "code_block"
BLOCK_TABLE = "table"
BLOCK_HTML_COMMENT = "html_comment"

# Blocks that may receive patches and are split into sentences.
_PATCHABLE_TYPES = frozenset(
    {BLOCK_PARAGRAPH, BLOCK_UNORDERED_LIST_ITEM, BLOCK_ORDERED_LIST_ITEM}
)

_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_UNORDERED_ITEM_RE = re.compile(r"^(\s*[-*+]\s+)(?=\S)")
_ORDERED_ITEM_RE = re.compile(r"^(\s*\d+[.)]\s+)(?=\S)")
_SHORT_HASH_CHARS = 8

# A bare structural-inventory entry: an optional list marker followed by a
# single file/module/path identifier and nothing else. Such blocks enumerate
# the repository; they are not explanatory documentation and must never be
# offered as replacement targets.
_INVENTORY_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_IDENTIFIER_RE = re.compile(
    r"^[\w.\-]+(?:/[\w.\-]+)+$"                       # a path: src/module.py
    r"|^[\w\-]+\.(?:py|js|ts|go|java|rb|rs|md|txt|json|toml|ya?ml|ini|cfg|lock|sh)$"
    r"|^[\w\-]+(?:\.[\w\-]+)+$"                       # dotted module/package
)


def is_structural_inventory(text: str) -> bool:
    """True when a block is a bare file/module inventory entry, not prose.

    Inventory (never a patch target)::

        - `src/git_change_detector.py`
        - src/git_change_detector.py
        1. `src/module.py`
        - `requirements.txt`

    Explanatory prose (stays patchable)::

        - `src/git_change_detector.py`: Extracts changed files and metadata.
        - The router uses `src/summary_change_router.py` to score sections.
        - `pytest` runs the complete offline test suite.

    The rule is deliberately narrow: after removing one list marker, backticks
    and terminal punctuation, the *entire* remainder must be a single
    identifier-looking token. Anything explanatory keeps its patchability.
    """
    body = (text or "").strip()
    if not body or "\n" in body:
        return False
    body = _INVENTORY_MARKER_RE.sub("", body, count=1)
    body = body.replace("`", "").strip().rstrip(".,;:")
    if not body or " " in body or "\t" in body:
        return False
    return bool(_IDENTIFIER_RE.match(body))


class SummaryIndexError(RuntimeError):
    """The stored index is unreadable or structurally invalid."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def summary_index_path(repo_path: str | Path = ".") -> Path:
    return Path(repo_path) / SKELETON_DIRECTORY / SUMMARY_INDEX_NAME


def default_index_source(repo_path: str | Path = ".") -> Path:
    """The index describes the *reviewable* summary.

    Falls back to the permanent baseline only when the updated summary does
    not exist yet (first initialization), because the baseline is the only
    document available at that point.
    """
    updated = updated_summary_path(repo_path)
    if updated.exists():
        return updated
    return original_summary_path(repo_path)


# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has
    have how in into is it its may might must not of on or other our over
    should so such than that the their then there these they this those to
    use used uses using was were what when where which while who will with
    within would you your
    """.split()
)

_RAW_TOKEN_RE = re.compile(r"[A-Za-z0-9_./\\-]+")
_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+")
_SEPARATORS_RE = re.compile(r"[_\-./\\]+")
MAX_KEYWORDS = 25


def extract_keywords(text: str, limit: int = MAX_KEYWORDS) -> list[str]:
    """Deterministic lowercase keywords for a piece of Markdown text.

    Recognizes technical forms: ``snake_case``, ``kebab-case``, ``camelCase``,
    dotted/slashed paths (``src/summary_change_router.py``) and abbreviations
    such as ``CI/CD``. The full technical token is kept alongside its parts;
    ordering is first-seen, duplicates and stop words are dropped.
    """
    keywords: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        token = token.lower()
        if len(token) < 2 or token.isdigit() or token in _STOPWORDS:
            return
        if token not in seen:
            seen.add(token)
            keywords.append(token)

    for raw in _RAW_TOKEN_RE.findall(text or ""):
        stripped = raw.strip("._-/\\")
        if not stripped:
            continue
        # Keep the whole technical token (path, dotted name, CI/CD, camelCase).
        has_separator = bool(_SEPARATORS_RE.search(stripped))
        is_mixed_case = stripped != stripped.lower() and stripped != stripped.upper()
        if has_separator or is_mixed_case:
            add(stripped)
        # ...and its useful parts.
        for part in _SEPARATORS_RE.split(stripped):
            if not part:
                continue
            add(part)
            for piece in _CAMEL_RE.findall(part):
                add(piece)

    return keywords[:limit]


# ---------------------------------------------------------------------------
# Sentence splitting (deterministic, stdlib only)
# ---------------------------------------------------------------------------
_ABBREVIATIONS = frozenset(
    """
    e.g i.e etc vs cf al approx dept est fig no vol ver sec min max
    mr mrs ms dr prof sr jr st inc ltd co corp dept univ
    """.split()
)
_TERMINATORS = ".!?"
_CLOSERS = ')"\']’”'
_OPENERS = '("\'[‘“'


def _is_abbreviation_before(text: str, dot_index: int) -> bool:
    """True when the '.' at ``dot_index`` ends a known abbreviation/initial."""
    start = dot_index
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "."):
        start -= 1
    token = text[start:dot_index].rstrip(".")
    if not token:
        return False
    if len(token) == 1 and token.isalpha():
        return True  # single-letter initial such as "J. Smith"
    return token.lower() in _ABBREVIATIONS


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans ``(start, end)`` within ``text``; end is exclusive.

    Conservative: only splits on ``.``/``!``/``?`` that are followed by
    whitespace and then a capital letter, digit, or opening quote. Decimals
    (``1.5``), dotted paths (``base_updated_summary.md``), inline code spans,
    abbreviations (``e.g.``), and initials are never split. Spans exclude the
    whitespace between sentences and never overlap.
    """
    spans: list[tuple[int, int]] = []
    length = len(text)
    index = 0
    while index < length and text[index].isspace():
        index += 1
    sentence_start = index
    in_code = False

    while index < length:
        char = text[index]
        if char == "`":
            in_code = not in_code
            index += 1
            continue
        if in_code or char not in _TERMINATORS:
            index += 1
            continue

        # Consume repeated terminators ("...", "?!") then closing punctuation.
        end = index
        while end < length and text[end] in _TERMINATORS:
            end += 1
        last_dot = end - 1
        while end < length and text[end] in _CLOSERS:
            end += 1

        if text[last_dot] == "." and _is_abbreviation_before(text, last_dot):
            index = end
            continue

        lookahead = end
        while lookahead < length and text[lookahead].isspace():
            lookahead += 1
        at_document_end = lookahead >= length
        starts_new_sentence = not at_document_end and (
            text[lookahead].isupper()
            or text[lookahead].isdigit()
            or text[lookahead] in _OPENERS
        )
        if not (at_document_end or starts_new_sentence):
            index = end
            continue
        # Require whitespace (or end of text) after the terminator so that
        # "file.md" and "1.5" are never treated as boundaries.
        if not at_document_end and not text[end : end + 1].isspace():
            index = end
            continue

        if end > sentence_start:
            spans.append((sentence_start, end))
        sentence_start = lookahead
        index = lookahead if lookahead > end else end

    trailing = text[sentence_start:].rstrip()
    if trailing:
        spans.append((sentence_start, sentence_start + len(trailing)))
    return spans


# ---------------------------------------------------------------------------
# Line / block scanning
# ---------------------------------------------------------------------------
@dataclass
class _Line:
    number: int  # one-based
    start: int  # inclusive document offset
    end: int  # exclusive document offset (newline not included)
    content: str


@dataclass
class _RawBlock:
    block_type: str
    first: int  # index into the line list
    last: int
    generated: bool = False


@dataclass
class _RawSection:
    heading: str
    level: int
    heading_line: Optional[int]
    generated: bool = False
    blocks: list[_RawBlock] = field(default_factory=list)


def _scan_lines(text: str) -> list[_Line]:
    lines: list[_Line] = []
    offset = 0
    number = 1
    for raw in (text or "").splitlines(keepends=True):
        content = raw
        if content.endswith("\n"):
            content = content[:-1]
        if content.endswith("\r"):
            content = content[:-1]
        lines.append(_Line(number, offset, offset + len(content), content))
        offset += len(raw)
        number += 1
    return lines


def _is_block_start(content: str) -> bool:
    """True when the line begins a new block (ends a lazy continuation)."""
    stripped = content.strip()
    if not stripped:
        return True
    return bool(
        _ATX_HEADING_RE.match(content)
        or _FENCE_RE.match(content)
        or _UNORDERED_ITEM_RE.match(content)
        or _ORDERED_ITEM_RE.match(content)
        or stripped.startswith(">")
        or stripped.startswith("|")
        or stripped.startswith("<!--")
    )


def _parse_structure(lines: list[_Line]) -> list[_RawSection]:
    """Split lines into sections and their content blocks."""
    sections: list[_RawSection] = [
        _RawSection(heading="", level=0, heading_line=None)
    ]
    in_generated = False
    index = 0
    total = len(lines)

    while index < total:
        content = lines[index].content
        stripped = content.strip()

        if not stripped:
            index += 1
            continue

        # Fenced code block (opaque; headings inside must not create sections).
        fence = _FENCE_RE.match(content)
        if fence is not None:
            marker = fence.group(1)[0]
            start = index
            index += 1
            while index < total:
                closing = _FENCE_RE.match(lines[index].content)
                if closing is not None and closing.group(1)[0] == marker:
                    index += 1
                    break
                index += 1
            sections[-1].blocks.append(
                _RawBlock(BLOCK_CODE, start, index - 1, in_generated)
            )
            continue

        # HTML comment (possibly multi-line); may carry generated markers.
        if stripped.startswith("<!--"):
            start = index
            while index < total and "-->" not in lines[index].content:
                index += 1
            last = min(index, total - 1)
            index = last + 1
            comment_text = "\n".join(
                line.content for line in lines[start : last + 1]
            )
            is_start_marker = UPDATE_MARKER_START in comment_text
            is_end_marker = UPDATE_MARKER_END in comment_text
            if is_start_marker:
                in_generated = True
            sections[-1].blocks.append(
                _RawBlock(
                    BLOCK_HTML_COMMENT,
                    start,
                    last,
                    in_generated or is_start_marker or is_end_marker,
                )
            )
            if is_end_marker:
                in_generated = False
            continue

        heading = _ATX_HEADING_RE.match(content)
        if heading is not None:
            sections.append(
                _RawSection(
                    heading=heading.group(2).strip(),
                    level=len(heading.group(1)),
                    heading_line=index,
                    generated=in_generated,
                )
            )
            index += 1
            continue

        if stripped.startswith("|"):
            start = index
            while index < total and lines[index].content.strip().startswith("|"):
                index += 1
            sections[-1].blocks.append(
                _RawBlock(BLOCK_TABLE, start, index - 1, in_generated)
            )
            continue

        if stripped.startswith(">"):
            start = index
            while index < total and lines[index].content.strip().startswith(">"):
                index += 1
            sections[-1].blocks.append(
                _RawBlock(BLOCK_BLOCKQUOTE, start, index - 1, in_generated)
            )
            continue

        unordered = _UNORDERED_ITEM_RE.match(content)
        ordered = _ORDERED_ITEM_RE.match(content)
        if unordered is not None or ordered is not None:
            block_type = (
                BLOCK_UNORDERED_LIST_ITEM if unordered is not None
                else BLOCK_ORDERED_LIST_ITEM
            )
            start = index
            index += 1
            # Lazy continuation lines belong to this item.
            while index < total and not _is_block_start(lines[index].content):
                index += 1
            sections[-1].blocks.append(
                _RawBlock(block_type, start, index - 1, in_generated)
            )
            continue

        # Ordinary paragraph: consecutive lines until a new block starts.
        start = index
        index += 1
        while index < total and not _is_block_start(lines[index].content):
            index += 1
        sections[-1].blocks.append(
            _RawBlock(BLOCK_PARAGRAPH, start, index - 1, in_generated)
        )

    if not sections[0].blocks:
        sections.pop(0)  # no preamble before the first heading
    return sections


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------
def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:_SHORT_HASH_CHARS]


def _content_offset(block_type: str, text: str) -> int:
    """Characters from the block start to where its prose begins."""
    if block_type == BLOCK_UNORDERED_LIST_ITEM:
        match = _UNORDERED_ITEM_RE.match(text)
        return len(match.group(1)) if match else 0
    if block_type == BLOCK_ORDERED_LIST_ITEM:
        match = _ORDERED_ITEM_RE.match(text)
        return len(match.group(1)) if match else 0
    return 0


def _skeleton_id_map(skeleton: Optional[SummarySkeleton]) -> dict[str, list[str]]:
    """Heading path -> skeleton section ids, in skeleton order."""
    mapping: dict[str, list[str]] = {}
    if skeleton is None:
        return mapping
    for section in skeleton.sections:
        mapping.setdefault(section.path, []).append(section.section_id)
    return mapping


def _line_for_offset(lines: list[_Line], offset: int) -> int:
    for line in lines:
        if line.start <= offset <= line.end:
            return line.number
    return lines[-1].number if lines else 1


def build_summary_index(
    markdown_text: str,
    skeleton: Optional[SummarySkeleton] = None,
    source_path: Optional[str] = None,
) -> dict:
    """Parse Markdown into the deterministic section/block/sentence index."""
    lines = _scan_lines(markdown_text)
    raw_sections = _parse_structure(lines)
    skeleton_ids = _skeleton_id_map(skeleton)

    taken_section_ids: set[str] = set()
    heading_stack: list[tuple[int, str]] = []  # (level, heading)
    sections: list[dict] = []
    block_occurrences: dict[tuple[str, str, str], int] = {}

    for order, raw_section in enumerate(raw_sections, start=1):
        if raw_section.heading_line is None:
            heading_path = ["(preamble)"]
            path_key = "(preamble)"
        else:
            while heading_stack and heading_stack[-1][0] >= raw_section.level:
                heading_stack.pop()
            heading_path = [h for _, h in heading_stack] + [raw_section.heading]
            heading_stack.append((raw_section.level, raw_section.heading))
            path_key = " > ".join(heading_path)

        available = skeleton_ids.get(path_key)
        if available:
            section_id = available.pop(0)
        else:
            section_id = unique_id(taken_section_ids, slugify(path_key))
        taken_section_ids.add(section_id)

        section_source = ""
        if raw_section.blocks:
            first = lines[raw_section.blocks[0].first]
            last = lines[raw_section.blocks[-1].last]
            section_source = markdown_text[first.start : last.end]

        blocks: list[dict] = []
        for block_order, raw_block in enumerate(raw_section.blocks, start=1):
            first_line = lines[raw_block.first]
            last_line = lines[raw_block.last]
            start_offset = first_line.start
            end_offset = last_line.end
            text = markdown_text[start_offset:end_offset]

            block_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            short = block_hash[:_SHORT_HASH_CHARS]
            key = (section_id, raw_block.block_type, short)
            block_occurrences[key] = block_occurrences.get(key, 0) + 1
            block_id = (
                f"{section_id}-{raw_block.block_type}-{short}-"
                f"{block_occurrences[key]}"
            )

            offset_in_block = _content_offset(raw_block.block_type, text)
            content_text = text[offset_in_block:]
            # A bare file/module inventory entry is structure, not prose: it is
            # recorded but never offered as a patch target. (block_id and
            # content_hash derive from the text alone, so this metadata never
            # changes existing identifiers.)
            structural_inventory = is_structural_inventory(text)
            patchable = (
                raw_block.block_type in _PATCHABLE_TYPES
                and not raw_block.generated
                and not structural_inventory
            )

            sentences: list[dict] = []
            if raw_block.block_type in _PATCHABLE_TYPES:
                sentence_occurrences: dict[str, int] = {}
                for local_start, local_end in split_sentences(content_text):
                    block_start = offset_in_block + local_start
                    block_end = offset_in_block + local_end
                    sentence_text = text[block_start:block_end]
                    sentence_hash = hashlib.sha256(
                        sentence_text.encode("utf-8")
                    ).hexdigest()
                    sentence_short = sentence_hash[:_SHORT_HASH_CHARS]
                    sentence_occurrences[sentence_short] = (
                        sentence_occurrences.get(sentence_short, 0) + 1
                    )
                    absolute_start = start_offset + block_start
                    absolute_end = start_offset + block_end
                    sentences.append(
                        {
                            "sentence_id": (
                                f"{block_id}-sentence-{sentence_short}-"
                                f"{sentence_occurrences[sentence_short]}"
                            ),
                            "text": sentence_text,
                            "content_hash": sentence_hash,
                            "keywords": extract_keywords(sentence_text),
                            "block_start_offset": block_start,
                            "block_end_offset": block_end,
                            "source_start_offset": absolute_start,
                            "source_end_offset": absolute_end,
                            "start_line": _line_for_offset(lines, absolute_start),
                            "end_line": _line_for_offset(lines, max(absolute_end - 1, absolute_start)),
                        }
                    )

            blocks.append(
                {
                    "block_id": block_id,
                    "block_type": raw_block.block_type,
                    "order": block_order,
                    "text": text,
                    "content_text": content_text,
                    "content_hash": block_hash,
                    "keywords": extract_keywords(content_text),
                    "patchable": patchable,
                    "structural_inventory": structural_inventory,
                    "generated": raw_block.generated,
                    "source_start_offset": start_offset,
                    "source_end_offset": end_offset,
                    "start_line": first_line.number,
                    "end_line": last_line.number,
                    "sentences": sentences,
                }
            )

        sections.append(
            {
                "section_id": section_id,
                "heading": raw_section.heading,
                "heading_level": raw_section.level,
                "heading_path": heading_path,
                "order": order,
                "generated": raw_section.generated,
                "content_hash": content_hash(section_source),
                "keywords": extract_keywords(
                    f"{raw_section.heading} {section_source}"
                ),
                "blocks": blocks,
            }
        )

    encoded = (markdown_text or "").encode("utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "path": source_path,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "size_bytes": len(encoded),
        },
        "sections": sections,
    }


# ---------------------------------------------------------------------------
# Persistence and lifecycle
# ---------------------------------------------------------------------------
def serialize_summary_index(index: dict) -> str:
    """Deterministic JSON text for an index (identical input -> identical bytes)."""
    return json.dumps(index, indent=2, ensure_ascii=False) + "\n"


def write_summary_index(index: dict, destination: str | Path) -> Path:
    """Atomically write the index JSON (temp file + ``os.replace``)."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(serialize_summary_index(index), encoding="utf-8")
    os.replace(temporary, path)
    return path


def load_summary_index(path: str | Path) -> dict:
    """Load an index. Raises ``FileNotFoundError`` / ``SummaryIndexError``."""
    index_path = Path(path)
    if not index_path.exists():
        raise FileNotFoundError(f"Summary index not found: {index_path}")
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SummaryIndexError(f"Summary index is not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise SummaryIndexError("Summary index root must be a JSON object.")
    if not isinstance(data.get("source"), dict) or "sections" not in data:
        raise SummaryIndexError("Summary index is missing required fields.")
    return data


def is_summary_index_current(
    index: dict,
    markdown_text: str,
    source_path: Optional[str] = None,
) -> bool:
    """True when the index matches the current source exactly.

    Requires a supported schema version, a matching source path (when one is
    supplied), and an identical document SHA-256. Modification time is never
    consulted, and section hashes alone are never sufficient.
    """
    if not isinstance(index, dict):
        return False
    if index.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        return False
    source = index.get("source")
    if not isinstance(source, dict):
        return False
    if source_path is not None and source.get("path") != source_path:
        return False
    digest = hashlib.sha256((markdown_text or "").encode("utf-8")).hexdigest()
    return source.get("sha256") == digest


@dataclass
class EnsureResult:
    """Outcome of :func:`ensure_summary_index`."""

    status: str  # "created" | "rebuilt" | "current"
    index_path: Path
    index: dict
    reason: str = ""


def ensure_summary_index(
    source_path: str | Path,
    index_path: str | Path,
    skeleton_path: Optional[str | Path] = None,
    relative_source: Optional[str] = None,
) -> EnsureResult:
    """Create, rebuild, or keep the index, reporting which happened.

    Rebuilds when the stored document hash differs, the schema is unsupported,
    or the stored JSON is malformed; leaves the file untouched when current.
    """
    source = Path(source_path)
    markdown_text = source.read_text(encoding="utf-8")
    stored_source_name = relative_source if relative_source is not None else str(source)

    skeleton: Optional[SummarySkeleton] = None
    if skeleton_path is not None and Path(skeleton_path).exists():
        try:
            skeleton = load_summary_skeleton(skeleton_path)
        except (json.JSONDecodeError, TypeError, ValueError):
            skeleton = None  # a broken skeleton must not block indexing

    destination = Path(index_path)
    status = "created"
    reason = "index did not exist"
    if destination.exists():
        try:
            existing = load_summary_index(destination)
        except (SummaryIndexError, FileNotFoundError) as error:
            status, reason = "rebuilt", f"existing index was invalid: {error}"
        else:
            if is_summary_index_current(existing, markdown_text, stored_source_name):
                return EnsureResult("current", destination, existing, "source hash matched")
            status, reason = "rebuilt", "source hash did not match the stored index"

    index = build_summary_index(markdown_text, skeleton, stored_source_name)
    write_summary_index(index, destination)
    return EnsureResult(status, destination, index, reason)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
EXIT_CURRENT = 0
EXIT_MISSING = 1
EXIT_STALE = 2
EXIT_INVALID = 3


def _relative_source_name(source: Path, repo_path: str | Path) -> str:
    try:
        return str(source.resolve().relative_to(Path(repo_path).resolve()))
    except ValueError:
        return str(source)


def main(argv: Optional[list[str]] = None) -> int:
    """``--preview`` prints, ``--check`` verifies, ``--write`` rebuilds."""
    parser = argparse.ArgumentParser(
        description="Build the summary paragraph/sentence placement index."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preview", action="store_true", help="Print JSON; write nothing.")
    group.add_argument("--check", action="store_true", help="Report freshness; write nothing.")
    group.add_argument("--write", action="store_true", help="Create or rebuild the index.")
    parser.add_argument("--repo-path", default=".", help="Repository root.")
    parser.add_argument("--source", default=None, help="Markdown summary to index.")
    arguments = parser.parse_args(argv)

    repo_path = arguments.repo_path
    source = (
        Path(arguments.source) if arguments.source else default_index_source(repo_path)
    )
    if not source.exists():
        print(f"[summary-index] source summary not found: {source}", file=sys.stderr)
        return EXIT_MISSING

    destination = summary_index_path(repo_path)
    relative_source = _relative_source_name(source, repo_path)
    markdown_text = source.read_text(encoding="utf-8")

    if arguments.preview:
        index = build_summary_index(markdown_text, _load_skeleton_quietly(repo_path), relative_source)
        print(serialize_summary_index(index), end="")
        print(
            f"[summary-index] previewed {len(index['sections'])} section(s) from "
            f"{relative_source}; no files were written.",
            file=sys.stderr,
        )
        return EXIT_CURRENT

    if arguments.check:
        if not destination.exists():
            print(f"[summary-index] missing: {destination}", file=sys.stderr)
            return EXIT_MISSING
        try:
            existing = load_summary_index(destination)
        except SummaryIndexError as error:
            print(f"[summary-index] invalid: {error}", file=sys.stderr)
            return EXIT_INVALID
        if is_summary_index_current(existing, markdown_text, relative_source):
            print(f"[summary-index] current: {destination}", file=sys.stderr)
            return EXIT_CURRENT
        print(f"[summary-index] stale: {destination}", file=sys.stderr)
        return EXIT_STALE

    result = ensure_summary_index(
        source, destination, summary_skeleton_path(repo_path), relative_source
    )
    print(
        f"[summary-index] {result.status}: {result.index_path} ({result.reason})",
        file=sys.stderr,
    )
    return EXIT_CURRENT


def _load_skeleton_quietly(repo_path: str | Path) -> Optional[SummarySkeleton]:
    path = summary_skeleton_path(repo_path)
    if not path.exists():
        return None
    try:
        return load_summary_skeleton(path)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
