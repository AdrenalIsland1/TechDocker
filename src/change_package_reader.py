"""One centralized reader that normalizes change packages for consumers.

Change packages exist in three producer formats and every consumer must accept
all of them:

* **v1** — ``changed_files`` with basic path/status only (no ``what_changed``),
* **v2** — hunks with verbose per-line ``added_lines``/``removed_lines`` arrays,
* **v3** — hunks with coherent ``change_blocks`` (multiline ``added_text``/
  ``removed_text``, true ranges, per-block symbols and truncation).

Scorers, the patch planner, and the preview normalize through here instead of
re-implementing v2/v3 parsing. The normalized hunk exposes coherent block facts
(ranges, removed/added text, symbols, summary, truncation) plus convenience
``removed_lines``/``added_lines`` (line-text lists) and aggregate
``removed_text``/``added_text`` so line-oriented consumers change minimally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass
class NormalizedBlock:
    """A coherent run of consecutive changed lines (schema-v3 change unit)."""

    change_type: str  # added | deleted | modified
    old_start_line: Optional[int]
    old_line_count: int
    old_end_line: Optional[int]
    new_start_line: Optional[int]
    new_line_count: int
    new_end_line: Optional[int]
    removed_text: str
    added_text: str
    summary: str = ""
    symbols: list[str] = field(default_factory=list)
    text_truncated: bool = False

    @property
    def removed_line_texts(self) -> list[str]:
        return self.removed_text.split("\n") if self.old_line_count > 0 else []

    @property
    def added_line_texts(self) -> list[str]:
        return self.added_text.split("\n") if self.new_line_count > 0 else []


@dataclass
class NormalizedHunk:
    """One hunk with its coherent change blocks, format-independent."""

    hunk_header: str
    old_start_line: Optional[int]
    old_line_count: Optional[int]
    old_end_line: Optional[int]
    new_start_line: Optional[int]
    new_line_count: Optional[int]
    new_end_line: Optional[int]
    change_type: str
    summary: str
    symbols: list[str]
    hunk_text_truncated: bool
    blocks: list[NormalizedBlock] = field(default_factory=list)

    @property
    def removed_lines(self) -> list[str]:
        """Removed line texts across all blocks, in diff-body order."""
        return [text for block in self.blocks for text in block.removed_line_texts]

    @property
    def added_lines(self) -> list[str]:
        """Added line texts across all blocks, in diff-body order."""
        return [text for block in self.blocks for text in block.added_line_texts]

    @property
    def removed_text(self) -> str:
        return "\n".join(block.removed_text for block in self.blocks if block.removed_text)

    @property
    def added_text(self) -> str:
        return "\n".join(block.added_text for block in self.blocks if block.added_text)


@dataclass
class NormalizedFile:
    """One changed file with its normalized hunks."""

    path: str
    old_path: Optional[str]
    status: str
    change_type: str
    additions: Optional[int]
    deletions: Optional[int]
    binary: bool
    binary_note: Optional[str]
    hunks: list[NormalizedHunk] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hunk normalization (the single v2/v3 compatibility point)
# ---------------------------------------------------------------------------
def _line_text(line: Any) -> str:
    if isinstance(line, dict):
        return str(line.get("text", "") or "")
    return str(line or "")


def _blocks_from_v2_lines(hunk: Mapping[str, Any]) -> list[NormalizedBlock]:
    """A v2 hunk (per-line arrays) becomes one aggregate block.

    v2 flattened the diff body, so precise per-run grouping is unrecoverable;
    a single block carrying all removed/added text preserves exactly what
    line-oriented consumers already used.
    """
    removed = [_line_text(line) for line in hunk.get("removed_lines") or []]
    added = [_line_text(line) for line in hunk.get("added_lines") or []]
    if not removed and not added:
        return []

    old_count = len(removed)
    new_count = len(added)
    if old_count and new_count:
        change_type = "modified"
    elif new_count:
        change_type = "added"
    else:
        change_type = "deleted"

    line_truncated = any(
        isinstance(line, dict) and line.get("text_truncated")
        for line in (*(hunk.get("removed_lines") or []), *(hunk.get("added_lines") or []))
    )
    return [
        NormalizedBlock(
            change_type=change_type,
            old_start_line=hunk.get("old_start_line") if old_count else None,
            old_line_count=old_count,
            old_end_line=hunk.get("old_end_line") if old_count else None,
            new_start_line=hunk.get("new_start_line") if new_count else None,
            new_line_count=new_count,
            new_end_line=hunk.get("new_end_line") if new_count else None,
            removed_text="\n".join(removed),
            added_text="\n".join(added),
            summary=str(hunk.get("summary", "") or ""),
            symbols=list(hunk.get("symbols") or []),
            text_truncated=bool(hunk.get("hunk_text_truncated")) or line_truncated,
        )
    ]


def _block_from_v3(block: Mapping[str, Any]) -> NormalizedBlock:
    return NormalizedBlock(
        change_type=str(block.get("change_type", "") or ""),
        old_start_line=block.get("old_start_line"),
        old_line_count=int(block.get("old_line_count") or 0),
        old_end_line=block.get("old_end_line"),
        new_start_line=block.get("new_start_line"),
        new_line_count=int(block.get("new_line_count") or 0),
        new_end_line=block.get("new_end_line"),
        removed_text=str(block.get("removed_text", "") or ""),
        added_text=str(block.get("added_text", "") or ""),
        summary=str(block.get("summary", "") or ""),
        symbols=list(block.get("symbols") or []),
        text_truncated=bool(block.get("text_truncated")),
    )


def normalize_hunk(hunk: Mapping[str, Any]) -> NormalizedHunk:
    """Normalize one hunk dict (v2 per-line arrays OR v3 change_blocks)."""
    if hunk.get("change_blocks") is not None:  # schema v3
        blocks = [_block_from_v3(block) for block in hunk.get("change_blocks") or []
                  if isinstance(block, dict)]
    else:  # schema v2 (or a hunk without block/line detail)
        blocks = _blocks_from_v2_lines(hunk)

    return NormalizedHunk(
        hunk_header=str(hunk.get("hunk_header", "") or ""),
        old_start_line=hunk.get("old_start_line"),
        old_line_count=hunk.get("old_line_count"),
        old_end_line=hunk.get("old_end_line"),
        new_start_line=hunk.get("new_start_line"),
        new_line_count=hunk.get("new_line_count"),
        new_end_line=hunk.get("new_end_line"),
        change_type=str(hunk.get("change_type", "") or ""),
        summary=str(hunk.get("summary", "") or ""),
        symbols=list(hunk.get("symbols") or []),
        hunk_text_truncated=bool(hunk.get("hunk_text_truncated")),
        blocks=blocks,
    )


# ---------------------------------------------------------------------------
# File / package iteration (v1 objects, v1/v2/v3 dicts)
# ---------------------------------------------------------------------------
def _normalize_file_entry(entry: Any) -> NormalizedFile:
    if isinstance(entry, dict):
        status = str(entry.get("status") or entry.get("change_type") or "")
        what_changed = entry.get("what_changed") or []
        hunks = [normalize_hunk(hunk) for hunk in what_changed if isinstance(hunk, dict)]
        return NormalizedFile(
            path=str(entry.get("path") or ""),
            old_path=entry.get("old_path"),
            status=status,
            change_type=str(entry.get("change_type") or status),
            additions=entry.get("additions"),
            deletions=entry.get("deletions"),
            binary=bool(entry.get("binary", False)),
            binary_note=entry.get("binary_note"),
            hunks=hunks,
        )
    if isinstance(entry, str):  # bare v1 path
        return NormalizedFile(entry, None, "", "", None, None, False, None, [])
    # ChangedFile-like object (v1)
    status = getattr(entry, "change_type", "") or ""
    return NormalizedFile(
        path=str(getattr(entry, "path", "") or ""),
        old_path=getattr(entry, "old_path", None),
        status=status,
        change_type=status,
        additions=None,
        deletions=None,
        binary=False,
        binary_note=None,
        hunks=[],
    )


def iter_normalized_files(change_package: Mapping[str, Any]) -> list[NormalizedFile]:
    """Every changed file of a v1/v2/v3 package as :class:`NormalizedFile`."""
    return [
        _normalize_file_entry(entry)
        for entry in change_package.get("changed_files") or []
    ]


def change_package_schema_version(change_package: Mapping[str, Any]) -> Optional[int]:
    version = change_package.get("schema_version")
    return version if isinstance(version, int) else None


def file_module_stem(path: str) -> str:
    return Path(path).stem if path else ""
