"""Parse the generated Markdown summary into sections.

The active TechDocker pipeline generates its own Markdown summaries with
deterministic ATX headings, so this intentionally simple parser replaces the
legacy DOCX parser for the summary path: headings are exactly ``#``/``##``/
``###`` lines (fenced code blocks are ignored), and section ids are stable
slugs of the heading path.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional

_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_TRAILING_COLON_RE = re.compile(r"\s*:\s*[-–—]?\s*$")


@dataclass
class MarkdownSection:
    """One heading-delimited section of a Markdown summary."""

    section_id: str
    heading: str
    level: int
    parent_id: Optional[str]
    path: str
    order: int
    content: str = ""
    content_hash: Optional[str] = None


def slugify(text: str) -> str:
    """Lowercase, dash-separated slug of a heading or path."""
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug or "section"


def unique_id(taken: set[str], base_slug: str) -> str:
    """Return ``base_slug``, suffixed with ``-2``, ``-3`` ... if already taken."""
    if base_slug not in taken:
        return base_slug
    counter = 2
    while f"{base_slug}-{counter}" in taken:
        counter += 1
    return f"{base_slug}-{counter}"


def normalize_heading(heading: str) -> str:
    """Lowercased heading, surrounding space and trailing colon removed."""
    return _TRAILING_COLON_RE.sub("", (heading or "").strip()).lower()


def content_hash(content: str) -> Optional[str]:
    """Stable sha256 of a section's content (None when empty)."""
    stripped = (content or "").strip()
    if not stripped:
        return None
    return hashlib.sha256(stripped.encode("utf-8")).hexdigest()


def _is_fence(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("```") or stripped.startswith("~~~")


def parse_markdown_sections(text: str) -> list[MarkdownSection]:
    """Split Markdown text into heading-delimited sections.

    ATX headings only; headings inside fenced code blocks are ignored.
    Parentage follows heading levels (a ``##`` under the preceding ``#``),
    ``path`` joins the ancestor headings, and duplicate headings get ``-2``/
    ``-3`` id suffixes.
    """
    sections: list[MarkdownSection] = []
    taken_ids: set[str] = set()
    stack: list[MarkdownSection] = []
    content_lines: dict[str, list[str]] = {}
    current: Optional[MarkdownSection] = None

    in_fence = False
    for line in (text or "").splitlines():
        if _is_fence(line):
            in_fence = not in_fence
            if current is not None:
                content_lines[current.section_id].append(line)
            continue

        match = None if in_fence else _ATX_HEADING_RE.match(line)
        if match is None:
            if current is not None:
                content_lines[current.section_id].append(line)
            continue

        level = len(match.group(1))
        heading = match.group(2).strip()

        while stack and stack[-1].level >= level:
            stack.pop()
        parent = stack[-1] if stack else None

        path = f"{parent.path} > {heading}" if parent else heading
        section_id = unique_id(taken_ids, slugify(path))
        taken_ids.add(section_id)

        current = MarkdownSection(
            section_id=section_id,
            heading=heading,
            level=level,
            parent_id=parent.section_id if parent else None,
            path=path,
            order=len(sections) + 1,
        )
        sections.append(current)
        stack.append(current)
        content_lines[section_id] = []

    for section in sections:
        section.content = "\n".join(content_lines[section.section_id]).strip()
        section.content_hash = content_hash(section.content)

    return sections
