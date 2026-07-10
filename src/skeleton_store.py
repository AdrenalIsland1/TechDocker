"""Load, save, and query the persistent document skeleton.

The skeleton is a JSON snapshot of a document's heading structure, built once
by :mod:`src.document_skeleton_builder`. Routine pushes read this small file
instead of re-parsing the full DOCX; the skeleton only changes when the
document's structure changes (a new heading/section is added).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SkeletonSection:
    """One heading/section of the document.

    ``classification`` distinguishes confirmed headings from probable ones
    (score 60-79): probable sections are routable targets like any other, but
    carry their review status so a future phase can treat them differently.
    """

    section_id: str
    heading: str
    level: int
    parent_id: Optional[str]
    path: str
    order: int
    summary: str = ""
    content_hash: Optional[str] = None
    source_document: Optional[str] = None
    classification: str = "heading"
    score: Optional[int] = None


@dataclass
class DocumentSkeleton:
    """The stored heading structure of one document."""

    source_document: str
    project_id: str
    generated_at: str
    sections: list[SkeletonSection] = field(default_factory=list)


def slugify(text: str) -> str:
    """Lowercase, dash-separated slug of a heading or path."""
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug or "section"


def unique_section_id(skeleton_ids: set[str], base_slug: str) -> str:
    """Return ``base_slug``, suffixed with ``-2``, ``-3`` ... if already taken."""
    if base_slug not in skeleton_ids:
        return base_slug
    counter = 2
    while f"{base_slug}-{counter}" in skeleton_ids:
        counter += 1
    return f"{base_slug}-{counter}"


def load_skeleton(path: str | Path) -> DocumentSkeleton:
    """Load a skeleton JSON file. Raises ``FileNotFoundError`` when absent."""
    skeleton_path = Path(path)
    if not skeleton_path.exists():
        raise FileNotFoundError(f"Skeleton not found: {skeleton_path}")

    data = json.loads(skeleton_path.read_text(encoding="utf-8"))
    return DocumentSkeleton(
        source_document=data.get("source_document", ""),
        project_id=data.get("project_id", ""),
        generated_at=data.get("generated_at", ""),
        sections=[
            SkeletonSection(**entry) for entry in data.get("sections", [])
        ],
    )


def save_skeleton(skeleton: DocumentSkeleton, path: str | Path) -> Path:
    """Write the skeleton as JSON, creating parent directories as needed."""
    skeleton_path = Path(path)
    skeleton_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source_document": skeleton.source_document,
        "project_id": skeleton.project_id,
        "generated_at": skeleton.generated_at,
        "sections": [asdict(section) for section in skeleton.sections],
    }
    skeleton_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return skeleton_path


def find_section_by_id(
    skeleton: DocumentSkeleton, section_id: str
) -> Optional[SkeletonSection]:
    """Return the section with this id, or ``None``."""
    for section in skeleton.sections:
        if section.section_id == section_id:
            return section
    return None


# Trailing colon endings (":", ":-", ":–", ":—") are separators, not part of
# the heading name: "Validation Rules:" must match a lookup for
# "Validation Rules". Only colon-led endings are stripped, so genuinely
# hyphenated headings ("Built-in") are unaffected.
_TRAILING_COLON_RE = re.compile(r"\s*:\s*[-–—]?\s*$")


def normalize_heading(heading: str) -> str:
    """Lowercased heading with surrounding space and trailing colon removed."""
    return _TRAILING_COLON_RE.sub("", (heading or "").strip()).lower()


def find_section_by_heading(
    skeleton: DocumentSkeleton, heading: str
) -> Optional[SkeletonSection]:
    """Return the first section whose heading matches.

    Matching is case-insensitive and ignores trailing colon endings on either
    side, so "Validation Rules" finds a section titled "Validation Rules:".
    """
    wanted = normalize_heading(heading)
    for section in skeleton.sections:
        if normalize_heading(section.heading) == wanted:
            return section
    return None


def find_section_containing(
    skeleton: DocumentSkeleton, keywords: list[str]
) -> Optional[SkeletonSection]:
    """Return the first section whose heading contains any keyword."""
    lowered = [keyword.lower() for keyword in keywords]
    for section in skeleton.sections:
        heading = section.heading.lower()
        if any(keyword in heading for keyword in lowered):
            return section
    return None


def append_section(
    skeleton: DocumentSkeleton,
    heading: str,
    level: int = 1,
    parent_id: Optional[str] = None,
) -> SkeletonSection:
    """Append a new section to the skeleton (in memory) and return it.

    The section id is a stable slug derived from the heading path, made unique
    against existing ids. The caller is responsible for saving the skeleton.
    """
    parent = find_section_by_id(skeleton, parent_id) if parent_id else None
    path = f"{parent.path} > {heading}" if parent else heading

    existing_ids = {section.section_id for section in skeleton.sections}
    section_id = unique_section_id(existing_ids, slugify(path))

    section = SkeletonSection(
        section_id=section_id,
        heading=heading,
        level=level,
        parent_id=parent.section_id if parent else None,
        path=path,
        order=len(skeleton.sections) + 1,
        summary="",
        content_hash=None,
        source_document=skeleton.source_document,
    )
    skeleton.sections.append(section)
    return section
