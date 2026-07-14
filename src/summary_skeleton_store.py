"""Load, save, and query ``artifacts/skeletons/base_skeleton.json``.

The summary skeleton is the routing structure of the active pipeline. It is
rewritten only when: it is missing (initial creation), a new heading/
subheading is added, or an explicit rebuild is run. Normal updates to
existing sections never touch it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from src.markdown_summary_parser import normalize_heading, slugify, unique_id


@dataclass
class SummarySection:
    """One section of the summary skeleton."""

    section_id: str
    heading: str
    level: int
    parent_id: Optional[str]
    path: str
    order: int
    content_hash: Optional[str] = None


@dataclass
class SummarySkeleton:
    """The stored structure of the Markdown summary."""

    project_id: str
    source_summary_path: str
    generated_at: str
    sections: list[SummarySection] = field(default_factory=list)


def load_summary_skeleton(path: str | Path) -> SummarySkeleton:
    """Load a skeleton JSON file. Raises ``FileNotFoundError`` when absent."""
    skeleton_path = Path(path)
    if not skeleton_path.exists():
        raise FileNotFoundError(f"Summary skeleton not found: {skeleton_path}")

    data = json.loads(skeleton_path.read_text(encoding="utf-8"))
    return SummarySkeleton(
        project_id=data.get("project_id", ""),
        source_summary_path=data.get("source_summary_path", ""),
        generated_at=data.get("generated_at", ""),
        sections=[SummarySection(**entry) for entry in data.get("sections", [])],
    )


def save_summary_skeleton(skeleton: SummarySkeleton, path: str | Path) -> Path:
    """Write the skeleton as JSON, creating parent directories as needed."""
    skeleton_path = Path(path)
    skeleton_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "project_id": skeleton.project_id,
        "source_summary_path": skeleton.source_summary_path,
        "generated_at": skeleton.generated_at,
        "sections": [asdict(section) for section in skeleton.sections],
    }
    skeleton_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return skeleton_path


def find_section_by_id(
    skeleton: SummarySkeleton, section_id: str
) -> Optional[SummarySection]:
    for section in skeleton.sections:
        if section.section_id == section_id:
            return section
    return None


def find_section_by_heading(
    skeleton: SummarySkeleton, heading: str
) -> Optional[SummarySection]:
    """First section whose heading matches (case/trailing-colon insensitive)."""
    wanted = normalize_heading(heading)
    for section in skeleton.sections:
        if normalize_heading(section.heading) == wanted:
            return section
    return None


def find_best_section_by_keywords(
    skeleton: SummarySkeleton, keywords: list[str]
) -> Optional[SummarySection]:
    """Section whose heading/path matches the most keywords (heading weighs double)."""
    best: Optional[SummarySection] = None
    best_score = 0
    for section in skeleton.sections:
        heading = section.heading.lower()
        path = section.path.lower()
        score = sum(
            2 if keyword.lower() in heading else 1 if keyword.lower() in path else 0
            for keyword in keywords
        )
        if score > best_score:
            best, best_score = section, score
    return best


def append_section(
    skeleton: SummarySkeleton,
    heading: str,
    level: int = 2,
    parent_id: Optional[str] = None,
) -> SummarySection:
    """Append a new section (in memory) and return it; caller saves."""
    parent = find_section_by_id(skeleton, parent_id) if parent_id else None
    path = f"{parent.path} > {heading}" if parent else heading

    taken = {section.section_id for section in skeleton.sections}
    section = SummarySection(
        section_id=unique_id(taken, slugify(path)),
        heading=heading,
        level=level,
        parent_id=parent.section_id if parent else None,
        path=path,
        order=len(skeleton.sections) + 1,
        content_hash=None,
    )
    skeleton.sections.append(section)
    return section


def update_section_metadata(
    skeleton: SummarySkeleton, section_id: str, **fields
) -> Optional[SummarySection]:
    """Update attributes of one section in memory; returns it (or ``None``)."""
    section = find_section_by_id(skeleton, section_id)
    if section is None:
        return None
    for name, value in fields.items():
        if not hasattr(section, name):
            raise AttributeError(f"SummarySection has no field {name!r}")
        setattr(section, name, value)
    return section
