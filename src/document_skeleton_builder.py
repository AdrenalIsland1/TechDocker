"""Build the persistent document skeleton from a DOCX (one-time full parse).

Run as ``python3 -m src.document_skeleton_builder``.

This is the only place the full feature-based DOCX parser runs in the
automation flow: it converts the configured document's heading hierarchy into
a small JSON skeleton (``artifacts/skeletons/<project_id>_skeleton.json``).
Subsequent pushes route their updates using that JSON instead of re-parsing
the whole document.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.docx_parser import parse_docx_document
from src.project_resolver import ProjectConfig, resolve_project
from src.skeleton_store import (
    DocumentSkeleton,
    SkeletonSection,
    save_skeleton,
    slugify,
    unique_section_id,
)

SKELETON_DIRECTORY = Path("artifacts") / "skeletons"


def skeleton_path_for(project: ProjectConfig, repo_path: str | Path = ".") -> Path:
    """Where the skeleton JSON for this project lives."""
    return Path(repo_path) / SKELETON_DIRECTORY / f"{project.project_id}_skeleton.json"


def _content_hash(content: list[dict[str, Any]]) -> Optional[str]:
    """Stable hash of a section's direct paragraph texts (None when empty)."""
    texts = [paragraph.get("text", "") for paragraph in content]
    if not any(texts):
        return None
    joined = "\n".join(texts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_skeleton(
    document_path: str | Path,
    project_id: str,
    source_document: str,
) -> DocumentSkeleton:
    """Parse the DOCX once and flatten its headings into a skeleton.

    Both confirmed headings (score >= 80 or official styles) and probable
    headings (score 60-79, stored by the parser as review metadata) become
    skeleton sections, so the change router can target either. Entries are
    merged in document order and parentage is rebuilt with a level stack.
    """
    parsed = parse_docx_document(str(document_path))

    entries: list[dict[str, Any]] = []

    def walk(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            entries.append(
                {
                    "heading": node.get("title", "").strip(),
                    "level": int(node.get("level", 1)),
                    "position": int(node.get("position", 0)),
                    "classification": "heading",
                    "score": node.get("score"),
                    "content_hash": _content_hash(node.get("content", [])),
                }
            )
            walk(node.get("children", []))

    walk(parsed.get("headings", []))

    for probable in parsed.get("probable_headings", []):
        entries.append(
            {
                "heading": probable.get("text", "").strip(),
                # Probable headings may have no reliable level; treat them as
                # top-level rather than inventing depth.
                "level": int(probable.get("predicted_level") or 1),
                "position": int(probable.get("position", 0)),
                "classification": "probable_heading",
                "score": probable.get("score"),
                "content_hash": None,
            }
        )

    entries.sort(key=lambda entry: entry["position"])

    skeleton = DocumentSkeleton(
        source_document=source_document,
        project_id=project_id,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

    taken_ids: set[str] = set()
    stack: list[SkeletonSection] = []

    for entry in entries:
        while stack and stack[-1].level >= entry["level"]:
            stack.pop()
        parent = stack[-1] if stack else None

        path = f"{parent.path} > {entry['heading']}" if parent else entry["heading"]
        section_id = unique_section_id(taken_ids, slugify(path))
        taken_ids.add(section_id)

        section = SkeletonSection(
            section_id=section_id,
            heading=entry["heading"],
            level=entry["level"],
            parent_id=parent.section_id if parent else None,
            path=path,
            order=len(skeleton.sections) + 1,
            summary="",
            content_hash=entry["content_hash"],
            source_document=source_document,
            classification=entry["classification"],
            score=entry["score"],
        )
        skeleton.sections.append(section)
        stack.append(section)

    return skeleton


def build_and_save_skeleton(
    project: ProjectConfig,
    repo_path: str | Path = ".",
) -> tuple[DocumentSkeleton, Path]:
    """Build the skeleton for a project's document and write its JSON."""
    document_path = Path(repo_path) / project.document_location
    if not document_path.exists():
        raise FileNotFoundError(
            f"Configured document not found: {document_path} "
            f"(from document_location of {project.repository_name!r})"
        )

    skeleton = build_skeleton(
        document_path,
        project_id=project.project_id,
        source_document=project.document_location,
    )
    path = skeleton_path_for(project, repo_path)
    save_skeleton(skeleton, path)
    return skeleton, path


def main() -> int:
    """Entry point for ``python3 -m src.document_skeleton_builder``."""
    project = resolve_project("TechDocker")
    skeleton, path = build_and_save_skeleton(project)

    print("=" * 60)
    print("TechDocker Document Skeleton Builder")
    print("=" * 60)
    print(f"Source document: {skeleton.source_document}")
    print(f"Project ID:      {skeleton.project_id}")
    print(f"Sections:        {len(skeleton.sections)}")
    for section in skeleton.sections:
        indent = "  " * (section.level - 1)
        tag = f" ({section.classification}, score {section.score})"
        print(
            f"  {indent}[{section.section_id}] {section.heading} "
            f"(level {section.level}){tag}"
        )
    print(f"Skeleton written to: {path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
