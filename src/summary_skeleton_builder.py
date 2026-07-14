"""Build ``artifacts/skeletons/base_skeleton.json`` from the Markdown summary.

Run as ``python3 -m src.summary_skeleton_builder``.

The skeleton is based on the **original** baseline summary
(``base_original_summary.md``); when the router later needs a new section it
is appended to the skeleton by the updater, without rebuilding from the
reviewable copy. ``base_updated_summary.md`` is only used as a fallback
source when the baseline does not exist. Parsing uses the simple Markdown
parser — the legacy DOCX parser is not involved.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.markdown_summary_parser import parse_markdown_sections
from src.project_summary_generator import (
    original_summary_path,
    updated_summary_path,
)
from src.summary_skeleton_store import (
    SummarySection,
    SummarySkeleton,
    save_summary_skeleton,
)

SKELETON_DIRECTORY = Path("artifacts") / "skeletons"
SKELETON_NAME = "base_skeleton.json"

PROJECT_ID = "techdocker"

# Headings of automated update blocks are transient content, not structure:
# they must not become routing targets when the skeleton is rebuilt.
_UPDATE_BLOCK_HEADING_PREFIX = "automated change update"


def summary_skeleton_path(repo_path: str | Path = ".") -> Path:
    return Path(repo_path) / SKELETON_DIRECTORY / SKELETON_NAME


def default_summary_source(repo_path: str | Path = ".") -> Path:
    """The skeleton is based on the original baseline summary.

    Falls back to the updated copy only when the baseline does not exist.
    """
    original = original_summary_path(repo_path)
    if original.exists():
        return original
    return updated_summary_path(repo_path)


def build_summary_skeleton(
    summary_path: str | Path,
    project_id: str = PROJECT_ID,
    source_summary_path: str | None = None,
) -> SummarySkeleton:
    """Parse a Markdown summary file into a skeleton."""
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")

    sections = parse_markdown_sections(path.read_text(encoding="utf-8"))

    skeleton = SummarySkeleton(
        project_id=project_id,
        source_summary_path=source_summary_path or str(path),
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )
    for parsed in sections:
        if parsed.heading.strip().lower().startswith(_UPDATE_BLOCK_HEADING_PREFIX):
            continue
        skeleton.sections.append(
            SummarySection(
                section_id=parsed.section_id,
                heading=parsed.heading,
                level=parsed.level,
                parent_id=parsed.parent_id,
                path=parsed.path,
                order=len(skeleton.sections) + 1,
                content_hash=parsed.content_hash,
            )
        )
    return skeleton


def build_and_save_summary_skeleton(
    repo_path: str | Path = ".",
    source: str | Path | None = None,
) -> tuple[SummarySkeleton, Path]:
    """Build the skeleton from the default (or given) source and save it."""
    source_path = Path(source) if source else default_summary_source(repo_path)
    try:
        relative_source = str(source_path.relative_to(Path(repo_path).resolve()))
    except ValueError:
        relative_source = str(source_path)

    skeleton = build_summary_skeleton(
        source_path, source_summary_path=relative_source
    )
    path = summary_skeleton_path(repo_path)
    save_summary_skeleton(skeleton, path)
    return skeleton, path


def main() -> int:
    skeleton, path = build_and_save_summary_skeleton()

    print("=" * 60)
    print("TechDocker Summary Skeleton Builder")
    print("=" * 60)
    print(f"Source summary: {skeleton.source_summary_path}")
    print(f"Sections:       {len(skeleton.sections)}")
    for section in skeleton.sections:
        indent = "  " * (section.level - 1)
        print(f"  {indent}[{section.section_id}] {section.heading} (level {section.level})")
    print(f"Skeleton written to: {path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
