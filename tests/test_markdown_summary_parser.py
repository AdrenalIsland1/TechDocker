"""Tests for the Markdown summary parser."""

from __future__ import annotations

from src.markdown_summary_parser import (
    content_hash,
    normalize_heading,
    parse_markdown_sections,
)

SAMPLE = """# Project Technical Summary

Intro paragraph.

## System Overview

The system does things.

## Core Modules

- module a

### Helpers

Helper details.

## Testing Strategy

```python
# this comment looks like a heading but is inside a fence
value = 1
```

Real testing content.
"""


def test_parses_headings_with_levels_and_order():
    sections = parse_markdown_sections(SAMPLE)
    headings = [(s.heading, s.level, s.order) for s in sections]
    assert headings == [
        ("Project Technical Summary", 1, 1),
        ("System Overview", 2, 2),
        ("Core Modules", 2, 3),
        ("Helpers", 3, 4),
        ("Testing Strategy", 2, 5),
    ]


def test_ignores_headings_inside_code_fences():
    sections = parse_markdown_sections(SAMPLE)
    assert not any("comment" in s.heading for s in sections)
    testing = next(s for s in sections if s.heading == "Testing Strategy")
    # Fence content stays inside the section content.
    assert "value = 1" in testing.content
    assert "Real testing content." in testing.content


def test_parent_child_relationships_and_paths():
    sections = parse_markdown_sections(SAMPLE)
    by_heading = {s.heading: s for s in sections}

    assert by_heading["System Overview"].parent_id == "project-technical-summary"
    assert by_heading["Helpers"].parent_id == (
        "project-technical-summary-core-modules"
    )
    assert by_heading["Helpers"].path == (
        "Project Technical Summary > Core Modules > Helpers"
    )
    assert by_heading["Testing Strategy"].parent_id == "project-technical-summary"


def test_duplicate_headings_get_suffixed_ids():
    text = "## Notes\n\na\n\n## Notes\n\nb\n\n## Notes\n\nc\n"
    sections = parse_markdown_sections(text)
    assert [s.section_id for s in sections] == ["notes", "notes-2", "notes-3"]


def test_content_hashes_are_stable_and_content_sensitive():
    first = parse_markdown_sections(SAMPLE)
    second = parse_markdown_sections(SAMPLE)
    assert [s.content_hash for s in first] == [s.content_hash for s in second]

    changed = parse_markdown_sections(SAMPLE.replace("does things", "does MORE"))
    overview_a = next(s for s in first if s.heading == "System Overview")
    overview_b = next(s for s in changed if s.heading == "System Overview")
    assert overview_a.content_hash != overview_b.content_hash

    assert content_hash("") is None
    assert content_hash("x") == content_hash("  x  ")


def test_normalize_heading_strips_trailing_colon():
    assert normalize_heading("Validation Rules:") == "validation rules"
    assert normalize_heading("Built-in") == "built-in"
