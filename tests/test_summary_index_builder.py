"""Tests for the deterministic summary placement index.

Fully offline: temporary directories only, no network, no LLM provider, and
never touches the real repository artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.markdown_summary_parser import slugify
from src.summary_index_builder import (
    BLOCK_BLOCKQUOTE,
    BLOCK_CODE,
    BLOCK_HTML_COMMENT,
    BLOCK_ORDERED_LIST_ITEM,
    BLOCK_PARAGRAPH,
    BLOCK_TABLE,
    BLOCK_UNORDERED_LIST_ITEM,
    EXIT_CURRENT,
    EXIT_INVALID,
    EXIT_MISSING,
    EXIT_STALE,
    SCHEMA_VERSION,
    SummaryIndexError,
    build_summary_index,
    ensure_summary_index,
    extract_keywords,
    is_summary_index_current,
    load_summary_index,
    main,
    serialize_summary_index,
    split_sentences,
    summary_index_path,
    write_summary_index,
)
from src.summary_skeleton_store import SummarySkeleton, append_section

SIMPLE = """\
# Project Technical Summary

Intro paragraph under the H1 heading.

## Architecture

The router selects a section. The builder writes the index.

## Testing Strategy

Tests run offline.
"""


def blocks_of(index, section_id):
    for section in index["sections"]:
        if section["section_id"] == section_id:
            return section["blocks"]
    raise AssertionError(f"section {section_id} not found")


def all_blocks(index):
    return [b for s in index["sections"] for b in s["blocks"]]


# ---------------------------------------------------------------------------
# section parsing
# ---------------------------------------------------------------------------
def test_basic_h1_h2_section_parsing():
    index = build_summary_index(SIMPLE)
    headings = [(s["heading"], s["heading_level"]) for s in index["sections"]]
    assert headings == [
        ("Project Technical Summary", 1),
        ("Architecture", 2),
        ("Testing Strategy", 2),
    ]
    assert index["schema_version"] == SCHEMA_VERSION


def test_intro_paragraph_belongs_to_h1():
    index = build_summary_index(SIMPLE)
    intro = index["sections"][0]["blocks"]
    assert len(intro) == 1
    assert intro[0]["text"] == "Intro paragraph under the H1 heading."
    assert intro[0]["block_type"] == BLOCK_PARAGRAPH


def test_nested_h3_h4_heading_paths():
    text = (
        "# Top\n\n## Middle\n\n### Deep\n\nDeep prose here.\n\n"
        "#### Deeper\n\nDeeper prose here.\n"
    )
    index = build_summary_index(text)
    paths = [s["heading_path"] for s in index["sections"]]
    assert paths == [
        ["Top"],
        ["Top", "Middle"],
        ["Top", "Middle", "Deep"],
        ["Top", "Middle", "Deep", "Deeper"],
    ]
    assert index["sections"][3]["heading_level"] == 4


def test_section_ids_align_with_supplied_skeleton():
    skeleton = SummarySkeleton(
        project_id="techdocker", source_summary_path="s.md", generated_at=""
    )
    top = append_section(skeleton, "Project Technical Summary", level=1)
    append_section(skeleton, "Architecture", level=2, parent_id=top.section_id)
    append_section(skeleton, "Testing Strategy", level=2, parent_id=top.section_id)

    index = build_summary_index(SIMPLE, skeleton)
    ids = [s["section_id"] for s in index["sections"]]
    assert ids == [s.section_id for s in skeleton.sections]
    # Not an independently invented id for the same heading path.
    assert ids[1] == slugify("Project Technical Summary > Architecture")


def test_duplicate_headings_map_deterministically():
    text = "# Top\n\n## Notes\n\nFirst body.\n\n## Notes\n\nSecond body.\n"
    first = build_summary_index(text)
    second = build_summary_index(text)
    ids = [s["section_id"] for s in first["sections"]]
    assert ids == [s["section_id"] for s in second["sections"]]
    assert len(set(ids)) == len(ids)  # unique
    assert ids[2].endswith("-2")


# ---------------------------------------------------------------------------
# block types
# ---------------------------------------------------------------------------
def test_multi_line_paragraph_is_one_block():
    text = "# T\n\nLine one of the paragraph\ncontinues on line two here.\n"
    index = build_summary_index(text)
    blocks = index["sections"][0]["blocks"]
    assert len(blocks) == 1
    assert blocks[0]["text"] == (
        "Line one of the paragraph\ncontinues on line two here."
    )
    assert blocks[0]["start_line"] == 3 and blocks[0]["end_line"] == 4


def test_unordered_and_ordered_list_items():
    text = "# T\n\n- alpha item\n- beta item\n\n1. first step\n2. second step\n"
    blocks = build_summary_index(text)["sections"][0]["blocks"]
    kinds = [b["block_type"] for b in blocks]
    assert kinds == [
        BLOCK_UNORDERED_LIST_ITEM, BLOCK_UNORDERED_LIST_ITEM,
        BLOCK_ORDERED_LIST_ITEM, BLOCK_ORDERED_LIST_ITEM,
    ]
    assert blocks[0]["text"] == "- alpha item"
    assert blocks[0]["content_text"] == "alpha item"   # marker stripped
    assert blocks[2]["content_text"] == "first step"
    assert all(b["patchable"] for b in blocks)


def test_nested_list_items_are_indexed():
    text = "# T\n\n- parent item\n  - nested child item\n- sibling item\n"
    blocks = build_summary_index(text)["sections"][0]["blocks"]
    texts = [b["text"] for b in blocks]
    assert texts == ["- parent item", "  - nested child item", "- sibling item"]
    assert blocks[1]["content_text"] == "nested child item"


def test_blockquote_is_indexed_conservatively():
    text = "# T\n\n> Quoted guidance line.\n> Second quoted line.\n"
    blocks = build_summary_index(text)["sections"][0]["blocks"]
    assert blocks[0]["block_type"] == BLOCK_BLOCKQUOTE
    # Conservative: markers preserved exactly, not a patch target.
    assert blocks[0]["text"].startswith("> Quoted")
    assert blocks[0]["patchable"] is False


def test_fenced_code_block_is_opaque_and_not_patchable():
    text = "# T\n\n```python\ndef f():\n    return 1\n```\n"
    blocks = build_summary_index(text)["sections"][0]["blocks"]
    assert blocks[0]["block_type"] == BLOCK_CODE
    assert blocks[0]["patchable"] is False
    assert blocks[0]["sentences"] == []
    assert blocks[0]["text"].startswith("```python") and blocks[0]["text"].endswith("```")


def test_heading_like_text_inside_code_fence_creates_no_section():
    text = "# Real\n\n```text\n# Not A Heading\n## Also Not\n```\n\n## Actual\n\nBody.\n"
    index = build_summary_index(text)
    headings = [s["heading"] for s in index["sections"]]
    assert headings == ["Real", "Actual"]


def test_markdown_table_is_opaque_and_not_patchable():
    text = "# T\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"
    blocks = build_summary_index(text)["sections"][0]["blocks"]
    assert blocks[0]["block_type"] == BLOCK_TABLE
    assert blocks[0]["patchable"] is False
    assert blocks[0]["sentences"] == []
    assert blocks[0]["text"].count("\n") == 2


def test_html_comment_is_not_patchable():
    text = "# T\n\n<!-- an ordinary note -->\n\nProse follows.\n"
    blocks = build_summary_index(text)["sections"][0]["blocks"]
    assert blocks[0]["block_type"] == BLOCK_HTML_COMMENT
    assert blocks[0]["patchable"] is False


def test_techdocker_generated_region_is_not_a_placement_candidate():
    text = (
        "# T\n\n## Testing Strategy\n\nOriginal prose stays patchable.\n\n"
        "<!-- TECHDOCKER_UPDATE_START -->\n"
        "### Automated Change Update - 2026-01-01\n\n"
        "Repository: TechDocker\n\n"
        "- modified: src/example.py\n\n"
        "<!-- TECHDOCKER_UPDATE_END -->\n"
    )
    index = build_summary_index(text)

    generated_blocks = [b for b in all_blocks(index) if b["generated"]]
    assert generated_blocks, "generated region should be detected"
    assert all(not b["patchable"] for b in generated_blocks)
    # The generated heading's section is flagged too.
    generated_sections = [s for s in index["sections"] if s["generated"]]
    assert [s["heading"] for s in generated_sections] == [
        "Automated Change Update - 2026-01-01"
    ]
    # Ordinary prose outside the region is still patchable.
    ordinary = [b for b in all_blocks(index) if not b["generated"]]
    assert any(b["patchable"] and "Original prose" in b["text"] for b in ordinary)
    # Markers themselves are never candidates.
    markers = [b for b in all_blocks(index) if b["block_type"] == BLOCK_HTML_COMMENT]
    assert len(markers) == 2 and all(not b["patchable"] for b in markers)


# ---------------------------------------------------------------------------
# sentence splitting
# ---------------------------------------------------------------------------
def texts(spans, source):
    return [source[a:b] for a, b in spans]


def test_sentence_splitting_normal_prose():
    text = "The router picks a section. The builder writes the index! Does it work?"
    assert texts(split_sentences(text), text) == [
        "The router picks a section.",
        "The builder writes the index!",
        "Does it work?",
    ]


def test_decimal_numbers_are_not_split():
    text = "Version 1.5 is current. Threshold 0.75 applies."
    assert texts(split_sentences(text), text) == [
        "Version 1.5 is current.",
        "Threshold 0.75 applies.",
    ]


def test_common_abbreviations_are_not_split():
    text = "Use a provider, e.g. Ollama, locally. Dr. Smith agreed. See i.e. That case."
    result = texts(split_sentences(text), text)
    assert "Use a provider, e.g. Ollama, locally." in result
    assert any("Dr. Smith agreed." in part for part in result)
    assert not any(part.strip() in {"e.g.", "Dr.", "i.e."} for part in result)


def test_urls_module_names_and_inline_code_preserved():
    text = (
        "See https://example.com/docs.html for details. "
        "Edit src/summary_change_router.py now. "
        "The file `base_updated_summary.md` changes."
    )
    result = texts(split_sentences(text), text)
    assert "See https://example.com/docs.html for details." in result
    assert "Edit src/summary_change_router.py now." in result
    assert "The file `base_updated_summary.md` changes." in result


def test_sentences_never_overlap_or_duplicate():
    text = "First one. Second one. Third one."
    spans = split_sentences(text)
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
    assert "".join(texts(spans, text)).replace(" ", "") == text.replace(" ", "")


# ---------------------------------------------------------------------------
# offsets, lines, hashes
# ---------------------------------------------------------------------------
RICH = (
    "# Top\n\nIntro sentence one. Intro sentence two.\n\n"
    "## Second\n\n- list item prose here\n\n"
    "Paragraph after the list.\n"
)


def test_exact_source_offsets_for_blocks():
    index = build_summary_index(RICH)
    for block in all_blocks(index):
        start, end = block["source_start_offset"], block["source_end_offset"]
        assert RICH[start:end] == block["text"]


def test_exact_source_offsets_for_sentences():
    index = build_summary_index(RICH)
    found = 0
    for block in all_blocks(index):
        for sentence in block["sentences"]:
            start, end = sentence["source_start_offset"], sentence["source_end_offset"]
            assert RICH[start:end] == sentence["text"]
            # block-relative offsets agree with the block text
            assert (
                block["text"][sentence["block_start_offset"] : sentence["block_end_offset"]]
                == sentence["text"]
            )
            found += 1
    assert found >= 3


def test_one_based_inclusive_line_numbers():
    index = build_summary_index(RICH)
    intro = index["sections"][0]["blocks"][0]
    assert intro["start_line"] == 3 and intro["end_line"] == 3
    lines = RICH.splitlines()
    assert lines[intro["start_line"] - 1] == intro["text"]


def test_document_section_block_and_sentence_hashes():
    import hashlib

    index = build_summary_index(RICH)
    assert index["source"]["sha256"] == hashlib.sha256(RICH.encode()).hexdigest()
    assert index["source"]["size_bytes"] == len(RICH.encode())
    for section in index["sections"]:
        assert section["content_hash"]
        for block in section["blocks"]:
            assert block["content_hash"] == hashlib.sha256(
                block["text"].encode()
            ).hexdigest()
            for sentence in block["sentences"]:
                assert sentence["content_hash"] == hashlib.sha256(
                    sentence["text"].encode()
                ).hexdigest()


# ---------------------------------------------------------------------------
# keywords
# ---------------------------------------------------------------------------
def test_keyword_extraction_is_deterministic():
    text = "The summary router writes the index."
    assert extract_keywords(text) == extract_keywords(text)
    assert "summary" in extract_keywords(text)
    assert "the" not in extract_keywords(text)  # stop word


def test_technical_token_tokenization():
    keywords = extract_keywords(
        "summary_change_router SummaryChangeRouter summary-change-router "
        "src/summary_change_router.py base_updated_summary.md CI/CD"
    )
    for expected in ("summary", "change", "router", "base", "updated", "ci", "cd"):
        assert expected in keywords, expected
    assert "summary_change_router" in keywords          # snake_case whole token
    assert "summarychangerouter" in keywords            # camelCase whole token
    assert "src/summary_change_router.py" in keywords   # dotted path whole token
    assert "ci/cd" in keywords


# ---------------------------------------------------------------------------
# identifier stability
# ---------------------------------------------------------------------------
def test_identical_input_produces_stable_ids_and_bytes():
    first = build_summary_index(RICH)
    second = build_summary_index(RICH)
    assert serialize_summary_index(first) == serialize_summary_index(second)
    assert [b["block_id"] for b in all_blocks(first)] == [
        b["block_id"] for b in all_blocks(second)
    ]


def test_unrelated_insertion_keeps_unchanged_block_ids():
    before = build_summary_index(RICH)
    changed = RICH.replace(
        "## Second\n", "## Inserted\n\nBrand new paragraph.\n\n## Second\n"
    )
    after = build_summary_index(changed)

    unchanged = "Paragraph after the list."
    before_id = next(b["block_id"] for b in all_blocks(before) if b["text"] == unchanged)
    after_id = next(b["block_id"] for b in all_blocks(after) if b["text"] == unchanged)
    assert before_id == after_id


def test_duplicate_block_text_gets_unique_ids():
    text = "# T\n\nRepeated body text.\n\nRepeated body text.\n"
    blocks = build_summary_index(text)["sections"][0]["blocks"]
    assert blocks[0]["text"] == blocks[1]["text"]
    assert blocks[0]["content_hash"] == blocks[1]["content_hash"]
    assert blocks[0]["block_id"] != blocks[1]["block_id"]
    assert blocks[0]["block_id"].endswith("-1") and blocks[1]["block_id"].endswith("-2")


def test_editing_one_block_does_not_corrupt_others():
    before = build_summary_index(RICH)
    after = build_summary_index(RICH.replace("Intro sentence two.", "Intro sentence TWO."))

    edited_before = before["sections"][0]["blocks"][0]
    edited_after = after["sections"][0]["blocks"][0]
    assert edited_before["block_id"] != edited_after["block_id"]
    assert edited_before["content_hash"] != edited_after["content_hash"]

    untouched = "Paragraph after the list."
    assert next(b["block_id"] for b in all_blocks(before) if b["text"] == untouched) == (
        next(b["block_id"] for b in all_blocks(after) if b["text"] == untouched)
    )


# ---------------------------------------------------------------------------
# persistence, staleness, ensure
# ---------------------------------------------------------------------------
def test_missing_index_and_hash_states(tmp_path):
    index = build_summary_index(RICH, source_path="s.md")
    assert is_summary_index_current(index, RICH, "s.md") is True
    assert is_summary_index_current(index, RICH + "\nextra\n", "s.md") is False
    assert is_summary_index_current(index, RICH, "other.md") is False

    unsupported = dict(index, schema_version=999)
    assert is_summary_index_current(unsupported, RICH, "s.md") is False

    with pytest.raises(FileNotFoundError):
        load_summary_index(tmp_path / "nope.json")


def test_malformed_index_json_is_reported_clearly(tmp_path):
    path = tmp_path / "base_summary_index.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SummaryIndexError, match="not valid JSON"):
        load_summary_index(path)

    path.write_text(json.dumps({"sections": []}), encoding="utf-8")
    with pytest.raises(SummaryIndexError, match="missing required fields"):
        load_summary_index(path)


def test_write_and_load_roundtrip(tmp_path):
    index = build_summary_index(RICH, source_path="s.md")
    path = write_summary_index(index, tmp_path / "idx.json")
    assert load_summary_index(path) == index
    assert not (tmp_path / "idx.json.tmp").exists()  # atomic temp cleaned up


def make_summary_repo(tmp_path, text=RICH):
    summaries = tmp_path / "artifacts" / "summaries"
    summaries.mkdir(parents=True)
    source = summaries / "base_updated_summary.md"
    source.write_text(text, encoding="utf-8")
    return source


def test_ensure_creates_rebuilds_and_skips(tmp_path):
    source = make_summary_repo(tmp_path)
    destination = summary_index_path(tmp_path)

    created = ensure_summary_index(source, destination, relative_source="s.md")
    assert created.status == "created" and destination.exists()

    unchanged_bytes = destination.read_bytes()
    current = ensure_summary_index(source, destination, relative_source="s.md")
    assert current.status == "current"
    assert destination.read_bytes() == unchanged_bytes  # no rewrite

    source.write_text(RICH + "\nA new trailing paragraph.\n", encoding="utf-8")
    rebuilt = ensure_summary_index(source, destination, relative_source="s.md")
    assert rebuilt.status == "rebuilt"
    assert destination.read_bytes() != unchanged_bytes


def test_ensure_rebuilds_over_invalid_json(tmp_path):
    source = make_summary_repo(tmp_path)
    destination = summary_index_path(tmp_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("{broken", encoding="utf-8")

    result = ensure_summary_index(source, destination, relative_source="s.md")
    assert result.status == "rebuilt"
    assert "invalid" in result.reason
    assert load_summary_index(destination)["schema_version"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_preview_writes_nothing(tmp_path, capsys):
    make_summary_repo(tmp_path)
    code = main(["--preview", "--repo-path", str(tmp_path)])
    captured = capsys.readouterr()

    assert code == EXIT_CURRENT
    assert not summary_index_path(tmp_path).exists()
    payload = json.loads(captured.out)               # stdout is pure JSON
    assert payload["schema_version"] == SCHEMA_VERSION
    assert "no files were written" in captured.err   # diagnostics on stderr


def test_cli_check_exit_codes(tmp_path, capsys):
    source = make_summary_repo(tmp_path)

    assert main(["--check", "--repo-path", str(tmp_path)]) == EXIT_MISSING
    assert not summary_index_path(tmp_path).exists()

    assert main(["--write", "--repo-path", str(tmp_path)]) == EXIT_CURRENT
    assert main(["--check", "--repo-path", str(tmp_path)]) == EXIT_CURRENT

    source.write_text(RICH + "\nStale-making paragraph.\n", encoding="utf-8")
    assert main(["--check", "--repo-path", str(tmp_path)]) == EXIT_STALE

    summary_index_path(tmp_path).write_text("{broken", encoding="utf-8")
    assert main(["--check", "--repo-path", str(tmp_path)]) == EXIT_INVALID


def test_cli_check_writes_nothing(tmp_path):
    source = make_summary_repo(tmp_path)
    main(["--write", "--repo-path", str(tmp_path)])
    index_bytes = summary_index_path(tmp_path).read_bytes()
    source_bytes = source.read_bytes()

    main(["--check", "--repo-path", str(tmp_path)])

    assert summary_index_path(tmp_path).read_bytes() == index_bytes
    assert source.read_bytes() == source_bytes


def test_cli_write_modifies_only_the_index(tmp_path):
    source = make_summary_repo(tmp_path)
    original = tmp_path / "artifacts" / "summaries" / "base_original_summary.md"
    original.write_text("# Baseline\n\nUntouched baseline.\n", encoding="utf-8")
    skeleton = summary_index_path(tmp_path).parent / "base_skeleton.json"
    skeleton.parent.mkdir(parents=True, exist_ok=True)
    skeleton.write_text(json.dumps({"project_id": "x", "sections": []}), encoding="utf-8")

    before = {
        "source": source.read_bytes(),
        "original": original.read_bytes(),
        "skeleton": skeleton.read_bytes(),
    }
    assert main(["--write", "--repo-path", str(tmp_path)]) == EXIT_CURRENT

    assert summary_index_path(tmp_path).exists()
    assert source.read_bytes() == before["source"]
    assert original.read_bytes() == before["original"]
    assert skeleton.read_bytes() == before["skeleton"]


# ---------------------------------------------------------------------------
# offline guarantees
# ---------------------------------------------------------------------------
def test_module_imports_no_llm_or_network():
    source = Path("src/summary_index_builder.py").read_text(encoding="utf-8")
    for forbidden in (
        "llm_provider", "llm_change_analyzer", "ollama", "urllib",
        "requests", "socket", "http.client",
    ):
        assert forbidden not in source, forbidden
