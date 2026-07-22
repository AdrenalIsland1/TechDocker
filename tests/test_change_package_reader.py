"""Tests for the centralized change-package normalization API.

Consumers read v1/v2/v3 packages through here; these prove all three formats
normalize to the same coherent block facts.
"""

from __future__ import annotations

import json

from src.change_package_reader import (
    iter_normalized_files,
    normalize_hunk,
)

V2_HUNK = {
    "hunk_header": "@@ -52,2 +52,3 @@ def f",
    "old_start_line": 52, "old_line_count": 2, "old_end_line": 53,
    "new_start_line": 52, "new_line_count": 3, "new_end_line": 54,
    "change_type": "modified",
    "summary": "Modified function f with 2 added and 1 removed lines.",
    "symbols": ["f"],
    "hunk_text_truncated": False,
    "added_lines": [
        {"line_number": 52, "text": "new one"},
        {"line_number": 53, "text": "new two"},
    ],
    "removed_lines": [{"line_number": 52, "text": "old one"}],
}

V3_HUNK = {
    "hunk_header": "@@ -52,2 +52,3 @@ def f",
    "old_start_line": 52, "old_line_count": 2, "old_end_line": 53,
    "new_start_line": 52, "new_line_count": 3, "new_end_line": 54,
    "change_type": "modified",
    "summary": "Modified function f with 2 added and 1 removed lines.",
    "symbols": ["f"],
    "hunk_text_truncated": False,
    "change_blocks": [
        {
            "block_index": 1, "change_type": "modified",
            "old_start_line": 52, "old_line_count": 1, "old_end_line": 52,
            "new_start_line": 52, "new_line_count": 1, "new_end_line": 52,
            "removed_text": "old one", "added_text": "new one",
            "summary": "Replaced old line 52 with new line 52.",
            "symbols": ["f"], "text_truncated": False,
        },
        {
            "block_index": 2, "change_type": "added",
            "old_start_line": None, "old_line_count": 0, "old_end_line": None,
            "new_start_line": 53, "new_line_count": 1, "new_end_line": 53,
            "removed_text": "", "added_text": "new two",
            "summary": "Added new line 53.",
            "symbols": ["f"], "text_truncated": False,
        },
    ],
}


# ---------------------------------------------------------------------------
# v2 / v3 normalize to the same line facts
# ---------------------------------------------------------------------------
def test_v3_hunk_exposes_blocks_and_aggregate_text():
    hunk = normalize_hunk(V3_HUNK)
    assert len(hunk.blocks) == 2
    assert hunk.removed_lines == ["old one"]
    assert hunk.added_lines == ["new one", "new two"]
    assert hunk.added_text == "new one\nnew two"
    assert hunk.removed_text == "old one"
    assert hunk.symbols == ["f"]


def test_v2_hunk_normalizes_to_equivalent_line_facts():
    hunk = normalize_hunk(V2_HUNK)
    # v2 flattens to a single aggregate block, but exposes the same line facts.
    assert hunk.removed_lines == ["old one"]
    assert hunk.added_lines == ["new one", "new two"]
    assert hunk.symbols == ["f"]


def test_v2_and_v3_yield_identical_line_facts():
    v2, v3 = normalize_hunk(V2_HUNK), normalize_hunk(V3_HUNK)
    assert v2.added_lines == v3.added_lines
    assert v2.removed_lines == v3.removed_lines
    assert v2.added_text == v3.added_text
    assert v2.removed_text == v3.removed_text


def test_pure_addition_block_has_no_removed_lines():
    hunk = normalize_hunk(V3_HUNK)
    added_block = hunk.blocks[1]
    assert added_block.removed_line_texts == []
    assert added_block.added_line_texts == ["new two"]


# ---------------------------------------------------------------------------
# file-level iteration for v1/v2/v3
# ---------------------------------------------------------------------------
def test_iter_files_v1_bare_entries():
    package = {"schema_version": 1, "changed_files": [
        {"path": "a.py", "change_type": "modified"},
        "b.txt",
    ]}
    files = iter_normalized_files(package)
    assert [f.path for f in files] == ["a.py", "b.txt"]
    assert all(f.hunks == [] for f in files)  # no hunk detail in v1


def test_iter_files_v2_and_v3_have_normalized_hunks():
    for hunk in (V2_HUNK, V3_HUNK):
        package = {"changed_files": [
            {"path": "src/f.py", "status": "modified", "additions": 2,
             "deletions": 1, "binary": False, "what_changed": [hunk]}
        ]}
        files = iter_normalized_files(package)
        assert len(files) == 1
        assert files[0].hunks[0].added_lines == ["new one", "new two"]


def test_binary_file_has_no_hunks():
    package = {"changed_files": [
        {"path": "logo.png", "status": "modified", "binary": True,
         "binary_note": "Binary file changed; textual hunks were not extracted.",
         "additions": None, "deletions": None, "what_changed": []}
    ]}
    normalized = iter_normalized_files(package)[0]
    assert normalized.binary is True
    assert normalized.hunks == []


# ---------------------------------------------------------------------------
# JSON round trip
# ---------------------------------------------------------------------------
def test_v3_hunk_survives_json_round_trip():
    reloaded = json.loads(json.dumps(V3_HUNK))
    hunk = normalize_hunk(reloaded)
    assert hunk.added_lines == ["new one", "new two"]
    assert hunk.blocks[0].summary == "Replaced old line 52 with new line 52."
