"""Schema-v3 change-block grouping, ranges, truncation, and symbol attribution.

Pure and offline: :func:`parse_unified_diff` operates on strings, and symbol
attribution uses AST helpers. No git, no network, no LLM.
"""

from __future__ import annotations

from src.git_change_detector import _attach_block_symbols, _hunk_to_dict
from src.git_diff_parser import (
    block_change_summary,
    extract_python_symbols,
    parse_unified_diff,
)


def blocks(diff, **caps):
    return parse_unified_diff(diff, **caps)[0].blocks


# ---------------------------------------------------------------------------
# grouping rules
# ---------------------------------------------------------------------------
def test_standard_modified_block():
    b = blocks("@@ -1,2 +1,2 @@\n context\n-old\n+new")
    assert len(b) == 1
    assert b[0].change_type == "modified"
    assert b[0].removed_text == "old" and b[0].added_text == "new"


def test_consecutive_removed_then_added_is_one_modified_block():
    b = blocks("@@ -1,4 +1,4 @@\n ctx\n-r1\n-r2\n+a1\n+a2\n+a3")
    assert len(b) == 1
    assert b[0].change_type == "modified"
    assert b[0].removed_text == "r1\nr2"
    assert b[0].added_text == "a1\na2\na3"
    assert b[0].old_line_count == 2 and b[0].new_line_count == 3


def test_unchanged_context_splits_blocks():
    b = blocks("@@ -1,4 +1,4 @@\n-a\n+b\n unchanged\n-c\n+d")
    assert len(b) == 2
    assert b[0].added_text == "b" and b[1].added_text == "d"


def test_multiple_blocks_in_one_hunk_have_sequential_indexes():
    b = blocks("@@ -1,6 +1,6 @@\n-a\n+b\n ctx1\n+c\n ctx2\n-d")
    assert [blk.block_index for blk in b] == [1, 2, 3]
    assert [blk.change_type for blk in b] == ["modified", "added", "deleted"]


def test_pure_addition_block():
    b = blocks("@@ -1,1 +1,3 @@\n keep\n+x\n+y")
    assert len(b) == 1
    blk = b[0]
    assert blk.change_type == "added"
    assert blk.old_line_count == 0
    assert blk.old_start_line is None and blk.old_end_line is None
    assert blk.removed_text == ""
    assert blk.added_text == "x\ny"


def test_pure_deletion_block():
    b = blocks("@@ -1,3 +1,1 @@\n keep\n-x\n-y")
    assert len(b) == 1
    blk = b[0]
    assert blk.change_type == "deleted"
    assert blk.new_line_count == 0
    assert blk.new_start_line is None and blk.new_end_line is None
    assert blk.added_text == ""
    assert blk.removed_text == "x\ny"


# ---------------------------------------------------------------------------
# range mathematics
# ---------------------------------------------------------------------------
def test_range_math_for_modified_block():
    # old lines 55-56 replaced; new lines 60-65.
    diff = (
        "@@ -54,3 +59,7 @@\n ctx\n-o1\n-o2\n+n1\n+n2\n+n3\n+n4\n+n5\n+n6"
    )
    blk = blocks(diff)[0]
    assert (blk.old_start_line, blk.old_end_line, blk.old_line_count) == (55, 56, 2)
    assert (blk.new_start_line, blk.new_end_line, blk.new_line_count) == (60, 65, 6)
    assert blk.summary == "Replaced old lines 55–56 with new lines 60–65."


def test_block_summaries_are_factual():
    assert block_change_summary("added", None, None, 63, 71) == "Added new lines 63–71."
    assert block_change_summary("deleted", 40, 45, None, None) == "Deleted old lines 40–45."
    assert block_change_summary("added", None, None, 5, 5) == "Added new line 5."


# ---------------------------------------------------------------------------
# preserved edge cases: +++/---, no-newline
# ---------------------------------------------------------------------------
def test_plus_and_minus_prefixed_content_preserved():
    # git emits a real "++value" as "+++value" and "--value" as "---value".
    blk = blocks("@@ -1,1 +1,1 @@\n---value\n+++value")[0]
    assert blk.change_type == "modified"
    assert blk.removed_text == "--value"
    assert blk.added_text == "++value"


def test_no_newline_marker_does_not_split_or_count():
    diff = (
        "@@ -1,1 +1,1 @@\n"
        "-old last line\n"
        "\\ No newline at end of file\n"
        "+new last line\n"
        "\\ No newline at end of file"
    )
    b = blocks(diff)
    assert len(b) == 1  # marker never splits the block
    assert b[0].change_type == "modified"
    assert b[0].removed_text == "old last line"
    assert b[0].added_text == "new last line"
    assert b[0].old_line_count == 1 and b[0].new_line_count == 1  # not advanced


# ---------------------------------------------------------------------------
# truncation: true counts/ranges survive, text_truncated is explicit
# ---------------------------------------------------------------------------
def test_very_long_single_line_truncates_text_not_count():
    long_line = "z" * 500
    blk = blocks(f"@@ -1,1 +1,1 @@\n+{long_line}", max_line_chars=100)[0]
    assert blk.new_line_count == 1  # true count preserved
    assert len(blk.added_text) == 100  # stored text capped
    assert blk.text_truncated is True


def test_total_block_char_budget_drops_lines_but_keeps_counts():
    diff = "@@ -1,5 +1,5 @@\n+aaaa\n+bbbb\n+cccc\n+dddd\n+eeee"
    # Budget only fits the first couple of lines.
    blk = blocks(diff, max_hunk_text_chars=8)[0]
    assert blk.new_line_count == 5  # every added line still counted
    assert blk.new_start_line == 1 and blk.new_end_line == 5  # true range
    assert blk.text_truncated is True
    assert len(blk.added_text.split("\n")) < 5  # some line text omitted


def test_per_side_line_cap_truncates_text_keeps_range():
    diff = "@@ -1,4 +1,4 @@\n+a\n+b\n+c\n+d"
    blk = blocks(diff, max_lines_per_hunk=2)[0]
    assert blk.new_line_count == 4
    assert blk.new_start_line == 1 and blk.new_end_line == 4
    assert blk.text_truncated is True


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def test_grouping_is_deterministic_and_repeatable():
    diff = "@@ -1,6 +1,6 @@\n-a\n+b\n ctx\n+c\n ctx2\n-d"
    first = [(x.block_index, x.change_type, x.added_text, x.removed_text) for x in blocks(diff)]
    second = [(x.block_index, x.change_type, x.added_text, x.removed_text) for x in blocks(diff)]
    assert first == second


# ---------------------------------------------------------------------------
# per-block symbol attribution
# ---------------------------------------------------------------------------
POST = """\
class Engine:
    def start(self):
        run_new()
        return 1


def helper():
    return added_helper()
"""

PRE = """\
class Engine:
    def start(self):
        run_old()
        return 1
"""


def test_symbols_new_side_for_additions_old_side_for_deletions():
    # Line 8 (def helper) region is a pure addition; method start (line 3) is
    # modified.
    diff = (
        "@@ -1,4 +1,9 @@\n"
        " class Engine:\n"
        "     def start(self):\n"
        "-        run_old()\n"
        "+        run_new()\n"
        "         return 1\n"
        "+\n"
        "+\n"
        "+def helper():\n"
        "+    return added_helper()"
    )
    hunk = parse_unified_diff(diff)[0]
    _attach_block_symbols(hunk, extract_python_symbols(POST), extract_python_symbols(PRE))
    modified = next(b for b in hunk.blocks if b.change_type == "modified")
    added = next(b for b in hunk.blocks if b.change_type == "added")
    assert "Engine.start" in modified.symbols  # method dotted name
    assert "helper" in added.symbols


def test_symbols_empty_on_syntax_error_source():
    diff = "@@ -1,1 +1,1 @@\n-a\n+b"
    hunk = parse_unified_diff(diff)[0]
    _attach_block_symbols(
        hunk, extract_python_symbols("def broken(:\n"), extract_python_symbols("x=1")
    )
    assert hunk.blocks[0].symbols == []  # syntax error -> no symbols


def test_symbols_empty_for_non_python():
    diff = "@@ -1,1 +1,1 @@\n-a\n+b"
    hunk = parse_unified_diff(diff)[0]
    # Non-Python files pass empty symbol lists.
    _attach_block_symbols(hunk, [], [])
    assert hunk.blocks[0].symbols == []


# ---------------------------------------------------------------------------
# serialized v3 shape
# ---------------------------------------------------------------------------
def test_serialized_hunk_has_change_blocks_and_no_line_arrays():
    hunk = parse_unified_diff("@@ -1,2 +1,2 @@\n ctx\n-old\n+new")[0]
    _attach_block_symbols(hunk, [], [])
    d = _hunk_to_dict(hunk, "summary", [])
    assert "change_blocks" in d
    assert "added_lines" not in d and "removed_lines" not in d
    block = d["change_blocks"][0]
    assert set(block) == {
        "block_index", "change_type", "old_start_line", "old_line_count",
        "old_end_line", "new_start_line", "new_line_count", "new_end_line",
        "removed_text", "added_text", "summary", "symbols", "text_truncated",
    }
