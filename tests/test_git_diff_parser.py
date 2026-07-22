"""Tests for the pure, offline unified-diff and Python-symbol parser."""

from __future__ import annotations

from src.git_diff_parser import (
    build_hunk_summary,
    enclosing_symbol_objects,
    end_line,
    extract_python_symbols,
    header_symbol,
    hunk_change_type,
    parse_hunk_header,
    parse_unified_diff,
)


# ---------------------------------------------------------------------------
# file headers vs. real +++/--- content inside a hunk
# ---------------------------------------------------------------------------
def test_file_headers_outside_hunk_are_ignored():
    diff = "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n-old\n+new\n"
    hunk = parse_unified_diff(diff)[0]
    assert hunk.added_count == 1 and hunk.removed_count == 1
    assert hunk.added_lines[0].text == "new"
    assert hunk.removed_lines[0].text == "old"


def test_added_content_line_beginning_with_plus_plus_is_retained():
    # In the patch, an added source line "++value" appears as "+++value".
    diff = "@@ -1,1 +1,2 @@\n context\n+++value\n"
    hunk = parse_unified_diff(diff)[0]
    assert hunk.added_count == 1
    assert hunk.added_lines[0].text == "++value"


def test_removed_content_line_beginning_with_minus_minus_is_retained():
    # A removed source line "--value" appears as "---value".
    diff = "@@ -1,2 +1,1 @@\n context\n---value\n"
    hunk = parse_unified_diff(diff)[0]
    assert hunk.removed_count == 1
    assert hunk.removed_lines[0].text == "--value"


def test_line_numbers_correct_after_plus_plus_minus_minus_content():
    diff = (
        "@@ -10,3 +10,3 @@\n"
        " context_10\n"
        "---removed_value\n"   # removed content "--removed_value" at old line 11
        "+++added_value\n"     # added content "++added_value" at new line 11
        " context_12\n"
    )
    hunk = parse_unified_diff(diff)[0]
    assert hunk.removed_lines[0].line_number == 11
    assert hunk.removed_lines[0].text == "--removed_value"
    assert hunk.added_lines[0].line_number == 11
    assert hunk.added_lines[0].text == "++added_value"


# ---------------------------------------------------------------------------
# character-based truncation caps
# ---------------------------------------------------------------------------
def test_single_extremely_long_line_is_char_truncated_and_flagged():
    long_text = "x" * 5000
    hunk = parse_unified_diff(
        f"@@ -0,0 +1,1 @@\n+{long_text}\n", max_line_chars=100
    )[0]
    assert hunk.added_count == 1
    assert len(hunk.added_lines[0].text) == 100
    assert hunk.added_lines[0].text_truncated is True
    assert hunk.hunk_text_truncated is True


def test_total_hunk_chars_cap_stops_storing_but_keeps_counts():
    # Five 100-char lines; a 250-char total budget stores only the first two.
    body = "".join("+" + "a" * 100 + "\n" for _ in range(5))
    hunk = parse_unified_diff(
        "@@ -0,0 +1,5 @@\n" + body, max_hunk_text_chars=250
    )[0]
    assert hunk.added_count == 5            # true count preserved
    assert len(hunk.added_lines) == 2       # only two fit in the budget
    assert hunk.hunk_text_truncated is True


def test_line_count_cap_truncation_still_works():
    body = "".join(f"+line{i}\n" for i in range(10))
    hunk = parse_unified_diff("@@ -0,0 +1,10 @@\n" + body, max_lines_per_hunk=4)[0]
    assert hunk.added_count == 10
    assert len(hunk.added_lines) == 4
    assert hunk.hunk_text_truncated is True


def test_true_counts_correct_under_all_caps():
    body = "".join("+" + "z" * 300 + "\n" for _ in range(6))
    hunk = parse_unified_diff(
        "@@ -0,0 +1,6 @@\n" + body,
        max_lines_per_hunk=100,
        max_line_chars=50,
        max_hunk_text_chars=120,
    )[0]
    assert hunk.added_count == 6                       # metadata count intact
    assert all(cl.text_truncated for cl in hunk.added_lines)
    assert len(hunk.added_lines) <= 3                  # 50-char lines, 120 budget
    assert hunk.hunk_text_truncated is True


def test_truncation_is_deterministic():
    body = "".join("+" + "q" * 400 + "\n" for _ in range(5))
    diff = "@@ -0,0 +1,5 @@\n" + body
    first = parse_unified_diff(diff, max_line_chars=100, max_hunk_text_chars=250)[0]
    second = parse_unified_diff(diff, max_line_chars=100, max_hunk_text_chars=250)[0]
    assert [(cl.line_number, cl.text, cl.text_truncated) for cl in first.added_lines] == [
        (cl.line_number, cl.text, cl.text_truncated) for cl in second.added_lines
    ]
    assert first.added_count == second.added_count == 5


def test_ordinary_hunk_unaffected_by_caps():
    hunk = parse_unified_diff(MODIFIED_DIFF)[0]
    assert hunk.hunk_text_truncated is False
    assert all(not cl.text_truncated for cl in hunk.added_lines)
    assert all(not cl.text_truncated for cl in hunk.removed_lines)


# ---------------------------------------------------------------------------
# hunk header parsing and range maths
# ---------------------------------------------------------------------------
def test_parse_standard_modified_hunk_header():
    assert parse_hunk_header("@@ -40,24 +42,31 @@") == (40, 24, 42, 31, "")


def test_omitted_counts_default_to_one():
    assert parse_hunk_header("@@ -10 +10,3 @@ function_name") == (
        10, 1, 10, 3, "function_name"
    )
    assert parse_hunk_header("@@ -5 +5 @@") == (5, 1, 5, 1, "")


def test_end_line_and_zero_count_rules():
    assert end_line(42, 31) == 72
    assert end_line(1, 20) == 20
    assert end_line(0, 0) is None
    assert end_line(15, 0) is None


def test_hunk_change_type_vocabulary():
    assert hunk_change_type(0, 20) == "added"
    assert hunk_change_type(8, 0) == "deleted"
    assert hunk_change_type(24, 31) == "modified"


def test_added_hunk_zero_old_count():
    hunk = parse_unified_diff("@@ -0,0 +1,2 @@\n+first\n+second\n")[0]
    assert hunk.old_start_line == 0
    assert hunk.old_line_count == 0
    assert hunk.old_end_line is None
    assert hunk.new_start_line == 1
    assert hunk.new_line_count == 2
    assert hunk.new_end_line == 2
    assert hunk.change_type == "added"


def test_deleted_hunk_zero_new_count():
    hunk = parse_unified_diff(
        "@@ -15,3 +0,0 @@\n-a\n-b\n-c\n"
    )[0]
    assert hunk.old_start_line == 15
    assert hunk.old_end_line == 17
    assert hunk.new_start_line == 0
    assert hunk.new_line_count == 0
    assert hunk.new_end_line is None
    assert hunk.change_type == "deleted"


# ---------------------------------------------------------------------------
# hunk body: line-number tracking
# ---------------------------------------------------------------------------
MODIFIED_DIFF = (
    "@@ -10,4 +10,5 @@\n"
    " context_a\n"
    "-old_line_11\n"
    "+new_line_11\n"
    "+new_line_12\n"
    " context_b\n"
    " context_c\n"
)


def test_added_line_numbers_tracked():
    hunk = parse_unified_diff(MODIFIED_DIFF)[0]
    assert [(cl.line_number, cl.text) for cl in hunk.added_lines] == [
        (11, "new_line_11"),
        (12, "new_line_12"),
    ]


def test_removed_line_numbers_tracked():
    hunk = parse_unified_diff(MODIFIED_DIFF)[0]
    assert [(cl.line_number, cl.text) for cl in hunk.removed_lines] == [
        (11, "old_line_11"),
    ]


def test_multiple_hunks_in_one_file():
    diff = (
        "@@ -1,2 +1,3 @@\n context\n+added_here\n context2\n"
        "@@ -20,2 +21,1 @@\n-removed_here\n context3\n"
    )
    hunks = parse_unified_diff(diff)
    assert len(hunks) == 2
    assert hunks[0].added_lines[0].line_number == 2
    assert hunks[1].removed_lines[0].line_number == 20


def test_no_newline_marker_is_ignored():
    diff = (
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "\\ No newline at end of file\n"
        "+new\n"
        "\\ No newline at end of file\n"
    )
    hunk = parse_unified_diff(diff)[0]
    assert hunk.added_count == 1 and hunk.removed_count == 1
    assert hunk.added_lines[0].text == "new"
    assert all("No newline" not in cl.text for cl in hunk.added_lines)


def test_plus_plus_and_minus_minus_headers_not_counted():
    diff = (
        "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n-old\n+new\n"
    )
    hunk = parse_unified_diff(diff)[0]
    assert hunk.added_count == 1 and hunk.removed_count == 1


def test_large_hunk_truncation_is_flagged_not_silent():
    body = "".join(f"+line{i}\n" for i in range(10))
    hunk = parse_unified_diff("@@ -0,0 +1,10 @@\n" + body, max_lines_per_hunk=4)[0]
    assert hunk.added_count == 10           # true count preserved
    assert len(hunk.added_lines) == 4       # stored lines capped
    assert hunk.hunk_text_truncated is True


def test_parsing_is_deterministic():
    first = parse_unified_diff(MODIFIED_DIFF)
    second = parse_unified_diff(MODIFIED_DIFF)
    assert [h.hunk_header for h in first] == [h.hunk_header for h in second]
    assert [(cl.line_number, cl.text) for cl in first[0].added_lines] == [
        (cl.line_number, cl.text) for cl in second[0].added_lines
    ]


# ---------------------------------------------------------------------------
# Python symbol detection
# ---------------------------------------------------------------------------
PY_SOURCE = '''\
import os


def top_level():
    return 1


async def fetch_data():
    return 2


class Widget:
    def create(self):
        return 3

    async def load(self):
        return 4
'''


def test_top_level_function_detection():
    symbols = {s.qualified_name: s.kind for s in extract_python_symbols(PY_SOURCE)}
    assert symbols["top_level"] == "function"


def test_async_function_detection():
    symbols = {s.qualified_name: s.kind for s in extract_python_symbols(PY_SOURCE)}
    assert symbols["fetch_data"] == "async function"


def test_class_and_method_detection():
    symbols = {s.qualified_name: s.kind for s in extract_python_symbols(PY_SOURCE)}
    assert symbols["Widget"] == "class"
    assert symbols["Widget.create"] == "method"
    assert symbols["Widget.load"] == "async function"


def test_enclosing_symbol_is_innermost():
    symbols = extract_python_symbols(PY_SOURCE)
    # line 13 is inside Widget.create
    names = [s.qualified_name for s in enclosing_symbol_objects(symbols, [13])]
    assert names == ["Widget.create"]


def test_syntax_error_returns_empty_without_crashing():
    assert extract_python_symbols("def broken(:\n    pass\n") == []


def test_non_python_returns_no_symbols_via_empty_enclosing():
    # A caller only runs extraction on .py files; empty source -> no symbols.
    assert extract_python_symbols("") == []


# ---------------------------------------------------------------------------
# deterministic hunk summaries
# ---------------------------------------------------------------------------
def test_summary_added_with_symbol():
    hunk = parse_unified_diff("@@ -0,0 +1,20 @@\n" + "+x\n" * 20)[0]
    symbols = extract_python_symbols(
        "def parse_diff_hunks():\n" + "    pass\n" * 25
    )
    primary = enclosing_symbol_objects(symbols, [1])[0]
    assert build_hunk_summary(hunk, primary) == (
        "Added 20 lines in function parse_diff_hunks."
    )


def test_summary_deleted_with_class_symbol():
    hunk = parse_unified_diff("@@ -15,8 +0,0 @@\n" + "-y\n" * 8)[0]
    symbols = extract_python_symbols(
        "class LegacyParser:\n" + "    x = 1\n" * 30
    )
    primary = enclosing_symbol_objects(symbols, [16])[0]
    assert build_hunk_summary(hunk, primary) == (
        "Deleted 8 lines from class LegacyParser."
    )


def test_summary_modified_with_and_without_symbol():
    hunk = parse_unified_diff(MODIFIED_DIFF)[0]  # 2 added, 1 removed
    assert build_hunk_summary(hunk) == "Modified 2 added and 1 removed lines."

    header = header_symbol("def create_change_package(self):")
    assert build_hunk_summary(hunk, None, header) == (
        "Modified function create_change_package with 2 added and 1 removed lines."
    )


def test_header_symbol_fallback_parsing():
    assert header_symbol("def foo(a, b):") == ("function", "foo")
    assert header_symbol("async def bar():") == ("async function", "bar")
    assert header_symbol("class Baz:") == ("class", "Baz")
    assert header_symbol("eee") is None
    assert header_symbol("") is None
