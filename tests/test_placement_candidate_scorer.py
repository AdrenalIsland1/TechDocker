"""Tests for deterministic placement scoring inside one selected section.

Indexes are built with the real ``summary_index_builder`` rather than
hand-written dicts, so offsets/ids/hashes are genuine. Fully offline: no LLM
provider, no network, no writes to real artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.placement_candidate_scorer import (
    APPEND_TO_SECTION,
    MIN_CANDIDATE_SCORE,
    SPECIFIC_COMPONENTS,
    USE_EXISTING_CANDIDATE,
    PlacementIndexError,
    extract_placement_signals,
    main,
    score_placement_candidates,
)
from src.summary_index_builder import build_summary_index

SECTION = "repository-automation"


def index_for(body: str, heading: str = "Repository Automation", **kwargs):
    """Build a real index for a one-section document."""
    markdown = f"# {heading}\n\n{body}\n"
    return build_summary_index(markdown, **kwargs), markdown


def section_id_of(index, heading_fragment="repository"):
    for section in index["sections"]:
        if heading_fragment in section["section_id"]:
            return section["section_id"]
    return index["sections"][0]["section_id"]


def package(
    path="src/summary_change_router.py",
    *,
    status="modified",
    symbols=(),
    added=(),
    removed=(),
    summary_text="1 file changed",
    hunk_summary="Modified function route_change.",
    old_path=None,
    binary=False,
    schema_version=2,
):
    hunks = []
    if not binary:
        hunks = [
            {
                "hunk_header": "@@ -10,5 +10,7 @@",
                "summary": hunk_summary,
                "symbols": list(symbols),
                "added_lines": [
                    {"line_number": i + 1, "text": text}
                    for i, text in enumerate(added)
                ],
                "removed_lines": [
                    {"line_number": i + 1, "text": text}
                    for i, text in enumerate(removed)
                ],
            }
        ]
    return {
        "schema_version": schema_version,
        "generated_summary": summary_text,
        "changed_files": [
            {
                "path": path,
                "old_path": old_path,
                "status": status,
                "binary": binary,
                "what_changed": hunks,
            }
        ],
    }


def score(body, pkg, **kwargs):
    index, markdown = index_for(body)
    return score_placement_candidates(
        pkg, index, section_id_of(index), source_markdown=markdown, **kwargs
    )


# ---------------------------------------------------------------------------
# direct evidence
# ---------------------------------------------------------------------------
def test_exact_changed_function_matches_one_sentence():
    body = (
        "The pipeline is orchestrated end to end. "
        "Routing is performed by route_change for every push. "
        "Reports are rendered afterwards."
    )
    assessment = score(body, package(symbols=["route_change"]))
    top = assessment.top
    assert top.candidate_type == "sentence"
    assert "route_change" in top.text
    assert top.score_breakdown["symbol_match"] == 16.0
    assert "route_change" in top.matched_signals


def test_exact_changed_class_matches_a_paragraph():
    body = (
        "The ReportBuilder assembles output.\n\n"
        "Unrelated deployment notes live here."
    )
    assessment = score(body, package(symbols=["ReportBuilder"]))
    assert "ReportBuilder" in assessment.top.text
    assert assessment.top.score_breakdown["symbol_match"] == 16.0


def test_module_stem_matches_existing_prose():
    body = (
        "Placement is handled by summary_change_router internals.\n\n"
        "Unrelated notes about deployment."
    )
    assessment = score(body, package())
    assert "summary_change_router" in assessment.top.text
    assert assessment.top.score_breakdown["module_match"] == 14.0


def test_exact_path_mention_matches_existing_prose():
    body = (
        "Routing lives in src/summary_change_router.py today.\n\n"
        "Unrelated notes about deployment."
    )
    assessment = score(body, package())
    assert assessment.top.score_breakdown["path_match"] == 14.0
    # The stem is not paid twice when the full path already matched.
    assert "module_match" not in assessment.top.score_breakdown


def test_removed_line_terms_identify_stale_text():
    body = (
        "The legacy heuristic uses a hardcoded threshold constant.\n\n"
        "Deployment happens through the workflow."
    )
    pkg = package(removed=["hardcoded threshold constant removed"], symbols=[])
    assessment = score(body, pkg)
    assert "hardcoded" in assessment.top.text
    assert assessment.top.score_breakdown["removed_evidence"] > 0


def test_added_line_terms_identify_extension_location():
    body = (
        "Scoring explains every decision with a breakdown.\n\n"
        "Totally unrelated prose about colours."
    )
    pkg = package(added=["breakdown scoring explains"], symbols=[])
    assessment = score(body, pkg)
    assert "breakdown" in assessment.top.text
    assert assessment.top.score_breakdown["added_evidence"] > 0


def test_hunk_summary_contributes():
    body = "Validation of fallback behaviour is documented here.\n\nOther prose."
    with_hunk = score(body, package(hunk_summary="Modified fallback validation.", symbols=[]))
    without = score(body, package(hunk_summary="", symbols=[]))
    assert with_hunk.top.score > without.top.score


def test_generated_summary_contributes_at_lower_weight():
    body = "Deployment scheduling notes live here.\n\nUnrelated colours prose."
    from src.placement_candidate_scorer import PLACEMENT_WEIGHTS

    assessment = score(
        body,
        package(summary_text="scheduling deployment", symbols=[], hunk_summary=""),
    )
    assert PLACEMENT_WEIGHTS["summary_overlap"] < PLACEMENT_WEIGHTS["symbol_match"]
    assert assessment.top.score_breakdown.get("summary_overlap", 0) > 0


def test_repeated_tokens_do_not_inflate_scores():
    repeated = "fallback " * 60
    body = f"The {repeated} behaviour is documented.\n\nOther prose."
    pkg = package(removed=["fallback"] * 40, symbols=[], hunk_summary="fallback")
    assessment = score(body, pkg)
    from src.placement_candidate_scorer import PLACEMENT_CAPS, PLACEMENT_WEIGHTS

    cap = PLACEMENT_CAPS["removed_evidence"] * PLACEMENT_WEIGHTS["removed_evidence"]
    assert assessment.top.score_breakdown.get("removed_evidence", 0) <= cap


def test_score_breakdown_sums_to_total():
    assessment = score(
        "Routing is performed by route_change in src/summary_change_router.py.",
        package(symbols=["route_change"]),
    )
    for candidate in assessment.candidates:
        assert candidate.score == pytest.approx(sum(candidate.score_breakdown.values()))


def test_symbol_does_not_match_inside_longer_word():
    body = "The router component is documented here.\n\nOther prose entirely."
    assessment = score(body, package(symbols=["route"], hunk_summary="", summary_text=""))
    # "route" must not match inside "router".
    assert assessment.top.score_breakdown.get("symbol_match", 0) == 0


# ---------------------------------------------------------------------------
# granularity and diversification
# ---------------------------------------------------------------------------
def test_concentrated_evidence_chooses_sentence():
    body = (
        "General introduction prose. "
        "Dispatch is handled by route_change during a push. "
        "Closing remarks about reporting."
    )
    assessment = score(body, package(symbols=["route_change"]))
    assert assessment.top.candidate_type == "sentence"
    assert "one sentence" in assessment.top.granularity_reason


def test_distributed_evidence_chooses_block():
    body = (
        "The route_change entry point is documented. "
        "Later the summary_change_router module is described in detail."
    )
    assessment = score(body, package(symbols=["route_change"]))
    assert assessment.top.candidate_type == "block"
    assert "distributed" in assessment.top.granularity_reason


def test_shortlist_is_diversified_across_blocks():
    body = (
        "route_change dispatches the update.\n\n"
        "route_change is also mentioned here.\n\n"
        "And route_change appears in this third block too."
    )
    assessment = score(body, package(symbols=["route_change"]))
    block_ids = [c.block_id for c in assessment.candidates]
    assert len(block_ids) == len(set(block_ids))  # one per block


def test_block_and_its_child_sentence_never_share_the_shortlist():
    body = (
        "First sentence names route_change. Second sentence names route_change too.\n\n"
        "An unrelated paragraph about colours."
    )
    assessment = score(body, package(symbols=["route_change"]))
    for candidate in assessment.candidates:
        siblings = [
            other for other in assessment.candidates
            if other is not candidate and other.block_id == candidate.block_id
        ]
        assert siblings == []


def test_at_most_one_sentence_per_block():
    body = (
        "Alpha mentions route_change. Beta mentions route_change. "
        "Gamma mentions route_change."
    )
    assessment = score(body, package(symbols=["route_change"]))
    sentence_blocks = [
        c.block_id for c in assessment.candidates if c.candidate_type == "sentence"
    ]
    assert len(sentence_blocks) == len(set(sentence_blocks))


# ---------------------------------------------------------------------------
# block kinds
# ---------------------------------------------------------------------------
def test_unordered_list_item_candidate():
    body = "- Routing is performed by route_change for each push.\n- Unrelated item."
    assessment = score(body, package(symbols=["route_change"]))
    assert assessment.top.block_type == "unordered_list_item"
    assert "route_change" in assessment.top.text


def test_ordered_list_item_candidate():
    body = "1. Routing is performed by route_change for each push.\n2. Unrelated item."
    assessment = score(body, package(symbols=["route_change"]))
    assert assessment.top.block_type == "ordered_list_item"


def test_multi_sentence_paragraph_offsets_are_exact():
    body = (
        "Opening statement here. "
        "Dispatch is handled by route_change today. "
        "Closing statement here."
    )
    index, markdown = index_for(body)
    assessment = score_placement_candidates(
        package(symbols=["route_change"]),
        index,
        section_id_of(index),
        source_markdown=markdown,
    )
    for candidate in assessment.candidates:
        start, end = candidate.source_start_offset, candidate.source_end_offset
        assert markdown[start:end] == candidate.text


# ---------------------------------------------------------------------------
# context bounding
# ---------------------------------------------------------------------------
def test_previous_and_next_context_is_bounded_and_correct():
    from src.placement_candidate_scorer import (
        MAX_CONTEXT_EXCERPT_CHARS,
        MAX_PARENT_TEXT_CHARS,
    )

    filler = "Filler prose about unrelated topics. " * 30
    body = f"{filler}\n\nDispatch uses route_change now.\n\n{filler}"
    assessment = score(body, package(symbols=["route_change"]))
    context = assessment.top.context

    assert context["section_heading"] == "Repository Automation"
    assert len(context["previous_excerpt"]) <= MAX_CONTEXT_EXCERPT_CHARS
    assert len(context["next_excerpt"]) <= MAX_CONTEXT_EXCERPT_CHARS
    assert context["previous_block_id"] and context["next_block_id"]
    if assessment.top.candidate_type == "sentence":
        assert len(context["parent_block_text"]) <= MAX_PARENT_TEXT_CHARS


def test_only_the_selected_section_is_scored():
    markdown = (
        "# Doc\n\n## Repository Automation\n\nDispatch uses route_change.\n\n"
        "## Other Area\n\nAlso mentions route_change prominently.\n"
    )
    index = build_summary_index(markdown)
    target = next(
        s["section_id"] for s in index["sections"] if "repository" in s["section_id"]
    )
    assessment = score_placement_candidates(
        package(symbols=["route_change"]), index, target, source_markdown=markdown
    )
    assert assessment.candidates
    assert all(c.section_id == target for c in assessment.candidates)
    assert all("Also mentions" not in c.text for c in assessment.candidates)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def test_repeated_scoring_is_byte_stable():
    body = "Dispatch uses route_change.\n\nAnother paragraph mentions routing."
    pkg = package(symbols=["route_change"])
    first = score(body, pkg).to_dict()
    second = score(body, pkg).to_dict()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_ties_break_by_block_order():
    body = "Shared wording exactly.\n\nShared wording exactly."
    pkg = package(symbols=[], hunk_summary="shared wording", summary_text="")
    assessment = score(body, pkg)
    assert assessment.candidates[0].score == assessment.candidates[1].score
    assert assessment.candidates[0].block_order < assessment.candidates[1].block_order


# ---------------------------------------------------------------------------
# excluded content
# ---------------------------------------------------------------------------
EXCLUDED_BODIES = {
    "code_block": "```python\ndef route_change():\n    return 1\n```",
    "table": "| module | symbol |\n| --- | --- |\n| src/x.py | route_change |",
    "html_comment": "<!-- route_change lives in src/summary_change_router.py -->",
    "generated": (
        "<!-- TECHDOCKER_UPDATE_START -->\n"
        "Changed: route_change in src/summary_change_router.py\n"
        "<!-- TECHDOCKER_UPDATE_END -->"
    ),
}


@pytest.mark.parametrize("kind,body", sorted(EXCLUDED_BODIES.items()))
def test_excluded_content_never_produces_candidates(kind, body):
    assessment = score(body, package(symbols=["route_change"]))
    # Even though the exact symbol/path is present, nothing is scorable.
    assert assessment.recommendation == APPEND_TO_SECTION
    assert all(c.score == 0 for c in assessment.candidates)
    assert not any("route_change" in c.text for c in assessment.candidates)


def test_exact_symbol_in_excluded_content_cannot_outrank_prose():
    body = (
        "```text\nroute_change route_change route_change\n```\n\n"
        "A short prose line mentioning route_change once."
    )
    assessment = score(body, package(symbols=["route_change"]))
    assert assessment.top.block_type == "paragraph"
    assert "```" not in assessment.top.text


def test_non_patchable_blocks_are_excluded():
    index, markdown = index_for("```\nroute_change\n```")
    section = index["sections"][0]
    assert all(not b["patchable"] for b in section["blocks"])
    assessment = score_placement_candidates(
        package(symbols=["route_change"]), index, section["section_id"],
        source_markdown=markdown,
    )
    assert assessment.recommendation == APPEND_TO_SECTION


# ---------------------------------------------------------------------------
# change-package shapes
# ---------------------------------------------------------------------------
def test_schema_v2_package_with_hunks():
    body = "Dispatch is handled by route_change today.\n\nOther prose."
    assessment = score(body, package(symbols=["route_change"]))
    assert assessment.recommendation == USE_EXISTING_CANDIDATE


def test_v1_style_package_without_hunks():
    body = "Routing lives in src/summary_change_router.py.\n\nOther prose."
    v1 = {
        "generated_summary": "1 file changed",
        "changed_files": [
            {"path": "src/summary_change_router.py", "change_type": "modified",
             "old_path": None}
        ],
    }
    assessment = score(body, v1)
    assert assessment.top.score_breakdown["path_match"] == 14.0


def test_added_file_uses_path_evidence():
    body = "Reporting is produced by pr_summary_report helpers.\n\nOther prose."
    assessment = score(
        body, package(path="src/pr_summary_report.py", status="added", symbols=[])
    )
    assert assessment.top.score_breakdown["module_match"] == 14.0


def test_deleted_file_uses_removed_evidence():
    body = "The obsolete demo updater wrote DOCX output.\n\nUnrelated prose."
    pkg = package(
        path="src/demo_docx_updater.py",
        status="deleted",
        symbols=[],
        removed=["obsolete demo updater DOCX"],
    )
    assessment = score(body, pkg)
    assert assessment.top.score_breakdown["removed_evidence"] > 0


def test_renamed_file_uses_old_and_new_paths():
    body = "Legacy behaviour is described in old_router prose.\n\nOther prose."
    pkg = package(path="src/new_router.py", old_path="src/old_router.py",
                  status="renamed", symbols=[])
    signals = extract_placement_signals(pkg)
    assert "src/old_router.py" in signals.paths
    assert "old_router" in signals.modules
    assessment = score(body, pkg)
    assert assessment.top.score_breakdown.get("module_match", 0) > 0


def test_binary_file_falls_back_to_path_evidence():
    body = "Branding assets live in the logo file.\n\nUnrelated prose."
    pkg = package(path="assets/logo.png", binary=True, symbols=[])
    signals = extract_placement_signals(pkg)
    assert "logo" in signals.modules
    assessment = score(body, pkg)  # must not raise
    assert assessment.section_id


def test_large_change_evidence_stays_bounded():
    pkg = {
        "schema_version": 2,
        "generated_summary": "many files changed",
        "changed_files": [
            {
                "path": f"src/module_{i:03d}.py",
                "status": "modified",
                "what_changed": [
                    {
                        "summary": f"Modified function handler_{i}.",
                        "symbols": [f"handler_{i}"],
                        "added_lines": [
                            {"line_number": n, "text": f"token_{i}_{n} value"}
                            for n in range(50)
                        ],
                        "removed_lines": [],
                    }
                    for _ in range(20)
                ],
            }
            for i in range(60)
        ],
    }
    signals = extract_placement_signals(pkg)
    assert len(signals.paths) == 60          # all files aggregated
    assert len(signals.added_tokens) < 5000  # diff text stays bounded
    first = extract_placement_signals(pkg).added_tokens
    assert first == signals.added_tokens     # deterministic


# ---------------------------------------------------------------------------
# recommendations, confidence, safety
# ---------------------------------------------------------------------------
def test_strong_result_is_confident_and_unambiguous():
    body = (
        "Dispatch is handled by route_change in src/summary_change_router.py.\n\n"
        "Completely unrelated prose about colours and shapes."
    )
    assessment = score(body, package(symbols=["route_change"]))
    assert assessment.recommendation == USE_EXISTING_CANDIDATE
    assert assessment.ambiguous is False
    assert assessment.confidence > 0.5


def test_close_results_are_marked_ambiguous():
    body = "Dispatch uses route_change.\n\nDispatch uses route_change."
    assessment = score(body, package(symbols=["route_change"]))
    assert assessment.ambiguous is True
    assert "review" in assessment.reasoning.lower()


def test_low_evidence_recommends_append_not_no_change():
    body = "Entirely unrelated prose about colours.\n\nMore unrelated prose."
    assessment = score(
        body, package(symbols=["zzz_unrelated_symbol"], hunk_summary="", summary_text="")
    )
    assert assessment.recommendation == APPEND_TO_SECTION
    assert assessment.recommendation != "no_change_needed"
    assert assessment.top.score < MIN_CANDIDATE_SCORE


def test_empty_section_recommends_append():
    index = build_summary_index("# Doc\n\n## Empty Area\n\n")
    section_id = index["sections"][-1]["section_id"]
    assessment = score_placement_candidates(package(), index, section_id)
    assert assessment.recommendation == APPEND_TO_SECTION
    assert assessment.candidates == []


def test_stale_index_is_rejected():
    index, markdown = index_for("Dispatch uses route_change.")
    with pytest.raises(PlacementIndexError, match="stale"):
        score_placement_candidates(
            package(symbols=["route_change"]),
            index,
            section_id_of(index),
            source_markdown=markdown + "\nAn edit that invalidates the hash.\n",
        )


def test_unsupported_schema_is_rejected():
    index, markdown = index_for("Dispatch uses route_change.")
    broken = dict(index, schema_version=999)
    with pytest.raises(PlacementIndexError, match="Unsupported"):
        score_placement_candidates(package(), broken, section_id_of(index))


def test_missing_section_is_rejected_not_substituted():
    index, _ = index_for("Dispatch uses route_change.")
    with pytest.raises(PlacementIndexError, match="not present"):
        score_placement_candidates(package(), index, "no-such-section-id")


def test_malformed_index_is_rejected():
    with pytest.raises(PlacementIndexError, match="sections"):
        score_placement_candidates(package(), {"schema_version": 1}, "x")


# ---------------------------------------------------------------------------
# offline guarantees / no writes
# ---------------------------------------------------------------------------
def test_module_imports_no_llm_or_network():
    source = Path("src/placement_candidate_scorer.py").read_text(encoding="utf-8")
    for forbidden in (
        "llm_provider", "llm_change_analyzer", "ollama", "urllib",
        "requests", "socket", "http.client",
    ):
        assert forbidden not in source, forbidden


def test_scoring_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    body = "Dispatch uses route_change.\n\nOther prose."
    score(body, package(symbols=["route_change"]))
    assert list(tmp_path.iterdir()) == []


def test_cli_is_read_only(tmp_path, capsys):
    index, markdown = index_for("Dispatch uses route_change.\n\nOther prose.")
    index_file = tmp_path / "index.json"
    package_file = tmp_path / "change.json"
    source_file = tmp_path / "summary.md"
    index_file.write_text(json.dumps(index), encoding="utf-8")
    package_file.write_text(json.dumps(package(symbols=["route_change"])), encoding="utf-8")
    source_file.write_text(markdown, encoding="utf-8")
    before = {p: p.read_bytes() for p in (index_file, package_file, source_file)}

    code = main(
        [
            "--change-package", str(package_file),
            "--index", str(index_file),
            "--section-id", section_id_of(index),
            "--source", str(source_file),
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    payload = json.loads(captured.out)
    assert payload["recommendation"] == USE_EXISTING_CANDIDATE
    assert "no files were written" in captured.err
    for path, content in before.items():
        assert path.read_bytes() == content


def test_cli_reports_stale_index_clearly(tmp_path, capsys):
    index, markdown = index_for("Dispatch uses route_change.")
    index_file = tmp_path / "index.json"
    package_file = tmp_path / "change.json"
    source_file = tmp_path / "summary.md"
    index_file.write_text(json.dumps(index), encoding="utf-8")
    package_file.write_text(json.dumps(package()), encoding="utf-8")
    source_file.write_text(markdown + "\nedited\n", encoding="utf-8")

    code = main(
        [
            "--change-package", str(package_file),
            "--index", str(index_file),
            "--section-id", section_id_of(index),
            "--source", str(source_file),
        ]
    )
    assert code == 3
    assert "stale" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# structural-inventory safety (correction 1)
# ---------------------------------------------------------------------------
from src.summary_index_builder import is_structural_inventory  # noqa: E402


@pytest.mark.parametrize(
    "line",
    [
        "- `src/git_change_detector.py`",
        "- src/git_change_detector.py",
        "1. `src/module.py`",
        "2) src/module.py",
        "- `tests/test_summary_pipeline.py`",
        "- `requirements.txt`",
        "* `pyproject.toml`",
        "`src/summary_updater.py`",
    ],
)
def test_bare_inventory_entries_are_detected(line):
    assert is_structural_inventory(line) is True


@pytest.mark.parametrize(
    "line",
    [
        "- `src/git_change_detector.py`: Extracts changed files and Git diff metadata.",
        "- The router uses `src/summary_change_router.py` to score real sections.",
        "- `pytest` runs the complete offline test suite.",
        "Routing lives in src/summary_change_router.py today.",
        "- Section routing scores the actual sections of the summary.",
    ],
)
def test_explanatory_items_are_not_inventory(line):
    assert is_structural_inventory(line) is False


def test_new_index_marks_inventory_non_patchable():
    index, _ = index_for("- `src/git_change_detector.py`\n- `src/summary_updater.py`")
    blocks = index["sections"][0]["blocks"]
    assert blocks
    for block in blocks:
        assert block["structural_inventory"] is True
        assert block["patchable"] is False


def test_inventory_metadata_does_not_change_ids_or_hashes():
    body = "- `src/git_change_detector.py`\n\nExplanatory prose about routing."
    index, _ = index_for(body)
    blocks = index["sections"][0]["blocks"]
    import hashlib

    for block in blocks:
        # ids/hashes derive from text alone, unaffected by the new metadata.
        assert block["content_hash"] == hashlib.sha256(
            block["text"].encode()
        ).hexdigest()
        assert block["content_hash"][:8] in block["block_id"]


def test_inventory_blocks_are_excluded_from_candidates():
    body = (
        "- `src/summary_change_router.py`\n"
        "- `src/git_change_detector.py`\n\n"
        "Routing is performed by route_change during each push."
    )
    assessment = score(body, package(symbols=["route_change"]))
    for candidate in assessment.candidates:
        assert not is_structural_inventory(candidate.text)
    assert "route_change" in assessment.top.text


def test_older_index_without_the_field_is_defensively_filtered():
    body = "- `src/summary_change_router.py`\n- `src/git_change_detector.py`"
    index, markdown = index_for(body)
    # Simulate a pre-existing index: drop the field and force patchable.
    for section in index["sections"]:
        for block in section["blocks"]:
            block.pop("structural_inventory", None)
            block["patchable"] = True

    assessment = score_placement_candidates(
        package(), index, section_id_of(index), source_markdown=markdown
    )
    assert assessment.recommendation == APPEND_TO_SECTION
    assert all(c.score == 0 for c in assessment.candidates)


def test_real_style_bare_core_modules_list_recommends_append():
    # Mirrors the real base_updated_summary.md "Core Modules" section.
    body = "\n".join(
        f"- `src/{name}.py`"
        for name in ("automation_demo", "change_summary_generator", "git_change_detector")
    )
    assessment = score(body, package(symbols=[], hunk_summary="", summary_text="1 file changed"))
    assert assessment.recommendation == APPEND_TO_SECTION
    assert assessment.candidates == [] or all(c.score == 0 for c in assessment.candidates)


# ---------------------------------------------------------------------------
# weak evidence must not authorize replacement (correction 2)
# ---------------------------------------------------------------------------
def test_category_only_score_recommends_append():
    # Topical agreement only: no symbol, module, path, or diff-line evidence.
    # (The file stem must not appear in the prose, or that would be a
    # legitimate specific `module_match`.)
    body = "Deployment and workflow scheduling notes are documented here."
    pkg = package(
        path=".github/workflows/zz_pipeline_conf.yml", symbols=[],
        hunk_summary="", summary_text="",
    )
    assessment = score(body, pkg)
    assert assessment.top.score > 0
    assert set(assessment.top.score_breakdown) <= {"category_overlap"}
    assert assessment.recommendation == APPEND_TO_SECTION


def test_broad_only_score_above_the_minimum_still_recommends_append():
    # Category + generated-summary overlap clears MIN_CANDIDATE_SCORE, but
    # neither ties the change to *this* text, so replacement is not authorized.
    body = "Deployment and workflow scheduling notes are documented here."
    pkg = package(
        path=".github/workflows/zz_pipeline_conf.yml", symbols=[],
        hunk_summary="", summary_text="deployment workflow scheduling",
    )
    assessment = score(body, pkg)
    broad = {"category_overlap", "summary_overlap", "keyword_overlap"}

    assert assessment.top.score > MIN_CANDIDATE_SCORE
    assert set(assessment.top.score_breakdown) <= broad
    assert not set(assessment.top.score_breakdown) & SPECIFIC_COMPONENTS
    assert assessment.recommendation == APPEND_TO_SECTION
    assert "No safe existing target" in assessment.reasoning


def test_exact_symbol_permits_existing_candidate():
    body = "Dispatch is handled by route_change during each push of the pipeline."
    assessment = score(body, package(symbols=["route_change"]))
    assert assessment.recommendation == USE_EXISTING_CANDIDATE
    assert "symbol_match" in assessment.top.score_breakdown


def test_removed_line_overlap_permits_existing_candidate():
    body = "The legacy heuristic uses a hardcoded threshold constant for scoring."
    pkg = package(
        path="notes/unrelated.txt", symbols=[],
        removed=["hardcoded threshold constant"], hunk_summary="",
    )
    assessment = score(body, pkg)
    assert assessment.recommendation == USE_EXISTING_CANDIDATE
    assert "removed_evidence" in assessment.top.score_breakdown


def test_specific_candidate_outranks_broad_leader():
    body = (
        "Deployment workflow scheduling and coordination notes are here.\n\n"
        "route_change dispatches updates."
    )
    assessment = score(
        body,
        package(path=".github/workflows/zz_pipeline_conf.yml",
                symbols=["route_change"], hunk_summary="", summary_text=""),
    )
    assert assessment.recommendation == USE_EXISTING_CANDIDATE
    # The specific (symbol) candidate leads, not the broad category-only one.
    assert set(assessment.top.score_breakdown) & SPECIFIC_COMPONENTS
    assert "route_change" in assessment.top.text


# ---------------------------------------------------------------------------
# matched signals (correction 4)
# ---------------------------------------------------------------------------
def test_nonzero_scored_candidates_expose_matched_signals():
    body = (
        "Dispatch is handled by route_change during each push.\n\n"
        "Deployment scheduling notes live here."
    )
    assessment = score(body, package(symbols=["route_change"]))
    for candidate in assessment.candidates:
        if candidate.score > 0:
            assert candidate.matched_signals, candidate.text


def test_matched_signals_are_bounded_and_deterministic():
    from src.placement_candidate_scorer import MAX_MATCHED_SIGNALS

    tokens = " ".join(f"token{i}" for i in range(60))
    body = f"Routing uses route_change with {tokens} in the pipeline."
    pkg = package(symbols=["route_change"], added=[tokens], removed=[tokens])
    first = score(body, pkg)
    second = score(body, pkg)
    assert first.top.matched_signals == second.top.matched_signals
    assert len(first.top.matched_signals) <= MAX_MATCHED_SIGNALS
