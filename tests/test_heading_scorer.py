"""Unit tests for the pure scoring logic in :mod:`src.heading_scorer`."""

from __future__ import annotations

from src import heading_scorer
from src.docx_feature_extractor import ParagraphFeatures, color_is_colored


def make_features(text: str = "Sample heading", **overrides) -> ParagraphFeatures:
    """Build a ParagraphFeatures with body-like defaults, then apply overrides.

    Text-derived fields are computed from ``text`` so tests stay realistic;
    formatting flags default to ordinary body text.
    """
    base = dict(
        position=1,
        style_name="Normal",
        word_count=len(text.split()),
        sentence_count=heading_scorer.count_sentences(text),
        numbering_prefix=heading_scorer.extract_numbering_prefix(text),
        is_title_case=heading_scorer.is_title_case(text),
        is_all_caps=heading_scorer.is_all_caps(text),
        ends_with_full_stop=heading_scorer.has_full_stop_ending(text),
        ends_with_colon=heading_scorer.has_colon_ending(text),
        is_word_list_item=False,
        is_bold=False,
        is_italic=False,
        is_underlined=False,
        font_size=11.0,
        font_family="Calibri",
        font_color=None,
        is_colored=False,
        alignment=None,
        spacing_before=None,
        spacing_after=None,
        left_indent=None,
        first_line_indent=None,
        is_official_heading=False,
        official_heading_level=None,
        is_title_style=False,
        body_font_size=11.0,
        body_font_family="Calibri",
        body_font_color=None,
        repeated_formatting_pattern=False,
    )
    base.update(overrides)
    return ParagraphFeatures(text=text, **base)


# --- numbering / sentence / colon text helpers -----------------------------
def test_numbering_prefix_extraction():
    assert heading_scorer.extract_numbering_prefix("1 Introduction") == "1"
    assert heading_scorer.extract_numbering_prefix("1. Introduction") == "1"
    assert heading_scorer.extract_numbering_prefix("1) Introduction") == "1"
    assert heading_scorer.extract_numbering_prefix("1: Introduction") == "1"
    assert heading_scorer.extract_numbering_prefix("1 - Introduction") == "1"
    assert heading_scorer.extract_numbering_prefix("1.2 API") == "1.2"
    assert heading_scorer.extract_numbering_prefix("3.4.1 Authentication") == "3.4.1"
    assert heading_scorer.extract_numbering_prefix("2.1: API Configuration") == "2.1"


def test_numbering_prefix_ignores_mid_text_numbers():
    assert heading_scorer.extract_numbering_prefix("See section 2 for details") is None
    assert heading_scorer.extract_numbering_prefix("The API returns 200 OK") is None


def test_numbering_prefix_ignores_years():
    # Four-digit numbers are not section numbering.
    assert heading_scorer.extract_numbering_prefix("1999 was a good year") is None


def test_sentence_counting_handles_decimals():
    assert heading_scorer.count_sentences("2.1 API Configuration") == 1
    assert heading_scorer.count_sentences("First sentence. Second sentence.") == 2
    assert heading_scorer.count_sentences("A single clause") == 1


def test_colon_endings():
    assert heading_scorer.has_colon_ending("Overview:")
    assert heading_scorer.has_colon_ending("Overview:-")
    assert heading_scorer.has_colon_ending("Overview:–")
    assert heading_scorer.has_colon_ending("Overview:—")
    assert not heading_scorer.has_colon_ending("Overview")


# --- official + hard-negative priorities -----------------------------------
def test_official_heading_scores_100():
    features = make_features(
        "Architecture", is_official_heading=True, official_heading_level=1
    )
    result = heading_scorer.score_paragraph(features)
    assert result.score == 100
    assert result.classification == "heading"
    assert result.detection_method == "official_word_heading_style"
    assert result.predicted_level == 1


def test_official_heading_level_two_preserved():
    features = make_features(
        "Sub Section", is_official_heading=True, official_heading_level=2
    )
    result = heading_scorer.score_paragraph(features)
    assert result.predicted_level == 2


def test_note_colon_forces_normal_content():
    result = heading_scorer.score_paragraph(
        make_features("Note: This endpoint is deprecated.")
    )
    assert result.score == 0
    assert result.classification == "normal_content"
    assert result.detection_method == "hard_negative_rule"
    assert "hard_negative_prefix:note" in result.signals


def test_note_colon_hyphen_forces_normal_content():
    result = heading_scorer.score_paragraph(
        make_features("Note:- Restart the service after installation.")
    )
    assert result.score == 0
    assert result.classification == "normal_content"


def test_note_space_dash_forces_normal_content():
    result = heading_scorer.score_paragraph(
        make_features("Note - Additional setup is required.")
    )
    assert result.score == 0
    assert "hard_negative_prefix:note" in result.signals


def test_link_colon_forces_normal_content():
    result = heading_scorer.score_paragraph(make_features("Link: https://example.com"))
    assert result.score == 0
    assert "hard_negative_prefix:link" in result.signals


def test_link_colon_hyphen_forces_normal_content():
    result = heading_scorer.score_paragraph(make_features("Link:- Internal repository"))
    assert result.score == 0


def test_hard_negative_is_case_insensitive():
    assert heading_scorer.hard_negative_prefix("note: lower case") == "note"
    assert heading_scorer.hard_negative_prefix("LINK - upper case") == "link"
    assert heading_scorer.hard_negative_prefix("Note:- mixed") == "note"


def test_link_in_middle_of_sentence_is_not_hard_negative():
    assert heading_scorer.hard_negative_prefix(
        "This paragraph contains a link to another service."
    ) is None
    result = heading_scorer.score_paragraph(
        make_features("This paragraph contains a link to another service.")
    )
    assert result.detection_method == "formatting_heuristic"


def test_notebook_prefix_is_not_hard_negative():
    # "Notebook" starts with "note" but is not a Note line.
    assert heading_scorer.hard_negative_prefix("Notebook configuration guide") is None


def test_hard_negative_overrides_positive_signals_and_combination():
    # Bold + colon ending would normally score highly; the Note rule wins.
    features = make_features("Note: Critical Section:", is_bold=True)
    result = heading_scorer.score_paragraph(features)
    assert result.score == 0
    assert result.classification == "normal_content"


# --- individual scoring rules ----------------------------------------------
def test_numbering_prefix_adds_25():
    result = heading_scorer.score_paragraph(make_features("3.4.1 Authentication"))
    assert "numbering_prefix:+25" in result.signals
    assert result.predicted_level == 3


def test_more_than_15_words_subtracts_20():
    text = " ".join(["word"] * 20)
    result = heading_scorer.score_paragraph(make_features(text))
    assert "many_words:-20" in result.signals


def test_multiple_sentences_subtracts_20():
    result = heading_scorer.score_paragraph(
        make_features("First sentence here. Second sentence here.")
    )
    assert "multiple_sentences:-20" in result.signals


def test_full_stop_ending_subtracts_10():
    result = heading_scorer.score_paragraph(make_features("This is a heading."))
    assert "full_stop_ending:-10" in result.signals


def test_colored_text_subtracts_20():
    result = heading_scorer.score_paragraph(
        make_features("Colored Heading", font_color="FF0000", is_colored=True)
    )
    assert "colored_text:-20" in result.signals


def test_color_detection_rules():
    assert color_is_colored("FF0000") is True
    assert color_is_colored("000000") is False  # black is not coloured
    assert color_is_colored(None) is False
    assert color_is_colored("auto") is False
    assert color_is_colored("theme:ACCENT_1 (5)") is True
    assert color_is_colored("theme:TEXT_1 (13)") is False


# --- combination + boundaries ----------------------------------------------
def test_numbering_bold_colon_combination_scores_at_least_90():
    features = make_features("2.1 API Configuration:", is_bold=True)
    result = heading_scorer.score_paragraph(features)
    assert result.score >= 90
    assert "combination:numbering+bold+colon:min_90" in result.signals
    # Confirmed heading via the ordinary thresholds, level from numbering depth.
    assert result.classification == "heading"
    assert result.predicted_level == 2


def test_combination_with_colon_hyphen_ending():
    features = make_features("3.4.1 Authentication:-", is_bold=True)
    result = heading_scorer.score_paragraph(features)
    assert result.score >= 90
    assert "combination:numbering+bold+colon:min_90" in result.signals
    assert result.predicted_level == 3


def test_numbering_without_bold_does_not_trigger_combination():
    features = make_features("2.1 API Configuration:", is_bold=False)
    result = heading_scorer.score_paragraph(features)
    assert "combination:numbering+bold+colon:min_90" not in result.signals
    assert result.score < 90


def test_bold_colon_without_numbering_does_not_trigger_combination():
    features = make_features("API Configuration:", is_bold=True)
    result = heading_scorer.score_paragraph(features)
    assert features.numbering_prefix is None
    assert "combination:numbering+bold+colon:min_90" not in result.signals


def test_numbering_bold_without_colon_does_not_trigger_combination():
    features = make_features("2.1 API Configuration", is_bold=True)
    result = heading_scorer.score_paragraph(features)
    assert "combination:numbering+bold+colon:min_90" not in result.signals


def test_hard_negative_cannot_be_rescued_by_combination():
    # Force every combination precondition true on a Note line: the hard
    # negative must still win outright.
    features = make_features(
        "Note:- 2.1 Deployment Steps:",
        is_bold=True,
        numbering_prefix="2.1",
        ends_with_colon=True,
    )
    result = heading_scorer.score_paragraph(features)
    assert result.score == 0
    assert result.classification == "normal_content"
    assert result.detection_method == "hard_negative_rule"
    assert "combination:numbering+bold+colon:min_90" not in result.signals


def test_combination_signal_appears_exactly_once():
    features = make_features("2.1 API Configuration:", is_bold=True)
    result = heading_scorer.score_paragraph(features)
    assert result.signals.count("combination:numbering+bold+colon:min_90") == 1


def test_short_bold_larger_text_is_a_likely_heading():
    features = make_features(
        "System Overview",
        is_bold=True,
        font_size=16.0,
        spacing_before=12.0,
    )
    result = heading_scorer.score_paragraph(features)
    assert result.score >= 60
    assert result.classification in ("heading", "probable_heading")


def test_score_never_exceeds_100():
    features = make_features(
        "2.1 Everything Enabled:",
        is_bold=True,
        is_underlined=True,
        font_size=20.0,
        font_family="Arial",
        spacing_before=12.0,
        spacing_after=12.0,
        repeated_formatting_pattern=True,
    )
    result = heading_scorer.score_paragraph(features)
    assert result.score == 100


def test_score_never_below_0():
    text = " ".join(["sentence"] * 20) + ". Another sentence."
    features = make_features(
        text,
        is_word_list_item=True,
        is_colored=True,
        font_color="FF0000",
    )
    result = heading_scorer.score_paragraph(features)
    assert result.score == 0


def test_classification_thresholds():
    assert heading_scorer.classify(85) == "heading"
    assert heading_scorer.classify(70) == "probable_heading"
    assert heading_scorer.classify(40) == "normal_content"
