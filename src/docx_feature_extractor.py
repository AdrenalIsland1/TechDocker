"""Extract visible formatting features from DOCX paragraphs.

This module is responsible for turning a python-docx ``Paragraph`` into a
:class:`ParagraphFeatures` record. It handles:

* run-level formatting with inheritance from the paragraph style chain,
* dominant / "mostly" formatting across visible runs (>= 75% for bold and
  underline),
* Word bullet/numbered list-item detection via the underlying ``w:numPr`` XML,
* font colour detection (RGB vs theme vs automatic/black),
* dynamic detection of the document's normal body formatting.

All accessors are defensive: missing or inherited formatting must never raise.

Basically extracts paragraph-level text and formatting data from the DOCX.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from docx.oxml.ns import qn

from src import heading_scorer

try:  # pragma: no cover - import guard for older python-docx
    from docx.enum.dml import MSO_COLOR_TYPE
except Exception:  # pragma: no cover
    MSO_COLOR_TYPE = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class ParagraphFeatures:
    """Every feature extracted for one non-empty paragraph."""

    text: str
    position: int
    style_name: str

    # text-derived
    word_count: int
    sentence_count: int
    numbering_prefix: Optional[str]
    is_title_case: bool
    is_all_caps: bool
    ends_with_full_stop: bool
    ends_with_colon: bool

    # run/format-derived
    is_word_list_item: bool
    is_bold: bool
    is_italic: bool
    is_underlined: bool
    font_size: Optional[float]
    font_family: Optional[str]
    font_color: Optional[str]
    is_colored: bool
    alignment: Optional[str]
    spacing_before: Optional[float]
    spacing_after: Optional[float]
    left_indent: Optional[float]
    first_line_indent: Optional[float]

    # official-heading facts
    is_official_heading: bool
    official_heading_level: Optional[int]
    is_title_style: bool

    # manual line breaks (w:br / w:cr) inside this single Word paragraph
    segments: list[str] = field(default_factory=list)
    has_internal_line_breaks: bool = False
    segment_count: int = 1

    # Word list metadata (resolved from word/numbering.xml)
    list_type: Optional[str] = None
    list_level: Optional[int] = None
    numbering_id: Optional[int] = None
    numbering_format: Optional[str] = None
    list_marker: Optional[str] = None
    display_text: Optional[str] = None

    # filled in later (document-level context)
    body_font_size: Optional[float] = None
    body_font_family: Optional[str] = None
    body_font_color: Optional[str] = None
    repeated_formatting_pattern: bool = False
    formatting_signature: Optional[tuple] = field(default=None)


@dataclass
class BodyFormatting:
    """The dynamically-detected "normal" formatting of a document."""

    font_size: Optional[float] = None
    font_family: Optional[str] = None
    font_color: Optional[str] = None


@dataclass
class DocumentDefaults:
    """Document-wide run defaults from ``w:docDefaults`` in styles.xml.

    Word stores the true base font here; paragraphs whose runs and styles set
    nothing explicit ultimately inherit these values.
    """

    font_size: Optional[float] = None
    font_family: Optional[str] = None


def read_document_defaults(word_document: Any) -> DocumentDefaults:
    """Read default run formatting from ``w:docDefaults`` (safe on failure)."""
    try:
        styles_element = word_document.styles.element
        doc_defaults = styles_element.find(qn("w:docDefaults"))
        if doc_defaults is None:
            return DocumentDefaults()
        rpr_default = doc_defaults.find(qn("w:rPrDefault"))
        rpr = rpr_default.find(qn("w:rPr")) if rpr_default is not None else None
        if rpr is None:
            return DocumentDefaults()
        size_el = rpr.find(qn("w:sz"))
        fonts_el = rpr.find(qn("w:rFonts"))
        font_size = None
        if size_el is not None:
            value = size_el.get(qn("w:val"))
            if value is not None:
                font_size = float(value) / 2.0  # half-points -> points
        font_family = (
            fonts_el.get(qn("w:ascii")) if fonts_el is not None else None
        )
        return DocumentDefaults(font_size=font_size, font_family=font_family)
    except Exception:
        return DocumentDefaults()


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _most_common(values: list[Any]) -> Optional[Any]:
    """Return the most common non-None value, or ``None`` if there are none.

    Used for size/family where ``None`` means "could not detect" and should be
    ignored in favour of any real value.
    """
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return Counter(filtered).most_common(1)[0][0]


def _most_common_including_none(values: list[Any]) -> Optional[Any]:
    """Return the most common value, counting ``None`` as a real value.

    Used for body colour, where ``None`` (automatic/default) is itself the
    meaningful "ordinary body colour" and must not be discarded just because a
    single paragraph carries an explicit colour.
    """
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _official_heading_level(style_name: str) -> Optional[int]:
    """Return the level of a "Heading N" style, or ``None``."""
    name = (style_name or "").strip().lower()
    if not name.startswith("heading "):
        return None
    try:
        level = int(name.split()[-1])
    except ValueError:
        return None
    return level if 1 <= level <= 9 else None


# ---------------------------------------------------------------------------
# Style-chain resolution (handles inherited run formatting)
# ---------------------------------------------------------------------------
def _resolve_bool_from_style(style: Any, attr: str) -> Optional[bool]:
    while style is not None:
        try:
            value = getattr(style.font, attr)
        except Exception:
            value = None
        if value is not None:
            return bool(value)
        style = getattr(style, "base_style", None)
    return None


def _resolve_bold(run: Any, style: Any) -> bool:
    value = run.font.bold
    if value is not None:
        return bool(value)
    inherited = _resolve_bool_from_style(style, "bold")
    return bool(inherited) if inherited is not None else False


def _resolve_italic(run: Any, style: Any) -> bool:
    value = run.font.italic
    if value is not None:
        return bool(value)
    inherited = _resolve_bool_from_style(style, "italic")
    return bool(inherited) if inherited is not None else False


def _underline_to_bool(value: Any) -> Optional[bool]:
    """Interpret a python-docx underline value as a tri-state boolean."""
    if value is None:
        return None
    if value is True:
        return True
    if value is False:
        return False
    # WD_UNDERLINE enum member: anything other than NONE counts as underlined.
    try:
        return str(value).upper().split()[0] != "NONE"
    except Exception:
        return bool(value)


def _resolve_underline(run: Any, style: Any) -> bool:
    resolved = _underline_to_bool(run.font.underline)
    if resolved is not None:
        return resolved
    while style is not None:
        try:
            resolved = _underline_to_bool(style.font.underline)
        except Exception:
            resolved = None
        if resolved is not None:
            return resolved
        style = getattr(style, "base_style", None)
    return False


def _resolve_size(run: Any, style: Any) -> Optional[float]:
    if run.font.size is not None:
        return float(run.font.size.pt)
    while style is not None:
        try:
            size = style.font.size
        except Exception:
            size = None
        if size is not None:
            return float(size.pt)
        style = getattr(style, "base_style", None)
    return None


def _resolve_name(run: Any, style: Any) -> Optional[str]:
    if run.font.name is not None:
        return run.font.name
    while style is not None:
        try:
            name = style.font.name
        except Exception:
            name = None
        if name is not None:
            return name
        style = getattr(style, "base_style", None)
    return None


# ---------------------------------------------------------------------------
# Colour detection
# ---------------------------------------------------------------------------
def _color_descriptor(color: Any) -> Optional[str]:
    """Describe a python-docx colour as a string, or ``None`` if inherited.

    Returns e.g. ``"FF0000"`` for RGB, ``"theme:ACCENT_1 (5)"`` for a theme
    colour, or ``"auto"`` for automatic.
    """
    if color is None:
        return None
    try:
        color_type = color.type
    except Exception:
        return None
    if color_type is None:
        return None
    try:
        if MSO_COLOR_TYPE is not None and color_type == MSO_COLOR_TYPE.RGB:
            rgb = color.rgb
            return str(rgb) if rgb is not None else None
        if MSO_COLOR_TYPE is not None and color_type == MSO_COLOR_TYPE.THEME:
            theme = color.theme_color
            return f"theme:{theme}" if theme is not None else None
    except Exception:
        return None
    return "auto"


def _resolve_color(run: Any, style: Any) -> Optional[str]:
    descriptor = _color_descriptor(run.font.color)
    if descriptor is not None:
        return descriptor
    while style is not None:
        try:
            descriptor = _color_descriptor(style.font.color)
        except Exception:
            descriptor = None
        if descriptor is not None:
            return descriptor
        style = getattr(style, "base_style", None)
    return None


def color_is_colored(descriptor: Optional[str]) -> bool:
    """Decide whether a colour descriptor represents visible (non-black) colour.

    Automatic colour, ``None`` (inherited default) and pure black are treated as
    normal body text. Explicit non-black RGB and accent/hyperlink theme colours
    count as coloured.
    """
    if not descriptor or descriptor == "auto":
        return False
    if descriptor.startswith("theme:"):
        upper = descriptor.upper()
        neutral = ("TEXT", "DARK", "LIGHT", "BACKGROUND")
        return not any(token in upper for token in neutral)
    return descriptor.upper() != "000000"


# ---------------------------------------------------------------------------
# Word list-item detection (via XML)
# ---------------------------------------------------------------------------
def _num_pr_is_active(num_pr: Any) -> bool:
    """Return ``True`` unless the ``numPr`` explicitly disables numbering."""
    if num_pr is None:
        return False
    num_id = num_pr.find(qn("w:numId"))
    if num_id is not None:
        value = num_id.get(qn("w:val"))
        if value is not None and value.strip() == "0":
            return False
    return True


def _style_has_numbering(style: Any) -> bool:
    """Return ``True`` when a paragraph style itself defines numbering.

    Covers built-in list styles such as "List Bullet"/"List Number", whose
    numbering lives in the style definition rather than the paragraph.
    """
    element = getattr(style, "_element", None)
    if element is None:
        return False
    p_pr = element.find(qn("w:pPr"))
    if p_pr is None:
        return False
    return _num_pr_is_active(p_pr.find(qn("w:numPr")))


def is_word_list_item(paragraph: Any) -> bool:
    """Return ``True`` when the paragraph is a real Word bullet/numbered item.

    Detected through the ``w:numPr`` numbering property, checked first as direct
    paragraph formatting and then on the paragraph style. A ``numId`` of 0
    explicitly means "no numbering" and is treated as not a list item.
    """
    p_pr = paragraph._p.pPr
    if p_pr is not None and _num_pr_is_active(p_pr.find(qn("w:numPr"))):
        return True
    return _style_has_numbering(paragraph.style)


# ---------------------------------------------------------------------------
# Run aggregation ("mostly" formatting)
# ---------------------------------------------------------------------------
def _visible_runs(paragraph: Any) -> list[Any]:
    return [run for run in paragraph.runs if run.text and run.text.strip()]


def _visible_weight(run: Any) -> int:
    """Number of visible (non-whitespace) characters in a run."""
    return sum(1 for character in run.text if not character.isspace())


def _weighted_ratio_true(runs: list[Any], flags: list[bool]) -> float:
    """Fraction of visible characters whose run has the flag set.

    Weighting by visible characters (rather than counting runs) stops a short
    run from carrying the same influence as a long one — a paragraph that is
    one long bold run plus a one-word plain run is still "mostly bold".
    Whitespace-only runs contribute no weight; zero total weight yields 0.0.
    """
    total = 0
    true_weight = 0
    for run, flag in zip(runs, flags):
        weight = _visible_weight(run)
        total += weight
        if flag:
            true_weight += weight
    if total == 0:
        return 0.0
    return true_weight / total


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_paragraph_features(
    paragraph: Any,
    position: int,
    list_info: Optional[Any] = None,
    defaults: Optional[DocumentDefaults] = None,
) -> ParagraphFeatures:
    """Build a :class:`ParagraphFeatures` for one paragraph.

    ``list_info`` is an optional resolved :class:`src.docx_numbering.ListInfo`;
    when omitted, only the plain ``w:numPr`` presence check is used.
    ``defaults`` (from :func:`read_document_defaults`) supplies the final
    fallback for font size/family when neither runs nor styles set them.
    Body-level fields (``body_font_size`` etc.) and the repeated-pattern flag
    are filled in later once the whole document has been scanned.
    """
    # .strip() removes surrounding whitespace but preserves the internal "\n"
    # characters that python-docx produces for manual line breaks (w:br/w:cr).
    text = paragraph.text.strip()

    # Manual line breaks inside one Word paragraph: expose each non-empty line
    # as a segment while keeping the paragraph itself intact.
    segments = [line.strip() for line in text.split("\n") if line.strip()]
    has_internal_line_breaks = len(segments) > 1
    style = paragraph.style
    style_name = (style.name or "").strip() if style is not None else ""

    runs = _visible_runs(paragraph)

    if runs:
        bold_flags = [_resolve_bold(run, style) for run in runs]
        underline_flags = [_resolve_underline(run, style) for run in runs]
        italic_flags = [_resolve_italic(run, style) for run in runs]
        sizes = [_resolve_size(run, style) for run in runs]
        families = [_resolve_name(run, style) for run in runs]
        colors = [_resolve_color(run, style) for run in runs]

        is_bold = _weighted_ratio_true(runs, bold_flags) >= 0.75
        is_underlined = _weighted_ratio_true(runs, underline_flags) >= 0.75
        is_italic = _weighted_ratio_true(runs, italic_flags) >= 0.5
        font_size = _most_common(sizes)
        font_family = _most_common(families)
        font_color = _most_common(colors)
    else:
        # No visible runs: fall back to the paragraph style only.
        is_bold = bool(_resolve_bool_from_style(style, "bold"))
        is_underlined = bool(_underline_to_bool(getattr(style.font, "underline", None))) if style else False
        is_italic = bool(_resolve_bool_from_style(style, "italic"))
        font_size = _resolve_size(_FakeRun(), style) if style else None
        font_family = _resolve_name(_FakeRun(), style) if style else None
        font_color = _resolve_color(_FakeRun(), style) if style else None

    # Fall back to the document-wide defaults (w:docDefaults) when neither the
    # runs nor the style chain define size/family explicitly.
    if defaults is not None:
        if font_size is None:
            font_size = defaults.font_size
        if font_family is None:
            font_family = defaults.font_family

    is_colored = color_is_colored(font_color)

    paragraph_format = paragraph.paragraph_format
    alignment = paragraph_format.alignment
    spacing_before = _length_pt(paragraph_format.space_before)
    spacing_after = _length_pt(paragraph_format.space_after)
    left_indent = _length_pt(paragraph_format.left_indent)
    first_line_indent = _length_pt(paragraph_format.first_line_indent)

    official_level = _official_heading_level(style_name)

    if list_info is not None:
        item_is_list = list_info.is_list_item
    else:
        item_is_list = is_word_list_item(paragraph)

    list_marker = getattr(list_info, "list_marker", None)
    display_text = f"{list_marker} {text}" if list_marker else None

    return ParagraphFeatures(
        text=text,
        position=position,
        style_name=style_name,
        word_count=len(text.split()),
        sentence_count=heading_scorer.count_sentences(text),
        numbering_prefix=heading_scorer.extract_numbering_prefix(text),
        is_title_case=heading_scorer.is_title_case(text),
        is_all_caps=heading_scorer.is_all_caps(text),
        ends_with_full_stop=heading_scorer.has_full_stop_ending(text),
        ends_with_colon=heading_scorer.has_colon_ending(text),
        segments=segments,
        has_internal_line_breaks=has_internal_line_breaks,
        segment_count=len(segments),
        list_type=getattr(list_info, "list_type", None),
        list_level=getattr(list_info, "list_level", None),
        numbering_id=getattr(list_info, "numbering_id", None),
        numbering_format=getattr(list_info, "numbering_format", None),
        list_marker=list_marker,
        display_text=display_text,
        is_word_list_item=item_is_list,
        is_bold=is_bold,
        is_italic=is_italic,
        is_underlined=is_underlined,
        font_size=font_size,
        font_family=font_family,
        font_color=font_color,
        is_colored=is_colored,
        alignment=(str(alignment) if alignment is not None else None),
        spacing_before=spacing_before,
        spacing_after=spacing_after,
        left_indent=left_indent,
        first_line_indent=first_line_indent,
        is_official_heading=official_level is not None,
        official_heading_level=official_level,
        is_title_style=style_name.lower() == "title",
    )


class _FakeRun:
    """A stand-in run whose font has no explicit formatting.

    Lets the style-resolution helpers be reused when a paragraph has no visible
    runs, without special-casing each resolver.
    """

    class _Font:
        size = None
        name = None
        color = None
        bold = None
        italic = None
        underline = None

    font = _Font()


def _length_pt(length: Any) -> Optional[float]:
    """Convert a python-docx ``Length`` to points, or ``None``."""
    if length is None:
        return None
    try:
        return float(length.pt)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Body-format detection
# ---------------------------------------------------------------------------
def detect_body_formatting(features: list[ParagraphFeatures]) -> BodyFormatting:
    """Estimate the document's normal body formatting.

    Prefers non-empty paragraphs using the ``Normal`` style, plus Word list
    items — list items are body content by definition, and in list-heavy
    documents they are often the majority of the ordinary text. If there are
    no such paragraphs, falls back to all non-heading paragraphs.
    """
    normals = [
        feature
        for feature in features
        if feature.text
        and (feature.style_name.lower() == "normal" or feature.is_word_list_item)
    ]

    # Mostly-bold/underlined paragraphs are heading-shaped, not ordinary text;
    # keep them out of the body vote whenever plainer candidates exist.
    plain = [
        feature
        for feature in normals
        if not feature.is_bold and not feature.is_underlined
    ]
    if plain:
        normals = plain
    if not normals:
        normals = [
            feature for feature in features if not feature.is_official_heading and feature.text
        ]
    if not normals:
        return BodyFormatting()

    return BodyFormatting(
        font_size=_most_common([feature.font_size for feature in normals]),
        font_family=_most_common([feature.font_family for feature in normals]),
        font_color=_most_common_including_none(
            [feature.font_color for feature in normals]
        ),
    )
