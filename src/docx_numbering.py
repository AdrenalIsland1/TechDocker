"""Resolve Word list metadata from the document's numbering definitions.

Word stores visible list markers ("1.", "a.", "•") outside the paragraph text,
in ``word/numbering.xml``: a paragraph's ``w:numPr`` carries a ``w:numId`` and
``w:ilvl`` which point (via a ``w:num`` instance) at a ``w:abstractNum`` level
definition holding the number format (``w:numFmt``) and marker template
(``w:lvlText``).

:class:`NumberingResolver` walks that chain and, for ordinary sequential lists,
reconstructs the visible marker. It is stateful: markers are counted in
document order, and deeper level counters reset whenever a shallower level
advances (matching Word's default restart behaviour).

When a numbering definition cannot be resolved, the marker is left as ``None``
and a warning is recorded — markers are never invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml.ns import qn

# Symbol-font bullet glyphs (Symbol/Wingdings private-use codepoints) that
# should be displayed as an ordinary bullet.
_SYMBOL_BULLET_CHARS = {"", "", "", "", "", ""}

# Number formats we can safely reconstruct sequentially.
_SEQUENTIAL_FORMATS = {
    "decimal",
    "lowerLetter",
    "upperLetter",
    "lowerRoman",
    "upperRoman",
}


@dataclass
class ListInfo:
    """Resolved list metadata for one paragraph."""

    is_list_item: bool = False
    list_type: Optional[str] = None  # "numbered" | "bullet" | "unknown"
    list_level: Optional[int] = None
    numbering_id: Optional[int] = None
    numbering_format: Optional[str] = None
    list_marker: Optional[str] = None


@dataclass
class _LevelDefinition:
    """One ``w:lvl`` entry from an abstract numbering definition."""

    num_fmt: Optional[str] = None
    lvl_text: Optional[str] = None
    start: int = 1


def _int_to_letter(value: int) -> str:
    """1 -> a, 2 -> b ... 27 -> aa (Word's letter numbering)."""
    result = ""
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        result = chr(ord("a") + remainder) + result
    return result


def _int_to_roman(value: int) -> str:
    """1 -> i, 4 -> iv ... (lower-case Roman numerals)."""
    pairs = (
        (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
        (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
        (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
    )
    result = ""
    for amount, numeral in pairs:
        while value >= amount:
            result += numeral
            value -= amount
    return result


def _format_counter(value: int, num_fmt: str) -> str:
    """Render one counter value in the given Word number format."""
    if num_fmt == "decimal":
        return str(value)
    if num_fmt == "lowerLetter":
        return _int_to_letter(value)
    if num_fmt == "upperLetter":
        return _int_to_letter(value).upper()
    if num_fmt == "lowerRoman":
        return _int_to_roman(value)
    if num_fmt == "upperRoman":
        return _int_to_roman(value).upper()
    return str(value)


class NumberingResolver:
    """Resolve ``w:numPr`` references against ``word/numbering.xml``.

    Create one resolver per document and call :meth:`resolve` on paragraphs in
    document order so sequential markers are counted correctly.
    """

    def __init__(self, word_document: Any) -> None:
        self.warnings: list[str] = []
        self._warned_num_ids: set[int] = set()
        # (numId -> {ilvl -> _LevelDefinition})
        self._definitions: dict[int, dict[int, _LevelDefinition]] = {}
        # (numId -> {ilvl -> current counter value})
        self._counters: dict[int, dict[int, int]] = {}
        self._load_definitions(word_document)

    # -- loading -----------------------------------------------------------
    def _load_definitions(self, word_document: Any) -> None:
        try:
            numbering_part = word_document.part.part_related_by(RT.NUMBERING)
        except (KeyError, ValueError):
            return  # document has no numbering definitions

        root = numbering_part.element

        abstract_levels: dict[str, dict[int, _LevelDefinition]] = {}
        for abstract in root.findall(qn("w:abstractNum")):
            abstract_id = abstract.get(qn("w:abstractNumId"))
            levels: dict[int, _LevelDefinition] = {}
            for lvl in abstract.findall(qn("w:lvl")):
                ilvl_value = lvl.get(qn("w:ilvl"))
                if ilvl_value is None:
                    continue
                num_fmt_el = lvl.find(qn("w:numFmt"))
                lvl_text_el = lvl.find(qn("w:lvlText"))
                start_el = lvl.find(qn("w:start"))
                try:
                    start = int(start_el.get(qn("w:val"))) if start_el is not None else 1
                except (TypeError, ValueError):
                    start = 1
                levels[int(ilvl_value)] = _LevelDefinition(
                    num_fmt=(
                        num_fmt_el.get(qn("w:val")) if num_fmt_el is not None else None
                    ),
                    lvl_text=(
                        lvl_text_el.get(qn("w:val")) if lvl_text_el is not None else None
                    ),
                    start=start,
                )
            if abstract_id is not None:
                abstract_levels[abstract_id] = levels

        for num in root.findall(qn("w:num")):
            num_id_value = num.get(qn("w:numId"))
            abstract_ref = num.find(qn("w:abstractNumId"))
            if num_id_value is None or abstract_ref is None:
                continue
            abstract_id = abstract_ref.get(qn("w:val"))
            if abstract_id in abstract_levels:
                self._definitions[int(num_id_value)] = abstract_levels[abstract_id]

    # -- numPr lookup ------------------------------------------------------
    @staticmethod
    def _num_pr_of(paragraph: Any) -> Optional[Any]:
        """Find the effective ``w:numPr``: direct formatting first, then style."""
        p_pr = paragraph._p.pPr
        if p_pr is not None:
            num_pr = p_pr.find(qn("w:numPr"))
            if num_pr is not None:
                return num_pr
        style = paragraph.style
        element = getattr(style, "_element", None)
        if element is not None:
            style_p_pr = element.find(qn("w:pPr"))
            if style_p_pr is not None:
                return style_p_pr.find(qn("w:numPr"))
        return None

    @staticmethod
    def _read_num_pr(num_pr: Any) -> tuple[Optional[int], int]:
        """Extract (numId, ilvl) from a ``w:numPr`` element."""
        num_id: Optional[int] = None
        ilvl = 0
        num_id_el = num_pr.find(qn("w:numId"))
        if num_id_el is not None:
            try:
                num_id = int(num_id_el.get(qn("w:val")))
            except (TypeError, ValueError):
                num_id = None
        ilvl_el = num_pr.find(qn("w:ilvl"))
        if ilvl_el is not None:
            try:
                ilvl = int(ilvl_el.get(qn("w:val")))
            except (TypeError, ValueError):
                ilvl = 0
        return num_id, ilvl

    # -- marker reconstruction ----------------------------------------------
    def _advance_counter(self, num_id: int, ilvl: int, definition: _LevelDefinition) -> int:
        """Advance the sequence counter for (numId, ilvl) and reset deeper ones."""
        counters = self._counters.setdefault(num_id, {})
        counters[ilvl] = counters.get(ilvl, definition.start - 1) + 1
        for deeper in [level for level in counters if level > ilvl]:
            del counters[deeper]
        return counters[ilvl]

    def _render_marker(
        self, num_id: int, ilvl: int, definition: _LevelDefinition
    ) -> Optional[str]:
        """Render the visible marker from the lvlText template, e.g. "%1." -> "1."."""
        counters = self._counters.get(num_id, {})
        template = definition.lvl_text or f"%{ilvl + 1}."
        marker = template
        for level in range(9):
            placeholder = f"%{level + 1}"
            if placeholder not in marker:
                continue
            level_definition = self._definitions.get(num_id, {}).get(
                level, _LevelDefinition()
            )
            value = counters.get(level)
            if value is None:
                # Referenced shallower level never appeared; assume its start.
                value = level_definition.start
            marker = marker.replace(
                placeholder,
                _format_counter(value, level_definition.num_fmt or "decimal"),
            )
        return marker

    def _warn_once(self, num_id: int, message: str) -> None:
        if num_id not in self._warned_num_ids:
            self._warned_num_ids.add(num_id)
            self.warnings.append(message)

    # -- public API ----------------------------------------------------------
    def resolve(self, paragraph: Any) -> ListInfo:
        """Resolve list metadata for one paragraph (call in document order)."""
        num_pr = self._num_pr_of(paragraph)
        if num_pr is None:
            return ListInfo()

        num_id, ilvl = self._read_num_pr(num_pr)
        if num_id is None or num_id == 0:
            # numId 0 explicitly disables numbering.
            return ListInfo()

        info = ListInfo(
            is_list_item=True,
            list_type="unknown",
            list_level=ilvl,
            numbering_id=num_id,
        )

        levels = self._definitions.get(num_id)
        if not levels:
            self._warn_once(
                num_id,
                f"List numbering definition numId={num_id} could not be "
                "resolved; list markers for it are unavailable.",
            )
            return info

        definition = levels.get(ilvl)
        if definition is None:
            self._warn_once(
                num_id,
                f"List level {ilvl} of numbering numId={num_id} is not "
                "defined; its marker is unavailable.",
            )
            return info

        info.numbering_format = definition.num_fmt

        if definition.num_fmt == "bullet":
            info.list_type = "bullet"
            raw = (definition.lvl_text or "").strip()
            if not raw or raw in _SYMBOL_BULLET_CHARS or ord(raw[0]) >= 0xE000:
                info.list_marker = "•"
            else:
                info.list_marker = raw
            return info

        if definition.num_fmt in _SEQUENTIAL_FORMATS:
            info.list_type = "numbered"
            self._advance_counter(num_id, ilvl, definition)
            info.list_marker = self._render_marker(num_id, ilvl, definition)
            return info

        # Unusual format (e.g. "none", "ordinalText"): keep the metadata but do
        # not invent a marker.
        info.list_type = "numbered" if definition.num_fmt else "unknown"
        self._warn_once(
            num_id,
            f"List numbering format {definition.num_fmt!r} (numId={num_id}) is "
            "not supported for marker reconstruction.",
        )
        return info
