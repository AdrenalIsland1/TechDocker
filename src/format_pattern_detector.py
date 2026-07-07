"""Detect repeated non-body formatting patterns across a document.

A paragraph earns the "repeated formatting" bonus only when its formatting
signature appears at least twice **and** differs meaningfully from the detected
body formatting. This prevents ordinary body paragraphs (which naturally repeat
their formatting many times) from being rewarded.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from src.docx_feature_extractor import BodyFormatting, ParagraphFeatures


def build_signature(features: ParagraphFeatures) -> tuple:
    """Build a hashable formatting signature for a paragraph.

    Font size is rounded so near-identical sizes collapse together.
    """
    return (
        round(features.font_size) if features.font_size is not None else None,
        features.font_family,
        features.is_bold,
        features.is_underlined,
        features.font_color,
        features.spacing_before,
        features.spacing_after,
        features.alignment,
    )


def _body_signature(body: BodyFormatting) -> tuple:
    """The reduced signature of ordinary body formatting."""
    return (
        round(body.font_size) if body.font_size is not None else None,
        body.font_family,
        False,  # body text is not bold
        False,  # body text is not underlined
        body.font_color,
    )


def _reduced(signature: tuple) -> tuple:
    """The style-defining part of a signature (ignores spacing/alignment)."""
    return signature[:5]


def _differs_from_body(signature: tuple, body: BodyFormatting) -> bool:
    """Return ``True`` when the signature is meaningfully non-body."""
    return _reduced(signature) != _body_signature(body)


def detect_repeated_patterns(
    features: list[ParagraphFeatures],
    body: BodyFormatting,
) -> None:
    """Populate ``formatting_signature`` and ``repeated_formatting_pattern``.

    Mutates each feature in place. A pattern counts as repeated when its
    signature occurs at least twice and differs from the body formatting.
    """
    for feature in features:
        feature.formatting_signature = build_signature(feature)

    counts: Counter[tuple] = Counter(
        feature.formatting_signature for feature in features
    )

    for feature in features:
        signature = feature.formatting_signature
        feature.repeated_formatting_pattern = (
            counts[signature] >= 2 and _differs_from_body(signature, body)
        )
