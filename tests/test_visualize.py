"""Tests for the visualisation panel builder."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest

from bacteriacv.config import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM
from bacteriacv.preprocess import STAGE_NAMES, preprocess
from bacteriacv.visualize import (
    SEGMENTATION_LIMITATION,
    STAGE_TITLES,
    annotate,
    build_visualization,
    confidence_level,
    encode_png,
    encode_png_base64,
    panel_sizes,
    shape_label_text,
)


def _image(success: bool = True) -> np.ndarray:
    """Build a simple test image with or without a dark area."""
    array = np.full((240, 320, 3), 210, dtype=np.uint8)
    if success:
        array[80:180, 90:230] = (20, 20, 220)
    else:
        array[:] = 235
    return array


# --- Confidence level ---


def test_confidence_level_boundaries() -> None:
    """The level must use the threshold from config, not a duplicated number."""
    assert confidence_level(CONFIDENCE_HIGH) == "high"
    assert confidence_level(CONFIDENCE_MEDIUM) == "medium"
    assert confidence_level(0.0) == "low"


def test_confidence_level_is_exhaustive() -> None:
    """Every value must fall into exactly one level."""
    levels = {confidence_level(value / 20) for value in range(21)}
    assert levels == {"high", "medium", "low"}


# --- Annotate ---


def test_annotate_changes_pixels() -> None:
    """Annotate must modify the image, not return a copy."""
    panel = np.full((224, 224, 3), 255, dtype=np.uint8)

    result = annotate(panel, ["baris satu", "baris dua"])

    assert not np.array_equal(result, panel)
    assert result.shape == panel.shape
    assert result.dtype == np.uint8


def test_annotate_does_not_mutate_input() -> None:
    """The original panel must stay intact so it can be reused."""
    panel = np.full((224, 224, 3), 255, dtype=np.uint8)

    annotate(panel, ["teks"])

    assert np.all(panel == 255)


def test_annotate_rejects_empty_lines() -> None:
    """With no text lines, annotate has nothing to do."""
    with pytest.raises(ValueError, match="annotate line is required"):
        annotate(np.zeros((10, 10, 3), dtype=np.uint8), [])


# --- PNG encoding ---


def test_encode_png_produces_png_signature() -> None:
    """The encoding result must really be a PNG."""
    payload = encode_png(np.zeros((32, 32, 3), dtype=np.uint8))

    assert payload[:8] == b"\x89PNG\r\n\x1a\n"


def test_encode_png_base64_decodes_to_png() -> None:
    """Base64 must decode back into the same PNG."""
    panel = _image()

    text = encode_png_base64(panel)

    assert base64.b64decode(text) == encode_png(panel)


# --- Bundle ---


def test_build_visualization_returns_five_panels() -> None:
    """The bundle must hold one panel per preprocessing stage."""
    prepared = preprocess(_image())

    bundle = build_visualization(prepared, "cocci", 0.9, "positive", 0.8)

    assert len(bundle.panels) == len(STAGE_NAMES) == 5
    assert bundle.stage_names == STAGE_NAMES


def test_build_visualization_panels_are_different() -> None:
    """The five panels must differ in content, not be copies of each other."""
    prepared = preprocess(_image())

    bundle = build_visualization(prepared, "cocci", 0.9, "positive", 0.8)

    assert len({panel.tobytes() for panel in bundle.panels}) == 5


def test_build_visualization_states_segmentation_is_not_validated() -> None:
    """The panel must state that shape is not yet validated from segmentation.

    This is a rule the project owner asked for. Dropping this note makes the
    panel show the mask as if it were evidence of cell shape.
    """
    prepared = preprocess(_image())

    bundle = build_visualization(prepared, "cocci", 0.9, "positive", 0.8)

    assert SEGMENTATION_LIMITATION in bundle.notes
    assert any("not yet validated" in note for note in bundle.notes)


def test_segmentation_limitation_uses_measured_numbers() -> None:
    """The figures in the note must come from the experiment, not be placeholders."""
    for number in ("1.30-1.43", "1.64-1.76", "1.03-4.19", "0.068"):
        assert number in SEGMENTATION_LIMITATION


def test_build_visualization_lists_failed_stages() -> None:
    """A failed stage must be named, not hidden."""
    prepared = preprocess(_image(success=False))

    bundle = build_visualization(prepared, "bacilli", 0.4, "negative", 0.3)

    assert bundle.failed_stages
    assert any("(failed)" in title for title in bundle.titles)
    assert any("Failed stages" in note for note in bundle.notes)


def test_build_visualization_keeps_prediction_when_segmentation_fails() -> None:
    """A segmentation failure must not delete the classification result."""
    prepared = preprocess(_image(success=False))

    bundle = build_visualization(prepared, "bacilli", 0.42, "negative", 0.31)

    assert bundle.object_count == 0
    assert len(bundle.panels) == 5


def test_build_visualization_marks_low_confidence() -> None:
    """The confidence level must also be readable in the note."""
    prepared = preprocess(_image())

    bundle = build_visualization(prepared, "bacilli", 0.42, "negative", 0.31)

    assert len(bundle.notes) >= 2


def test_build_visualization_rejects_out_of_range_confidence() -> None:
    """A confidence outside 0 to 1 must be rejected."""
    prepared = preprocess(_image())

    with pytest.raises(ValueError, match="shape_confidence"):
        build_visualization(prepared, "cocci", 1.4, "positive", 0.8)


def test_build_visualization_titles_match_stages() -> None:
    """Every stage must have a title."""
    assert set(STAGE_NAMES) == set(STAGE_TITLES)


def test_panel_sizes_reports_five_panels() -> None:
    """The size check must return one entry per panel."""
    prepared = preprocess(_image())

    sizes = panel_sizes(build_visualization(prepared, "cocci", 0.9, "positive", 0.8))

    assert len(sizes) == 5
    assert all(len(size) == 2 for size in sizes)


def test_shape_label_text_falls_back_on_invalid_index() -> None:
    """An index out of range must not crash."""
    assert shape_label_text(0) == "cocci"
    assert shape_label_text(9) == "tidak_diketahui"
    assert shape_label_text(-1) == "tidak_diketahui"
