"""Tes untuk penyusun panel visualisasi."""

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
    """Buat citra uji sederhana dengan atau tanpa area gelap."""
    array = np.full((240, 320, 3), 210, dtype=np.uint8)
    if success:
        array[80:180, 90:230] = (20, 20, 220)
    else:
        array[:] = 235
    return array


# --- Confidence level ---


def test_confidence_level_boundaries() -> None:
    """Level harus memakai ambang dari config, bukan angka duplikat."""
    assert confidence_level(CONFIDENCE_HIGH) == "tinggi"
    assert confidence_level(CONFIDENCE_MEDIUM) == "sedang"
    assert confidence_level(0.0) == "rendah"


def test_confidence_level_is_exhaustive() -> None:
    """Setiap nilai harus jatuh pada tepat satu level."""
    levels = {confidence_level(value / 20) for value in range(21)}
    assert levels == {"tinggi", "sedang", "rendah"}


# --- Annotate ---


def test_annotate_changes_pixels() -> None:
    """Annotate harus mengubah citra, bukan mengembalikan salinan."""
    panel = np.full((224, 224, 3), 255, dtype=np.uint8)

    result = annotate(panel, ["baris satu", "baris dua"])

    assert not np.array_equal(result, panel)
    assert result.shape == panel.shape
    assert result.dtype == np.uint8


def test_annotate_does_not_mutate_input() -> None:
    """Panel asal harus tetap utuh supaya bisa dipakai ulang."""
    panel = np.full((224, 224, 3), 255, dtype=np.uint8)

    annotate(panel, ["teks"])

    assert np.all(panel == 255)


def test_annotate_rejects_empty_lines() -> None:
    """Tanpa baris teks, annotate tidak punya gunanya."""
    with pytest.raises(ValueError, match="baris"):
        annotate(np.zeros((10, 10, 3), dtype=np.uint8), [])


# --- Enkode PNG ---


def test_encode_png_produces_png_signature() -> None:
    """Hasil enkode harus benar-benar PNG."""
    payload = encode_png(np.zeros((32, 32, 3), dtype=np.uint8))

    assert payload[:8] == b"\x89PNG\r\n\x1a\n"


def test_encode_png_base64_decodes_to_png() -> None:
    """Basis64 harus dapat dikembalikan menjadi PNG yang sama."""
    panel = _image()

    text = encode_png_base64(panel)

    assert base64.b64decode(text) == encode_png(panel)


# --- Bundle ---


def test_build_visualization_returns_five_panels() -> None:
    """Bundle harus memuat satu panel per tahap pra-pemrosesan."""
    prepared = preprocess(_image())

    bundle = build_visualization(prepared, "cocci", 0.9, "positive", 0.8)

    assert len(bundle.panels) == len(STAGE_NAMES) == 5
    assert bundle.stage_names == STAGE_NAMES


def test_build_visualization_panels_are_different() -> None:
    """Kelima panel harus berbeda isi, bukan salinan satu sama lain."""
    prepared = preprocess(_image())

    bundle = build_visualization(prepared, "cocci", 0.9, "positive", 0.8)

    assert len({panel.tobytes() for panel in bundle.panels}) == 5


def test_build_visualization_states_segmentation_is_not_validated() -> None:
    """Panel harus menulis bahwa bentuk belum tervalidasi dari segmentasi.

    Ini aturan yang diminta pemilik proyek. Menyingkirkan catatan ini membuat
    panel menampilkan mask seolah-olah mask itu bukti bentuk sel.
    """
    prepared = preprocess(_image())

    bundle = build_visualization(prepared, "cocci", 0.9, "positive", 0.8)

    assert SEGMENTATION_LIMITATION in bundle.notes
    assert any("belum tervalidasi" in note for note in bundle.notes)


def test_segmentation_limitation_uses_measured_numbers() -> None:
    """Angka pada catatan harus angka hasil eksperimen, bukan placeholder."""
    for number in ("1.30-1.43", "1.64-1.76", "1.03-4.19", "0.068"):
        assert number in SEGMENTATION_LIMITATION


def test_build_visualization_lists_failed_stages() -> None:
    """Tahap gagal harus disebut, bukan disembunyikan."""
    prepared = preprocess(_image(success=False))

    bundle = build_visualization(prepared, "bacilli", 0.4, "negative", 0.3)

    assert bundle.failed_stages
    assert any("gagal" in title for title in bundle.titles)
    assert any("Tahap gagal" in note for note in bundle.notes)


def test_build_visualization_keeps_prediction_when_segmentation_fails() -> None:
    """Kegagalan segmentasi tidak boleh menghapus hasil klasifikasi."""
    prepared = preprocess(_image(success=False))

    bundle = build_visualization(prepared, "bacilli", 0.42, "negative", 0.31)

    assert bundle.object_count == 0
    assert len(bundle.panels) == 5


def test_build_visualization_marks_low_confidence() -> None:
    """Level confidence harus ikut terbaca pada catatan."""
    prepared = preprocess(_image())

    bundle = build_visualization(prepared, "bacilli", 0.42, "negative", 0.31)

    assert len(bundle.notes) >= 2


def test_build_visualization_rejects_out_of_range_confidence() -> None:
    """Confidence di luar 0 sampai 1 harus ditolak."""
    prepared = preprocess(_image())

    with pytest.raises(ValueError, match="shape_confidence"):
        build_visualization(prepared, "cocci", 1.4, "positive", 0.8)


def test_build_visualization_titles_match_stages() -> None:
    """Setiap tahap harus punya judul."""
    assert set(STAGE_NAMES) == set(STAGE_TITLES)


def test_panel_sizes_reports_five_panels() -> None:
    """Pemeriksaan ukuran harus mengembalikan satu entri per panel."""
    prepared = preprocess(_image())

    sizes = panel_sizes(build_visualization(prepared, "cocci", 0.9, "positive", 0.8))

    assert len(sizes) == 5
    assert all(len(size) == 2 for size in sizes)


def test_shape_label_text_falls_back_on_invalid_index() -> None:
    """Indeks di luar rentang tidak boleh membuat crash."""
    assert shape_label_text(0) == "cocci"
    assert shape_label_text(9) == "tidak_diketahui"
    assert shape_label_text(-1) == "tidak_diketahui"
