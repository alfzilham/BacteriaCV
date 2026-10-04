"""Tes untuk pipeline pra-pemrosesan."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bacteriacv import preprocess as pp
from bacteriacv.config import (
    AUGMENT_ROTATION_DEGREES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def _synthetic(height: int = 600, width: int = 800, cells: int = 40, seed: int = 0):
    """Buat citra mikroskop palsu dengan sel berwarna gelap di latar terang."""
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), 232, dtype=np.uint8)
    for _ in range(cells):
        cy = int(rng.integers(40, height - 40))
        cx = int(rng.integers(40, width - 40))
        ry = int(rng.integers(6, 14))
        rx = int(rng.integers(6, 14))
        y0, y1 = max(0, cy - ry), min(height, cy + ry)
        x0, x1 = max(0, cx - rx), min(width, cx + rx)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        ellipse = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
        image[y0:y1, x0:x1][ellipse] = (110, 70, 150)
    return image


# --- Tahap 1: pemuatan ---


def test_load_image_reads_png(tmp_path) -> None:
    """PNG valid harus terbaca sebagai RGB uint8."""
    import cv2

    path = tmp_path / "citra.png"
    assert cv2.imwrite(str(path), cv2.cvtColor(_synthetic(200, 200), cv2.COLOR_RGB2BGR))

    image = pp.load_image(path)

    assert image.shape == (200, 200, 3)
    assert image.dtype == np.uint8


def test_load_image_reads_tiff(tmp_path) -> None:
    """TIFF harus terbaca, karena itu format DIBaS."""
    import cv2

    path = tmp_path / "citra.tif"
    assert cv2.imwrite(str(path), cv2.cvtColor(_synthetic(200, 200), cv2.COLOR_RGB2BGR))

    assert pp.load_image(path).shape == (200, 200, 3)


def test_load_image_missing_file_raises(tmp_path) -> None:
    """Berkas hilang harus ditolak dengan pesan jelas."""
    with pytest.raises(FileNotFoundError):
        pp.load_image(tmp_path / "hilang.tif")


def test_load_image_corrupt_file_raises(tmp_path) -> None:
    """Berkas rusak harus ditolak, bukan menghasilkan array kosong."""
    path = tmp_path / "rusak.tif"
    path.write_bytes(b"bukan gambar")

    with pytest.raises(ValueError):
        pp.load_image(path)


# --- Tahap 2: resize ---


def test_resize_produces_target_size() -> None:
    """Resize harus menghasilkan sisi IMAGE_SIZE sesuai ARCHITECTURE C1."""
    resized = pp.resize_image(_synthetic(600, 800))
    assert resized.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)


def test_resize_accepts_non_square_input() -> None:
    """Citra DIBaS tidak persegi, resize harus tetap menghasilkan persegi."""
    resized = pp.resize_image(_synthetic(1532, 2048))
    assert resized.shape[:2] == (IMAGE_SIZE, IMAGE_SIZE)


def test_resize_does_not_mutate_input() -> None:
    """Resize tidak boleh mengubah citra sumber."""
    original = _synthetic(300, 300)
    before = original.copy()
    pp.resize_image(original)
    assert np.array_equal(original, before)


# --- Tahap 3: normalisasi ---


def test_normalize_uses_imagenet_statistics() -> None:
    """Normalisasi memakai statistik ImageNet."""
    image = np.full((32, 32, 3), 255, dtype=np.uint8)

    normalized = pp.normalize(image)

    expected = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    assert normalized.dtype == np.float32
    assert normalized[0, 0, 0] == pytest.approx(expected, abs=1e-5)


def test_normalize_zero_maps_to_negative_mean_ratio() -> None:
    """Piksel hitam harus menghasilkan nilai negatif."""
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    normalized = pp.normalize(black)
    assert (normalized < 0).all()


def test_normalize_output_has_no_nan() -> None:
    """Normalisasi tidak boleh menghasilkan NaN atau inf."""
    normalized = pp.normalize(_synthetic(100, 100))
    assert np.isfinite(normalized).all()


# --- Tahap 4 dan 5: segmentasi ---


def test_segmentation_finds_objects_in_synthetic_image() -> None:
    """Citra dengan sel synthesetis harus menghasilkan objek."""
    mask, ok = pp.segment_cells(_synthetic(800, 800, cells=60, seed=3))

    assert ok is True
    assert mask.shape == (800, 800)
    assert mask.dtype == bool
    assert mask.sum() > 0


def test_segmentation_returns_false_on_blank_image() -> None:
    """Citra tanpa sel harus gagal dengan tenang, bukan melempar exception."""
    blank = np.full((400, 400, 3), 250, dtype=np.uint8)

    mask, ok = pp.segment_cells(blank)

    assert ok is False
    assert mask.sum() == 0


def test_segmentation_returns_false_on_uniform_dark_image() -> None:
    """Citra gelap seragam juga harus dianggap tidak ada sel."""
    dark = np.full((400, 400, 3), 10, dtype=np.uint8)

    _, ok = pp.segment_cells(dark)

    assert ok is False


def test_segmentation_mask_does_not_cover_whole_image() -> None:
    """Mask tidak boleh menutup seluruh citra, itu tanda ambang terbalik."""
    mask, ok = pp.segment_cells(_synthetic(600, 600, cells=30, seed=7))

    if ok:
        assert mask.mean() < 0.5, "mask menutup terlalu besar dari citra"


def test_segmentation_is_deterministic() -> None:
    """Segmentasi citra sama harus menghasilkan mask sama."""
    image = _synthetic(600, 600, cells=40, seed=11)

    first, _ = pp.segment_cells(image)
    second, _ = pp.segment_cells(image)

    assert np.array_equal(first, second)


# --- Pipeline penuh ---


def test_preprocess_returns_tensor_chw() -> None:
    """Output tensor harus B x 3 x 224 x 224 sesuai ARCHITECTURE C1."""
    result = pp.preprocess(_synthetic(600, 800))

    assert isinstance(result.tensor, torch.Tensor)
    assert result.tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert result.tensor.dtype == torch.float32


def test_preprocess_reports_stage_status() -> None:
    """Setiap tahap harus punya status yang dilaporkan."""
    result = pp.preprocess(_synthetic(600, 800))

    for stage in ("load", "resize", "normalize", "segment", "watershed"):
        assert stage in result.stage_ok, stage
        assert isinstance(result.stage_ok[stage], bool)


def test_preprocess_success_on_valid_image() -> None:
    """Citra dengan sel harus meluluskan semua tahap."""
    result = pp.preprocess(_synthetic(900, 900, cells=70, seed=5))

    assert all(result.stage_ok.values())
    assert result.object_count > 0


def test_preprocess_continues_when_segmentation_fails() -> None:
    """Kegagalan segmentasi tidak boleh menghentikan pipeline.

    Sesuai ARCHITECTURE bagian 7: lanjutkan dengan citra asli dan catat
    peringatan. Tensor untuk model harus tetap terbentuk.
    """
    blank = np.full((500, 500, 3), 250, dtype=np.uint8)

    result = pp.preprocess(blank)

    assert result.stage_ok["segment"] is False
    assert result.stage_ok["resize"] is True
    assert result.stage_ok["normalize"] is True
    assert result.tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert result.object_count == 0
    assert result.failed_panels == ("segment", "watershed")
    assert result.message is not None
    assert "kelompok sel" in result.message


def test_preprocess_continues_when_only_load_fails() -> None:
    """Citra tak terbaca harus menghasilkan status gagal, bukan exception."""
    result = pp.preprocess(np.full((300, 300, 3), 250, dtype=np.uint8))

    assert result.stage_ok["resize"] is True


def test_preprocess_accepts_path(tmp_path) -> None:
    """preprocess harus menerima path berkas, bukan hanya array."""
    import cv2

    path = tmp_path / "citra.png"
    assert cv2.imwrite(str(path), cv2.cvtColor(_synthetic(400, 400), cv2.COLOR_RGB2BGR))

    result = pp.preprocess(path)

    assert result.tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_preprocess_panels_are_five_stages() -> None:
    """Panel visualisasi harus lima tahap sesuai urutan pipeline."""
    result = pp.preprocess(_synthetic(600, 600, cells=40, seed=13))

    assert len(result.panels) == 5
    assert pp.STAGE_NAMES == ("original", "resized", "normalized", "segment", "watershed")


def test_panels_have_consistent_size() -> None:
    """Empat panel turunan harus 224; panel asli mengikuti ukuran citra sumber.

    Panel asli sengaja tidak diperkecil, karena segmentasi berjalan pada
    resolusi asli dan panel itu dipakai untuk menilai hasil segmentasi.
    """
    result = pp.preprocess(_synthetic(600, 800, cells=40, seed=17))

    assert result.panels[0].shape[:2] == (600, 800)
    for index, panel in enumerate(result.panels[1:], start=1):
        assert panel.shape[:2] == (IMAGE_SIZE, IMAGE_SIZE), f"panel {index}: {panel.shape}"


def test_failing_segmentation_marks_panels() -> None:
    """Panel tahap yang gagal harus ditandai pada hasil."""
    blank = np.full((400, 400, 3), 250, dtype=np.uint8)

    result = pp.preprocess(blank)

    assert result.failed_panels == ("segment", "watershed")


# --- Augmentasi ---


def test_augment_returns_requested_count() -> None:
    """Augmentasi harus menghasilkan jumlah varian yang diminta."""
    image = _synthetic(224, 224)

    variants = pp.augment_train_variants(image, count=4, seed=0)

    assert len(variants) == 4


def test_augment_preserves_shape_and_dtype() -> None:
    """Varian harus bentuk dan tipe sama agar bisa di-resize dan di-normalisasi."""
    image = _synthetic(300, 300)

    variants = pp.augment_train_variants(image, count=3, seed=0)

    for variant in variants:
        assert variant.shape == image.shape
        assert variant.dtype == np.uint8


def test_augment_is_deterministic_with_same_seed() -> None:
    """Seed sama harus menghasilkan varian sama."""
    image = _synthetic(224, 224)

    first = pp.augment_train_variants(image, count=2, seed=42)
    second = pp.augment_train_variants(image, count=2, seed=42)

    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_augment_differs_with_different_seed() -> None:
    """Seed berbeda harus menghasilkan varian berbeda."""
    image = _synthetic(224, 224)

    first = pp.augment_train_variants(image, count=2, seed=1)
    second = pp.augment_train_variants(image, count=2, seed=2)

    assert not all(np.array_equal(a, b) for a, b in zip(first, second))


def test_augment_zero_count_returns_empty() -> None:
    """Jumlah nol harus daftar kosong, bukan error."""
    assert pp.augment_train_variants(_synthetic(100, 100), count=0) == []


def test_augment_rejects_negative_count() -> None:
    """Jumlah negatif harus ditolak."""
    with pytest.raises(ValueError):
        pp.augment_train_variants(_synthetic(100, 100), count=-1)


def test_augment_is_not_reachable_from_preprocess() -> None:
    """preprocess tidak boleh pernah memicu augmentasi.

    Pemeriksaan memakai bytecode, bukan pencarian teks pada docstring, karena
    preprocess memang menyebut augment_train_variants di dalam penjelasannya.
    Yang outlaw adalah pemanggilan fungsi itu, bukan penyebutan namanya.
    """
    import dis
    import inspect

    loaded = [
        instruction.argval
        for instruction in dis.get_instructions(pp.preprocess)
        if instruction.opname == "LOAD_GLOBAL"
    ]
    assert "augment_train_variants" not in loaded

    assert "augment" not in inspect.signature(pp.preprocess).parameters


def test_rotation_stays_within_bound() -> None:
    """Rotasi harus memakai batas dari config, bukan angka di kode."""
    import inspect

    source = inspect.getsource(pp.augment_train_variants)
    assert "AUGMENT_ROTATION_DEGREES" in source
    assert str(AUGMENT_ROTATION_DEGREES) not in source, "angka tidak boleh ditulis langsung"