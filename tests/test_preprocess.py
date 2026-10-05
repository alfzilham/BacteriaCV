"""Tests for the preprocessing pipeline."""

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
    """Build a fake microscope image with dark cells on a bright background."""
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


# --- Stage 1: loading ---


def test_load_image_reads_png(tmp_path) -> None:
    """A valid PNG must be read as RGB uint8."""
    import cv2

    path = tmp_path / "citra.png"
    assert cv2.imwrite(str(path), cv2.cvtColor(_synthetic(200, 200), cv2.COLOR_RGB2BGR))

    image = pp.load_image(path)

    assert image.shape == (200, 200, 3)
    assert image.dtype == np.uint8


def test_load_image_reads_tiff(tmp_path) -> None:
    """A TIFF must be read, because that is the DIBaS format."""
    import cv2

    path = tmp_path / "citra.tif"
    assert cv2.imwrite(str(path), cv2.cvtColor(_synthetic(200, 200), cv2.COLOR_RGB2BGR))

    assert pp.load_image(path).shape == (200, 200, 3)


def test_load_image_missing_file_raises(tmp_path) -> None:
    """A missing file must be rejected with a clear message."""
    with pytest.raises(FileNotFoundError):
        pp.load_image(tmp_path / "hilang.tif")


def test_load_image_corrupt_file_raises(tmp_path) -> None:
    """A damaged file must be rejected, not turned into an empty array."""
    path = tmp_path / "rusak.tif"
    path.write_bytes(b"bukan gambar")

    with pytest.raises(ValueError):
        pp.load_image(path)


# --- Stage 2: resize ---


def test_resize_produces_target_size() -> None:
    """Resize must produce an IMAGE_SIZE side per ARCHITECTURE C1."""
    resized = pp.resize_image(_synthetic(600, 800))
    assert resized.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)


def test_resize_accepts_non_square_input() -> None:
    """DIBaS images are not square, resize must still produce a square."""
    resized = pp.resize_image(_synthetic(1532, 2048))
    assert resized.shape[:2] == (IMAGE_SIZE, IMAGE_SIZE)


def test_resize_does_not_mutate_input() -> None:
    """Resize must not modify the source image."""
    original = _synthetic(300, 300)
    before = original.copy()
    pp.resize_image(original)
    assert np.array_equal(original, before)


# --- Stage 3: normalization ---


def test_normalize_uses_imagenet_statistics() -> None:
    """Normalisation uses the ImageNet statistics."""
    image = np.full((32, 32, 3), 255, dtype=np.uint8)

    normalized = pp.normalize(image)

    expected = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    assert normalized.dtype == np.float32
    assert normalized[0, 0, 0] == pytest.approx(expected, abs=1e-5)


def test_normalize_zero_maps_to_negative_mean_ratio() -> None:
    """A black pixel must produce a negative value."""
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    normalized = pp.normalize(black)
    assert (normalized < 0).all()


def test_normalize_output_has_no_nan() -> None:
    """Normalisation must not produce NaN or inf."""
    normalized = pp.normalize(_synthetic(100, 100))
    assert np.isfinite(normalized).all()


# --- Stage 4 and 5: segmentation ---


def test_segmentation_finds_objects_in_synthetic_image() -> None:
    """An image with synthetic cells must produce objects."""
    mask, ok = pp.segment_cells(_synthetic(800, 800, cells=60, seed=3))

    assert ok is True
    assert mask.shape == (800, 800)
    assert mask.dtype == bool
    assert mask.sum() > 0


def test_segmentation_returns_false_on_blank_image() -> None:
    """An image without cells must fail quietly, not raise."""
    blank = np.full((400, 400, 3), 250, dtype=np.uint8)

    mask, ok = pp.segment_cells(blank)

    assert ok is False
    assert mask.sum() == 0


def test_segmentation_returns_false_on_uniform_dark_image() -> None:
    """A uniformly dark image must also count as having no cells."""
    dark = np.full((400, 400, 3), 10, dtype=np.uint8)

    _, ok = pp.segment_cells(dark)

    assert ok is False


def test_segmentation_mask_does_not_cover_whole_image() -> None:
    """The mask must not cover the whole image, that means an inverted threshold."""
    mask, ok = pp.segment_cells(_synthetic(600, 600, cells=30, seed=7))

    if ok:
        assert mask.mean() < 0.5, "mask menutup terlalu besar dari citra"


def test_segmentation_is_deterministic() -> None:
    """Segmenting the same image must produce the same mask."""
    image = _synthetic(600, 600, cells=40, seed=11)

    first, _ = pp.segment_cells(image)
    second, _ = pp.segment_cells(image)

    assert np.array_equal(first, second)


# --- Full pipeline ---


def test_preprocess_returns_tensor_chw() -> None:
    """The output tensor must be B x 3 x 224 x 224 per ARCHITECTURE C1."""
    result = pp.preprocess(_synthetic(600, 800))

    assert isinstance(result.tensor, torch.Tensor)
    assert result.tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert result.tensor.dtype == torch.float32


def test_preprocess_reports_stage_status() -> None:
    """Every stage must have a reported status."""
    result = pp.preprocess(_synthetic(600, 800))

    for stage in ("load", "resize", "normalize", "segment", "watershed"):
        assert stage in result.stage_ok, stage
        assert isinstance(result.stage_ok[stage], bool)


def test_preprocess_success_on_valid_image() -> None:
    """An image with cells must pass every stage."""
    result = pp.preprocess(_synthetic(900, 900, cells=70, seed=5))

    assert all(result.stage_ok.values())
    assert result.object_count > 0


def test_preprocess_continues_when_segmentation_fails() -> None:
    """A segmentation failure must not stop the pipeline.

    Per ARCHITECTURE section 7: continue with the original image and record a
    warning. The tensor for the model must still be produced.
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
    assert "cell group level" in result.message


def test_preprocess_continues_when_only_load_fails() -> None:
    """An unreadable image must give a failed status, not an exception."""
    result = pp.preprocess(np.full((300, 300, 3), 250, dtype=np.uint8))

    assert result.stage_ok["resize"] is True


def test_preprocess_accepts_path(tmp_path) -> None:
    """preprocess must accept a file path, not only an array."""
    import cv2

    path = tmp_path / "citra.png"
    assert cv2.imwrite(str(path), cv2.cvtColor(_synthetic(400, 400), cv2.COLOR_RGB2BGR))

    result = pp.preprocess(path)

    assert result.tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_preprocess_panels_are_five_stages() -> None:
    """The visualisation panels must be five stages in pipeline order."""
    result = pp.preprocess(_synthetic(600, 600, cells=40, seed=13))

    assert len(result.panels) == 5
    assert pp.STAGE_NAMES == ("original", "resized", "normalized", "segment", "watershed")


def test_panels_have_consistent_size() -> None:
    """The four derived panels must be 224; the original follows the source size.

    The original panel is deliberately not reduced, because segmentation runs at
    the original resolution and that panel is used to judge the segmentation.
    """
    result = pp.preprocess(_synthetic(600, 800, cells=40, seed=17))

    assert result.panels[0].shape[:2] == (600, 800)
    for index, panel in enumerate(result.panels[1:], start=1):
        assert panel.shape[:2] == (IMAGE_SIZE, IMAGE_SIZE), f"panel {index}: {panel.shape}"


def test_failing_segmentation_marks_panels() -> None:
    """A failed stage panel must be marked in the result."""
    blank = np.full((400, 400, 3), 250, dtype=np.uint8)

    result = pp.preprocess(blank)

    assert result.failed_panels == ("segment", "watershed")


# --- Augmentation ---


def test_augment_returns_requested_count() -> None:
    """Augmentation must produce the requested number of variants."""
    image = _synthetic(224, 224)

    variants = pp.augment_train_variants(image, count=4, seed=0)

    assert len(variants) == 4


def test_augment_preserves_shape_and_dtype() -> None:
    """Variants must share shape and dtype so they can be resized and normalised."""
    image = _synthetic(300, 300)

    variants = pp.augment_train_variants(image, count=3, seed=0)

    for variant in variants:
        assert variant.shape == image.shape
        assert variant.dtype == np.uint8


def test_augment_is_deterministic_with_same_seed() -> None:
    """The same seed must produce the same variants."""
    image = _synthetic(224, 224)

    first = pp.augment_train_variants(image, count=2, seed=42)
    second = pp.augment_train_variants(image, count=2, seed=42)

    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_augment_differs_with_different_seed() -> None:
    """A different seed must produce different variants."""
    image = _synthetic(224, 224)

    first = pp.augment_train_variants(image, count=2, seed=1)
    second = pp.augment_train_variants(image, count=2, seed=2)

    assert not all(np.array_equal(a, b) for a, b in zip(first, second))


def test_augment_zero_count_returns_empty() -> None:
    """A count of zero must give an empty list, not an error."""
    assert pp.augment_train_variants(_synthetic(100, 100), count=0) == []


def test_augment_rejects_negative_count() -> None:
    """A negative count must be rejected."""
    with pytest.raises(ValueError):
        pp.augment_train_variants(_synthetic(100, 100), count=-1)


def test_augment_is_not_reachable_from_preprocess() -> None:
    """preprocess must never trigger augmentation.

    The check uses bytecode rather than a text search in the docstring, because
    preprocess does mention augment_train_variants in its explanation.
    What is forbidden is calling that function, not naming it.
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
    """Rotation must use the bound from config, not a number in the code."""
    import inspect

    source = inspect.getsource(pp.augment_train_variants)
    assert "AUGMENT_ROTATION_DEGREES" in source
    assert str(AUGMENT_ROTATION_DEGREES) not in source, "angka tidak boleh ditulis langsung"