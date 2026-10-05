"""Preprocessing pipeline for bacterial microscope images.

The training and inference stages share this module, per ARCHITECTURE
section 1. The only difference is augmentation: augmentation may only be
triggered through augment_train_variants, which preprocess never calls.
preprocess has no augmentation parameter, so the inference path has no way
to trigger it.

Segmentation runs at the original resolution, not at 224 x 224. The reason
is scale dependent: at the original resolution one pixel is about
0.048 microns, so a 1 micron bacterial cell is about 21 pixels across. At
224 x 224 one pixel is about 0.43 microns, so the same cell is only 2.3
pixels and a 5 pixel morphological footprint is almost as large as the
cell it is meant to separate. The consequence is written up in
docs/SPEC.md section 8.

A failure in any stage does not stop the pipeline. The stage is marked as
failed, its panel stays empty, and the model tensor is still produced, per
ARCHITECTURE section 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage import morphology
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from .config import (
    AUGMENT_HORIZONTAL_FLIP,
    AUGMENT_ROTATION_DEGREES,
    AUGMENT_SCALE_RANGE,
    AUGMENT_SHIFT_FRACTION,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    SEGMENT_DILATION_DISK,
    SEGMENT_EROSION_DISK,
    SEGMENT_MIN_OBJECT_AREA,
    SEGMENT_MIN_PEAK_DISTANCE,
)

STAGE_NAMES = ("original", "resized", "normalized", "segment", "watershed")

BLUR_KERNEL = (5, 5)

SEGMENTATION_FAILURE_MESSAGE = (
    "Segmentasi pada level kelompok sel, bukan sel individual; "
    "sel bersentuhan pada citra Gram 100x."
)


@dataclass(frozen=True)
class PreprocessResult:
    """Result of the preprocessing pipeline.

    Attributes:
        tensor: A 3 x 224 x 224 tensor ready for the backbone.
        panels: Five RGB images for visualisation, ordered per STAGE_NAMES.
        stage_ok: Success status of each stage.
        failed_panels: Names of the stages that failed, for marking in the interface.
        object_count: Number of objects from segmentation, zero when it failed.
        message: The warning to show the user, None when there is none.
    """

    tensor: torch.Tensor
    panels: tuple[np.ndarray, ...]
    stage_ok: dict[str, bool]
    failed_panels: tuple[str, ...]
    object_count: int
    message: str | None = None


def load_image(path: Path | str) -> np.ndarray:
    """Read an image from disk as an RGB array.

    Args:
        path: Image file location.

    Returns:
        Array RGB bertipe uint8.

    Raises:
        FileNotFoundError: When the file does not exist.
        ValueError: When the file cannot be read as an image.
    """
    location = Path(path)
    if not location.is_file():
        raise FileNotFoundError(f"Citra tidak ditemukan: {location.name}")

    raw = np.frombuffer(location.read_bytes(), dtype=np.uint8)
    decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError(f"Citra tidak dapat dibaca: {location.name}")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _as_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure the image is uint8 RGB without modifying the original."""
    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if array.ndim == 3 and array.shape[2] == 4:
        return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_BGRA2RGB)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Bentuk citra tidak didukung: {array.shape}")
    return array.astype(np.uint8, copy=False)


def resize_image(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize the image to a square size.

    Args:
        image: The source image array.
        size: Sisi target dalam piksel.

    Returns:
        An RGB array with side length size.
    """
    return cv2.resize(_as_rgb(image), (size, size), interpolation=cv2.INTER_AREA)


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalise intensity using the ImageNet statistics.

    Args:
        image: Array RGB uint8.

    Returns:
        A float32 array ranging roughly from minus two to two.
    """
    array = _as_rgb(image).astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    return (array - mean) / std


def segment_cells(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Segment bacterial areas with opening, watershed, and an area filter.

    Args:
        image: Array RGB pada resolusi asli.

    Returns:
        A (binary_mask, succeeded) tuple. When no object meets the
        area threshold, the mask is empty and succeeded is False.
    """
    array = _as_rgb(image)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    mask = binary > 0
    if mask.mean() > 0.5:
        mask = ~mask
    if mask.sum() == 0:
        return np.zeros(gray.shape, dtype=bool), False

    eroded = morphology.erosion(mask, morphology.disk(SEGMENT_EROSION_DISK))
    opened = morphology.dilation(eroded, morphology.disk(SEGMENT_DILATION_DISK))
    if opened.sum() == 0:
        return np.zeros(gray.shape, dtype=bool), False

    distance = ndi.distance_transform_edt(opened)
    peaks = peak_local_max(
        distance, min_distance=SEGMENT_MIN_PEAK_DISTANCE, labels=opened
    )
    if len(peaks) == 0:
        return np.zeros(gray.shape, dtype=bool), False

    markers = np.zeros(opened.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = watershed(-distance, markers, mask=opened)

    # Filter luas tanpa loop regionprops. Versi loop membandingkan seluruh
    # labelled image with one label per region, so it costs
    # O(num_regions x image_size). On a 2048 x 1532 DIBaS image with
    # hundreds of regions that is several seconds per image. Computing through
    # np.unique and np.bincount yields exactly the same mask.
    flat = labels.ravel()
    unique_labels, inverse = np.unique(flat, return_inverse=True)
    sizes = np.bincount(inverse, minlength=unique_labels.size)
    sizes[unique_labels == 0] = 0

    kept = unique_labels[sizes >= SEGMENT_MIN_OBJECT_AREA]
    keep = np.isin(labels, kept)
    return keep, bool(keep.any())


def count_objects(mask: np.ndarray | None) -> int:
    """Count connected components on the segmentation mask.

    Args:
        mask: The binary mask from segment_cells, or None when segmentation failed.

    Returns:
        Jumlah komponen terhubung berlabel.
    """
    if mask is None or not mask.any():
        return 0
    labeled, count = ndi.label(mask)
    return int(count)


def _mask_panel(mask: np.ndarray | None, size: int = IMAGE_SIZE) -> np.ndarray:
    """Build an RGB image from a binary mask at the panel size.

    Args:
        mask: An original resolution binary mask, or None when segmentation failed.
        size: The desired panel side length.

    Returns:
        An RGB array of size by size, or an empty array when mask is None.
    """
    if mask is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    scaled = cv2.resize(
        mask.astype(np.uint8) * 255, (size, size), interpolation=cv2.INTER_NEAREST
    )
    return np.repeat(scaled[..., None], 3, axis=2)


def _boundary_panel(
    mask: np.ndarray | None, base: np.ndarray, size: int = IMAGE_SIZE
) -> np.ndarray:
    """Build an RGB image with the mask outline over the base image."""
    if mask is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    panel = cv2.resize(base, (size, size), interpolation=cv2.INTER_AREA)
    shrunk = cv2.resize(
        (mask.astype(np.uint8) * 255),
        (size, size),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    if not shrunk.any():
        return panel
    edges = shrunk ^ cv2.erode(
        shrunk.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
    ).astype(bool)
    panel[edges] = (255, 255, 255)
    return panel


def _to_tensor(normalized: np.ndarray) -> torch.Tensor:
    """Convert a normalised HWC array into a float32 CHW tensor."""
    chw = np.ascontiguousarray(normalized.transpose(2, 0, 1))
    return torch.from_numpy(chw).float()


def preprocess(image: np.ndarray | Path | str) -> PreprocessResult:
    """Run the preprocessing pipeline without augmentation.

    This function has no augmentation parameter and never calls
    augment_train_variants, so the inference path cannot trigger
    augmentation by accident.

    Args:
        image: An RGB array, or an image file path.

    Returns:
        A PreprocessResult holding the tensor, five panels, and the stage status.

    Raises:
        FileNotFoundError: When a path is given but the file does not exist.
        ValueError: When the image shape is not supported.
    """
    stage_ok: dict[str, bool] = {}
    failed: list[str] = []

    source = load_image(image) if isinstance(image, (str, Path)) else _as_rgb(image)
    stage_ok["load"] = True

    resized = resize_image(source)
    stage_ok["resize"] = resized.shape[:2] == (IMAGE_SIZE, IMAGE_SIZE)
    if not stage_ok["resize"]:
        failed.append("resized")

    normalized = normalize(resized)
    stage_ok["normalize"] = bool(np.isfinite(normalized).all())
    if not stage_ok["normalize"]:
        failed.append("normalized")

    mask, segmented = segment_cells(source)
    stage_ok["segment"] = segmented
    stage_ok["watershed"] = segmented
    if not segmented:
        failed.extend(["segment", "watershed"])
        mask = None
    count = count_objects(mask)

    panels = (
        source,
        resized,
        _denormalized_panel(normalized),
        _mask_panel(mask),
        _boundary_panel(mask, source),
    )

    return PreprocessResult(
        tensor=_to_tensor(normalized),
        panels=panels,
        stage_ok=stage_ok,
        failed_panels=tuple(failed),
        object_count=count,
        message=SEGMENTATION_FAILURE_MESSAGE if not segmented else None,
    )


def preprocess_tensor(image: np.ndarray | Path | str) -> torch.Tensor:
    """Build only the model tensor, without segmentation and without panels.

    This path is used by feature extraction for training. Segmentation is skipped
    because ARCHITECTURE section 3 states segmentation is not needed for inference:
    shape and Gram status classification comes from the image, not the mask.
    Segmentation stays in preprocess for the visualisation panels.

    On a 2048 x 1532 DIBaS image, segmentation takes about 2.5 seconds.
    Running it once per augmented image makes feature extraction for
    2070 rows takes more than an hour without improving the model metrics
    at all.

    Args:
        image: An RGB array, or an image file path.

    Returns:
        A 3 x 224 x 224 tensor ready for the backbone.

    Raises:
        FileNotFoundError: When a path is given but the file does not exist.
        ValueError: When the image shape is not supported.
    """
    source = load_image(image) if isinstance(image, (str, Path)) else _as_rgb(image)
    return _to_tensor(normalize(resize_image(source)))


def _denormalized_panel(normalized: np.ndarray) -> np.ndarray:
    """Return the normalised image to the uint8 range for display."""
    array = normalized * np.array(IMAGENET_STD, dtype=np.float32)
    array = array + np.array(IMAGENET_MEAN, dtype=np.float32)
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def augment_train_variants(
    image: np.ndarray, count: int, seed: int | None = None
) -> list[np.ndarray]:
    """Build augmentation variants for the train data only.

    This function is never called by preprocess, so augmentation does not
    leak into the inference path.

    Args:
        image: A uint8 RGB image array.
        count: The number of variants requested.
        seed: A seed so the result is reproducible.

    Returns:
        A list of image variants, each with the same shape and dtype as the
        source image.

    Raises:
        ValueError: When count is negative.
    """
    if count < 0:
        raise ValueError(f"Jumlah varian tidak boleh negatif: {count}")
    if count == 0:
        return []

    source = _as_rgb(image)
    rng = np.random.default_rng(seed)
    height, width = source.shape[:2]
    variants: list[np.ndarray] = []

    for _ in range(count):
        angle = float(rng.uniform(-AUGMENT_ROTATION_DEGREES, AUGMENT_ROTATION_DEGREES))
        scale = float(rng.uniform(*AUGMENT_SCALE_RANGE))
        shift_x = float(rng.uniform(-AUGMENT_SHIFT_FRACTION, AUGMENT_SHIFT_FRACTION)) * width
        shift_y = float(rng.uniform(-AUGMENT_SHIFT_FRACTION, AUGMENT_SHIFT_FRACTION)) * height

        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
        matrix[0, 2] += shift_x
        matrix[1, 2] += shift_y

        rotated = cv2.warpAffine(
            source, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101
        )
        if AUGMENT_HORIZONTAL_FLIP and rng.random() < 0.5:
            rotated = cv2.flip(rotated, 1)
        variants.append(rotated)

    return variants