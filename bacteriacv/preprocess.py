"""Pipeline pra-pemrosesan untuk citra mikroskopis bakteri.

Modul ini dipakai bersama oleh tahap pelatihan dan tahap inferensi, sesuai
ARCHITECTURE bagian 1. Kebedaan keduanya hanya augmentasi: augmentasi boleh
dipanggil hanya lewat augment_train_variants, yang tidak pernah dipanggil
preprocess. Tidak ada parameter augmentasi pada preprocess, sehingga jalur
inferensi tidak punya jalan untuk memicunya.

Segmentasi berjalan pada resolusi asli, bukan pada 224 x 224. Alasannya
bersifat skala: pada resolusi asli satu piksel setara sekitar 0,048 mikron,
sehingga sel bakteri 1 mikron berdiameter sekitar 21 piksel. Pada 224 x 224
satu piksel setara sekitar 0,43 mikron, sehingga sel yang sama hanya 2,3
piksel dan footprint morfologis berukuran 5 piksel hampir sama besar dengan
sel yang hendak dipisahkan. Konsekuensinya ditulis di docs/SPEC.md bagian 8.

Kegagalan tahap manapun tidak menghentikan pipeline. Tahap ditandai gagal,
panelnya kosong, dan tensor untuk model tetap terbentuk, sesuai ARCHITECTURE
bagian 7.
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
    """Hasil pipeline pra-pemrosesan.

    Attributes:
        tensor: Tensor 3 x 224 x 224 siap masuk backbone.
        panels: Lima citra RGB untuk visualisasi, urut sesuai STAGE_NAMES.
        stage_ok: Status keberhasilan tiap tahap.
        failed_panels: Nama tahap yang gagal, untuk penandaan di antarmuka.
        object_count: Jumlah objek hasil segmentasi, nol bila gagal.
        message: Peringatan yang harus ditampilkan pengguna, None bila tidak ada.
    """

    tensor: torch.Tensor
    panels: tuple[np.ndarray, ...]
    stage_ok: dict[str, bool]
    failed_panels: tuple[str, ...]
    object_count: int
    message: str | None = None


def load_image(path: Path | str) -> np.ndarray:
    """Baca citra dari disk sebagai array RGB.

    Args:
        path: Lokasi berkas citra.

    Returns:
        Array RGB bertipe uint8.

    Raises:
        FileNotFoundError: Bila berkas tidak ada.
        ValueError: Bila berkas tidak dapat dibaca sebagai citra.
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
    """Pastikan citra berupa RGB uint8 tanpa mengubah aslinya."""
    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if array.ndim == 3 and array.shape[2] == 4:
        return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_BGRA2RGB)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Bentuk citra tidak didukung: {array.shape}")
    return array.astype(np.uint8, copy=False)


def resize_image(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize citra ke ukuran persegi.

    Args:
        image: Array citra sumber.
        size: Sisi target dalam piksel.

    Returns:
        Array RGB dengan sisi size.
    """
    return cv2.resize(_as_rgb(image), (size, size), interpolation=cv2.INTER_AREA)


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalisasi intensitas dengan statistik ImageNet.

    Args:
        image: Array RGB uint8.

    Returns:
        Array float32 dengan rentang sekitar minus dua sampai dua.
    """
    array = _as_rgb(image).astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    return (array - mean) / std


def segment_cells(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Segmentasi area bakteri dengan opening, watershed, dan filter luas.

    Args:
        image: Array RGB pada resolusi asli.

    Returns:
        Pasangan (mask_biner, berhasil). Bila tidak ada objek yang memenuhi
        ambang luas, mask kosong dan berhasil bernilai False.
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
    # citra berlabel dengan satu label untuk tiap region, sehingga biayanya
    # O(jumlah_region x ukuran_citra). Pada citra DIBaS 2048 x 1532 dengan
    # ratusan region, itu beberapa detik per citra. Penghitungan lewat
    # np.unique dan np.bincount menghasilkan mask yang sama persis.
    flat = labels.ravel()
    unique_labels, inverse = np.unique(flat, return_inverse=True)
    sizes = np.bincount(inverse, minlength=unique_labels.size)
    sizes[unique_labels == 0] = 0

    kept = unique_labels[sizes >= SEGMENT_MIN_OBJECT_AREA]
    keep = np.isin(labels, kept)
    return keep, bool(keep.any())


def count_objects(mask: np.ndarray | None) -> int:
    """Hitung komponen terhubung pada mask segmentasi.

    Args:
        mask: Mask biner dari segment_cells, atau None bila segmentasi gagal.

    Returns:
        Jumlah komponen terhubung berlabel.
    """
    if mask is None or not mask.any():
        return 0
    labeled, count = ndi.label(mask)
    return int(count)


def _mask_panel(mask: np.ndarray | None, size: int = IMAGE_SIZE) -> np.ndarray:
    """Bentuk citra RGB dari mask biner pada ukuran panel.

    Args:
        mask: Mask biner resolusi asli, atau None bila segmentasi gagal.
        size: Sisi panel yang diinginkan.

    Returns:
        Array RGB ukuran size kali size, atau array kosong bila mask None.
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
    """Bentuk citra RGB dengan garis batas mask di atas citra dasar."""
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
    """Ubah array HWC ternormalisasi menjadi tensor CHW float32."""
    chw = np.ascontiguousarray(normalized.transpose(2, 0, 1))
    return torch.from_numpy(chw).float()


def preprocess(image: np.ndarray | Path | str) -> PreprocessResult:
    """Jalankan pipeline pra-pemrosesan tanpa augmentasi.

    Fungsi ini tidak punya parameter augmentasi dan tidak memanggil
    augment_train_variants, sehingga jalur inferensi tidak dapat memicu
    augmentasi secara tidak sengaja.

    Args:
        image: Array citra RGB, atau path berkas citra.

    Returns:
        PreprocessResult berisi tensor, lima panel, dan status tiap tahap.

    Raises:
        FileNotFoundError: Bila path diberikan tapi berkasnya tidak ada.
        ValueError: Bila bentuk citra tidak didukung.
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
    """Bangun tensor model saja, tanpa segmentasi dan tanpa panel.

    Jalur ini dipakai ekstraksi fitur untuk pelatihan. Segmentasi dilewati
    karena ARCHITECTURE bagian 3 menyatakan segmentasi tidak dibutuhkan inferensi:
    klasifikasi bentuk dan status Gram berasal dari citra, bukan dari mask.
    Segmentasi tetap ada di preprocess untuk kebutuhan panel visualisasi.

    Pada citra DIBaS 2048 x 1532, segmentasi memakan sekitar 2,5 detik.
    Menjalankannya sekali per citra_augmented membuat ekstraksi fitur untuk
    2070 baris memakan lebih dari satu jam, tanpa memperbaiki metrik model
    sedikit pun.

    Args:
        image: Array citra RGB, atau path berkas citra.

    Returns:
        Tensor 3 x 224 x 224 siap masuk backbone.

    Raises:
        FileNotFoundError: Bila path diberikan tapi berkasnya tidak ada.
        ValueError: Bila bentuk citra tidak didukung.
    """
    source = load_image(image) if isinstance(image, (str, Path)) else _as_rgb(image)
    return _to_tensor(normalize(resize_image(source)))


def _denormalized_panel(normalized: np.ndarray) -> np.ndarray:
    """Kembalikan citra ternormalisasi ke rentang uint8 untuk ditampilkan."""
    array = normalized * np.array(IMAGENET_STD, dtype=np.float32)
    array = array + np.array(IMAGENET_MEAN, dtype=np.float32)
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def augment_train_variants(
    image: np.ndarray, count: int, seed: int | None = None
) -> list[np.ndarray]:
    """Buat varian augmentasi untuk data latih saja.

    Fungsi ini tidak pernah dipanggil preprocess, sehingga augmentasi tidak
    dapat bocor ke jalur inferensi.

    Args:
        image: Array citra RGB uint8.
        count: Jumlah varian yang diminta.
        seed: Seed agar hasil dapat direproduksi.

    Returns:
        Daftar varian citra, masing-masing dengan bentuk dan tipe sama seperti
        citra sumber.

    Raises:
        ValueError: Bila count negatif.
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