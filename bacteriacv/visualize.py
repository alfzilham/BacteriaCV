"""Penyusun panel visualisasi untuk antarmuka web.

DESIGN bagian 3 menetapkan satu panel dengan tombol alih, bukan lima panel
sekaligus. Modul ini karena itu menyiapkan kelima panel sebagai berkas terpisah
lalu membakar annotate ke masing-masing, supaya tombol alih tidak perlu
menulis ulang teks setiap kali panel berganti.

Aturan yang tidak boleh dilanggar: panel wajib menulis bahwa klasifikasi bentuk
belum tervalidasi dari objek hasil segmentasi. Angka pada catatan itu berasal
dari eksperimen pada data DIBaS dan tidak boleh diganti tanpa eksperimen baru.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np

from .config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    LOW_CONFIDENCE_WARNING,
    SHAPE_LABELS,
)
from .preprocess import STAGE_NAMES, PreprocessResult

STAGE_TITLES = {
    "original": "Citra asal",
    "resized": "Resize 224 x 224",
    "normalized": "Normalisasi ImageNet",
    "segment": "Mask segmentasi",
    "watershed": "Batas objek",
}

# Bukti dari eksperimen segmentasi pada DIBaS. Angka ini hasil pengukuran,
# bukan estimasi. Elongasi memakai metrik major/minor axis.
SEGMENTATION_LIMITATION = (
    "Bentuk sel belum tervalidasi dari segmentasi: elongasi median kokus "
    "1.30-1.43 dan batang 1.64-1.76, tetapi rentang intraspesies 1.03-4.19 "
    "melampaui selisih antargrup 0.068."
)

FONT_SCALE = 0.42
FONT_THICKNESS = 1
TEXT_COLOR = (255, 255, 255)
STRIP_COLOR = (0, 0, 0)
LINE_HEIGHT = 18


@dataclass(frozen=True)
class VisualizationBundle:
    """Kelima panel yang sudah diberi annotate.

    Attributes:
        panels: Lima citra RGB berannotasi, urut sesuai stage_names.
        stage_names: Kunci tahap untuk tombol alih.
        titles: Judul tahap sesuai stage_names.
        failed_stages: Tahap yang gagal, disorot di antarmuka.
        object_count: Jumlah objek segmentasi, nol bila gagal.
        notes: Catatan yang harus tampil di bawah panel.
    """

    panels: tuple[np.ndarray, ...]
    stage_names: tuple[str, ...]
    titles: tuple[str, ...]
    failed_stages: tuple[str, ...]
    object_count: int
    notes: tuple[str, ...]


def confidence_level(value: float) -> str:
    """Ubah confidence menjadi level verbal.

    Args:
        value: Confidence antara nol dan satu.

    Returns:
        "tinggi", "sedang", atau "rendah".
    """
    if value >= CONFIDENCE_HIGH:
        return "tinggi"
    if value >= CONFIDENCE_MEDIUM:
        return "sedang"
    return "rendah"


def annotate(
    panel: np.ndarray, lines: list[str], strip_alpha: float = 0.65
) -> np.ndarray:
    """Bakar teks ke panel dengan latar gelap.

    Args:
        panel: Array RGB uint8.
        lines: Baris teks yang ditulis di kiri atas.
        strip_alpha: Kelembutan latar, nol berarti transparan.

    Returns:
        Salinan panel dengan teks terburna.

    Raises:
        ValueError: Bila lines kosong.
    """
    if not lines:
        raise ValueError("Setidaknya satu baris annotate wajib ada.")

    result = np.ascontiguousarray(panel.copy())
    height = LINE_HEIGHT * len(lines)
    strip = result[:height].astype(np.float32)
    result[:height] = (
        strip * (1.0 - strip_alpha) + np.array(STRIP_COLOR, dtype=np.float32) * strip_alpha
    ).astype(np.uint8)

    for index, line in enumerate(lines):
        cv2.putText(
            result,
            line,
            (6, LINE_HEIGHT * index + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )
    return result


def encode_png(panel: np.ndarray) -> bytes:
    """Ubah panel RGB menjadi PNG di memori.

    Args:
        panel: Array RGB uint8.

    Returns:
        Isi berkas PNG.
    """
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Panel tidak dapat dikodekan menjadi PNG.")
    return buffer.tobytes()


def encode_png_base64(panel: np.ndarray) -> str:
    """Ubah panel RGB menjadi PNG basis64 untuk disisipkan ke HTML.

    Args:
        panel: Array RGB uint8.

    Returns:
        Teks PNG basis64 tanpa prefiks data URI.
    """
    return base64.b64encode(encode_png(panel)).decode("ascii")


def _headline(
    shape_label: str,
    shape_confidence: float,
    gram_label: str,
    gram_confidence: float,
) -> str:
    """Susun baris ringkasan prediksi untuk annotate."""
    return (
        f"Bentuk: {shape_label} ({shape_confidence:.0%}, "
        f"{confidence_level(shape_confidence)})   "
        f"Gram: {gram_label} ({gram_confidence:.0%}, "
        f"{confidence_level(gram_confidence)})"
    )


def build_visualization(
    result: PreprocessResult,
    shape_label: str,
    shape_confidence: float,
    gram_label: str,
    gram_confidence: float,
) -> VisualizationBundle:
    """Bangun kelima panel berannotasi dari hasil pra-pemrosesan.

    Args:
        result: Hasil preprocess.
        shape_label: Nama kelas bentuk hasil prediksi.
        shape_confidence: Confidence prediksi bentuk.
        gram_label: Nama kelas status Gram hasil prediksi.
        gram_confidence: Confidence prediksi Gram.

    Returns:
        VisualizationBundle siap dikirim ke antarmuka.

    Raises:
        ValueError: Bila confidence di luar rentang nol sampai satu.
    """
    for name, value in (
        ("shape_confidence", shape_confidence),
        ("gram_confidence", gram_confidence),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} harus antara 0 dan 1, dapat {value}")

    notes: list[str] = [LOW_CONFIDENCE_WARNING, SEGMENTATION_LIMITATION]
    if result.failed_panels:
        notes.insert(
            0,
            "Tahap gagal: " + ", ".join(result.failed_panels) + ". "
            "Panel terkait dikosongkan, klasifikasi tetap dihitung.",
        )

    base = [_headline(shape_label, shape_confidence, gram_label, gram_confidence)]
    base.extend(notes)

    titles: list[str] = []
    panels: list[np.ndarray] = []
    for stage, panel in zip(STAGE_NAMES, result.panels):
        title = STAGE_TITLES[stage]
        if stage in result.failed_panels:
            title = f"{title} (gagal)"
        titles.append(title)
        panels.append(annotate(panel, base[:2] + [title] + base[2:4]))

    return VisualizationBundle(
        panels=tuple(panels),
        stage_names=STAGE_NAMES,
        titles=tuple(titles),
        failed_stages=result.failed_panels,
        object_count=result.object_count,
        notes=tuple(notes),
    )


def panel_sizes(bundle: VisualizationBundle) -> list[tuple[int, int]]:
    """Kembalikan ukuran tiap panel untuk pemeriksaan antarmuka.

    Args:
        bundle: Bundle visualisasi.

    Returns:
        Daftar pasangan (tinggi, lebar) untuk tiap panel.
    """
    return [panel.shape[:2] for panel in bundle.panels]


def shape_label_text(index: int) -> str:
    """Ubah indeks kelas bentuk menjadi label, dengan fallback aman."""
    if 0 <= index < len(SHAPE_LABELS):
        return SHAPE_LABELS[index]
    return "tidak_diketahui"
