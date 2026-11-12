"""Logika prediksi untuk antarmuka web, terpisah dari rute HTTP.

Service tidak mengenal FastAPI. Kegagalan validasi dilaporkan lewat
UploadError yang membawa kode status, lalu controller yang mengubahnya
menjadi respons HTTP. Pesan dan kode status sama persis dengan versi
sebelum restrukturisasi MVC.
"""

from __future__ import annotations

import cv2
import numpy as np

from bacteriacv.config import (
    LOW_CONFIDENCE_WARNING,
    LOW_CONFIDENCE_WARNING_KEY,
    MAX_UPLOAD_BYTES,
)
from bacteriacv.infer import check_suffix
from bacteriacv.visualize import encode_png_base64


class UploadError(ValueError):
    """Masukan dari pengguna ditolak, beserta kode status HTTP-nya."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def validate_name(name: str) -> None:
    """Tolak ekstensi berkas yang tidak didukung.

    Raises:
        UploadError: 415 bila ekstensi tidak didukung.
    """
    try:
        check_suffix(name)
    except ValueError as error:
        raise UploadError(415, str(error)) from error


def validate_payload(payload: bytes) -> None:
    """Tolak berkas kosong atau berkas yang melebihi batas ukuran.

    Raises:
        UploadError: 400 bila kosong, 413 bila melebihi MAX_UPLOAD_BYTES.
    """
    if not payload:
        raise UploadError(400, "Empty file")
    if len(payload) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise UploadError(413, f"File exceeds the {limit_mb} MB limit")


def parse_stage(raw: str) -> int:
    """Ubah parameter query stage menjadi angka.

    Raises:
        UploadError: 400 bila bukan angka.
    """
    try:
        return int(raw)
    except ValueError as error:
        raise UploadError(400, "stage parameter must be a number") from error


def decode_image(payload: bytes) -> np.ndarray:
    """Convert file content into an RGB image.

    Args:
        payload: The image file content.

    Returns:
        A uint8 RGB array.

    Raises:
        UploadError: 400 when the file content cannot be read as an image.
    """
    raw = np.frombuffer(payload, dtype=np.uint8)
    decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if decoded is None:
        raise UploadError(400, "File cannot be read as an image")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def build_payload(result, stage_index: int) -> dict:
    """Susun respons JSON prediksi untuk satu tahap yang dipilih.

    Args:
        result: Hasil Predictor.predict.
        stage_index: Indeks tahap yang ditampilkan.

    Returns:
        Kamus respons, dengan field yang sama seperti sebelum restrukturisasi.

    Raises:
        UploadError: 400 bila indeks tahap di luar rentang.
    """
    visualization = result.visualization
    panels = visualization.panels
    if not 0 <= stage_index < len(panels):
        raise UploadError(
            400, f"Stage out of range, must be 0 to {len(panels) - 1}"
        )

    prediction = result.prediction
    return {
        "prediction": prediction.to_dict(),
        "stage_index": stage_index,
        "stage_name": visualization.stage_names[stage_index],
        "stage_key": visualization.title_keys[stage_index],
        "stage_title": visualization.titles[stage_index],
        "stage_names": list(visualization.stage_names),
        # stage_keys are i18n keys, stage_texts the English titles. The
        # client translates the keys and falls back to the texts.
        "stage_keys": list(visualization.title_keys),
        "stage_texts": list(visualization.titles),
        "panel": encode_png_base64(panels[stage_index]),
        # All five panels are sent at once so the toggle does not have to
        # re-upload the image. What is shown on screen stays one.
        "panels": [encode_png_base64(panel) for panel in panels],
        # Same rule as the stages: note_keys for translation,
        # notes_text as the English fallback. notes is a documented alias
        # of notes_text so a client without a dictionary still renders.
        "note_keys": list(prediction.note_keys),
        "notes_text": list(visualization.notes),
        "notes": list(visualization.notes),
        "disclaimer": LOW_CONFIDENCE_WARNING,
        "disclaimer_key": LOW_CONFIDENCE_WARNING_KEY,
        "failed_stages": list(visualization.failed_stages),
    }
