"""Antarmuka web BacteriaCV dengan FastAPI.

DESIGN bagian 3 menetapkan satu panel citra dengan tombol alih tahap, bukan
lima panel sekaligus. Karena itu predict mengembalikan satu panel untuk tahap
yang dipilih, bukan lima berkas.

Batas penting:
1. Checkpoint dimuat sekali saat aplikasi mulai, bukan per permintaan.
2. Kegagalan segmentasi tidak menghentikan prediksi, dan status tahap dikirim
   ke antarmuka agar panel kosong bisa diberi tanda.
3. Catatan keterbatasan segmentasi selalu ikut dalam respons.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import cv2.utils.logging
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from bacteriacv.config import (
    ALLOWED_SUFFIXES,
    CHECKPOINT_DIR,
    LOW_CONFIDENCE_WARNING,
    MAX_UPLOAD_BYTES,
    STATIC_DIR,
)
from bacteriacv.infer import DEFAULT_CHECKPOINT, Predictor, check_suffix
from bacteriacv.visualize import encode_png_base64

LOGGER = logging.getLogger("bacteriacv.app")

STAGE_INDEX_FIELD = "stage"
EVALUATION_NAME = "evaluation.json"


class AppState:
    """Pemredict bersama untuk seluruh permintaan.

    Model dibekukan dan tidak punya state per citra, jadi satu instans dipakai
    untuk semua permintaan. Membangun ulang per permintaan akan memuat bobot
    backbone sekitar 100 MB setiap kali.
    """

    def __init__(self, checkpoint_path: Path | None = None) -> None:
        """Muat checkpoint head sekali.

        Args:
            checkpoint_path: Lokasi checkpoint. Default-nya DEFAULT_CHECKPOINT.

        Raises:
            FileNotFoundError: Bila checkpoint tidak ada.
        """
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT)
        self.predictor = Predictor(self.checkpoint_path)


def build_lifespan(checkpoint_path: Path | None = None):
    """Buat pengelola siklus hidup yang memuat checkpoint saat aplikasi mulai.

    Args:
        checkpoint_path: Lokasi checkpoint. Bawaannya DEFAULT_CHECKPOINT.

    Returns:
        Fungsi async context manager untuk argumen lifespan FastAPI.
    """

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """Bangun Predictor saat aplikasi mulai dan lepas saat berhenti.

        Args:
            application: Aplikasi FastAPI.

        Yields:
            Aplikasi setelah Predictor siap, lalu dilepas saat berhenti.
        """
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        application.state.bacteria = AppState(checkpoint_path)
        LOGGER.info(
            "Checkpoint dimuat: %s", application.state.bacteria.checkpoint_path.name
        )
        try:
            yield
        finally:
            application.state.bacteria = None

    return lifespan


def create_app(checkpoint_path: Path | None = None) -> FastAPI:
    """Bangun aplikasi FastAPI.

    Args:
        checkpoint_path: Lokasi checkpoint. Bawaannya DEFAULT_CHECKPOINT.

    Returns:
        Aplikasi yang siap dijalankan dengan uvicorn.
    """
    app = FastAPI(title="BacteriaCV", lifespan=build_lifespan(checkpoint_path))

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        """Sajikan halaman utama.

        Returns:
            Isi app/static/index.html sebagai HTML.
        """
        page = STATIC_DIR / "index.html"
        if not page.is_file():
            raise HTTPException(status_code=500, detail="index.html not found")
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/api/health")
    async def health(request: Request) -> dict:
        """Laporkan status aplikasi dan keberadaan checkpoint.

        Args:
            request: Permintaan HTTP.

        Returns:
            Dictionary status sederhana.
        """
        state = getattr(request.app.state, "bacteria", None)
        return {
            "status": "siap" if state is not None else "belum siap",
            "checkpoint": state.checkpoint_path.name if state else None,
            "max_upload_bytes": MAX_UPLOAD_BYTES,
        }

    @app.post("/api/predict")
    async def predict(request: Request, file: UploadFile) -> JSONResponse:
        """Prediksi satu citra dan kembalikan satu panel sesuai tahap.

        Args:
            request: Permintaan HTTP.
            file: Berkas citra yang diunggah.

        Returns:
            JSON berisi prediksi, stage yang dipilih, dan panel PNG basis64.

        Raises:
            HTTPException: Bila predict belum siap, berkas tidak valid, atau
                citra tidak dapat dibaca.
        """
        state: AppState | None = getattr(request.app.state, "bacteria", None)
        if state is None:
            raise HTTPException(status_code=503, detail="Model is not ready")

        name = file.filename or "tanpa_nama"
        try:
            check_suffix(name)
        except ValueError as error:
            raise HTTPException(status_code=415, detail=str(error)) from error

        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        if not payload:
            raise HTTPException(status_code=400, detail="Empty file")
        if len(payload) > MAX_UPLOAD_BYTES:
            limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {limit_mb} MB limit",
            )

        stage = request.query_params.get(STAGE_INDEX_FIELD, "0")
        try:
            stage_index = int(stage)
        except ValueError as error:
            raise HTTPException(
                status_code=400, detail="stage parameter must be a number"
            ) from error

        try:
            image = _decode(payload)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        try:
            result = state.predictor.predict(image)
        except Exception as error:  # noqa: BLE001 - laporkan, jangan tumpahkan
            LOGGER.exception("Prediksi gagal")
            raise HTTPException(status_code=500, detail=str(error)) from error

        panels = result.visualization.panels
        if not 0 <= stage_index < len(panels):
            raise HTTPException(
                status_code=400, detail=f"Stage out of range, must be 0 to {len(panels) - 1}"
            )

        prediction = result.prediction
        return JSONResponse(
            {
                "prediction": prediction.to_dict(),
                "stage_index": stage_index,
                "stage_name": result.visualization.stage_names[stage_index],
                "stage_title": result.visualization.titles[stage_index],
                "stage_names": list(result.visualization.stage_names),
                "stage_titles": list(result.visualization.titles),
                "panel": encode_png_base64(panels[stage_index]),
                # Lima panel dikirim sekaligus supaya tombol alih tidak perlu
                # unggah ulang citra. Yang ditampilkan di layar tetap satu.
                "panels": [encode_png_base64(panel) for panel in panels],
                "notes": list(result.visualization.notes),
                "disclaimer": LOW_CONFIDENCE_WARNING,
                "failed_stages": list(result.visualization.failed_stages),
            }
        )

    @app.get("/api/report")
    async def report() -> JSONResponse:
        """Sajikan hasil evaluasi terakhir bila tersedia.

        Returns:
            Isi evaluation.json, atau 404 bila evaluasi belum dijalankan.
        """
        path = CHECKPOINT_DIR / EVALUATION_NAME
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Evaluation has not been run yet")
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    return app


def _decode(payload: bytes) -> np.ndarray:
    """Ubah isi berkas menjadi citra RGB.

    Args:
        payload: Isi berkas citra.

    Returns:
        Array RGB uint8.

    Raises:
        ValueError: When the file content cannot be read as an image.
    """
    raw = np.frombuffer(payload, dtype=np.uint8)
    decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("File cannot be read as an image")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


app = create_app()
