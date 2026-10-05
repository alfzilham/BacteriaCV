"""The BacteriaCV web interface with FastAPI.

DESIGN section 3 establishes one image panel with a stage toggle, not
five panels at once. Therefore predict returns one panel for the selected
stage, not five files.

Important limits:
1. The checkpoint is loaded once at application startup, not per request.
2. A segmentation failure does not stop the prediction, and the stage status is
   sent to the interface so an empty panel can be marked.
3. The segmentation limitation note is always included in the response.
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
    LOW_CONFIDENCE_WARNING_KEY,
    MAX_UPLOAD_BYTES,
    STATIC_DIR,
)
from bacteriacv.infer import DEFAULT_CHECKPOINT, Predictor, check_suffix
from bacteriacv.visualize import encode_png_base64

LOGGER = logging.getLogger("bacteriacv.app")

STAGE_INDEX_FIELD = "stage"
EVALUATION_NAME = "evaluation.json"


class AppState:
    """The shared predictor for every request.

    The model is frozen and has no per image state, so one instance serves
    all requests. Rebuilding per request would load the backbone weights,
    roughly 100 MB, every time.
    """

    def __init__(self, checkpoint_path: Path | None = None) -> None:
        """Load the head checkpoint once.

        Args:
            checkpoint_path: The checkpoint location. Defaults to DEFAULT_CHECKPOINT.

        Raises:
            FileNotFoundError: When the checkpoint does not exist.
        """
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT)
        self.predictor = Predictor(self.checkpoint_path)


def build_lifespan(checkpoint_path: Path | None = None):
    """Create the lifespan manager that loads the checkpoint at startup.

    Args:
        checkpoint_path: The checkpoint location. Defaults to DEFAULT_CHECKPOINT.

    Returns:
        An async context manager function for the FastAPI lifespan argument.
    """

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """Build the Predictor at startup and release it at shutdown.

        Args:
            application: The FastAPI application.

        Yields:
            The application once the Predictor is ready, released at shutdown.
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
    """Build the FastAPI application.

    Args:
        checkpoint_path: The checkpoint location. Defaults to DEFAULT_CHECKPOINT.

    Returns:
        An application ready to run under uvicorn.
    """
    app = FastAPI(title="BacteriaCV", lifespan=build_lifespan(checkpoint_path))

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        """Serve the main page.

        Returns:
            The content of app/static/index.html as HTML.
        """
        page = STATIC_DIR / "index.html"
        if not page.is_file():
            raise HTTPException(status_code=500, detail="index.html not found")
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/api/health")
    async def health(request: Request) -> dict:
        """Report the application status and whether the checkpoint is present.

        Args:
            request: The HTTP request.

        Returns:
            A simple status dictionary.
        """
        state = getattr(request.app.state, "bacteria", None)
        return {
            "status": "siap" if state is not None else "belum siap",
            "checkpoint": state.checkpoint_path.name if state else None,
            "max_upload_bytes": MAX_UPLOAD_BYTES,
        }

    @app.post("/api/predict")
    async def predict(request: Request, file: UploadFile) -> JSONResponse:
        """Predict one image and return one panel for the given stage.

        Args:
            request: The HTTP request.
            file: The uploaded image file.

        Returns:
            JSON holding the prediction, the selected stage, and a base64 PNG panel.

        Raises:
            HTTPException: When predict is not ready, the file is invalid, or
                the image cannot be read.
        """
        state: AppState | None = getattr(request.app.state, "bacteria", None)
        if state is None:
            raise HTTPException(status_code=503, detail="Model is not ready")

        name = file.filename or "unnamed"
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
        except Exception as error:  # noqa: BLE001 - report it, do not leak it
            LOGGER.exception("Prediction failed")
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
                "stage_key": result.visualization.title_keys[stage_index],
                "stage_title": result.visualization.titles[stage_index],
                "stage_names": list(result.visualization.stage_names),
                # stage_keys are i18n keys, stage_texts the English titles. The
                # client translates the keys and falls back to the texts.
                "stage_keys": list(result.visualization.title_keys),
                "stage_texts": list(result.visualization.titles),
                "panel": encode_png_base64(panels[stage_index]),
                # All five panels are sent at once so the toggle does not have to
                # re-upload the image. What is shown on screen stays one.
                "panels": [encode_png_base64(panel) for panel in panels],
                # Same rule as the stages: note_keys for translation,
                # notes_text as the English fallback. notes is a documented alias
                # of notes_text so a client without a dictionary still renders.
                "note_keys": list(prediction.note_keys),
                "notes_text": list(result.visualization.notes),
                "notes": list(result.visualization.notes),
                "disclaimer": LOW_CONFIDENCE_WARNING,
                "disclaimer_key": LOW_CONFIDENCE_WARNING_KEY,
                "failed_stages": list(result.visualization.failed_stages),
            }
        )

    @app.get("/api/report")
    async def report() -> JSONResponse:
        """Serve the most recent evaluation result when available.

        Returns:
            The content of evaluation.json, or 404 when no evaluation has been run.
        """
        path = CHECKPOINT_DIR / EVALUATION_NAME
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Evaluation has not been run yet")
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    return app


def _decode(payload: bytes) -> np.ndarray:
    """Convert file content into an RGB image.

    Args:
        payload: The image file content.

    Returns:
        A uint8 RGB array.

    Raises:
        ValueError: When the file content cannot be read as an image.
    """
    raw = np.frombuffer(payload, dtype=np.uint8)
    decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("File cannot be read as an image")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


app = create_app()
