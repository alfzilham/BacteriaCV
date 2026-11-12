"""Controller API: /api/health, /api/predict, /api/report.

Controller hanya menerjemahkan HTTP. Validasi dan susunan respons ada di
app/services/prediction_service.py, model ada di paket bacteriacv.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.core.lifespan import AppState
from app.services import prediction_service as service
from app.services.prediction_service import UploadError
from bacteriacv.config import CHECKPOINT_DIR, MAX_UPLOAD_BYTES

LOGGER = logging.getLogger("bacteriacv.app")

STAGE_INDEX_FIELD = "stage"
EVALUATION_NAME = "evaluation.json"

router = APIRouter(prefix="/api")


@router.get("/health")
async def health(request: Request) -> dict:
    """Report the application status and whether the checkpoint is present.

    Args:
        request: The HTTP request.

    Returns:
        A simple status dictionary.
    """
    state = getattr(request.app.state, "bacteria", None)
    return {
        "status": "ready" if state is not None else "not ready",
        "checkpoint": state.checkpoint_path.name if state else None,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
    }


@router.post("/predict")
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

    try:
        service.validate_name(file.filename or "unnamed")
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        service.validate_payload(payload)
        stage_index = service.parse_stage(
            request.query_params.get(STAGE_INDEX_FIELD, "0")
        )
        image = service.decode_image(payload)
    except UploadError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from error

    try:
        result = state.predictor.predict(image)
    except Exception as error:  # noqa: BLE001 - report it, do not leak it
        LOGGER.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(error)) from error

    try:
        body = service.build_payload(result, stage_index)
    except UploadError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from error
    return JSONResponse(body)


@router.get("/report")
async def report() -> JSONResponse:
    """Serve the most recent evaluation result when available.

    Returns:
        The content of evaluation.json, or 404 when no evaluation has been run.
    """
    path = CHECKPOINT_DIR / EVALUATION_NAME
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Evaluation has not been run yet")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
