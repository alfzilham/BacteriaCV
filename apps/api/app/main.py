"""The BacteriaCV web interface with FastAPI, structured as MVC.

- Model: the bacteriacv package (preprocess, model, infer, visualize).
- View: app/views/templates (HTML) and app/views/static (CSS, JS, icons).
- Controller: app/controllers (HTTP routes), helped by app/services.

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

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.controllers import api_controller, page_controller
from app.core.lifespan import build_lifespan
from bacteriacv.config import STATIC_DIR


def create_app(checkpoint_path: Path | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        checkpoint_path: The checkpoint location. Defaults to DEFAULT_CHECKPOINT.

    Returns:
        An application ready to run under uvicorn.
    """
    app = FastAPI(title="BacteriaCV", lifespan=build_lifespan(checkpoint_path))

    # Situs web (Vercel) memanggil API ini dari origin lain. Daftar origin
    # diambil dari CORS_ORIGINS (dipisah koma). Kosong berarti CORS mati.
    origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    app.include_router(page_controller.router)
    app.include_router(api_controller.router)
    return app


app = create_app()
