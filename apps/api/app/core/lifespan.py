"""Siklus hidup aplikasi: checkpoint dimuat sekali saat startup.

Isi modul ini dipindahkan apa adanya dari app/main.py saat restrukturisasi MVC.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import cv2.utils.logging
from fastapi import FastAPI

from bacteriacv.infer import DEFAULT_CHECKPOINT, Predictor

LOGGER = logging.getLogger("bacteriacv.app")


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
            "Checkpoint loaded: %s", application.state.bacteria.checkpoint_path.name
        )
        try:
            yield
        finally:
            application.state.bacteria = None

    return lifespan
