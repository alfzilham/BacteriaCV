"""Controller halaman: menyajikan template utama."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from bacteriacv.config import TEMPLATES_DIR

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the main page.

    Returns:
        The content of app/views/templates/index.html as HTML.
    """
    page = TEMPLATES_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=500, detail="index.html not found")
    return HTMLResponse(page.read_text(encoding="utf-8"))
