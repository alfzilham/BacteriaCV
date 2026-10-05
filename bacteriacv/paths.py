"""Project folder and file locations.

Every path is kept relative to the project root so no absolute path can
leak into the code (AGENT.md section 3 rule 5).
"""

from __future__ import annotations

from pathlib import Path

# paths.py lives in bacteriacv/, so parents[0] is bacteriacv and
# parents[1] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ZIPS_DIR = RAW_DIR / "zips"
IMAGES_DIR = RAW_DIR / "images"
MANIFEST_PATH = RAW_DIR / "zips_manifest.csv"
UNREADABLE_PATH = RAW_DIR / "unreadable.csv"
INDEX_PATH = DATA_DIR / "index.csv"


def ensure_data_dirs() -> None:
    """Create the required data folders if they do not exist yet."""
    ZIPS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)