"""Lokasi folder dan berkas proyek.

Semua path disimpan relatif terhadap root proyek agar tidak ada path absolut
yang bocor ke dalam kode (AGENT.md bagian 3 aturan 5).
"""

from __future__ import annotations

from pathlib import Path

# paths.py berada di bacteriacv/, sehingga parents[0] adalah bacteriacv dan
# parents[1] adalah root proyek.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ZIPS_DIR = RAW_DIR / "zips"
IMAGES_DIR = RAW_DIR / "images"
MANIFEST_PATH = RAW_DIR / "zips_manifest.csv"
UNREADABLE_PATH = RAW_DIR / "unreadable.csv"
INDEX_PATH = DATA_DIR / "index.csv"


def ensure_data_dirs() -> None:
    """Buat folder data yang dibutuhkan bila belum ada."""
    ZIPS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)