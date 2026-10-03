"""Fixture bersama untuk tes BacteriaCV.

Folder sementara dibuat di dalam root proyek, bukan di folder temp sistem.
Alasannya,``_relative_posix`` sengaja menolak citra di luar root proyek agar
index.csv tidak pernah memuat path absolut. Folder temp sistem berada di drive
C:, sehingga tidak bisa dipakai untuk membuat citra palsu.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from bacteriacv.paths import PROJECT_ROOT

TEMP_DIR_NAME = ".pytest_tmp"


@pytest.fixture()
def project_tmp_dir() -> Iterator[Path]:
    """Folder sementara di dalam root proyek, dibersihkan setelah tes."""
    target = PROJECT_ROOT / TEMP_DIR_NAME
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    try:
        yield target
    finally:
        shutil.rmtree(target, ignore_errors=True)


@pytest.fixture(autouse=True)
def _cleanup_stray_temp_dir() -> Iterator[None]:
    """Pastikan folder sementara tidak tertinggal bila tes gagal mendadak."""
    yield
    stray = PROJECT_ROOT / TEMP_DIR_NAME
    if stray.is_dir():
        shutil.rmtree(stray, ignore_errors=True)