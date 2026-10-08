"""Shared fixtures for the BacteriaCV tests.

The temporary folder is created inside the project root, not in the system temp folder.
The reason is that ``_relative_posix`` deliberately rejects images outside the project root so
index.csv can never hold an absolute path. The system temp folder is on the C: drive,
so it cannot be used to build fake images.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from bacteriacv.paths import PROJECT_ROOT

# The pid keeps two concurrent pytest processes in this repo from deleting
# each other's folder. The name must be stable within a process and unique
# across processes, so the pid is the only discriminator that works.
TEMP_DIR_NAME = f".pytest_tmp_{os.getpid()}"


@pytest.fixture()
def project_tmp_dir() -> Iterator[Path]:
    """A temporary folder inside the project root, cleaned up after the tests."""
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
    """Make sure no temporary folder is left behind if a test dies unexpectedly."""
    yield
    stray = PROJECT_ROOT / TEMP_DIR_NAME
    if stray.is_dir():
        shutil.rmtree(stray, ignore_errors=True)