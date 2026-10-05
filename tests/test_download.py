"""Unit tests for the download logic, the manifest, and error handling."""

from __future__ import annotations

import csv
import hashlib
import zipfile
from pathlib import Path

import pytest

from bacteriacv.datasets import download as download_module
from bacteriacv.datasets.download import (
    MANIFEST_FIELDS,
    DownloadResult,
    _discard_partial,
    download_species,
    sha256_of,
    write_manifest,
)
from bacteriacv.datasets.species_map import SPECIES

SAMPLE = SPECIES[0]


def _write_zip(path: Path, names: list[str], payload: bytes = b"isi") -> None:
    """Build a fake ZIP archive for the tests."""
    with zipfile.ZipFile(path, "w") as bundle:
        for name in names:
            bundle.writestr(name, payload)


def test_sha256_of_matches_hashlib(tmp_path: Path) -> None:
    """The computed hash must match hashlib directly."""
    target = tmp_path / "berkas.bin"
    target.write_bytes(b"abc" * 5000)
    assert sha256_of(target) == hashlib.sha256(b"abc" * 5000).hexdigest()


def test_download_species_skips_existing_file(tmp_path: Path) -> None:
    """An archive that already exists must not be downloaded again."""
    zips = tmp_path / "zips"
    zips.mkdir()
    _write_zip(zips / f"{SAMPLE.zip_name}.zip", ["a.tif"])

    result = download_species(SAMPLE, zips)

    assert result.skipped is True
    assert result.path.is_file()


def test_download_species_rejects_empty_file(tmp_path: Path) -> None:
    """A zero byte archive is not a valid download and must be fetched again."""
    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / f"{SAMPLE.zip_name}.zip").write_bytes(b"")

    called: list[str] = []
    original = download_module._download_once

    def fake_download(url: str, target: Path) -> None:
        called.append(url)
        _write_zip(target, ["a.tif"])

    download_module._download_once = fake_download
    try:
        result = download_species(SAMPLE, zips)
    finally:
        download_module._download_once = original

    assert called == [SAMPLE.url]
    assert result.skipped is False


def test_download_species_retries_then_succeeds(tmp_path: Path) -> None:
    """The first attempt fails, the second attempt succeeds."""
    zips = tmp_path / "zips"
    zips.mkdir()
    attempts: list[str] = []

    def flaky_download(url: str, target: Path) -> None:
        attempts.append(url)
        if len(attempts) == 1:
            raise TimeoutError("koneksi menganggur")
        _write_zip(target, ["a.tif"])

    original = download_module._download_once
    download_module._download_once = flaky_download
    try:
        result = download_species(SAMPLE, zips)
    finally:
        download_module._download_once = original

    assert len(attempts) == 2
    assert result.skipped is False


def test_download_species_raises_after_all_retries(tmp_path: Path) -> None:
    """A total failure must be raised as a RuntimeError."""
    zips = tmp_path / "zips"
    zips.mkdir()
    attempts: list[str] = []

    def always_fails(url: str, target: Path) -> None:
        attempts.append(url)
        raise TimeoutError("selalu gagal")

    original = download_module._download_once
    original_wait = download_module._RETRY_WAIT_SEC
    download_module._download_once = always_fails
    download_module._RETRY_WAIT_SEC = 0
    try:
        with pytest.raises(RuntimeError, match="Failed to download"):
            download_species(SAMPLE, zips)
    finally:
        download_module._download_once = original
        download_module._RETRY_WAIT_SEC = original_wait

    assert len(attempts) == download_module._RETRY


def test_discard_partial_does_not_raise_when_locked(tmp_path: Path) -> None:
    """Cleaning up leftover downloads must not raise a new error."""
    partial = tmp_path / f"{SAMPLE.zip_name}.zip.part"
    partial.write_bytes(b"sebagian")

    _discard_partial(tmp_path / f"{SAMPLE.zip_name}.zip")

    assert not partial.exists()


def test_discard_partial_tolerates_missing_file(tmp_path: Path) -> None:
    """Having no temporary files at all must still be safe."""
    _discard_partial(tmp_path / "tidak_ada.zip")


def test_download_retries_cleanup_never_masks_original_error(tmp_path: Path) -> None:
    """The original error must stay visible even when cleanup fails.

    This is a regression guard for an earlier bug, where a PermissionError from
    unlink masked the real TimeoutError.
    """
    zips = tmp_path / "zips"
    zips.mkdir()

    def failing_download(url: str, target: Path) -> None:
        partial = target.with_suffix(target.suffix + ".part")
        partial.write_bytes(b"bekas")
        partial.chmod(0o444)
        raise TimeoutError("koneksi menganggur")

    original = download_module._download_once
    original_wait = download_module._RETRY_WAIT_SEC
    download_module._download_once = failing_download
    download_module._RETRY_WAIT_SEC = 0
    try:
        with pytest.raises(RuntimeError) as info:
            download_species(SAMPLE, zips)
    finally:
        download_module._download_once = original
        download_module._RETRY_WAIT_SEC = original_wait

    assert "Failed to download" in str(info.value)


def test_write_manifest_columns_and_values(tmp_path: Path) -> None:
    """The manifest must carry every column it promises."""
    zips = tmp_path / "zips"
    zips.mkdir()
    archive = zips / f"{SAMPLE.zip_name}.zip"
    _write_zip(archive, ["a.tif"], payload=b"x" * 100)

    result = download_species(SAMPLE, zips)
    manifest = tmp_path / "zips_manifest.csv"
    write_manifest([result], manifest)

    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert tuple(rows[0]) == MANIFEST_FIELDS
    assert rows[0]["species_id"] == SAMPLE.species_id
    assert rows[0]["canonical_name"] == SAMPLE.display_name
    assert rows[0]["bytes"] == str(archive.stat().st_size)
    assert rows[0]["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()


def test_write_manifest_uses_lf_and_utf8(tmp_path: Path) -> None:
    """The manifest must be UTF-8 without BOM to stay consistent with index.csv."""
    zips = tmp_path / "zips"
    zips.mkdir()
    _write_zip(zips / f"{SAMPLE.zip_name}.zip", ["a.tif"])
    result = download_species(SAMPLE, zips)

    manifest = tmp_path / "manifest.csv"
    write_manifest([result], manifest)
    raw = manifest.read_bytes()

    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw


def test_insecure_context_is_used_only_for_downloads() -> None:
    """The lax TLS context may only be used on the download path."""
    context = download_module._insecure_context()
    assert context.verify_mode.name == "CERT_NONE"
    assert context.check_hostname is False


def test_download_result_fields_are_consistent(tmp_path: Path) -> None:
    """The DownloadResult values must match the file content."""
    zips = tmp_path / "zips"
    zips.mkdir()
    archive = zips / f"{SAMPLE.zip_name}.zip"
    _write_zip(archive, ["a.tif"])

    result = download_species(SAMPLE, zips)

    assert isinstance(result, DownloadResult)
    assert result.size_bytes == archive.stat().st_size
    assert result.species is SAMPLE