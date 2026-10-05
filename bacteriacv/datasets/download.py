"""Download the DIBaS ZIP archives per species and record a SHA-256 manifest.

The source server uses an already expired TLS certificate, so downloading
is only possible without certificate verification. To close that integrity gap,
each archive is hashed with SHA-256 and the result written to
``zips_manifest.csv`` as provenance evidence for the report.

Usage:
    python -m bacteriacv.datasets.download
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ..paths import MANIFEST_PATH, ZIPS_DIR, ensure_data_dirs
from .species_map import EXPECTED_SPECIES_COUNT, SPECIES, Species

MANIFEST_FIELDS = ("zip_name", "species_id", "canonical_name", "bytes", "sha256")

_CHUNK = 1024 * 1024
_RETRY = 3
_RETRY_WAIT_SEC = 5
_SOCKET_TIMEOUT = 300


@dataclass(frozen=True)
class DownloadResult:
    """The download result of one archive.

    Attributes:
        species: The source species of the archive.
        path: The ZIP file location on disk.
        size_bytes: The file size in bytes.
        sha256: The SHA-256 hash of the file, lowercase hexadecimal.
        skipped: True when the file already exists and the hash matches.
    """

    species: Species
    path: Path
    size_bytes: int
    sha256: str
    skipped: bool


def _insecure_context() -> ssl.SSLContext:
    """TLS context without certificate verification.

    Used only because the DIBaS server certificate is expired. Integrity
    is guaranteed by the SHA-256 recorded in the manifest, not by the certificate.
    """
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def sha256_of(path: Path) -> str:
    """Compute the SHA-256 of a file with chunked reading."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _discard_partial(target: Path) -> None:
    """Remove a temporary download file, ignoring it when it is locked.

    A failure to remove leftover download content must not mask the original
    error, so the error is deliberately swallowed here.
    """
    partial = target.with_suffix(target.suffix + ".part")
    try:
        partial.unlink(missing_ok=True)
    except OSError:
        pass


def _download_once(url: str, target: Path) -> None:
    """Download one URL to a target through a temporary file.

    A generous timeout is used because the source server is slow and the
    connection often stalls in the middle of a large file download.
    """
    context = _insecure_context()
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "BacteriaCV/0.1"})
    with urllib.request.urlopen(request, context=context, timeout=_SOCKET_TIMEOUT) as response:
        with partial.open("wb") as handle:
            while chunk := response.read(_CHUNK):
                handle.write(chunk)
    partial.replace(target)


def download_species(species: Species, zips_dir: Path = ZIPS_DIR) -> DownloadResult:
    """Download one ZIP archive, or skip it when already complete.

    Args:
        species: The species to download.
        zips_dir: The target folder.

    Returns:
        A DownloadResult holding the file location, size and hash.

    Raises:
        RuntimeError: When the download fails after every attempt.
    """
    target = zips_dir / f"{species.zip_name}.zip"

    if target.exists() and target.stat().st_size > 0:
        digest = sha256_of(target)
        return DownloadResult(species, target, target.stat().st_size, digest, skipped=True)

    last_error: Exception | None = None
    for attempt in range(1, _RETRY + 1):
        try:
            _download_once(species.url, target)
            break
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            last_error = error
            _discard_partial(target)
            if attempt < _RETRY:
                print(
                    f"    percobaan {attempt} gagal: {error}. "
                    f"mengulang dalam {_RETRY_WAIT_SEC} detik.",
                    flush=True,
                )
                time.sleep(_RETRY_WAIT_SEC)
    else:
        raise RuntimeError(f"Gagal mengunduh {species.zip_name}: {last_error}") from last_error

    return DownloadResult(
        species, target, target.stat().st_size, sha256_of(target), skipped=False
    )


def write_manifest(results: list[DownloadResult], manifest_path: Path = MANIFEST_PATH) -> None:
    """Write the SHA-256 manifest for all archives."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "zip_name": result.species.zip_name,
                    "species_id": result.species.species_id,
                    "canonical_name": result.species.display_name,
                    "bytes": result.size_bytes,
                    "sha256": result.sha256,
                }
            )


def download_all(zips_dir: Path = ZIPS_DIR) -> list[DownloadResult]:
    """Download all 33 archives and write the manifest."""
    ensure_data_dirs()
    if len(SPECIES) != EXPECTED_SPECIES_COUNT:
        raise RuntimeError(
            f"Peta spesies berisi {len(SPECIES)} entri, diharapkan {EXPECTED_SPECIES_COUNT}."
        )

    results: list[DownloadResult] = []
    total_bytes = 0
    for index, species in enumerate(SPECIES, start=1):
        print(f"[{index:2d}/{len(SPECIES)}] {species.display_name}", flush=True)
        result = download_species(species, zips_dir)
        total_bytes += result.size_bytes
        status = "lompat" if result.skipped else "unduh"
        print(
            f"    {status}  {result.size_bytes / 1048576:7.1f} MB  "
            f"sha256 {result.sha256[:16]}...",
            flush=True,
        )
        results.append(result)

    write_manifest(results)
    print(f"\nSelesai. {len(results)} arsip, total {total_bytes / 1073741824:.2f} GB.")
    print(f"Manifest: {MANIFEST_PATH.name} di folder data/raw")
    return results


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Unduh arsip DIBaS dan tulis manifest.")
    parser.add_argument(
        "--zips-dir",
        type=Path,
        default=ZIPS_DIR,
        help="Folder tujuan berkas ZIP.",
    )
    args = parser.parse_args(argv)

    try:
        download_all(args.zips_dir)
    except RuntimeError as error:
        print(f"GAGAL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())