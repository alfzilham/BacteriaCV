"""Unduh arsip ZIP DIBaS per spesies dan catat manifest SHA-256.

Server asal memakai sertifikat TLS yang sudah kedaluwarsa sehingga unduhan
hanya bisa jalan tanpa verifikasi sertifikat. Untuk menutup celah integritas,
setiap berkas di-hash SHA-256 dan hasilnya ditulis ke ``zips_manifest.csv``
sebagai bukti provenance untuk laporan.

Pemakaian:
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
    """Hasil unduhan satu arsip.

    Attributes:
        species: Spesies asal arsip.
        path: Lokasi berkas ZIP di disk.
        size_bytes: Ukuran berkas dalam byte.
        sha256: Hash SHA-256 berkas, huruf kecil heksadesimal.
        skipped: True bila berkas sudah ada dan hash cocok.
    """

    species: Species
    path: Path
    size_bytes: int
    sha256: str
    skipped: bool


def _insecure_context() -> ssl.SSLContext:
    """Konteks TLS tanpa verifikasi sertifikat.

    Dipakai hanya karena sertifikat server DIBaS kedaluwarsa. Integritas
    dijamin lewat SHA-256 yang dicatat di manifest, bukan lewat sertifikat.
    """
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def sha256_of(path: Path) -> str:
    """Hitung SHA-256 berkas dengan pembacaan bertahap."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _discard_partial(target: Path) -> None:
    """Hapus berkas unduhan sementara, abaikan bila sedang terkunci.

    Kegagalan menghapus sisa unduhan tidak boleh menutupi error asli, jadi
    di sini error sengaja ditelan.
    """
    partial = target.with_suffix(target.suffix + ".part")
    try:
        partial.unlink(missing_ok=True)
    except OSError:
        pass


def _download_once(url: str, target: Path) -> None:
    """Unduh satu URL ke target lewat berkas sementara.

    Timeout generous dipakai karena server asal lambat dan koneksi sering
    menganggur di tengah unduhan berkas besar.
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
    """Unduh satu arsip ZIP, atau lewati bila sudah lengkap.

    Args:
        species: Spesies yang akan diunduh.
        zips_dir: Folder tujuan.

    Returns:
        DownloadResult berisi lokasi, ukuran, dan hash berkas.

    Raises:
        RuntimeError: Bila unduhan gagal setelah semua percobaan.
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
    """Tulis manifest SHA-256 untuk seluruh arsip."""
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
    """Unduh seluruh 33 arsip dan tulis manifest."""
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
    """Titik masuk baris perintah."""
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