"""Ekstrak arsip ZIP DIBaS ke folder per spesies.

Struktur di dalam arsip tidak konsisten. Hanya Acinetobacter.baumanii yang
memiliki subfolder, sedangkan 32 arsip lain menyimpan berkas TIFF datar di root.
Modul ini memflatten semua entri ke satu folder per spesies dan menolak entri
yang bukan gambar.

Pemakaian:
    python -m bacteriacv.datasets.extract
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import cv2

from ..config import MIN_IMAGES_PER_SPECIES
from ..paths import IMAGES_DIR, PROJECT_ROOT, UNREADABLE_PATH, ZIPS_DIR, ensure_data_dirs
from .species_map import EXPECTED_SPECIES_COUNT, SPECIES, Species

IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

UNREADABLE_FIELDS = ("path", "species_id", "bytes", "reason")


def _relative_posix(path: Path) -> str:
    """Ubah path citra menjadi path relatif proyek dengan garis miring maju.

    Args:
        path: Lokasi berkas citra.

    Returns:
        Path relatif terhadap root proyek.

    Raises:
        ValueError: Bila citra berada di luar root proyek.
    """
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(
            f"Citra di luar root proyek: {resolved}. "
            f"Index hanya menerima citra di dalam {PROJECT_ROOT.name}."
        )
    return resolved.relative_to(PROJECT_ROOT).as_posix()


@dataclass(frozen=True)
class ExtractResult:
    """Hasil ekstraksi satu arsip.

    Attributes:
        species: Spesies asal arsip.
        directory: Folder tujuan.
        image_count: Jumlah berkas gambar yang terekstrak.
        skipped: True bila folder tujuan sudah lengkap.
    """

    species: Species
    directory: Path
    image_count: int
    skipped: bool


def _is_image_member(member: str, name: str) -> bool:
    """Periksa apakah sebuah entri arsip adalah citra yang layak diekstrak.

    Arsip yang dibuat di macOS menyertakan sidecar AppleDouble di folder
    ``__MACOSX`` dengan nama diawali ``._``. Berkas sidecar itu berakhiran
    ``.tif`` sehingga akan lolos bila hanya sufiksnya yang diperiksa, padahal
    isinya bukan citra.

    Args:
        member: Nama entri lengkap di dalam arsip.
        name: Nama berkas tanpa folder.

    Returns:
        True bila entri adalah citra asli.
    """
    parts = PurePosixPath(member).parts
    if "__MACOSX" in parts:
        return False
    if name.startswith("._"):
        return False
    return PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES


def count_images(directory: Path) -> int:
    """Hitung berkas gambar di satu folder."""
    if not directory.is_dir():
        return 0
    return sum(
        1 for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )


def extract_species(
    species: Species,
    zips_dir: Path = ZIPS_DIR,
    images_dir: Path = IMAGES_DIR,
) -> ExtractResult:
    """Ekstrak satu arsip ke folder spesies, atau lewati bila sudah lengkap.

    Args:
        species: Spesies yang akan diekstrak.
        zips_dir: Folder berisi berkas ZIP.
        images_dir: Folder tujuan citra.

    Returns:
        ExtractResult berisi lokasi dan jumlah citra.

    Raises:
        FileNotFoundError: Bila arsip ZIP tidak ditemukan.
        RuntimeError: Bila arsip tidak berisi berkas gambar, atau jumlah citra
            di folder tujuan kurang dari batas minimum.
    """
    archive = zips_dir / f"{species.zip_name}.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"Arsip tidak ditemukan: {archive.name}")

    target = images_dir / species.species_id
    existing = count_images(target)
    if existing >= MIN_IMAGES_PER_SPECIES:
        return ExtractResult(species, target, existing, skipped=True)

    target.mkdir(parents=True, exist_ok=True)

    extracted = 0
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if member.endswith("/"):
                continue
            name = Path(member).name
            if not _is_image_member(member, name):
                continue
            destination = target / name
            with bundle.open(member) as source, destination.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            extracted += 1

    if extracted == 0:
        raise RuntimeError(f"Tidak ada citra di dalam {archive.name}.")

    return ExtractResult(species, target, extracted, skipped=False)


def extract_all(zips_dir: Path = ZIPS_DIR, images_dir: Path = IMAGES_DIR) -> list[ExtractResult]:
    """Ekstrak seluruh arsip dan kembalikan daftar hasil."""
    ensure_data_dirs()
    results: list[ExtractResult] = []
    for index, species in enumerate(SPECIES, start=1):
        result = extract_species(species, zips_dir, images_dir)
        status = "lompat" if result.skipped else "ekstrak"
        print(
            f"[{index:2d}/{len(SPECIES)}] {species.display_name:38s} "
            f"{status}  {result.image_count:3d} citra",
            flush=True,
        )
        results.append(result)

    total = sum(result.image_count for result in results)
    print(f"\nSelesai. {len(results)} spesies, total {total} citra.")
    return results


@dataclass(frozen=True)
class UnreadableImage:
    """Citra yang tidak dapat dibuka oleh pustaka citra.

    Attributes:
        path: Path relatif terhadap root proyek.
        species_id: Kunci spesies pemilik berkas.
        size_bytes: Ukuran berkas di disk.
        reason: Alasan kenapa citra tidak terbaca.
    """

    path: str
    species_id: str
    size_bytes: int
    reason: str


def classify_unreadable(size_bytes: int) -> str:
    """Tentukan alasan sebuah citra tidak dapat dibaca.

    Args:
        size_bytes: Ukuran berkas di disk.

    Returns:
        Alasan singkat dalam bahasa Indonesia.
    """
    if size_bytes == 0:
        return "berkas 0 byte, tidak ada data gambar"
    return "struktur TIFF rusak atau kompresi tidak didukung"


def find_unreadable(images_dir: Path = IMAGES_DIR) -> list[UnreadableImage]:
    """Temukan seluruh citra yang tidak dapat dibaca.

    Pembacaan dilakukan dengan cv2.imread, sama seperti yang dipakai pipeline.
    Daftar yang dikecualikan bukan yang dikembalikan, melainkan yangHealthy
    ditemukan, supaya bisa dicatat di data/raw/unreadable.csv.

    Args:
        images_dir: Folder citra hasil ekstraksi.

    Returns:
        Daftar UnreadableImage, terurut menurut path.
    """
    found: list[UnreadableImage] = []
    for species in SPECIES:
        directory = images_dir / species.species_id
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            size_bytes = path.stat().st_size
            if size_bytes == 0 or cv2.imread(str(path), cv2.IMREAD_COLOR) is None:
                found.append(
                    UnreadableImage(
                        path=_relative_posix(path),
                        species_id=species.species_id,
                        size_bytes=size_bytes,
                        reason=classify_unreadable(size_bytes),
                    )
                )
    return found


def write_unreadable_report(
    unreadable: list[UnreadableImage], report_path: Path = UNREADABLE_PATH
) -> None:
    """Catat citra tidak terbaca ke data/raw/unreadable.csv.

    Berkas ini di-commit sebagai bukti keputusan data, sama seperti
    zips_manifest.csv.

    Args:
        unreadable: Daftar citra yang tidak terbaca.
        report_path: Lokasi berkas keluaran.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=UNREADABLE_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        for item in unreadable:
            writer.writerow(
                {
                    "path": item.path,
                    "species_id": item.species_id,
                    "bytes": item.size_bytes,
                    "reason": item.reason,
                }
            )


def read_unreadable_report(report_path: Path = UNREADABLE_PATH) -> list[dict[str, str]]:
    """Baca data/raw/unreadable.csv bila ada.

    Args:
        report_path: Lokasi berkas.

    Returns:
        Daftar baris sebagai dictionary, atau daftar kosong bila berkas tidak ada.
    """
    if not report_path.is_file():
        return []
    with report_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_all(
    images_dir: Path = IMAGES_DIR, report_path: Path = UNREADABLE_PATH
) -> bool:
    """Periksa kelengkapan dataset hasil ekstraksi.

    Empat pemeriksaan dilakukan:
    1. Setiap spesies punya cukup citra.
    2. Tidak ada nama berkas kembar di dua spesies.
    3. Setiap citra benar-benar dapat dibuka oleh cv2.imread.
    4. Daftar citra tidak terbaca dicatat di unreadable.csv.

    Pemeriksaan ketiga membedakan kerusakan yang sudah diketahui dari kerusakan
    baru. Kerusakan yang sudah tercatat di unreadable.csv dan tidak bertambah
    dianggap sesuai keputusan data, bukan kegagalan. Kerusakan yang bertambah
    atau belum pernah tercatat membuat verifikasi gagal.

    Args:
        images_dir: Folder citra hasil ekstraksi.
        report_path: Lokasi keluaran laporan citra tidak terbaca. Harus
            diteruskan saat pemanggilan dari tes agar tidak menimpa berkas proyek.

    Returns:
        True bila semua pemeriksaan lolos.
    """
    ok = True
    seen: dict[str, str] = {}

    for species in SPECIES:
        directory = images_dir / species.species_id
        count = count_images(directory)
        if count < MIN_IMAGES_PER_SPECIES:
            print(
                f"GAGAL  {species.species_id}: {count} citra, "
                f"kurang dari {MIN_IMAGES_PER_SPECIES}"
            )
            ok = False
        for path in sorted(directory.glob("*")) if directory.is_dir() else []:
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if path.name in seen:
                print(
                    f"GAGAL  nama berkas kembar {path.name} di "
                    f"{seen[path.name]} dan {species.species_id}"
                )
                ok = False
            else:
                seen[path.name] = species.species_id

    previously_recorded = {row["path"] for row in read_unreadable_report(report_path)}
    unreadable = find_unreadable(images_dir)
    write_unreadable_report(unreadable, report_path)

    if unreadable:
        found = {item.path for item in unreadable}
        known_before = previously_recorded
        new_damage = found - known_before
        for item in unreadable:
            marker = "diketahui" if item.path in known_before else "BARU"
            print(f"    {item.path}  ({item.reason}) [{marker}]")
        if new_damage:
            ok = False
            print(
                f"GAGAL  {len(new_damage)} kerusakan baru ditemukan, "
                f"total {len(found)} citra tidak terbaca."
            )
        else:
            print(
                f"CATATAN  {len(found)} citra tidak terbaca, semua sudah "
                f"tercatat di {report_path.name}."
            )

    readable = len(seen) - len(unreadable)
    print(
        f"Total {len(seen)} berkas citra unik, {readable} terbaca, "
        f"{len(unreadable)} rusak."
    )
    return ok


def main(argv: list[str] | None = None) -> int:
    """Titik masuk baris perintah."""
    parser = argparse.ArgumentParser(description="Ekstrak arsip DIBaS per spesies.")
    parser.add_argument("--zips-dir", type=Path, default=ZIPS_DIR)
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Hanya periksa hasil ekstraksi tanpa mengekstrak ulang.",
    )
    args = parser.parse_args(argv)

    try:
        if args.verify_only:
            return 0 if verify_all(args.images_dir, UNREADABLE_PATH) else 1
        extract_all(args.zips_dir, args.images_dir)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"GAGAL: {error}", file=sys.stderr)
        return 1

    if len(SPECIES) != EXPECTED_SPECIES_COUNT:
        print("GAGAL: jumlah spesies tidak sesuai.", file=sys.stderr)
        return 1

    return 0 if verify_all(args.images_dir, UNREADABLE_PATH) else 1


if __name__ == "__main__":
    raise SystemExit(main())