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
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..paths import IMAGES_DIR, ZIPS_DIR, ensure_data_dirs
from .species_map import EXPECTED_SPECIES_COUNT, SPECIES, Species

IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

MIN_IMAGES_PER_SPECIES = 15


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


def verify_all(images_dir: Path = IMAGES_DIR) -> bool:
    """Periksa bahwa setiap spesies punya cukup citra dan nama file unik.

    Returns:
        True bila semua pemeriksaan lolos.
    """
    ok = True
    seen: dict[str, str] = {}

    for species in SPECIES:
        directory = images_dir / species.species_id
        count = count_images(directory)
        if count < MIN_IMAGES_PER_SPECIES:
            print(f"GAGAL  {species.species_id}: {count} citra, kurang dari {MIN_IMAGES_PER_SPECIES}")
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

    print(f"Total {len(seen)} berkas citra unik.")
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
            return 0 if verify_all(args.images_dir) else 1
        extract_all(args.zips_dir, args.images_dir)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"GAGAL: {error}", file=sys.stderr)
        return 1

    if len(SPECIES) != EXPECTED_SPECIES_COUNT:
        print("GAGAL: jumlah spesies tidak sesuai.", file=sys.stderr)
        return 1

    return 0 if verify_all(args.images_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())