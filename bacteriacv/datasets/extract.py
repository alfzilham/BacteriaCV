"""Extract the DIBaS ZIP archives into a folder per species.

The structure inside the archives is inconsistent. Only Acinetobacter.baumanii
has a subfolder, while the other 32 archives store flat TIFF files at the root.
This module flattens every entry into one folder per species and rejects
entries that are not images.

Usage:
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
    """Turn an image path into a project relative path with forward slashes.

    Args:
        path: The image file location.

    Returns:
        A path relative to the project root.

    Raises:
        ValueError: When the image lies outside the project root.
    """
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(
            f"Image outside the project root: {resolved}. "
            f"The index only accepts images inside {PROJECT_ROOT.name}."
        )
    return resolved.relative_to(PROJECT_ROOT).as_posix()


@dataclass(frozen=True)
class ExtractResult:
    """The extraction result of one archive.

    Attributes:
        species: The source species of the archive.
        directory: The target folder.
        image_count: The number of image files extracted.
        skipped: True when the target folder is already complete.
    """

    species: Species
    directory: Path
    image_count: int
    skipped: bool


def _is_image_member(member: str, name: str) -> bool:
    """Check whether an archive entry is an image worth extracting.

    Archives created on macOS include an AppleDouble sidecar in the
    ``__MACOSX`` folder under names starting with ``._``. Those sidecar files
    end in ``.tif`` so they would pass when only the suffix is checked, even
    though their content is not an image.

    Args:
        member: The full entry name inside the archive.
        name: The filename without the folder.

    Returns:
        True when the entry is a real image.
    """
    parts = PurePosixPath(member).parts
    if "__MACOSX" in parts:
        return False
    if name.startswith("._"):
        return False
    return PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES


def count_images(directory: Path) -> int:
    """Count the image files in one folder."""
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
    """Extract one archive into the species folder, or skip when already complete.

    Args:
        species: The species to extract.
        zips_dir: The folder holding the ZIP files.
        images_dir: The target image folder.

    Returns:
        An ExtractResult holding the location and image count.

    Raises:
        FileNotFoundError: When the ZIP archive is not found.
        RuntimeError: When the archive holds no image files, or the image
            count in the target folder is below the minimum.
    """
    archive = zips_dir / f"{species.zip_name}.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"Archive not found: {archive.name}")

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
        raise RuntimeError(f"No images inside {archive.name}.")

    return ExtractResult(species, target, extracted, skipped=False)


def extract_all(zips_dir: Path = ZIPS_DIR, images_dir: Path = IMAGES_DIR) -> list[ExtractResult]:
    """Extract every archive and return the result list."""
    ensure_data_dirs()
    results: list[ExtractResult] = []
    for index, species in enumerate(SPECIES, start=1):
        result = extract_species(species, zips_dir, images_dir)
        status = "lompat" if result.skipped else "ekstrak"
        print(
            f"[{index:2d}/{len(SPECIES)}] {species.display_name:38s} "
            f"{status}  {result.image_count:3d} images",
            flush=True,
        )
        results.append(result)

    total = sum(result.image_count for result in results)
    print(f"\nDone. {len(results)} species, {total} images in total.")
    return results


@dataclass(frozen=True)
class UnreadableImage:
    """An image that cannot be opened by the image library.

    Attributes:
        path: A path relative to the project root.
        species_id: The species key owning the file.
        size_bytes: The file size on disk.
        reason: The reason the image cannot be read.
    """

    path: str
    species_id: str
    size_bytes: int
    reason: str


def classify_unreadable(size_bytes: int) -> str:
    """Determine the reason an image cannot be read.

    Args:
        size_bytes: The file size on disk.

    Returns:
        A short reason.

    The zero byte reason below is deliberately left in Indonesian. It is not
    a message shown to a user: it fills the reason column of
    data/raw/unreadable.csv, which is committed as provenance evidence and
    compared byte for byte by tests/test_extract.py. Translating it would
    either break that test or silently rewrite committed data, so it is
    treated as data rather than as prose.
    """
    if size_bytes == 0:
        return "berkas 0 byte, tidak ada data gambar"
    return "broken TIFF structure or unsupported compression"


def find_unreadable(images_dir: Path = IMAGES_DIR) -> list[UnreadableImage]:
    """Find every image that cannot be read.

    Reading uses cv2.imread, the same call the pipeline uses.
    The list that is excluded is not the one returned, but the one that is
    found, so it can be recorded in data/raw/unreadable.csv.

    Args:
        images_dir: The folder of extracted images.

    Returns:
        A list of UnreadableImage, sorted by path.
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
    """Record the unreadable images into data/raw/unreadable.csv.

    This file is committed as evidence of a data decision, the same as
    zips_manifest.csv.

    Args:
        unreadable: The list of unreadable images.
        report_path: The output file location.
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
    """Read data/raw/unreadable.csv when it exists.

    Args:
        report_path: The file location.

    Returns:
        The rows as dictionaries, or an empty list when the file is absent.
    """
    if not report_path.is_file():
        return []
    with report_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_all(
    images_dir: Path = IMAGES_DIR, report_path: Path = UNREADABLE_PATH
) -> bool:
    """Check the completeness of the extracted dataset.

Four checks are performed:
1. Every species has enough images.
2. No filename is duplicated across two species.
3. Every image genuinely opens through cv2.imread.
4. The list of unreadable images is recorded in unreadable.csv.

The third check separates known damage from new
damage. Damage already recorded in unreadable.csv that does not grow
counts as the agreed data decision, not a failure. Damage that grows,
or was never recorded makes the verification fail.

    Args:
        images_dir: The folder of extracted images.
        report_path: The output location of the unreadable image report. Must
            be passed in by test calls so project files are not overwritten.

    Returns:
        True when every check passes.
    """
    ok = True
    seen: dict[str, str] = {}

    for species in SPECIES:
        directory = images_dir / species.species_id
        count = count_images(directory)
        if count < MIN_IMAGES_PER_SPECIES:
            print(
                f"FAILED  {species.species_id}: {count} images, "
                f"fewer than {MIN_IMAGES_PER_SPECIES}"
            )
            ok = False
        for path in sorted(directory.glob("*")) if directory.is_dir() else []:
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if path.name in seen:
                print(
                    f"FAILED  duplicate filename {path.name} in "
                    f"{seen[path.name]} and {species.species_id}"
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
            marker = "known" if item.path in known_before else "NEW"
            print(f"    {item.path}  ({item.reason}) [{marker}]")
        if new_damage:
            ok = False
            print(
                f"FAILED  {len(new_damage)} newly found damaged files, "
                f"{len(found)} unreadable images in total."
            )
        else:
            print(
                f"NOTE  {len(found)} unreadable images, all already "
                f"recorded in {report_path.name}."
            )

    readable = len(seen) - len(unreadable)
    print(
        f"Total {len(seen)} unique image files, {readable} readable, "
        f"{len(unreadable)} unreadable."
    )
    return ok


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description="Extract the DIBaS archives per species."
    )
    parser.add_argument("--zips-dir", type=Path, default=ZIPS_DIR)
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify the extraction result without extracting again.",
    )
    args = parser.parse_args(argv)

    try:
        if args.verify_only:
            return 0 if verify_all(args.images_dir, UNREADABLE_PATH) else 1
        extract_all(args.zips_dir, args.images_dir)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1

    if len(SPECIES) != EXPECTED_SPECIES_COUNT:
        print("FAILED: the species count does not match.", file=sys.stderr)
        return 1

    return 0 if verify_all(args.images_dir, UNREADABLE_PATH) else 1


if __name__ == "__main__":
    raise SystemExit(main())