"""Unit tests for archive extraction into a folder per species."""

from __future__ import annotations

import zipfile

import numpy as np
from pathlib import Path

import pytest

import cv2

from bacteriacv.datasets.extract import (
    IMAGE_SUFFIXES,
    MIN_IMAGES_PER_SPECIES,
    UnreadableImage,
    classify_unreadable,
    count_images,
    extract_all,
    extract_species,
    find_unreadable,
    read_unreadable_report,
    verify_all,
    write_unreadable_report,
)
from bacteriacv.datasets.species_map import SPECIES


def _write_valid_images(directory: Path, count: int, prefix: str = "citra") -> list[Path]:
    """Build a valid PNG that cv2 can genuinely open.

    The prefix keeps filenames from colliding between species.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(count):
        path = directory / f"{prefix}_{index:04d}.png"
        array = np.full((32, 32, 3), 180, dtype=np.uint8)
        array[8:24, 8:24] = (40, 90, 200)
        assert cv2.imwrite(str(path), array)
        paths.append(path)
    return paths

SAMPLE = SPECIES[0]
NESTED = SPECIES[8]


def _make_zip(path: Path, entries: dict[str, bytes]) -> Path:
    """Build a ZIP archive with the given content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as bundle:
        for name, payload in entries.items():
            bundle.writestr(name, payload)
    return path


def _full_archive(name: str, count: int) -> dict[str, bytes]:
    """Build an archive content with a given image count in a flat folder."""
    return {f"{name}_{index:04d}.tif": b"x" * 10 for index in range(1, count + 1)}


def test_extract_flat_archive(tmp_path: Path) -> None:
    """An archive without a subfolder must extract into one species folder."""
    zips = tmp_path / "zips"
    images = tmp_path / "images"
    _make_zip(zips / f"{SAMPLE.zip_name}.zip", _full_archive(SAMPLE.zip_name, 20))

    result = extract_species(SAMPLE, zips, images)

    assert result.image_count == 20
    assert result.skipped is False
    assert count_images(images / SAMPLE.species_id) == 20


def test_extract_nested_archive_is_flattened(tmp_path: Path) -> None:
    """An archive with a subfolder must still extract flat into the species folder."""
    zips = tmp_path / "zips"
    images = tmp_path / "images"
    entries = {f"{SAMPLE.zip_name}/{name}": payload for name, payload in _full_archive("a", 20).items()}
    _make_zip(zips / f"{SAMPLE.zip_name}.zip", entries)

    extract_species(SAMPLE, zips, images)

    files = sorted(p.name for p in (images / SAMPLE.species_id).iterdir())
    assert len(files) == 20
    assert all(Path(name).parent == Path(".") for name in files)


def test_extract_skips_non_image_entries(tmp_path: Path) -> None:
    """Non image files in the archive must be skipped."""
    zips = tmp_path / "zips"
    images = tmp_path / "images"
    entries = _full_archive(SAMPLE.zip_name, 20)
    entries["readme.txt"] = b"bukan citra"
    entries["__MACOSX/._sampel.tif"] = b"resource fork"
    _make_zip(zips / f"{SAMPLE.zip_name}.zip", entries)

    result = extract_species(SAMPLE, zips, images)

    assert result.image_count == 20
    assert not (images / SAMPLE.species_id / "readme.txt").exists()


def test_extract_skips_when_already_complete(tmp_path: Path) -> None:
    """A complete target folder must not be extracted again."""
    zips = tmp_path / "zips"
    images = tmp_path / "images"
    _make_zip(zips / f"{SAMPLE.zip_name}.zip", _full_archive(SAMPLE.zip_name, 20))

    first = extract_species(SAMPLE, zips, images)
    second = extract_species(SAMPLE, zips, images)

    assert first.skipped is False
    assert second.skipped is True
    assert second.image_count == first.image_count


def test_extract_missing_archive_raises(tmp_path: Path) -> None:
    """A missing archive must fail with a clear message."""
    with pytest.raises(FileNotFoundError, match="tidak ditemukan"):
        extract_species(SAMPLE, tmp_path / "zips", tmp_path / "images")


def test_extract_archive_without_images_raises(tmp_path: Path) -> None:
    """An archive without images must count as damaged."""
    zips = tmp_path / "zips"
    images = tmp_path / "images"
    _make_zip(zips / f"{SAMPLE.zip_name}.zip", {"catatan.txt": b"tidak ada citra"})

    with pytest.raises(RuntimeError, match="Tidak ada citra"):
        extract_species(SAMPLE, zips, images)


def test_count_images_ignores_other_files(tmp_path: Path) -> None:
    """The count only counts image files."""
    directory = tmp_path / "spesies"
    directory.mkdir()
    (directory / "a.tif").write_bytes(b"x")
    (directory / "b.png").write_bytes(b"x")
    (directory / "catatan.txt").write_text("x", encoding="utf-8")

    assert count_images(directory) == 2


def test_count_images_on_missing_directory(tmp_path: Path) -> None:
    """A folder that does not exist counts zero, not an error."""
    assert count_images(tmp_path / "belum_ada") == 0


def test_verify_all_passes_on_complete_dataset(project_tmp_dir: Path) -> None:
    """A dataset with every species complete and readable must pass verification."""
    zips = project_tmp_dir / "zips"
    images = project_tmp_dir / "images"
    for species in SPECIES:
        _write_valid_images(images / species.species_id, MIN_IMAGES_PER_SPECIES, species.species_id)

    assert verify_all(images, project_tmp_dir / "unreadable.csv") is True


def test_verify_all_fails_on_missing_species(project_tmp_dir: Path) -> None:
    """A species that has not been extracted must make verification fail."""
    images = project_tmp_dir / "images"
    assert verify_all(images, project_tmp_dir / "unreadable.csv") is False


def test_verify_all_fails_on_thin_species(project_tmp_dir: Path) -> None:
    """A species with too few images must count as failed."""
    images = project_tmp_dir / "images"
    for species in SPECIES:
        count = 3 if species is SAMPLE else MIN_IMAGES_PER_SPECIES
        _write_valid_images(images / species.species_id, count, species.species_id)

    assert verify_all(images, project_tmp_dir / "unreadable.csv") is False


def test_verify_all_fails_on_zero_byte_image(project_tmp_dir: Path) -> None:
    """A zero byte image must be detected even when the count is sufficient."""
    images = project_tmp_dir / "images"
    for species in SPECIES:
        _write_valid_images(images / species.species_id, MIN_IMAGES_PER_SPECIES, species.species_id)

    broken = images / SAMPLE.species_id / f"{SAMPLE.zip_name}_9999.tif"
    broken.write_bytes(b"")

    assert verify_all(images, project_tmp_dir / "unreadable.csv") is False


def test_verify_all_fails_on_corrupt_image(project_tmp_dir: Path) -> None:
    """An image with broken content must be detected even when its size is normal."""
    images = project_tmp_dir / "images"
    for species in SPECIES:
        _write_valid_images(images / species.species_id, MIN_IMAGES_PER_SPECIES, species.species_id)

    broken = images / SAMPLE.species_id / f"{SAMPLE.zip_name}_9998.tif"
    broken.write_bytes(b"bukan gambar tiff" * 200)

    assert verify_all(images, project_tmp_dir / "unreadable.csv") is False


def test_find_unreadable_reports_broken_files(project_tmp_dir: Path) -> None:
    """The unreadable image list must include the damaged file."""
    images = project_tmp_dir / "images"
    for species in SPECIES:
        _write_valid_images(images / species.species_id, MIN_IMAGES_PER_SPECIES, species.species_id)

    (images / SAMPLE.species_id / "rusak.tif").write_bytes(b"")

    found = find_unreadable(images)
    assert len(found) == 1
    assert found[0].species_id == SAMPLE.species_id
    assert found[0].size_bytes == 0


def test_classify_unreadable_distinguishes_causes() -> None:
    """The reason must tell an empty file apart from a damaged one."""
    assert "0 byte" in classify_unreadable(0)
    assert "0 byte" not in classify_unreadable(9471378)
    assert "TIFF" in classify_unreadable(9471378)


def test_verify_all_accepts_already_recorded_damage(project_tmp_dir: Path) -> None:
    """Damage already recorded earlier is not a new failure."""
    images = project_tmp_dir / "images"
    report = project_tmp_dir / "unreadable.csv"
    for species in SPECIES:
        _write_valid_images(
            images / species.species_id, MIN_IMAGES_PER_SPECIES, species.species_id
        )

    broken = images / SAMPLE.species_id / "sudah_dicatat.tif"
    broken.write_bytes(b"")

    # Record the damage first, then verify again.
    assert verify_all(images, report) is False
    assert report.is_file()
    assert verify_all(images, report) is True


def test_verify_all_rejects_new_damage(project_tmp_dir: Path) -> None:
    """Additional damage that was not recorded must fail."""
    images = project_tmp_dir / "images"
    report = project_tmp_dir / "unreadable.csv"
    for species in SPECIES:
        _write_valid_images(
            images / species.species_id, MIN_IMAGES_PER_SPECIES, species.species_id
        )

    known = images / SAMPLE.species_id / "sudah_dicatat.tif"
    known.write_bytes(b"")
    assert verify_all(images, report) is False

    fresh = images / SAMPLE.species_id / "baru_rusak.tif"
    fresh.write_bytes(b"")
    assert verify_all(images, report) is False


def test_write_unreadable_report_columns(project_tmp_dir: Path) -> None:
    """The report must carry the columns it promises."""
    import csv

    record = UnreadableImage(
        path="data/raw/images/x/a.tif",
        species_id="x",
        size_bytes=0,
        reason="berkas 0 byte",
    )
    target = project_tmp_dir / "unreadable.csv"
    write_unreadable_report([record], target)

    with target.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert tuple(rows[0]) == ("path", "species_id", "bytes", "reason")
    assert rows[0]["reason"] == "berkas 0 byte"


def test_read_unreadable_report_missing_file(project_tmp_dir: Path) -> None:
    """A missing report must yield an empty list."""
    assert read_unreadable_report(project_tmp_dir / "tidak_ada.csv") == []


def test_verify_all_detects_duplicate_filenames(project_tmp_dir: Path) -> None:
    """Duplicate filenames across two species must be detected."""
    images = project_tmp_dir / "images"
    for species in SPECIES:
        directory = images / species.species_id
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(MIN_IMAGES_PER_SPECIES):
            (directory / f"kembar_{index:04d}.png").write_bytes(b"x")

    assert verify_all(images, project_tmp_dir / "unreadable.csv") is False


def test_image_suffixes_cover_dibas_formats() -> None:
    """DIBaS only holds TIFF, the suffix list must still be complete."""
    assert ".tif" in IMAGE_SUFFIXES
    assert ".tiff" in IMAGE_SUFFIXES
    assert all(suffix.startswith(".") for suffix in IMAGE_SUFFIXES)


def test_nested_species_uses_its_own_folder(tmp_path: Path) -> None:
    """A species with a subfolder in the archive still goes into its own folder."""
    zips = tmp_path / "zips"
    images = tmp_path / "images"
    entries = {f"{NESTED.zip_name}/{k}": v for k, v in _full_archive("b", 20).items()}
    _make_zip(zips / f"{NESTED.zip_name}.zip", entries)

    extract_species(NESTED, zips, images)

    assert count_images(images / NESTED.species_id) == 20
    assert not (images / NESTED.zip_name).exists()