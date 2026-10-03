"""Tes unit untuk ekstraksi arsip ke folder per spesies."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from bacteriacv.datasets.extract import (
    IMAGE_SUFFIXES,
    MIN_IMAGES_PER_SPECIES,
    count_images,
    extract_all,
    extract_species,
    verify_all,
)
from bacteriacv.datasets.species_map import SPECIES

SAMPLE = SPECIES[0]
NESTED = SPECIES[8]


def _make_zip(path: Path, entries: dict[str, bytes]) -> Path:
    """Buat arsip ZIP dengan isi yang ditentukan."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as bundle:
        for name, payload in entries.items():
            bundle.writestr(name, payload)
    return path


def _full_archive(name: str, count: int) -> dict[str, bytes]:
    """Buat isi arsip dengan jumlah citra tertentu pada folder datar."""
    return {f"{name}_{index:04d}.tif": b"x" * 10 for index in range(1, count + 1)}


def test_extract_flat_archive(tmp_path: Path) -> None:
    """Arsip tanpa subfolder harus diekstrak ke satu folder spesies."""
    zips = tmp_path / "zips"
    images = tmp_path / "images"
    _make_zip(zips / f"{SAMPLE.zip_name}.zip", _full_archive(SAMPLE.zip_name, 20))

    result = extract_species(SAMPLE, zips, images)

    assert result.image_count == 20
    assert result.skipped is False
    assert count_images(images / SAMPLE.species_id) == 20


def test_extract_nested_archive_is_flattened(tmp_path: Path) -> None:
    """Arsip dengan subfolder harus tetap diekstrak datar ke folder spesies."""
    zips = tmp_path / "zips"
    images = tmp_path / "images"
    entries = {f"{SAMPLE.zip_name}/{name}": payload for name, payload in _full_archive("a", 20).items()}
    _make_zip(zips / f"{SAMPLE.zip_name}.zip", entries)

    extract_species(SAMPLE, zips, images)

    files = sorted(p.name for p in (images / SAMPLE.species_id).iterdir())
    assert len(files) == 20
    assert all(Path(name).parent == Path(".") for name in files)


def test_extract_skips_non_image_entries(tmp_path: Path) -> None:
    """Berkas non-gambar di arsip harus dilewati."""
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
    """Folder tujuan yang sudah lengkap tidak boleh diekstrak ulang."""
    zips = tmp_path / "zips"
    images = tmp_path / "images"
    _make_zip(zips / f"{SAMPLE.zip_name}.zip", _full_archive(SAMPLE.zip_name, 20))

    first = extract_species(SAMPLE, zips, images)
    second = extract_species(SAMPLE, zips, images)

    assert first.skipped is False
    assert second.skipped is True
    assert second.image_count == first.image_count


def test_extract_missing_archive_raises(tmp_path: Path) -> None:
    """Arsip yang tidak ada harus gagal dengan pesan jelas."""
    with pytest.raises(FileNotFoundError, match="tidak ditemukan"):
        extract_species(SAMPLE, tmp_path / "zips", tmp_path / "images")


def test_extract_archive_without_images_raises(tmp_path: Path) -> None:
    """Arsip tanpa citra harus dianggap rusak."""
    zips = tmp_path / "zips"
    images = tmp_path / "images"
    _make_zip(zips / f"{SAMPLE.zip_name}.zip", {"catatan.txt": b"tidak ada citra"})

    with pytest.raises(RuntimeError, match="Tidak ada citra"):
        extract_species(SAMPLE, zips, images)


def test_count_images_ignores_other_files(tmp_path: Path) -> None:
    """Penghitungan hanya menghitung berkas gambar."""
    directory = tmp_path / "spesies"
    directory.mkdir()
    (directory / "a.tif").write_bytes(b"x")
    (directory / "b.png").write_bytes(b"x")
    (directory / "catatan.txt").write_text("x", encoding="utf-8")

    assert count_images(directory) == 2


def test_count_images_on_missing_directory(tmp_path: Path) -> None:
    """Folder yang belum ada dihitung nol, bukan error."""
    assert count_images(tmp_path / "belum_ada") == 0


def test_verify_all_passes_on_complete_dataset(tmp_path: Path) -> None:
    """Dataset dengan semua spesies lengkap harus lolos verifikasi."""
    zips = tmp_path / "zips"
    images = tmp_path / "images"
    for species in SPECIES:
        count = MIN_IMAGES_PER_SPECIES
        _make_zip(zips / f"{species.zip_name}.zip", _full_archive(species.zip_name, count))

    extract_all(zips, images)

    assert verify_all(images) is True


def test_verify_all_fails_on_missing_species(tmp_path: Path) -> None:
    """Spesies yang belum diekstrak harus membuat verifikasi gagal."""
    images = tmp_path / "images"
    assert verify_all(images) is False


def test_verify_all_fails_on_thin_species(tmp_path: Path) -> None:
    """Spesies dengan citra terlalu sedikit harus dianggap gagal."""
    zips = tmp_path / "zips"
    images = tmp_path / "images"
    for species in SPECIES:
        count = 3 if species is SAMPLE else MIN_IMAGES_PER_SPECIES
        _make_zip(zips / f"{species.zip_name}.zip", _full_archive(species.zip_name, count))

    extract_all(zips, images)

    assert verify_all(images) is False


def test_verify_all_detects_duplicate_filenames(tmp_path: Path) -> None:
    """Nama berkas kembar di dua spesies harus terdeteksi."""
    images = tmp_path / "images"
    for species in SPECIES:
        directory = images / species.species_id
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(MIN_IMAGES_PER_SPECIES):
            (directory / f"citra_{index:04d}.tif").write_bytes(b"x")

    assert verify_all(images) is False


def test_image_suffixes_cover_dibas_formats() -> None:
    """DIBaS hanya memuat TIFF, daftar sufiks tetap harus lengkap."""
    assert ".tif" in IMAGE_SUFFIXES
    assert ".tiff" in IMAGE_SUFFIXES
    assert all(suffix.startswith(".") for suffix in IMAGE_SUFFIXES)


def test_nested_species_uses_its_own_folder(tmp_path: Path) -> None:
    """Spesies dengan subfolder di arsip tetap masuk foldernya sendiri."""
    zips = tmp_path / "zips"
    images = tmp_path / "images"
    entries = {f"{NESTED.zip_name}/{k}": v for k, v in _full_archive("b", 20).items()}
    _make_zip(zips / f"{NESTED.zip_name}.zip", entries)

    extract_species(NESTED, zips, images)

    assert count_images(images / NESTED.species_id) == 20
    assert not (images / NESTED.zip_name).exists()