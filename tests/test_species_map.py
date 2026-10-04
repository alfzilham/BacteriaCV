"""Tes unit untuk peta spesies dan lokasi folder proyek."""

from __future__ import annotations

import pytest

from bacteriacv.datasets.species_map import (
    BY_SPECIES_ID,
    BY_ZIP_NAME,
    EXPECTED_SPECIES_COUNT,
    SPECIES,
    SPECIES_IDS,
    BASE_URL,
    canonical_name,
    display_name,
)


def test_expected_species_count_is_33() -> None:
    """SPEC mewajibkan 33 spesies DIBaS."""
    assert len(SPECIES) == EXPECTED_SPECIES_COUNT == 33


def test_species_ids_are_unique() -> None:
    """species_id dipakai sebagai nama folder, jadi harus unik."""
    assert len(set(SPECIES_IDS)) == len(SPECIES_IDS)


def test_zip_names_are_unique() -> None:
    """Dua spesies tidak boleh menunjuk arsip yang sama."""
    assert len(BY_ZIP_NAME) == len(SPECIES)


def test_all_urls_are_unique() -> None:
    """Setiap spesies harus punya URL berbeda."""
    urls = [species.url for species in SPECIES]
    assert len(set(urls)) == len(urls)


# Enam nama arsip DIBaS yang ejaannya berbeda dari nama taksonom baku.
# Pasangan di sebelah kiri adalah nama arsip, di sebelah kanan nama kanonik.
TYPO_PAIRS: tuple[tuple[str, str], ...] = (
    ("Acinetobacter.baumanii", "Acinetobacter baumannii"),
    ("Actinomyces.israeli", "Actinomyces israelii"),
    ("Lactobacillus.jehnsenii", "Lactobacillus johnsonii"),
    ("Porfyromonas.gingivalis", "Porphyromonas gingivalis"),
    ("Staphylococcus.saprophiticus", "Staphylococcus saprophyticus"),
    ("Veionella", "Veillonella spp."),
)


@pytest.mark.parametrize("zip_name,expected", TYPO_PAIRS)
def test_archive_typo_is_normalized(zip_name: str, expected: str) -> None:
    """Nama arsip yang salah ketik harus dipetakan ke nama taksonom baku."""
    species = BY_ZIP_NAME[zip_name]
    assert canonical_name(species.species_id) == expected
    assert species.zip_name == zip_name, "nama arsip harus tetap apa adanya"


def test_typo_pairs_cover_every_typo() -> None:
    """Daftar pasangan harus memuat keenam salah ketik arsip."""
    assert len(TYPO_PAIRS) == 6
    for zip_name, _ in TYPO_PAIRS:
        assert zip_name in BY_ZIP_NAME, zip_name


def test_plantarum_is_not_a_typo() -> None:
    """Lactobacillus.plantarum memang ejaan benar di arsip.

    Salah ketik plantaru hanya ada di readme repository GitHub, bukan di nama
    arsip DIBaS.
    """
    plantarum = BY_SPECIES_ID["lactobacillus_plantarum"]
    assert plantarum.zip_name == "Lactobacillus.plantarum"
    assert plantarum.zip_name not in {zip for zip, _ in TYPO_PAIRS}


def test_plantarum_uses_working_url() -> None:
    """Nama plantaru di readme repo 404, nama yang benar adalah plantarum."""
    plantarum = BY_SPECIES_ID["lactobacillus_plantarum"]
    assert plantarum.url == f"{BASE_URL}Lactobacillus.plantarum.zip"


def test_johnsonii_kept_as_two_separate_labels() -> None:
    """Dua arsip johnsonii dipertahankan agar jumlah label tetap 33."""
    a = BY_SPECIES_ID["lactobacillus_johnsonii_a"]
    b = BY_SPECIES_ID["lactobacillus_johnsonii_b"]
    assert a.zip_name == "Lactobacillus.jehnsenii"
    assert b.zip_name == "Lactobacillus.johnsonii"
    assert a.canonical_name == b.canonical_name == "Lactobacillus johnsonii"


def test_johnsonii_variants_have_distinct_display_names() -> None:
    """Nama tampilan harus bisa membedakan dua label johnsonii."""
    assert display_name("lactobacillus_johnsonii_a") == "Lactobacillus johnsonii (A)"
    assert display_name("lactobacillus_johnsonii_b") == "Lactobacillus johnsonii (B)"


def test_other_species_have_no_variant_suffix() -> None:
    """Spesies tanpa varian tidak menambah sufiks pada nama tampilan."""
    assert display_name("escherichia_coli") == "Escherichia coli"


def test_species_ids_are_path_safe() -> None:
    """species_id menjadi nama folder, jadi harus aman tanpa spasi atau garis miring."""
    for species_id in SPECIES_IDS:
        assert species_id.islower()
        assert " " not in species_id
        assert "/" not in species_id
        assert "\\" not in species_id
        assert all(char.isalnum() or char == "_" for char in species_id)


def test_unknown_species_id_raises() -> None:
    """species_id tak dikenal harus gagal diam-diam, bukan mengembalikan nilai salah."""
    with pytest.raises(KeyError):
        canonical_name("tidak_ada_spesies_ini")


def test_zip_name_lookup_is_bidirectional() -> None:
    """Pencarian lewat zip_name harus mengembalikan spesies yang sama."""
    for species in SPECIES:
        assert BY_ZIP_NAME[species.zip_name] is species