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


def test_known_typos_are_normalized() -> None:
    """Lima nama arsip mengandung salah ketik dan harus dipetakan ke nama baku."""
    assert canonical_name("acinetobacter_baumannii") == "Acinetobacter baumannii"
    assert canonical_name("actinomyces_israelii") == "Actinomyces israelii"
    assert canonical_name("porphyromonas_gingivalis") == "Porphyromonas gingivalis"
    assert canonical_name("veillonella_spp") == "Veillonella spp."
    assert canonical_name("lactobacillus_plantarum") == "Lactobacillus plantarum"


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