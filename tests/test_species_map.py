"""Unit tests for the species map and the project folder locations."""

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
    """SPEC requires 33 DIBaS species."""
    assert len(SPECIES) == EXPECTED_SPECIES_COUNT == 33


def test_species_ids_are_unique() -> None:
    """species_id becomes the folder name, so it must be unique."""
    assert len(set(SPECIES_IDS)) == len(SPECIES_IDS)


def test_zip_names_are_unique() -> None:
    """Two species must not point at the same archive."""
    assert len(BY_ZIP_NAME) == len(SPECIES)


def test_all_urls_are_unique() -> None:
    """Every species must have a different URL."""
    urls = [species.url for species in SPECIES]
    assert len(set(urls)) == len(urls)


# Six DIBaS archive names spelled differently from the standard taxonomy name.
# The left side of a pair is the archive name, the right side the canonical name.
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
    """Misspelled archive names must map to the standard taxonomy name."""
    species = BY_ZIP_NAME[zip_name]
    assert canonical_name(species.species_id) == expected
    assert species.zip_name == zip_name, "nama arsip harus tetap apa adanya"


def test_typo_pairs_cover_every_typo() -> None:
    """The pair list must contain all six misspelled archives."""
    assert len(TYPO_PAIRS) == 6
    for zip_name, _ in TYPO_PAIRS:
        assert zip_name in BY_ZIP_NAME, zip_name


def test_plantarum_is_not_a_typo() -> None:
    """Lactobacillus.plantarum is spelled correctly in the archive.

    The misspelling plantaru only exists in the GitHub repository readme, not in
    the DIBaS archive names.
    """
    plantarum = BY_SPECIES_ID["lactobacillus_plantarum"]
    assert plantarum.zip_name == "Lactobacillus.plantarum"
    assert plantarum.zip_name not in {zip for zip, _ in TYPO_PAIRS}


def test_plantarum_uses_working_url() -> None:
    """The plantaru name in the repo readme is 404, the correct name is plantarum."""
    plantarum = BY_SPECIES_ID["lactobacillus_plantarum"]
    assert plantarum.url == f"{BASE_URL}Lactobacillus.plantarum.zip"


def test_johnsonii_kept_as_two_separate_labels() -> None:
    """The two johnsonii archives are kept so the label count stays at 33."""
    a = BY_SPECIES_ID["lactobacillus_johnsonii_a"]
    b = BY_SPECIES_ID["lactobacillus_johnsonii_b"]
    assert a.zip_name == "Lactobacillus.jehnsenii"
    assert b.zip_name == "Lactobacillus.johnsonii"
    assert a.canonical_name == b.canonical_name == "Lactobacillus johnsonii"


def test_johnsonii_variants_have_distinct_display_names() -> None:
    """The display name must tell the two johnsonii labels apart."""
    assert display_name("lactobacillus_johnsonii_a") == "Lactobacillus johnsonii (A)"
    assert display_name("lactobacillus_johnsonii_b") == "Lactobacillus johnsonii (B)"


def test_other_species_have_no_variant_suffix() -> None:
    """A species without a variant adds no suffix to the display name."""
    assert display_name("escherichia_coli") == "Escherichia coli"


def test_species_ids_are_path_safe() -> None:
    """species_id becomes the folder name, so it must be free of spaces and slashes."""
    for species_id in SPECIES_IDS:
        assert species_id.islower()
        assert " " not in species_id
        assert "/" not in species_id
        assert "\\" not in species_id
        assert all(char.isalnum() or char == "_" for char in species_id)


def test_unknown_species_id_raises() -> None:
    """An unknown species_id must fail rather than quietly return a wrong value."""
    with pytest.raises(KeyError):
        canonical_name("tidak_ada_spesies_ini")


def test_zip_name_lookup_is_bidirectional() -> None:
    """Looking up by zip_name must return the same species."""
    for species in SPECIES:
        assert BY_ZIP_NAME[species.zip_name] is species