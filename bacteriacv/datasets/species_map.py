"""The source of truth for mapping DIBaS species names.

Filenames inside the DIBaS archives contain several misspellings. This module
holds one table mapping archive names to canonical species names, used
by the download, extract, and index building scripts alike.

Rules:
1. The table is the only source of truth. Do not guess species names from
   filenames anywhere else.
2. DIBaS provides two archives for Lactobacillus johnsonii with different
   spellings. Both are kept as separate species labels so the label count
   stays at 33 as SPEC requires.
"""

from __future__ import annotations

from dataclasses import dataclass

BASE_URL = "https://doctoral.matinf.uj.edu.pl/database/dibas/"


@dataclass(frozen=True)
class Species:
    """One DIBaS species.

    Attributes:
        species_id: The stable key for folders and index columns.
        zip_name: The archive name without extension, same as the filenames inside.
        canonical_name: The species name per standard taxonomy.
        variant: The marker for the two johnsonii archives, None for other species.
    """

    species_id: str
    zip_name: str
    canonical_name: str
    variant: str | None = None

    @property
    def display_name(self) -> str:
        """The name to display, including the variant when present."""
        if self.variant is None:
            return self.canonical_name
        return f"{self.canonical_name} ({self.variant})"

    @property
    def url(self) -> str:
        """The ZIP archive URL for this species."""
        return f"{BASE_URL}{self.zip_name}.zip"


_SPECIES: tuple[Species, ...] = (
    Species("acinetobacter_baumannii", "Acinetobacter.baumanii", "Acinetobacter baumannii"),
    Species("actinomyces_israelii", "Actinomyces.israeli", "Actinomyces israelii"),
    Species("bacteroides_fragilis", "Bacteroides.fragilis", "Bacteroides fragilis"),
    Species("bifidobacterium_spp", "Bifidobacterium.spp", "Bifidobacterium spp."),
    Species("candida_albicans", "Candida.albicans", "Candida albicans"),
    Species("clostridium_perfringens", "Clostridium.perfringens", "Clostridium perfringens"),
    Species("enterococcus_faecium", "Enterococcus.faecium", "Enterococcus faecium"),
    Species("enterococcus_faecalis", "Enterococcus.faecalis", "Enterococcus faecalis"),
    Species("escherichia_coli", "Escherichia.coli", "Escherichia coli"),
    Species("fusobacterium_spp", "Fusobacterium", "Fusobacterium spp."),
    Species("lactobacillus_casei", "Lactobacillus.casei", "Lactobacillus casei"),
    Species("lactobacillus_crispatus", "Lactobacillus.crispatus", "Lactobacillus crispatus"),
    Species("lactobacillus_delbrueckii", "Lactobacillus.delbrueckii", "Lactobacillus delbrueckii"),
    Species("lactobacillus_gasseri", "Lactobacillus.gasseri", "Lactobacillus gasseri"),
    Species("lactobacillus_johnsonii_a", "Lactobacillus.jehnsenii", "Lactobacillus johnsonii", "A"),
    Species("lactobacillus_johnsonii_b", "Lactobacillus.johnsonii", "Lactobacillus johnsonii", "B"),
    Species("lactobacillus_paracasei", "Lactobacillus.paracasei", "Lactobacillus paracasei"),
    Species("lactobacillus_plantarum", "Lactobacillus.plantarum", "Lactobacillus plantarum"),
    Species("lactobacillus_reuteri", "Lactobacillus.reuteri", "Lactobacillus reuteri"),
    Species("lactobacillus_rhamnosus", "Lactobacillus.rhamnosus", "Lactobacillus rhamnosus"),
    Species("lactobacillus_salivarius", "Lactobacillus.salivarius", "Lactobacillus salivarius"),
    Species("listeria_monocytogenes", "Listeria.monocytogenes", "Listeria monocytogenes"),
    Species("micrococcus_spp", "Micrococcus.spp", "Micrococcus spp."),
    Species("neisseria_gonorrhoeae", "Neisseria.gonorrhoeae", "Neisseria gonorrhoeae"),
    Species("porphyromonas_gingivalis", "Porfyromonas.gingivalis", "Porphyromonas gingivalis"),
    Species("propionibacterium_acnes", "Propionibacterium.acnes", "Propionibacterium acnes"),
    Species("proteus_spp", "Proteus", "Proteus spp."),
    Species("pseudomonas_aeruginosa", "Pseudomonas.aeruginosa", "Pseudomonas aeruginosa"),
    Species("staphylococcus_aureus", "Staphylococcus.aureus", "Staphylococcus aureus"),
    Species("staphylococcus_epidermidis", "Staphylococcus.epidermidis", "Staphylococcus epidermidis"),
    Species("staphylococcus_saprophyticus", "Staphylococcus.saprophiticus", "Staphylococcus saprophyticus"),
    Species("streptococcus_agalactiae", "Streptococcus.agalactiae", "Streptococcus agalactiae"),
    Species("veillonella_spp", "Veionella", "Veillonella spp."),
)

SPECIES: tuple[Species, ...] = _SPECIES

BY_SPECIES_ID: dict[str, Species] = {s.species_id: s for s in _SPECIES}
BY_ZIP_NAME: dict[str, Species] = {s.zip_name: s for s in _SPECIES}

SPECIES_IDS: tuple[str, ...] = tuple(s.species_id for s in _SPECIES)

EXPECTED_SPECIES_COUNT = 33

# Species present in the dataset but not used for training, with the
# reason. The exclusion is recorded here, not quietly dropped from SPEC.
EXCLUDED_FROM_TRAINING: dict[str, str] = {
    "candida_albicans": (
        "Candida albicans adalah jamur, bukan bakteri, sehingga tidak memenuhi "
        "premis sistem yang mengklasifikasi bakteri. Sel jamur berukuran 5 sampai "
        "10 mikron, jauh lebih besar dari bakteri 1 sampai 2 mikron, sehingga "
        "masuknya kelas ini akan menaikkan F1-score tanpa menunjukkan kemampuan "
        "klasifikasi morfologi bakteri."
    ),
}

TRAINABLE_SPECIES: tuple[Species, ...] = tuple(
    s for s in _SPECIES if s.species_id not in EXCLUDED_FROM_TRAINING
)

TRAINABLE_SPECIES_IDS: tuple[str, ...] = tuple(
    s.species_id for s in TRAINABLE_SPECIES
)

EXPECTED_TRAINABLE_COUNT = 32


def exclusion_reason(species_id: str) -> str | None:
    """Return the reason a species is excluded, or None when it is not."""
    return EXCLUDED_FROM_TRAINING.get(species_id)


def is_trainable(species_id: str) -> bool:
    """Report whether a species is used for training."""
    return species_id not in EXCLUDED_FROM_TRAINING


def canonical_name(species_id: str) -> str:
    """Return the canonical species name for a species_id.

    Args:
        species_id: The species key, for example ``escherichia_coli``.

    Returns:
        The canonical name, for example ``Escherichia coli``.

    Raises:
        KeyError: When the species_id is unknown.
    """
    return BY_SPECIES_ID[species_id].canonical_name


def display_name(species_id: str) -> str:
    """Return the display name, including the variant when present."""
    return BY_SPECIES_ID[species_id].display_name