"""Sumber kebenaran untuk pemetaan nama spesies DIBaS.

Nama berkas di dalam arsip DIBaS mengandung beberapa salah ketik. Modul ini
menyimpan satu tabel yang memetakan nama arsip ke nama spesies kanonik, dipakai
bersama oleh skrip unduhan, ekstraksi, dan pembangunan index.

Aturan:
1. Tabel adalah satu-satunya sumber kebenaran. Jangan menebak nama spesies dari
   nama berkas di tempat lain.
2. DIBaS menyediakan dua arsip untuk Lactobacillus johnsonii dengan ejaan
   berbeda. Keduanya dipertahankan sebagai label spesies terpisah agar jumlah
   label tetap 33 sesuai SPEC.
"""

from __future__ import annotations

from dataclasses import dataclass

BASE_URL = "https://doctoral.matinf.uj.edu.pl/database/dibas/"


@dataclass(frozen=True)
class Species:
    """Satu spesies DIBaS.

    Attributes:
        species_id: Kunci stabil untuk folder dan kolom index.
        zip_name: Nama arsip tanpa ekstensi, sama dengan nama berkas di dalam arsip.
        canonical_name: Nama spesies menurut taksonom baku.
        variant: Penanda untuk dua arsip johnsonii, None untuk spesies lain.
    """

    species_id: str
    zip_name: str
    canonical_name: str
    variant: str | None = None

    @property
    def display_name(self) -> str:
        """Nama untuk ditampilkan, dengan varian bila ada."""
        if self.variant is None:
            return self.canonical_name
        return f"{self.canonical_name} ({self.variant})"

    @property
    def url(self) -> str:
        """URL arsip ZIP untuk spesies ini."""
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

# Spesies yang ada di dataset tetapi tidak dipakai untuk pelatihan, beserta
# alasannya. Pengecualian dicatat di sini, bukan dihapus dari SPEC secara diam-diam.
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
    """Kembalikan alasan pengecualian spesies, atau None bila tidak dikecualikan."""
    return EXCLUDED_FROM_TRAINING.get(species_id)


def is_trainable(species_id: str) -> bool:
    """Beri tahu apakah spesies dipakai untuk pelatihan."""
    return species_id not in EXCLUDED_FROM_TRAINING


def canonical_name(species_id: str) -> str:
    """Kembalikan nama spesies kanonik untuk species_id.

    Args:
        species_id: Kunci spesies, misalnya ``escherichia_coli``.

    Returns:
        Nama kanonik, misalnya ``Escherichia coli``.

    Raises:
        KeyError: Bila species_id tidak dikenal.
    """
    return BY_SPECIES_ID[species_id].canonical_name


def display_name(species_id: str) -> str:
    """Kembalikan nama tampilan, termasuk varian bila ada."""
    return BY_SPECIES_ID[species_id].display_name