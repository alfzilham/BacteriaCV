"""Lookup table pemetaan spesies DIBaS ke bentuk sel dan status Gram.

Modul ini hanya dipakai pada tahap pelatihan. Pada tahap inferensi, label spesies
tidak diketahui sehingga tabel ini tidak boleh dipanggil.

Catatan untuk laporan:
1. Bifidobacterium adalah Gram positif meskipun namanya mengandung "bacterium".
   Kesalahan yang sering terjadi adalah menganggapnya Gram negatif.
2. Neisseria gonorrhoeae berupa diplococci, bukan spiral.
3. Fusobacterium berupa fusiform bacillus, bukan spiral.
4. DIBaS tidak memuat satu pun spesies berbentuk spiral, sehingga Head A dilatih
   hanya pada dua kelas. Enum bentuk tetap memuat spiral sebagai penanda kelas
   yang tidak terisi.
5. Acinetobacter baumannii dan Porphyromonas gingivalis adalah coccobacillus,
   yaitu bentuk antara. Keduanya dipetakan ke bacilli karena sumbu batang tetap
   dominan pada bentuk tersebut.
6. Actinomyces israelii berupa filamen bercabang. Pada perbesaran 1000x selnya
   terlihat sebagai fragmen batang pendek, sehingga dipetakan ke bacilli. Ini
   satu-satunya spesies yang tidak benar-benar muat ke tiga kategori dasar.
7. Candida albicans tidak ada di tabel karena dikeluarkan dari pelatihan. Jamur,
   bukan bakteri. Lihat EXCLUDED_FROM_TRAINING pada datasets/species_map.py.
"""

from __future__ import annotations

from collections import Counter

import torch

from .config import GRAM_LABELS, SHAPE_LABELS
from .datasets.species_map import (
    EXCLUDED_FROM_TRAINING,
    TRAINABLE_SPECIES_IDS,
    display_name,
)

# Lookup table [L]. Kunci adalah species_id dari datasets/species_map.py.
LOOKUP: dict[str, tuple[str, str]] = {
    "acinetobacter_baumannii": ("bacilli", "negative"),
    "actinomyces_israelii": ("bacilli", "positive"),
    "bacteroides_fragilis": ("bacilli", "negative"),
    "bifidobacterium_spp": ("bacilli", "positive"),
    "clostridium_perfringens": ("bacilli", "positive"),
    "enterococcus_faecium": ("cocci", "positive"),
    "enterococcus_faecalis": ("cocci", "positive"),
    "escherichia_coli": ("bacilli", "negative"),
    "fusobacterium_spp": ("bacilli", "negative"),
    "lactobacillus_casei": ("bacilli", "positive"),
    "lactobacillus_crispatus": ("bacilli", "positive"),
    "lactobacillus_delbrueckii": ("bacilli", "positive"),
    "lactobacillus_gasseri": ("bacilli", "positive"),
    "lactobacillus_johnsonii_a": ("bacilli", "positive"),
    "lactobacillus_johnsonii_b": ("bacilli", "positive"),
    "lactobacillus_paracasei": ("bacilli", "positive"),
    "lactobacillus_plantarum": ("bacilli", "positive"),
    "lactobacillus_reuteri": ("bacilli", "positive"),
    "lactobacillus_rhamnosus": ("bacilli", "positive"),
    "lactobacillus_salivarius": ("bacilli", "positive"),
    "listeria_monocytogenes": ("bacilli", "positive"),
    "micrococcus_spp": ("cocci", "positive"),
    "neisseria_gonorrhoeae": ("cocci", "negative"),
    "porphyromonas_gingivalis": ("bacilli", "negative"),
    "propionibacterium_acnes": ("bacilli", "positive"),
    "proteus_spp": ("bacilli", "negative"),
    "pseudomonas_aeruginosa": ("bacilli", "negative"),
    "staphylococcus_aureus": ("cocci", "positive"),
    "staphylococcus_epidermidis": ("cocci", "positive"),
    "staphylococcus_saprophyticus": ("cocci", "positive"),
    "streptococcus_agalactiae": ("cocci", "positive"),
    "veillonella_spp": ("cocci", "negative"),
}

SHAPE_TO_INDEX: dict[str, int] = {
    label: index for index, label in enumerate(SHAPE_LABELS)
}
GRAM_TO_INDEX: dict[str, int] = {
    label: index for index, label in enumerate(GRAM_LABELS)
}


def to_targets(species_ids: list[str]) -> tuple[list[int], list[int]]:
    """Ubah daftar species_id menjadi indeks bentuk dan status Gram.

    Args:
        species_ids: Daftar species_id dari data/index.csv.

    Returns:
        Pasangan (indeks_bentuk, indeks_gram).

    Raises:
        KeyError: Bila ada species_id yang tidak ada di LOOKUP, termasuk yang
            sengaja dikecualikan seperti candida_albicans.
    """
    shapes: list[int] = []
    grams: list[int] = []
    for species_id in species_ids:
        shape, gram = LOOKUP[species_id]
        shapes.append(SHAPE_TO_INDEX[shape])
        grams.append(GRAM_TO_INDEX[gram])
    return shapes, grams


def unmapped_species(species_ids: list[str]) -> list[str]:
    """Kembalikan species_id yang tidak ada di lookup table.

    Berguna untuk melaporkan citra yang dibuang, sesuai SPEC bagian 3.

    Args:
        species_ids: Daftar species_id yang diperiksa.

    Returns:
        Daftar species_id yang tidak terpetakan, unik dan terurut.
    """
    missing = {species_id for species_id in species_ids if species_id not in LOOKUP}
    return sorted(missing)


def excluded_from_lookup() -> list[str]:
    """Kembalikan species_id yang sengaja dikeluarkan dari lookup.

    Returns:
        Daftar species_id yang dikecualikan, unik dan terurut.
    """
    return sorted(EXCLUDED_FROM_TRAINING)


def lookup_summary() -> dict[str, object]:
    """Ringkasan distribusi lookup table untuk keperluan laporan.

    Returns:
        Dictionary berisi jumlah spesies total, distribusi bentuk, dan
        distribusi status Gram.
    """
    shape_counts = Counter(shape for shape, _ in LOOKUP.values())
    gram_counts = Counter(gram for _, gram in LOOKUP.values())
    return {
        "total_species": len(LOOKUP),
        "shape": {label: shape_counts.get(label, 0) for label in SHAPE_LABELS},
        "gram": {label: gram_counts.get(label, 0) for label in GRAM_LABELS},
        "excluded": excluded_from_lookup(),
    }


def class_weights(targets: list[int], n_classes: int | None = None) -> torch.Tensor:
    """Hitung bobot kelas dari frekuensi data latih.

    Bobot tiap kelas memakai rumus sklearn, yaitu jumlah sampel dibagi jumlah
    kelas dikali frekuensi kelas tersebut. Rumus ini menghasilkan bobot satu
    untuk data seimbang dan bobot lebih besar pada kelas minor. Hanya data
    latih yang boleh dipakai.

    Args:
        targets: Daftar indeks kelas dari data latih.
        n_classes: Jumlah kelas total. Default-nya jumlah label bentuk.

    Returns:
        Tensor bobot sepanjang n_classes. Kelas yang tidak muncul di data latih
        diberi bobot satu.

    Raises:
        ValueError: Bila targets kosong atau hanya berisi satu kelas.
    """
    if n_classes is None:
        n_classes = len(SHAPE_LABELS)
    if not targets:
        raise ValueError("Daftar target kosong, bobot kelas tidak dapat dihitung.")

    counts = Counter(targets)
    if len(counts) < 2:
        raise ValueError("Hanya satu kelas pada data latih, bobot tidak dapat dihitung.")

    weights = [1.0] * n_classes
    for index, count in counts.items():
        if 0 <= index < n_classes:
            weights[index] = len(targets) / (n_classes * count)
    return torch.tensor(weights, dtype=torch.float32)