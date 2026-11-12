"""Lookup table mapping DIBaS species to cell shape and Gram status.

This module is only used at the training stage. At inference the species label
is not known, so this table must not be called.

Notes for the report:
1. Bifidobacterium is Gram positive even though its name contains "bacterium".
   The mistake that often occurs is treating it as Gram negative.
2. Neisseria gonorrhoeae is diplococci, not spiral.
3. Fusobacterium is a fusiform bacillus, not spiral.
4. DIBaS contains no spiral shaped species at all, so Head A is trained
   on only two classes. The shape enum still holds spiral as the marker of
   an unpopulated class.
5. Acinetobacter baumannii and Porphyromonas gingivalis are coccobacilli,
   an intermediate shape. Both are mapped to bacilli because the rod axis
   stays dominant in that shape.
6. Actinomyces israelii is branching filaments. At 1000x magnification its
   cells look like short rod fragments, so it is mapped to bacilli. This is
   the only species that does not truly fit the three base categories.
7. Candida albicans is absent from the table because it is excluded from
   training. A fungus, not a bacterium. See EXCLUDED_FROM_TRAINING in
   datasets/species_map.py.
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

# Lookup table [L]. The keys are species_id values from datasets/species_map.py.
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
    """Turn a list of species_id values into shape and Gram indices.

    Args:
        species_ids: The species_id values from data/index.csv.

    Returns:
        A (shape_index, gram_index) pair.

    Raises:
        KeyError: When a species_id is absent from LOOKUP, including those
            deliberately excluded such as candida_albicans.
    """
    shapes: list[int] = []
    grams: list[int] = []
    for species_id in species_ids:
        shape, gram = LOOKUP[species_id]
        shapes.append(SHAPE_TO_INDEX[shape])
        grams.append(GRAM_TO_INDEX[gram])
    return shapes, grams


def unmapped_species(species_ids: list[str]) -> list[str]:
    """Return the species_id values that are absent from the lookup table.

    Useful for reporting discarded images, per SPEC section 3.

    Args:
        species_ids: The species_id values to check.

    Returns:
        The unmapped species_id values, unique and sorted.
    """
    missing = {species_id for species_id in species_ids if species_id not in LOOKUP}
    return sorted(missing)


def excluded_from_lookup() -> list[str]:
    """Return the species_id values deliberately excluded from the lookup.

    Returns:
        The excluded species_id values, unique and sorted.
    """
    return sorted(EXCLUDED_FROM_TRAINING)


def lookup_summary() -> dict[str, object]:
    """Summary of the lookup table distribution for reporting purposes.

    Returns:
        A dictionary holding the total species count, the shape distribution,
        and the Gram status distribution.
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
    """Compute class weights from the frequency of the train data.

    Each class weight uses the sklearn formula, the sample count divided by the
    class count times that class frequency. The formula yields a weight of one
    for balanced data and a larger weight on the minority class. Only train
    data may be used.

    Args:
        targets: The class indices from the train data.
        n_classes: The total class count. Defaults to the number of shape labels.

    Returns:
        A weight tensor of length n_classes. A class that does not appear in
        the train data is given a weight of one.

    Raises:
        ValueError: When targets is empty or holds only one class.
    """
    if n_classes is None:
        n_classes = len(SHAPE_LABELS)
    if not targets:
        raise ValueError("The target list is empty, class weights cannot be computed.")

    counts = Counter(targets)
    if len(counts) < 2:
        raise ValueError("Only one class in the train data, weights cannot be computed.")

    weights = [1.0] * n_classes
    for index, count in counts.items():
        if 0 <= index < n_classes:
            weights[index] = len(targets) / (n_classes * count)
    return torch.tensor(weights, dtype=torch.float32)