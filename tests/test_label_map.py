"""Tests for the species lookup table of cell shape and Gram status."""

from __future__ import annotations

import pytest
import torch

from bacteriacv.config import GRAM_LABELS, N_GRAM_CLASSES, N_SHAPE_CLASSES, SHAPE_LABELS
from bacteriacv.datasets.species_map import (
    EXCLUDED_FROM_TRAINING,
    EXPECTED_TRAINABLE_COUNT,
    TRAINABLE_SPECIES,
    TRAINABLE_SPECIES_IDS,
    exclusion_reason,
    is_trainable,
)
from bacteriacv.label_map import (
    GRAM_TO_INDEX,
    LOOKUP,
    SHAPE_TO_INDEX,
    class_weights,
    excluded_from_lookup,
    lookup_summary,
    to_targets,
    unmapped_species,
)


def test_lookup_covers_every_trainable_species() -> None:
    """Every trained species must have a lookup entry."""
    assert set(LOOKUP) == set(TRAINABLE_SPECIES_IDS)
    assert len(LOOKUP) == EXPECTED_TRAINABLE_COUNT == 32


def test_lookup_excludes_candida() -> None:
    """Candida albicans is excluded from training, so it is not in the lookup."""
    assert "candida_albicans" not in LOOKUP
    assert not is_trainable("candida_albicans")


def test_exclusion_reason_is_documented() -> None:
    """Every excluded species must carry a written reason."""
    assert EXCLUDED_FROM_TRAINING
    for species_id in EXCLUDED_FROM_TRAINING:
        reason = exclusion_reason(species_id)
        assert reason and len(reason) > 40, species_id


def test_lookup_values_are_valid() -> None:
    """Lookup values must sit inside the valid labels."""
    for species_id, (shape, gram) in LOOKUP.items():
        assert shape in SHAPE_LABELS, f"{species_id}: {shape}"
        assert gram in GRAM_LABELS, f"{species_id}: {gram}"


def test_no_spiral_species_in_lookup() -> None:
    """DIBaS has no spiral species, whatever SHAPE_LABELS says."""
    from bacteriacv.config import SHAPE_LABELS_FULL, SHAPE_UNPOPULATED

    shapes = {shape for shape, _ in LOOKUP.values()}
    assert SHAPE_UNPOPULATED not in shapes
    assert shapes == {"cocci", "bacilli"}
    assert shapes < set(SHAPE_LABELS_FULL), "spiral harus tetap ada di enum lengkap"


def test_bifidobacterium_is_gram_positive() -> None:
    """Bifidobacterium is often wrongly taken as Gram negative because of its name."""
    assert LOOKUP["bifidobacterium_spp"][1] == "positive"


def test_actinomyces_is_gram_positive() -> None:
    """Actinomyces is Gram positive, not Gram negative."""
    assert LOOKUP["actinomyces_israelii"][1] == "positive"


def test_all_lactobacillus_are_gram_positive() -> None:
    """All Lactobacillus are Gram positive."""
    for species in TRAINABLE_SPECIES:
        if species.species_id.startswith("lactobacillus"):
            assert LOOKUP[species.species_id][1] == "positive", species.species_id


def test_fusobacterium_is_bacillus_not_spiral() -> None:
    """Fusobacterium is fusiform, not spiral."""
    assert LOOKUP["fusobacterium_spp"][0] == "bacilli"


def test_neisseria_is_coccus_not_spiral() -> None:
    """Neisseria is diplococci, not spiral."""
    assert LOOKUP["neisseria_gonorrhoeae"][0] == "cocci"


def test_coccobacillus_mapped_to_bacilli() -> None:
    """Acinetobacter and Porphyromonas are coccobacilli, mapped to bacilli."""
    assert LOOKUP["acinetobacter_baumannii"][0] == "bacilli"
    assert LOOKUP["porphyromonas_gingivalis"][0] == "bacilli"


def test_gram_negative_species_on_dibas() -> None:
    """The Gram negative species on DIBaS include both cocci and bacilli.

    Neisseria gonorrhoeae and Veillonella are Gram negative cocci, the rest are
    bacilli. This matters for the report: not every cocci is Gram positive.
    """
    negative = {
        species_id: shape
        for species_id, (shape, gram) in LOOKUP.items()
        if gram == "negative"
    }
    assert negative["neisseria_gonorrhoeae"] == "cocci"
    assert negative["veillonella_spp"] == "cocci"
    assert negative["escherichia_coli"] == "bacilli"
    assert negative["acinetobacter_baumannii"] == "bacilli"


def test_johnsonii_variants_have_same_labels() -> None:
    """The two johnsonii archives are the same taxonomic species."""
    assert LOOKUP["lactobacillus_johnsonii_a"] == LOOKUP["lactobacillus_johnsonii_b"]


def test_both_shape_classes_are_populated() -> None:
    """Both shape classes must have species, otherwise macro F1 means nothing."""
    shapes = {shape for shape, _ in LOOKUP.values()}
    assert shapes == set(SHAPE_LABELS), shapes


def test_both_gram_classes_are_populated() -> None:
    """Both Gram classes must have species."""
    grams = {gram for _, gram in LOOKUP.values()}
    assert grams == set(GRAM_LABELS), grams


def test_index_maps_are_reversible() -> None:
    """The index to label map must be reversible back to a label."""
    for label, index in SHAPE_TO_INDEX.items():
        assert SHAPE_LABELS[index] == label
    for label, index in GRAM_TO_INDEX.items():
        assert GRAM_LABELS[index] == label


def test_to_targets_converts_species_ids() -> None:
    """to_targets must turn species_id values into numeric indices."""
    shape, gram = to_targets(["escherichia_coli", "staphylococcus_aureus"])
    assert shape == [SHAPE_TO_INDEX["bacilli"], SHAPE_TO_INDEX["cocci"]]
    assert gram == [GRAM_TO_INDEX["negative"], GRAM_TO_INDEX["positive"]]


def test_to_targets_handles_empty_list() -> None:
    """An empty list yields two empty lists, not an error."""
    shape, gram = to_targets([])
    assert shape == []
    assert gram == []


def test_to_targets_rejects_unknown_species() -> None:
    """Species outside the table must be rejected, not silently skipped."""
    with pytest.raises(KeyError):
        to_targets(["spesies_yang_tidak_ada"])


def test_to_targets_rejects_candida() -> None:
    """Candida has no lookup entry, so it must be rejected."""
    with pytest.raises(KeyError):
        to_targets(["candida_albicans"])


def test_unmapped_species_lists_unknowns() -> None:
    """Unmapped species must be reportable without raising."""
    assert unmapped_species(["escherichia_coli", "bakteri_misterius"]) == [
        "bakteri_misterius"
    ]


def test_unmapped_species_deduplicates() -> None:
    """The same unmapped species must not be listed twice."""
    assert unmapped_species(["x", "x", "y"]) == ["x", "y"]


def test_unmapped_species_on_full_dataset_returns_empty() -> None:
    """Every trainable species must be mapped."""
    assert unmapped_species(list(TRAINABLE_SPECIES_IDS)) == []


def test_excluded_from_lookup_reports_candida() -> None:
    """An excluded species must be reported as unmapped."""
    assert "candida_albicans" in excluded_from_lookup()


def test_lookup_summary_counts_match_expected() -> None:
    """The label distribution must match the manual verification result."""
    summary = lookup_summary()
    assert summary["total_species"] == 32
    assert summary["shape"]["bacilli"] == 23
    assert summary["shape"]["cocci"] == 9
    assert summary["gram"]["positive"] == 23
    assert summary["gram"]["negative"] == 9


def test_class_weights_upweight_minority_class() -> None:
    """A minority class must get a larger weight than the majority.

    That the weights come from train data only cannot be tested at this
    function level because class_weights takes no split argument. That guarantee
    is tested at the caller level, in train.compute_class_weights.
    """
    weights = class_weights([0, 0, 0, 0, 1, 1])
    assert len(weights) == N_SHAPE_CLASSES
    assert weights[0] < weights[1], "kelas minor harus dapat bobot lebih besar"
    assert float(weights.sum()) > 0


def test_class_weights_follow_sklearn_formula() -> None:
    """The weights follow the sklearn formula: samples divided by (class x frequency)."""
    targets = [0] * 8 + [1] * 2
    weights = class_weights(targets)
    expected = [len(targets) / (2 * 8), len(targets) / (2 * 2)]
    assert float(weights[0]) == pytest.approx(expected[0])
    assert float(weights[1]) == pytest.approx(expected[1])


def test_class_weights_equal_for_balanced_data() -> None:
    """Balanced data gives every class the same weight."""
    weights = class_weights([0, 1, 0, 1])
    assert float(weights[0]) == pytest.approx(1.0)
    assert float(weights[1]) == pytest.approx(1.0)


def test_class_weights_rejects_empty() -> None:
    """An empty target list yields no sensible weights."""
    with pytest.raises(ValueError):
        class_weights([])


def test_class_weights_rejects_single_class() -> None:
    """A single class is not enough to compute unbalanced weights."""
    with pytest.raises(ValueError):
        class_weights([0, 0, 0])


def test_class_weights_respects_n_classes() -> None:
    """The weight length must equal the requested class count."""
    weights = class_weights([0, 0, 1], n_classes=N_GRAM_CLASSES)
    assert weights.shape == (N_GRAM_CLASSES,)
    assert isinstance(weights, torch.Tensor)


def test_class_weights_handles_missing_class() -> None:
    """A class that does not appear gets a weight of one, not zero."""
    weights = class_weights([0, 0, 1], n_classes=3)
    assert weights.shape == (3,)
    assert float(weights[2]) == 1.0