"""Tes untuk lookup table spesies ke bentuk sel dan status Gram."""

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
    """Semua spesies yang dilatih harus punya entri lookup."""
    assert set(LOOKUP) == set(TRAINABLE_SPECIES_IDS)
    assert len(LOOKUP) == EXPECTED_TRAINABLE_COUNT == 32


def test_lookup_excludes_candida() -> None:
    """Candida albicans dikeluarkan dari pelatihan, jadi tidak ada di lookup."""
    assert "candida_albicans" not in LOOKUP
    assert not is_trainable("candida_albicans")


def test_exclusion_reason_is_documented() -> None:
    """Setiap spesies yang dikecualikan harus punya alasan tertulis."""
    assert EXCLUDED_FROM_TRAINING
    for species_id in EXCLUDED_FROM_TRAINING:
        reason = exclusion_reason(species_id)
        assert reason and len(reason) > 40, species_id


def test_lookup_values_are_valid() -> None:
    """Nilai lookup harus berada di dalam label yang sah."""
    for species_id, (shape, gram) in LOOKUP.items():
        assert shape in SHAPE_LABELS, f"{species_id}: {shape}"
        assert gram in GRAM_LABELS, f"{species_id}: {gram}"


def test_no_spiral_species_in_lookup() -> None:
    """Keputusan D1: tidak ada spesies DIBaS dengan bentuk spiral."""
    assert all(shape in SHAPE_LABELS for shape, _ in LOOKUP.values())


def test_bifidobacterium_is_gram_positive() -> None:
    """Bifidobacterium sering salah dianggap Gram negatif karena namanya."""
    assert LOOKUP["bifidobacterium_spp"][1] == "positif"


def test_actinomyces_is_gram_positive() -> None:
    """Actinomyces adalah Gram positif, bukan negatif."""
    assert LOOKUP["actinomyces_israelii"][1] == "positif"


def test_all_lactobacillus_are_gram_positive() -> None:
    """Seluruh Lactobacillus adalah Gram positif."""
    for species in TRAINABLE_SPECIES:
        if species.species_id.startswith("lactobacillus"):
            assert LOOKUP[species.species_id][1] == "positif", species.species_id


def test_fusobacterium_is_bacillus_not_spiral() -> None:
    """Fusobacterium fusiform, bukan spiral."""
    assert LOOKUP["fusobacterium_spp"][0] == "bacilli"


def test_neisseria_is_coccus_not_spiral() -> None:
    """Neisseria diplococci, bukan spiral."""
    assert LOOKUP["neisseria_gonorrhoeae"][0] == "cocci"


def test_coccobacillus_mapped_to_bacilli() -> None:
    """Acinetobacter dan Porphyromonas adalah coccobacillus, dipetakan ke bacilli."""
    assert LOOKUP["acinetobacter_baumannii"][0] == "bacilli"
    assert LOOKUP["porphyromonas_gingivalis"][0] == "bacilli"


def test_gram_negative_species_on_dibas() -> None:
    """Spesies Gram negatif pada DIBaS mencakup kokus dan batang.

    Neisseria gonorrhoeae dan Veillonella adalah kokus Gram negatif, sisanya
    batang. Ini penting untuk laporan: tidak semua kokus adalah Gram positif.
    """
    negative = {
        species_id: shape
        for species_id, (shape, gram) in LOOKUP.items()
        if gram == "negatif"
    }
    assert negative["neisseria_gonorrhoeae"] == "cocci"
    assert negative["veillonella_spp"] == "cocci"
    assert negative["escherichia_coli"] == "bacilli"
    assert negative["acinetobacter_baumannii"] == "bacilli"


def test_johnsonii_variants_have_same_labels() -> None:
    """Dua arsip johnsonii adalah spesies taksonom sama."""
    assert LOOKUP["lactobacillus_johnsonii_a"] == LOOKUP["lactobacillus_johnsonii_b"]


def test_both_shape_classes_are_populated() -> None:
    """Kedua kelas bentuk harus punya spesies, kalau tidak F1 makro tidak bermakna."""
    shapes = {shape for shape, _ in LOOKUP.values()}
    assert shapes == set(SHAPE_LABELS), shapes


def test_both_gram_classes_are_populated() -> None:
    """Kedua kelas Gram harus punya spesies."""
    grams = {gram for _, gram in LOOKUP.values()}
    assert grams == set(GRAM_LABELS), grams


def test_index_maps_are_reversible() -> None:
    """Peta indeks ke label harus bisa dikembalikan ke label."""
    for label, index in SHAPE_TO_INDEX.items():
        assert SHAPE_LABELS[index] == label
    for label, index in GRAM_TO_INDEX.items():
        assert GRAM_LABELS[index] == label


def test_to_targets_converts_species_ids() -> None:
    """to_targets harus mengubah species_id menjadi indeks numerik."""
    shape, gram = to_targets(["escherichia_coli", "staphylococcus_aureus"])
    assert shape == [SHAPE_TO_INDEX["bacilli"], SHAPE_TO_INDEX["cocci"]]
    assert gram == [GRAM_TO_INDEX["negatif"], GRAM_TO_INDEX["positif"]]


def test_to_targets_handles_empty_list() -> None:
    """Daftar kosong menghasilkan dua daftar kosong, bukan error."""
    shape, gram = to_targets([])
    assert shape == []
    assert gram == []


def test_to_targets_rejects_unknown_species() -> None:
    """Spesies di luar tabel harus ditolak, bukan diam-diam dilewati."""
    with pytest.raises(KeyError):
        to_targets(["spesies_yang_tidak_ada"])


def test_to_targets_rejects_candida() -> None:
    """Candida tidak punya entri lookup, jadi harus ditolak."""
    with pytest.raises(KeyError):
        to_targets(["candida_albicans"])


def test_unmapped_species_lists_unknowns() -> None:
    """Spesies tak terpetakan harus bisa dilaporkan tanpa exception."""
    assert unmapped_species(["escherichia_coli", "bakteri_misterius"]) == [
        "bakteri_misterius"
    ]


def test_unmapped_species_deduplicates() -> None:
    """Spesies tak terpetaman yang sama tidak boleh diulang."""
    assert unmapped_species(["x", "x", "y"]) == ["x", "y"]


def test_unmapped_species_on_full_dataset_returns_empty() -> None:
    """Seluruh spesies trainable harus terpetakan."""
    assert unmapped_species(list(TRAINABLE_SPECIES_IDS)) == []


def test_excluded_from_lookup_reports_candida() -> None:
    """Spesies yang dieksklusi harus dilaporkan sebagai tak terpetakan."""
    assert "candida_albicans" in excluded_from_lookup()


def test_lookup_summary_counts_match_expected() -> None:
    """Distribusi label harus cocok dengan hasil verifikasi manual."""
    summary = lookup_summary()
    assert summary["total_species"] == 32
    assert summary["shape"]["bacilli"] == 23
    assert summary["shape"]["cocci"] == 9
    assert summary["gram"]["positif"] == 23
    assert summary["gram"]["negatif"] == 9


def test_class_weights_from_train_only() -> None:
    """Bobot kelas harus dihitung dari data latih saja."""
    weights = class_weights([0, 0, 0, 0, 1, 1])
    assert len(weights) == N_SHAPE_CLASSES
    assert weights[0] < weights[1], "kelas minor harus dapat bobot lebih besar"
    assert float(weights.sum()) > 0


def test_class_weights_follow_sklearn_formula() -> None:
    """Bobot mengikuti rumus sklearn: jumlah sampel dibagi (kelas x frekuensi)."""
    targets = [0] * 8 + [1] * 2
    weights = class_weights(targets)
    expected = [len(targets) / (2 * 8), len(targets) / (2 * 2)]
    assert float(weights[0]) == pytest.approx(expected[0])
    assert float(weights[1]) == pytest.approx(expected[1])


def test_class_weights_equal_for_balanced_data() -> None:
    """Data seimbang memberi bobot sama untuk tiap kelas."""
    weights = class_weights([0, 1, 0, 1])
    assert float(weights[0]) == pytest.approx(1.0)
    assert float(weights[1]) == pytest.approx(1.0)


def test_class_weights_rejects_empty() -> None:
    """Daftar target kosong tidak menghasilkan bobot yang masuk akal."""
    with pytest.raises(ValueError):
        class_weights([])


def test_class_weights_rejects_single_class() -> None:
    """Hanya satu kelas tidak cukup untuk menghitung bobot tidak seimbang."""
    with pytest.raises(ValueError):
        class_weights([0, 0, 0])


def test_class_weights_respects_n_classes() -> None:
    """Panjang bobot harus sama dengan jumlah kelas yang diminta."""
    weights = class_weights([0, 0, 1], n_classes=N_GRAM_CLASSES)
    assert weights.shape == (N_GRAM_CLASSES,)
    assert isinstance(weights, torch.Tensor)


def test_class_weights_handles_missing_class() -> None:
    """Kelas yang tidak muncul diberi bobot satu, bukan nol."""
    weights = class_weights([0, 0, 1], n_classes=3)
    assert weights.shape == (3,)
    assert float(weights[2]) == 1.0