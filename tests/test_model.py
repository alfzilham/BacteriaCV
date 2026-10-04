"""Tes untuk model dua head dengan backbone beku."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from bacteriacv.config import (
    FEATURE_DIM,
    GRAM_LABELS,
    N_GRAM_CLASSES,
    N_SHAPE_CLASSES,
    SHAPE_LABELS,
)
from bacteriacv.model import BacteriaNet, build_model, load_state_dicts, predict_batch


def test_backbone_is_frozen() -> None:
    """ARCHITECTURE C2: seluruh parameter backbone dibekukan."""
    model = build_model(pretrained=False)

    backbone_parameters = list(model.backbone.parameters())
    assert backbone_parameters, "backbone harus punya parameter"
    assert all(not p.requires_grad for p in backbone_parameters)


def test_only_heads_are_trainable() -> None:
    """Hanya head yang boleh diperbarui saat pelatihan."""
    model = build_model(pretrained=False)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "head harus bisa dilatih"
    assert all(n.startswith("head_") for n in trainable), trainable


def test_head_a_output_dimension() -> None:
    """Keputusan D1: Head A dua kelas bentuk."""
    model = build_model(pretrained=False)
    features = torch.randn(8, FEATURE_DIM)

    assert model.head_a(features).shape == (8, N_SHAPE_CLASSES)


def test_head_b_output_dimension() -> None:
    """Head B dua kelas status Gram."""
    model = build_model(pretrained=False)
    features = torch.randn(8, FEATURE_DIM)

    assert model.head_b(features).shape == (8, N_GRAM_CLASSES)


def test_head_a_softmax_sums_to_one() -> None:
    """Head A memakai softmax, jadi probabilitasnya berjumlah satu."""
    model = build_model(pretrained=False).eval()
    features = torch.randn(16, FEATURE_DIM)

    probabilities = model.head_a(features).softmax(dim=-1)

    assert torch.allclose(
        probabilities.sum(dim=-1), torch.ones(16), atol=1e-5
    )


def test_head_b_sigmoid_stays_in_range() -> None:
    """Head B memakai sigmoid, jadi probabilitasnya di rentang 0 sampai 1."""
    model = build_model(pretrained=False).eval()
    features = torch.randn(16, FEATURE_DIM)

    probabilities = model.head_b(features).sigmoid()

    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()


def test_heads_do_not_share_parameters() -> None:
    """Dua head harus punya parameter terpisah."""
    model = build_model(pretrained=False)

    a_ids = {id(p) for p in model.head_a.parameters()}
    b_ids = {id(p) for p in model.head_b.parameters()}

    assert not (a_ids & b_ids)


def test_backbone_stays_in_eval_mode_after_train() -> None:
    """Memanggil train() tidak boleh mengubah backbone ke mode training.

    BatchNorm dan Dropout pada backbone akan mengubah statistik fitur bila
    dibiarkan aktif.
    """
    model = build_model(pretrained=False)

    model.train()

    assert not model.backbone.training
    assert model.head_a.training


def test_extract_features_dimension() -> None:
    """Backbone menghasilkan vektor 2048-d sesuai ARCHITECTURE bagian 2."""
    model = build_model(pretrained=False).eval()
    batch = torch.randn(2, 3, 224, 224)

    with torch.inference_mode():
        features = model.extract_features(batch)

    assert features.shape == (2, FEATURE_DIM)


def test_extract_features_does_not_build_graph() -> None:
    """Ekstraksi fitur tidak boleh menyimpan graph komputasi."""
    model = build_model(pretrained=False).eval()
    batch = torch.randn(2, 3, 224, 224)

    features = model.extract_features(batch)

    assert not features.requires_grad


def test_predict_batch_returns_labels_and_confidence() -> None:
    """predict_batch harus mengembalikan label dan confidence kedua head."""
    model = build_model(pretrained=False).eval()
    features = torch.randn(6, FEATURE_DIM)

    shape, shape_conf, gram, gram_conf = predict_batch(model, features)

    assert shape.shape == (6,)
    assert shape_conf.shape == (6,)
    assert gram.shape == (6,)
    assert gram_conf.shape == (6,)


def test_predict_batch_confidence_in_range() -> None:
    """Confidence harus berada di rentang 0 sampai 1."""
    model = build_model(pretrained=False).eval()
    features = torch.randn(6, FEATURE_DIM)

    _, shape_conf, _, gram_conf = predict_batch(model, features)

    assert (shape_conf >= 0).all() and (shape_conf <= 1).all()
    assert (gram_conf >= 0).all() and (gram_conf <= 1).all()


def test_predict_batch_shape_indices_are_valid() -> None:
    """Indeks prediksi harus berada di dalam daftar label."""
    model = build_model(pretrained=False).eval()
    features = torch.randn(32, FEATURE_DIM)

    shape, _, gram, _ = predict_batch(model, features)

    assert (shape >= 0).all() and (shape < len(SHAPE_LABELS)).all()
    assert (gram >= 0).all() and (gram < len(GRAM_LABELS)).all()


def test_gram_confidence_matches_predicted_class() -> None:
    """Confidence Head B harus sesuai probabilitas kelas yang dipilih."""
    model = build_model(pretrained=False).eval()
    features = torch.randn(16, FEATURE_DIM)

    gram, gram_conf = predict_batch(model, features)[2:]

    probabilities = model.head_b(features).sigmoid()
    expected = torch.where(gram == 1, probabilities[:, 1], probabilities[:, 0])
    assert torch.allclose(gram_conf, expected, atol=1e-6)


def test_shape_confidence_is_max_probability() -> None:
    """Confidence Head A harus probabilitas terbesar."""
    model = build_model(pretrained=False).eval()
    features = torch.randn(16, FEATURE_DIM)

    shape, shape_conf = predict_batch(model, features)[:2]

    probabilities = model.head_a(features).softmax(dim=-1)
    assert torch.allclose(shape_conf, probabilities.max(dim=-1).values, atol=1e-6)
    assert torch.equal(shape, probabilities.argmax(dim=-1))


def test_state_dict_contains_only_heads() -> None:
    """Checkpoint hanya menyimpan head, bukan backbone.

    Backbone diambil ulang dari torchvision, jadi menyimpannya membuat
    checkpoint besar tanpa manfaat.
    """
    model = build_model(pretrained=False)

    state = model.head_state_dict()

    assert set(state) == {"head_a.weight", "head_a.bias", "head_b.weight", "head_b.bias"}
    assert all(not key.startswith("backbone") for key in state)


def test_state_dict_roundtrip_preserves_weights() -> None:
    """Bobot harus pulih persis setelah disimpan dan dimuat."""
    source = build_model(pretrained=False)
    state = source.head_state_dict()

    target = build_model(pretrained=False)
    load_state_dicts(target, state)

    for a, b in zip(source.head_a.parameters(), target.head_a.parameters()):
        assert torch.allclose(a, b)
    for a, b in zip(source.head_b.parameters(), target.head_b.parameters()):
        assert torch.allclose(a, b)


def test_load_state_dicts_rejects_missing_key() -> None:
    """Checkpoint yang tidak lengkap harus ditolak."""
    model = build_model(pretrained=False)

    with pytest.raises(RuntimeError):
        load_state_dicts(model, {"head_a.weight": torch.zeros(2, FEATURE_DIM)})


def test_model_is_module() -> None:
    """BacteriaNet harus berupa nn.Module agar bisa dilatih PyTorch."""
    assert issubclass(BacteriaNet, nn.Module)


def test_pretrained_flag_changes_backbone_weights() -> None:
    """Bendera pretrained harus benar-benar memuat bobot ImageNet."""
    without = build_model(pretrained=False)
    with_weights = build_model(pretrained=True)

    assert not torch.allclose(
        without.backbone.conv1.weight, with_weights.backbone.conv1.weight
    )