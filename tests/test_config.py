"""Tes untuk konfigurasi terpusat."""

from __future__ import annotations

import bacteriacv.config as config


def test_image_size_is_224() -> None:
    """SPEC bagian C1 menetapkan resize ke 224."""
    assert config.IMAGE_SIZE == 224


def test_n_shape_classes_is_two() -> None:
    """Keputusan D1: kelas spiral tidak memiliki spesies di DIBaS."""
    assert config.N_SHAPE_CLASSES == 2


def test_shape_labels_exclude_spiral() -> None:
    """Label aktif hanya cocci dan bacilli."""
    assert config.SHAPE_LABELS == ("cocci", "bacilli")


def test_shape_labels_full_keeps_spiral() -> None:
    """Enum lengkap tetap memuat spiral sebagai penanda kelas kosong."""
    assert config.SHAPE_LABELS_FULL == ("cocci", "bacilli", "spiral")
    assert config.SHAPE_UNPOPULATED == "spiral"
    assert config.SHAPE_UNPOPULATED not in config.SHAPE_LABELS


def test_gram_labels_are_positive_and_negative() -> None:
    """Head B tetap dua kelas status Gram."""
    assert config.GRAM_LABELS == ("positif", "negatif")
    assert config.N_GRAM_CLASSES == 2


def test_feature_dim_is_2048() -> None:
    """ResNet-50 menghasilkan 2048-d setelah global average pooling."""
    assert config.FEATURE_DIM == 2048


def test_head_sizes_match_feature_dim() -> None:
    """Ukuran head harus mengikuti dimensi fitur."""
    assert config.HEAD_A_SIZE == config.N_SHAPE_CLASSES
    assert config.HEAD_B_SIZE == config.N_GRAM_CLASSES


def test_augment_variants_is_four() -> None:
    """Keputusan D3: empat varian augmentasi pada data latih."""
    assert config.AUGMENT_VARIANTS == 4


def test_imagenet_stats_are_standard() -> None:
    """Normalisasi harus memakai statistik ImageNet yang dipakai torchvision."""
    assert config.IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert config.IMAGENET_STD == (0.229, 0.224, 0.225)


def test_seeds_are_pinned_and_distinct() -> None:
    """Seed harus deterministik dan berbeda antar tahap."""
    assert config.RANDOM_SEED == config.INDEX_SEED == 20260203
    assert config.TRAIN_SEED == 1337
    assert config.TRAIN_SEED != config.INDEX_SEED


def test_training_hyperparameters_are_sane() -> None:
    """Nilai pelatihan harus masuk akal untuk dataset kecil."""
    assert 0 < config.LEARNING_RATE < 1
    assert config.MAX_EPOCHS > 0
    assert config.EARLY_STOPPING_PATIENCE > 0
    assert config.BATCH_SIZE > 0
    assert config.N_FOLDS == 5


def test_paths_are_inside_project() -> None:
    """Semua folder keluaran harus di dalam root proyek."""
    from bacteriacv.paths import PROJECT_ROOT

    for path in (config.FEATURES_DIR, config.CHECKPOINT_DIR, config.APP_DIR, config.STATIC_DIR):
        assert path.is_relative_to(PROJECT_ROOT), path


def test_max_upload_is_20_megabytes() -> None:
    """DESIGN bagian 7 menetapkan batas 20 MB."""
    assert config.MAX_UPLOAD_BYTES == 20 * 1024 * 1024


def test_allowed_suffixes_cover_design_formats() -> None:
    """DESIGN bagian 5 menyebut PNG, JPG, dan TIFF."""
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        assert suffix in config.ALLOWED_SUFFIXES