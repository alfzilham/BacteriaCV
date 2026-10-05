"""Tes untuk konfigurasi terpusat."""

from __future__ import annotations

from pathlib import Path

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
    assert config.GRAM_LABELS == ("positive", "negative")
    assert config.N_GRAM_CLASSES == 2


def test_feature_dim_is_2048() -> None:
    """ResNet-50 menghasilkan 2048-d setelah global average pooling."""
    assert config.FEATURE_DIM == 2048


def test_head_sizes_match_class_count() -> None:
    """Ukuran head mengikuti jumlah kelas."""
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
    assert config.INDEX_SEED == 20260203
    assert config.TRAIN_SEED == 1337
    assert config.TRAIN_SEED != config.INDEX_SEED


def test_no_stale_random_seed_alias() -> None:
    """RANDOM_SEED sudah dihapus, hanya INDEX_SEED yang dipakai."""
    assert not hasattr(config, "RANDOM_SEED")


def test_training_hyperparameters_are_sane() -> None:
    """Nilai pelatihan harus berada di rentang yang bermakna.

    Batas longgar seperti 0 < LR < 1 tidak menangkap perubahan learning rate
    dari 1e-3 ke 0,9 yang akan merusak pelatihan.
    """
    assert 1e-5 <= config.LEARNING_RATE <= 1e-2, config.LEARNING_RATE
    assert 1e-6 <= config.WEIGHT_DECAY <= 1e-2, config.WEIGHT_DECAY
    assert 50 <= config.MAX_EPOCHS <= 500, config.MAX_EPOCHS
    assert 5 <= config.EARLY_STOPPING_PATIENCE <= 50, config.EARLY_STOPPING_PATIENCE
    assert 8 <= config.BATCH_SIZE <= 128, config.BATCH_SIZE
    assert config.N_FOLDS == 5
    assert config.MIN_DELTA > 0


def test_head_sizes_relate_to_feature_dim() -> None:
    """Ukuran head mengikuti jumlah kelas, bukan dimensi fitur."""
    assert config.HEAD_A_SIZE == config.N_SHAPE_CLASSES == 2
    assert config.HEAD_B_SIZE == config.N_GRAM_CLASSES == 2
    assert config.FEATURE_DIM == 2048
    assert config.HEAD_A_SIZE != config.FEATURE_DIM


def test_paths_are_inside_project() -> None:
    """Semua folder keluaran harus di dalam root proyek."""
    from bacteriacv.paths import PROJECT_ROOT

    for path in (config.FEATURES_DIR, config.CHECKPOINT_DIR, config.APP_DIR, config.STATIC_DIR):
        assert path.is_relative_to(PROJECT_ROOT), path


def test_max_upload_is_20_megabytes() -> None:
    """DESIGN bagian 7 menetapkan batas 20 MB."""
    assert config.MAX_UPLOAD_BYTES == 20 * 1024 * 1024


def _config_constant_names() -> set[str]:
    """Kumpulkan nama konstanta yang benar-benar dideklarasikan di config.py."""
    import ast

    from bacteriacv.paths import PROJECT_ROOT

    source = (PROJECT_ROOT / "bacteriacv" / "config.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    names.add(target.id)
    return names


def _scanned_source_files() -> list[Path]:
    """Daftar berkas yang wajib bebas duplikasi konstanta config.

    Cakupannya bukan hanya paket bacteriacv. Backend app/ juga mengimpor
    konstanta yang sama, jadi MAX_UPLOAD_BYTES atau ALLOWED_SUFFIXES yang
    ditulis ulang di sana harus tertangkap.
    """
    from bacteriacv.paths import PROJECT_ROOT

    files = sorted((PROJECT_ROOT / "bacteriacv").rglob("*.py"))
    app_dir = PROJECT_ROOT / "app"
    if app_dir.is_dir():
        files.extend(sorted(app_dir.rglob("*.py")))
    return [path for path in files if path.name != "config.py"]


def test_no_hyperparameter_duplicated_outside_config() -> None:
    """Konstanta config.py tidak boleh dideklarasikan ulang di modul lain.

    config.py menyatakan dirinya sebagai sumber kebenaran tunggal. Tanpa tes ini,
    mengubah config.INDEX_SEED tidak akan mengubah index.csv karena modul lain
    masih memakai salinan lokal.
    """
    import ast

    reserved = _config_constant_names()
    assert reserved, "config.py harus punya konstanta"

    offenders: list[str] = []
    for source in _scanned_source_files():
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in reserved:
                    offenders.append(f"{source.name}: {target.id}")

    assert not offenders, f"konstanta config diduplikasi di: {sorted(offenders)}"


def test_guard_covers_app_package_when_it_exists() -> None:
    """Penjaga duplikasi harus ikut memindai app/ begitu backend dibuat."""
    from bacteriacv.paths import PROJECT_ROOT

    scanned = {path.relative_to(PROJECT_ROOT).as_posix() for path in _scanned_source_files()}
    if (PROJECT_ROOT / "app").is_dir():
        assert any(name.startswith("app/") for name in scanned), scanned
    else:
        assert scanned, "minimal paket bacteriacv harus dipindai"


def test_build_index_imports_from_config() -> None:
    """build_index harus mengambil hyperparameter dari config, bukan menyalin."""
    from bacteriacv.datasets import build_index

    assert build_index.INDEX_SEED == config.INDEX_SEED
    assert build_index.N_FOLDS == config.N_FOLDS
    assert build_index.TRAIN_FRACTION == config.TRAIN_FRACTION
    assert build_index.VAL_FRACTION == config.VAL_FRACTION


def test_min_images_threshold_lives_in_config() -> None:
    """Ambang integritas data harus terpusat di config.py."""
    from bacteriacv.datasets import extract

    assert extract.MIN_IMAGES_PER_SPECIES == config.MIN_IMAGES_PER_SPECIES
    assert config.MIN_IMAGES_PER_SPECIES > 0


def test_allowed_suffixes_cover_design_formats() -> None:
    """DESIGN bagian 5 menyebut PNG, JPG, dan TIFF."""
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        assert suffix in config.ALLOWED_SUFFIXES