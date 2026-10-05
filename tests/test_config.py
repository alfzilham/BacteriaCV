"""Tests for the centralised configuration."""

from __future__ import annotations

from pathlib import Path

import bacteriacv.config as config


def test_image_size_is_224() -> None:
    """ARCHITECTURE C1 sets the resize to 224."""
    assert config.IMAGE_SIZE == 224


def test_n_shape_classes_is_two() -> None:
    """Decision D1: the spiral class has no species on DIBaS."""
    assert config.N_SHAPE_CLASSES == 2


def test_shape_labels_exclude_spiral() -> None:
    """The populated labels are only cocci and bacilli."""
    assert config.SHAPE_LABELS == ("cocci", "bacilli")


def test_shape_labels_full_keeps_spiral() -> None:
    """The full enum still holds spiral as the marker of an empty class."""
    assert config.SHAPE_LABELS_FULL == ("cocci", "bacilli", "spiral")
    assert config.SHAPE_UNPOPULATED == "spiral"
    assert config.SHAPE_UNPOPULATED not in config.SHAPE_LABELS


def test_gram_labels_are_positive_and_negative() -> None:
    """Head B stays at two Gram status classes."""
    assert config.GRAM_LABELS == ("positive", "negative")
    assert config.N_GRAM_CLASSES == 2


def test_feature_dim_is_2048() -> None:
    """ResNet-50 yields 2048-d after global average pooling."""
    assert config.FEATURE_DIM == 2048


def test_head_sizes_match_class_count() -> None:
    """The head size follows the class count."""
    assert config.HEAD_A_SIZE == config.N_SHAPE_CLASSES
    assert config.HEAD_B_SIZE == config.N_GRAM_CLASSES


def test_augment_variants_is_four() -> None:
    """Decision D3: four augmentation variants on the train data."""
    assert config.AUGMENT_VARIANTS == 4


def test_imagenet_stats_are_standard() -> None:
    """Normalisation must use the ImageNet statistics torchvision uses."""
    assert config.IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert config.IMAGENET_STD == (0.229, 0.224, 0.225)


def test_seeds_are_pinned_and_distinct() -> None:
    """Seeds must be deterministic and different per stage."""
    assert config.INDEX_SEED == 20260203
    assert config.TRAIN_SEED == 1337
    assert config.TRAIN_SEED != config.INDEX_SEED


def test_no_stale_random_seed_alias() -> None:
    """RANDOM_SEED is gone, only INDEX_SEED is used."""
    assert not hasattr(config, "RANDOM_SEED")


def test_training_hyperparameters_are_sane() -> None:
    """The training figures must sit in a meaningful range.

    A loose bound such as 0 < LR < 1 does not catch a learning rate change
    from 1e-3 to 0.9, which would break training.
    """
    assert 1e-5 <= config.LEARNING_RATE <= 1e-2, config.LEARNING_RATE
    assert 1e-6 <= config.WEIGHT_DECAY <= 1e-2, config.WEIGHT_DECAY
    assert 50 <= config.MAX_EPOCHS <= 500, config.MAX_EPOCHS
    assert 5 <= config.EARLY_STOPPING_PATIENCE <= 50, config.EARLY_STOPPING_PATIENCE
    assert 8 <= config.BATCH_SIZE <= 128, config.BATCH_SIZE
    assert config.N_FOLDS == 5
    assert config.MIN_DELTA > 0


def test_head_sizes_relate_to_feature_dim() -> None:
    """The head size follows the class count, not the feature dimension."""
    assert config.HEAD_A_SIZE == config.N_SHAPE_CLASSES == 2
    assert config.HEAD_B_SIZE == config.N_GRAM_CLASSES == 2
    assert config.FEATURE_DIM == 2048
    assert config.HEAD_A_SIZE != config.FEATURE_DIM


def test_paths_are_inside_project() -> None:
    """Every output folder must sit inside the project root."""
    from bacteriacv.paths import PROJECT_ROOT

    for path in (config.FEATURES_DIR, config.CHECKPOINT_DIR, config.APP_DIR, config.STATIC_DIR):
        assert path.is_relative_to(PROJECT_ROOT), path


def test_max_upload_is_20_megabytes() -> None:
    """DESIGN section 7 sets the 20 MB limit."""
    assert config.MAX_UPLOAD_BYTES == 20 * 1024 * 1024


def _config_constant_names() -> set[str]:
    """Collect the constant names actually declared in config.py."""
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
    """The files that must be free of duplicated config constants.

    The scope is not only the bacteriacv package. The app/ backend imports the
    same constants, so a rewritten MAX_UPLOAD_BYTES or ALLOWED_SUFFIXES there
    must be caught.
    """
    from bacteriacv.paths import PROJECT_ROOT

    files = sorted((PROJECT_ROOT / "bacteriacv").rglob("*.py"))
    app_dir = PROJECT_ROOT / "app"
    if app_dir.is_dir():
        files.extend(sorted(app_dir.rglob("*.py")))
    return [path for path in files if path.name != "config.py"]


def test_no_hyperparameter_duplicated_outside_config() -> None:
    """config.py constants must not be redeclared in another module.

    config.py declares itself the single source of truth. Without this test,
    changing config.INDEX_SEED would not change index.csv because another module
    would still use a local copy.
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
    """The duplicate guard must scan app/ too now that the backend exists."""
    from bacteriacv.paths import PROJECT_ROOT

    scanned = {path.relative_to(PROJECT_ROOT).as_posix() for path in _scanned_source_files()}
    if (PROJECT_ROOT / "app").is_dir():
        assert any(name.startswith("app/") for name in scanned), scanned
    else:
        assert scanned, "minimal paket bacteriacv harus dipindai"


def test_build_index_imports_from_config() -> None:
    """build_index must read its hyperparameters from config, not copy them."""
    from bacteriacv.datasets import build_index

    assert build_index.INDEX_SEED == config.INDEX_SEED
    assert build_index.N_FOLDS == config.N_FOLDS
    assert build_index.TRAIN_FRACTION == config.TRAIN_FRACTION
    assert build_index.VAL_FRACTION == config.VAL_FRACTION


def test_min_images_threshold_lives_in_config() -> None:
    """The data integrity threshold must be centralised in config.py."""
    from bacteriacv.datasets import extract

    assert extract.MIN_IMAGES_PER_SPECIES == config.MIN_IMAGES_PER_SPECIES
    assert config.MIN_IMAGES_PER_SPECIES > 0


def test_allowed_suffixes_cover_design_formats() -> None:
    """DESIGN section 5 names PNG, JPG, and TIFF."""
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        assert suffix in config.ALLOWED_SUFFIXES