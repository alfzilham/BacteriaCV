"""Frozen feature extraction and the two head training loop.

Feature caching lives in this module rather than a separate one, because
train.py is already listed in ARCHITECTURE section 4 and FEATURES_DIR already
exists in config. The consequence is a single source of truth.

Reason for caching: the backbone is frozen, so the feature vector of an image
does not change between epochs. Passing every image through the backbone each epoch
costs 76 ms times the image count per epoch. With caching the backbone runs
once for all train data, and later epochs only run the two
Linear(2048, 2) layers whose weights are a few thousand parameters.

Test data is never used for early stopping. Test data is evaluated
only once, after the best weights are restored. This rule is enforced by
test_train_does_not_use_test_for_early_stopping.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import cv2.utils.logging
import numpy as np
import torch
from torch import nn

from .config import (
    AUGMENT_VARIANTS,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EARLY_STOPPING_PATIENCE,
    FEATURE_DIM,
    FEATURES_DIR,
    GRAM_LABELS,
    IMAGE_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_DELTA,
    N_GRAM_CLASSES,
    N_SHAPE_CLASSES,
    SHAPE_LABELS,
    TRAIN_SEED,
    WEIGHT_DECAY,
)
from .label_map import class_weights, to_targets
from .model import BacteriaNet, build_model, save_checkpoint
from .paths import INDEX_PATH, PROJECT_ROOT
from .preprocess import augment_train_variants, load_image, preprocess_tensor

# Bump this value when the preprocessing pipeline changes, so an old cache is not
# reused for images already processed under different rules.
CACHE_VERSION = "features-v1"

MATRIX_NAME = "features.npy"
LABELS_NAME = "labels.npz"
REPORT_NAME = "training_report.json"
CHECKPOINT_NAME = "heads.pt"

EXTRACT_BATCH_SIZE = 16
PROGRESS_EVERY = 50

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
REQUIRED_SPLITS = (TRAIN_SPLIT, VAL_SPLIT)


# =====================================================================
# Feature store
# =====================================================================


@dataclass(frozen=True)
class FeatureStore:
    """Frozen feature vectors together with their supporting labels.

    Attributes:
        matrix: An N x FEATURE_DIM float32 matrix.
        species_ids: The species_id of each row, from which head labels are derived.
        splits: The split name of each row.
        paths: The source image path of each row, relative to the image folder.
    """

    matrix: np.ndarray
    species_ids: list[str]
    splits: list[str]
    paths: list[str]


def save_feature_store(store: FeatureStore, directory: Path | str) -> Path:
    """Write the feature store to disk.

    Args:
        store: The feature store being written.
        directory: The target folder, created when missing.

    Returns:
        The folder the files were written into.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    matrix = np.ascontiguousarray(store.matrix, dtype=np.float32)
    np.save(target / MATRIX_NAME, matrix)
    np.savez(
        target / LABELS_NAME,
        species_ids=np.array(store.species_ids, dtype=object).astype("U"),
        splits=np.array(store.splits, dtype=object).astype("U"),
        paths=np.array(store.paths, dtype=object).astype("U"),
    )
    return target


def load_feature_store(directory: Path | str) -> FeatureStore:
    """Read the feature store from disk.

    Args:
        directory: The folder holding features.npy and labels.npz.

    Returns:
        A FeatureStore recovered exactly as it was saved.

    Raises:
        FileNotFoundError: When the folder or the matrix file is missing.
        ValueError: When the matrix dimensions or label length are inconsistent.
    """
    source = Path(directory)
    matrix_path = source / MATRIX_NAME
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Feature store not found: {source.name}")

    matrix = np.load(matrix_path)
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"The feature column count does not match FEATURE_DIM, "
            f"check the matrix dimensions: shape {matrix.shape}"
        )

    labels = np.load(source / LABELS_NAME, allow_pickle=False)
    species_ids = [str(value) for value in labels["species_ids"]]
    splits = [str(value) for value in labels["splits"]]
    paths = [str(value) for value in labels["paths"]]

    lengths = {len(matrix), len(species_ids), len(splits), len(paths)}
    if len(lengths) != 1:
        raise ValueError(
            f"The label count does not match the feature row count: "
            f"{len(matrix)}, {len(species_ids)}, {len(splits)}, {len(paths)}"
        )

    return FeatureStore(matrix, species_ids, splits, paths)


def cache_signature(images_dir: Path | str, rows: list[dict], augment: bool) -> str:
    """Compute the cache fingerprint from image content and pipeline rules.

    The fingerprint holds the filename, size, and modification time of each image.
    Changing a single pixel in a source image therefore invalidates the cache.

    Args:
        images_dir: The base folder relative paths are computed from.
        rows: The index rows.
        augment: Whether augmentation is skipped during extraction.

    Returns:
        A 16 character hex fingerprint.
    """
    base = Path(images_dir)
    digest = hashlib.sha256()
    digest.update(
        f"{CACHE_VERSION}|augment={augment}|variants={AUGMENT_VARIANTS}"
        f"|size={IMAGE_SIZE}|dim={FEATURE_DIM}\n".encode()
    )

    for row in sorted(rows, key=lambda item: item["path"]):
        location = base / row["path"]
        try:
            stat = location.stat()
            marker = f"{stat.st_size}-{stat.st_mtime_ns}"
        except OSError:
            marker = "absent"
        digest.update(f"{row['path']}|{row.get('split', '')}|{marker}\n".encode())

    return digest.hexdigest()[:16]


def extract_features_for_rows(
    model: BacteriaNet,
    images_dir: Path | str,
    rows: list[dict],
    augment: bool = False,
    batch_size: int = EXTRACT_BATCH_SIZE,
    verbose: bool = False,
) -> FeatureStore:
    """Extract the feature vector for every index row.

    Augmentation applies only to rows with split train, per decision D3.
    The train row count becomes AUGMENT_VARIANTS times larger: one original
    image plus AUGMENT_VARIANTS - 1 augmented variants.

    Extraction uses preprocess_tensor, not preprocess, so segmentation
    does not run. Segmentation exists only for the visualisation panels and does
    not affect the model metrics.

    Args:
        model: The BacteriaNet model. Its backbone is frozen by the model itself.
        images_dir: The base folder for relative paths in the index rows.
        rows: Index rows with path, species_id and split keys.
        augment: When True, train rows get augmented variants.
        batch_size: The number of images per backbone inference.
        verbose: When True, print progress every PROGRESS_EVERY images.

    Returns:
        A FeatureStore holding the feature matrix and its labels.

    Raises:
        FileNotFoundError: When an image cannot be found.
    """
    base = Path(images_dir)
    tensors: list[torch.Tensor] = []
    species_ids: list[str] = []
    splits: list[str] = []
    paths: list[str] = []

    for position, row in enumerate(rows):
        relative = str(row["path"])
        image = load_image(base / relative)
        split = str(row.get("split", ""))

        if augment and split == TRAIN_SPLIT:
            variants = [
                image,
                *augment_train_variants(
                    image, AUGMENT_VARIANTS - 1, seed=TRAIN_SEED + position
                ),
            ]
        else:
            variants = [image]

        for offset, variant in enumerate(variants):
            label = relative if offset == 0 else f"{relative}#aug{offset}"
            tensors.append(preprocess_tensor(variant))
            species_ids.append(str(row["species_id"]))
            splits.append(split)
            paths.append(label)

        if verbose and (position + 1) % PROGRESS_EVERY == 0:
            print(
                f"Extracting {position + 1}/{len(rows)} images, "
                f"{len(tensors)} feature rows",
                flush=True,
            )

    matrix = np.zeros((len(tensors), FEATURE_DIM), dtype=np.float32)
    for start in range(0, len(tensors), batch_size):
        chunk = tensors[start : start + batch_size]
        batch = torch.stack(chunk)
        matrix[start : start + len(chunk)] = model.extract_features(batch).numpy()

    return FeatureStore(matrix, species_ids, splits, paths)


def build_or_load_features(
    model: BacteriaNet,
    images_dir: Path | str,
    rows: list[dict],
    augment: bool = False,
    cache_root: Path | str | None = None,
    verbose: bool = False,
) -> tuple[FeatureStore, Path, bool]:
    """Take features from cache when available, re-extract when not.

    Args:
        model: The BacteriaNet model.
        images_dir: The base folder for relative paths in the index rows.
        rows: The index rows.
        augment: When True, train rows get augmented variants.
        cache_root: The cache root folder. Defaults to FEATURES_DIR from config.
        verbose: When True, print progress during extraction.

    Returns:
        Three elements: (store, cache_folder, cache_used).
    """
    root = Path(cache_root) if cache_root is not None else FEATURES_DIR
    directory = root / f"store_{cache_signature(images_dir, rows, augment)}"

    if (directory / MATRIX_NAME).is_file():
        return load_feature_store(directory), directory, True

    store = extract_features_for_rows(
        model, images_dir, rows, augment=augment, verbose=verbose
    )
    save_feature_store(store, directory)
    return store, directory, False


# =====================================================================
# Metrics
# =====================================================================


def f1_macro(truth: np.ndarray, prediction: np.ndarray, n_classes: int) -> float:
    """Compute macro F1 without depending on sklearn.

    Args:
        truth: The true labels.
        prediction: The predicted labels.
        n_classes: The number of classes evaluated.

    Returns:
        The mean F1 per class. A class that never appears in either the true
        or the predicted labels is given a score of zero so an empty class does
        not raise the mean.
    """
    scores: list[float] = []
    for index in range(n_classes):
        true_positive = int(np.sum((truth == index) & (prediction == index)))
        false_positive = int(np.sum((truth != index) & (prediction == index)))
        false_negative = int(np.sum((truth == index) & (prediction != index)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores)) if scores else 0.0


def accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Compute accuracy as the proportion of correct predictions.

    Args:
        truth: The true labels.
        prediction: The predicted labels.

    Returns:
        A value between zero and one, zero when the arrays are empty.
    """
    if truth.size == 0:
        return 0.0
    return float(np.mean(truth == prediction))


def _head_predictions(
    model: BacteriaNet, features: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Take the predictions of both heads for a batch of features.

    Args:
        model: The BacteriaNet model.
        features: An N x FEATURE_DIM tensor.

    Returns:
        A (shape_prediction, gram_prediction) pair of integer arrays.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            shape = model.head_a(features).argmax(dim=-1)
            gram = (model.head_b(features)[:, 1].sigmoid() > 0.5).long()
    finally:
        if was_training:
            model.train()
    return shape.numpy(), gram.numpy()


def _evaluate(
    model: BacteriaNet,
    features: torch.Tensor,
    shape_truth: torch.Tensor,
    gram_truth: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Evaluate one split on all four metrics.

    Args:
        model: The BacteriaNet model.
        features: The feature matrix of all data.
        shape_truth: The shape labels of all data.
        gram_truth: The Gram labels of all data.
        mask: A row mask for the rows in this split.

    Returns:
        Four values: (shape_f1, gram_f1, shape_accuracy, gram_accuracy).
    """
    selected = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if selected.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0

    shape_pred, gram_pred = _head_predictions(model, features[selected])
    shape_true = shape_truth[selected].numpy()
    gram_true = gram_truth[selected].numpy()

    return (
        f1_macro(shape_true, shape_pred, N_SHAPE_CLASSES),
        f1_macro(gram_true, gram_pred, N_GRAM_CLASSES),
        accuracy(shape_true, shape_pred),
        accuracy(gram_true, gram_pred),
    )


# =====================================================================
# Training loop
# =====================================================================


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters. All defaults come from config.

    Attributes:
        epochs: The upper bound on epochs.
        patience: Epochs without improvement before stopping.
        learning_rate: The AdamW learning rate.
        weight_decay: The AdamW weight decay.
        min_delta: The minimum improvement for an epoch to count as an improvement.
        batch_size: Rows per optimizer step.
        seed: The seed for head initialisation and batch shuffling.
        checkpoint_dir: The folder checkpoints are written into.
    """

    epochs: int = MAX_EPOCHS
    patience: int = EARLY_STOPPING_PATIENCE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    min_delta: float = MIN_DELTA
    batch_size: int = BATCH_SIZE
    seed: int = TRAIN_SEED
    checkpoint_dir: Path = CHECKPOINT_DIR


@dataclass(frozen=True)
class TrainingReport:
    """Training result together with the final metrics.

    test_f1_shape and test_f1_gram are one element lists because the test
    data is only evaluated once. The list shape makes that rule visible in
    the report rather than merely implied.

    Attributes:
        epochs_run: The number of epochs actually run.
        best_epoch: The epoch with the best validation score, one based.
        best_val_f1_shape: The shape macro F1 at the best epoch.
        best_val_f1_gram: The Gram macro F1 at the best epoch.
        test_f1_shape: The shape macro F1 list on the test data, length one.
        test_f1_gram: The Gram macro F1 list on the test data, length one.
        test_accuracy_shape: The shape accuracy on the test data.
        test_accuracy_gram: The Gram accuracy on the test data.
        history: The per epoch history without any test metrics.
        checkpoint_path: The location of the best weights checkpoint.
        n_train: The number of train rows after augmentation.
        n_val: The number of validation rows.
        n_test: The number of test rows.
        class_counts: Train row counts per shape and Gram class.
        config: The training configuration used.
    """

    epochs_run: int
    best_epoch: int
    best_val_f1_shape: float
    best_val_f1_gram: float
    test_f1_shape: list[float]
    test_f1_gram: list[float]
    test_accuracy_shape: float
    test_accuracy_gram: float
    history: list[dict[str, float]]
    checkpoint_path: Path | None
    n_train: int
    n_val: int
    n_test: int
    class_counts: dict[str, dict[str, int]]
    config: dict

    def to_dict(self) -> dict:
        """Turn the report into a dictionary that can be serialised to JSON.

        Returns:
            A dictionary with no Path objects and no numpy arrays.
        """
        payload = asdict(self)
        checkpoint = payload.get("checkpoint_path")
        if checkpoint is not None:
            # The filename only. training_report.json is deployed too, so an
            # absolute path inside it would leak the folder structure of the
            # build machine and would break AGENT.md section 3 rule 5.
            payload["checkpoint_path"] = Path(checkpoint).name
        payload["labels"] = {
            "shape": list(SHAPE_LABELS),
            "gram": list(GRAM_LABELS),
        }
        return payload


def _split_masks(splits: list[str]) -> dict[str, torch.Tensor]:
    """Build the row mask for each split.

    Args:
        splits: The split name of each row.

    Returns:
        A dictionary of boolean masks for train, val and test.

    Raises:
        ValueError: When the train or validation split is empty.
    """
    masks = {
        name: torch.tensor([value == name for value in splits], dtype=torch.bool)
        for name in (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)
    }
    for name in REQUIRED_SPLITS:
        if not bool(masks[name].any()):
            raise ValueError(f"Split {name!r} is empty in the feature store.")
    return masks


def _safe_weights(targets: list[int], n_classes: int) -> torch.Tensor:
    """Compute class weights, falling back to one when there is only one class."""
    try:
        return class_weights(targets, n_classes)
    except ValueError:
        return torch.ones(n_classes, dtype=torch.float32)


def train(store: FeatureStore, config: TrainConfig | None = None) -> TrainingReport:
    """Train both heads on the frozen features.

    Args:
        store: The feature store from extraction.
        config: The training hyperparameters. Defaults to TrainConfig().

    Returns:
        A TrainingReport holding the history, the best validation metrics, and one
        test data evaluation.

    Raises:
        ValueError: When the feature dimensions do not match or a required split is empty.
        KeyError: When a species_id is absent from the lookup table.
    """
    settings = config or TrainConfig()

    matrix = np.ascontiguousarray(store.matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"The feature matrix must have shape (N, {FEATURE_DIM}), got {matrix.shape}"
        )

    masks = _split_masks(store.splits)
    shape_targets, gram_targets = to_targets(store.species_ids)

    features = torch.from_numpy(matrix)
    shape_tensor = torch.tensor(shape_targets, dtype=torch.long)
    gram_tensor = torch.tensor(gram_targets, dtype=torch.float32)

    train_mask = masks[TRAIN_SPLIT]
    train_positions = torch.nonzero(train_mask, as_tuple=False).reshape(-1)
    shape_weights = _safe_weights(
        [shape_targets[int(i)] for i in train_positions.tolist()], N_SHAPE_CLASSES
    )
    gram_weights = _safe_weights(
        [gram_targets[int(i)] for i in train_positions.tolist()], N_GRAM_CLASSES
    )

    loss_shape = nn.CrossEntropyLoss(weight=shape_weights)
    loss_gram = nn.BCEWithLogitsLoss(pos_weight=gram_weights[1].reshape(1))

    torch.manual_seed(settings.seed)
    model = build_model(pretrained=False)
    optimizer = torch.optim.AdamW(
        list(model.head_a.parameters()) + list(model.head_b.parameters()),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    generator = torch.Generator().manual_seed(settings.seed)

    history: list[dict[str, float]] = []
    best_score = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_shape = 0.0
    best_gram = 0.0
    since_improvement = 0
    n_train = int(train_positions.numel())
    batch_size = max(1, settings.batch_size)

    for epoch in range(1, settings.epochs + 1):
        model.train()
        order = torch.randperm(n_train, generator=generator)
        total_loss = 0.0

        for start in range(0, n_train, batch_size):
            batch_index = train_positions[order[start : start + batch_size]]
            batch_features = features[batch_index]

            optimizer.zero_grad(set_to_none=True)
            shape_loss = loss_shape(
                model.head_a(batch_features), shape_tensor[batch_index]
            )
            gram_loss = loss_gram(
                model.head_b(batch_features)[:, 1], gram_tensor[batch_index]
            )
            loss = shape_loss + gram_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch_index.numel()

        val_shape_f1, val_gram_f1, val_shape_acc, val_gram_acc = _evaluate(
            model, features, shape_tensor, gram_tensor, masks[VAL_SPLIT]
        )
        score = (val_shape_f1 + val_gram_f1) / 2

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / n_train,
                "val_f1_shape": val_shape_f1,
                "val_f1_gram": val_gram_f1,
                "val_accuracy_shape": val_shape_acc,
                "val_accuracy_gram": val_gram_acc,
            }
        )

        if score > best_score + settings.min_delta:
            best_score = score
            best_epoch = epoch
            best_shape = val_shape_f1
            best_gram = val_gram_f1
            best_state = model.head_state_dict()
            since_improvement = 0
        else:
            since_improvement += 1

        if since_improvement >= settings.patience:
            break

    if best_state is not None:
        model.head_a.load_state_dict(
            {k.removeprefix("head_a."): v for k, v in best_state.items() if k.startswith("head_a.")}
        )
        model.head_b.load_state_dict(
            {k.removeprefix("head_b."): v for k, v in best_state.items() if k.startswith("head_b.")}
        )

    checkpoint_path = save_checkpoint(model, Path(settings.checkpoint_dir) / CHECKPOINT_NAME)

    # The test data is evaluated exactly once, right here.
    test_shape_f1, test_gram_f1, test_shape_acc, test_gram_acc = _evaluate(
        model, features, shape_tensor, gram_tensor, masks[TEST_SPLIT]
    )

    train_mask = masks[TRAIN_SPLIT]
    return TrainingReport(
        epochs_run=len(history),
        best_epoch=best_epoch,
        best_val_f1_shape=best_shape,
        best_val_f1_gram=best_gram,
        test_f1_shape=[test_shape_f1],
        test_f1_gram=[test_gram_f1],
        test_accuracy_shape=test_shape_acc,
        test_accuracy_gram=test_gram_acc,
        history=history,
        checkpoint_path=checkpoint_path,
        n_train=n_train,
        n_val=int(masks[VAL_SPLIT].sum()),
        n_test=int(masks[TEST_SPLIT].sum()),
        class_counts={
            "shape": {
                SHAPE_LABELS[index]: int(
                    sum(1 for i in shape_targets if i == index)
                )
                for index in range(N_SHAPE_CLASSES)
            },
            "gram": {
                GRAM_LABELS[index]: int(sum(1 for i in gram_targets if i == index))
                for index in range(N_GRAM_CLASSES)
            },
        },
        config={
            "epochs": settings.epochs,
            "patience": settings.patience,
            "learning_rate": settings.learning_rate,
            "weight_decay": settings.weight_decay,
            "min_delta": settings.min_delta,
            "batch_size": settings.batch_size,
            "seed": settings.seed,
            # The folder name only, not an absolute path. See the note on
            # TrainingReport.to_dict.
            "checkpoint_dir": Path(settings.checkpoint_dir).name,
        },
    )


def read_index_rows(index_path: Path | str = INDEX_PATH) -> list[dict]:
    """Read data/index.csv into a list of dicts.

    Args:
        index_path: The index file location.

    Returns:
        A list of rows with path, species, species_id, split and fold keys.

    Raises:
        FileNotFoundError: When the index file does not exist.
    """
    location = Path(index_path)
    if not location.is_file():
        raise FileNotFoundError(f"Index file not found: {location.name}")

    with location.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    """Run feature caching then training from the command line.

    Args:
        argv: The argument list. Defaults to sys.argv.

    Returns:
        The exit code, zero when training finished.
    """
    parser = argparse.ArgumentParser(
        description="Train the two BacteriaCV heads on frozen backbone features."
    )
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Skip augmentation and use the features as they are.",
    )
    args = parser.parse_args(argv)

    # DIBaS TIFF files carry tag 33560 which OpenCV does not know. This warning
    # appears once per image and does not affect the read result.
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

    rows = read_index_rows(args.index)
    model = build_model(pretrained=True)
    store, directory, reused = build_or_load_features(
        model,
        args.root,
        rows,
        augment=not args.no_augment,
        cache_root=args.features_dir,
        verbose=True,
    )

    print(f"Index rows        : {len(rows)}")
    print(f"Feature rows      : {store.matrix.shape[0]}")
    print(f"Cache             : {directory.name} (reused: {reused})")

    report = train(
        store,
        TrainConfig(
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
        ),
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.checkpoint_dir / REPORT_NAME).write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )

    print(f"Epochs run        : {report.epochs_run} (best epoch {report.best_epoch})")
    print(f"Shape validation  : F1 {report.best_val_f1_shape:.4f}")
    print(f"Gram validation   : F1 {report.best_val_f1_gram:.4f}")
    print(f"Shape test        : F1 {report.test_f1_shape[0]:.4f} accuracy {report.test_accuracy_shape:.4f}")
    print(f"Gram test         : F1 {report.test_f1_gram[0]:.4f} accuracy {report.test_accuracy_gram:.4f}")
    print(f"Checkpoint        : {report.checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
