"""Ekstraksi fitur beku dan loop pelatihan dua head.

Feature caching diletakkan di modul ini, bukan modul terpisah, karena
train.py sudah tercatat di ARCHITECTURE bagian 4 dan FEATURES_DIR sudah ada di
config. Konsekuensinya satuberriesi.

Alasan caching: backbone dibekukan, sehingga vektor fitur sebuah citra tidak
berubah antar epoch. Kalau citra diteruskan lewat backbone setiap epoch, biaya
satu epoch adalah 76 ms kali jumlah citra. Dengan caching, backbone dijalankan
sekali untuk seluruh data latih, dan epoch berikutnya hanya menjalankan dua
lapisan Linear(2048, 2) yang bobotnya beberapa ribu parameter.

Data uji tidak pernah dipakai untuk early stopping. Data uji hanya dievaluasi
sekali, setelah bobot terbaik dipulihkan. Aturan ini ditegakkan oleh
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

# Bump nilai ini bila pipeline pra-pemrosesan berubah, supaya cache lama tidak
# dipakai untuk citra yang sudah diproses dengan aturan berbeda.
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
    """Vektor fitur beku beserta label yang menunjangnya.

    Attributes:
        matrix: Matriks N x FEATURE_DIM bertipe float32.
        species_ids: species_id tiap baris, untuk diturunkan jadi label head.
        splits: Nama split tiap baris.
        paths: Path citra asal tiap baris, relatif terhadap folder citra.
    """

    matrix: np.ndarray
    species_ids: list[str]
    splits: list[str]
    paths: list[str]


def save_feature_store(store: FeatureStore, directory: Path | str) -> Path:
    """Tulis feature store ke disk.

    Args:
        store: Feature store yang ditulis.
        directory: Folder tujuan, dibuat bila belum ada.

    Returns:
        Folder tempat berkas ditulis.
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
    """Baca feature store dari disk.

    Args:
        directory: Folder yang berisi features.npy dan labels.npz.

    Returns:
        FeatureStore yang pulih persis seperti saat disimpan.

    Raises:
        FileNotFoundError: Bila folder atau berkas matriks tidak ada.
        ValueError: Bila dimensi matriks atau panjang label tidak konsisten.
    """
    source = Path(directory)
    matrix_path = source / MATRIX_NAME
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Feature store tidak ditemukan: {source.name}")

    matrix = np.load(matrix_path)
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Jumlah kolom fitur tidak cocok dengan FEATURE_DIM, "
            f"periksa dimensi matriks: bentuk {matrix.shape}"
        )

    labels = np.load(source / LABELS_NAME, allow_pickle=False)
    species_ids = [str(value) for value in labels["species_ids"]]
    splits = [str(value) for value in labels["splits"]]
    paths = [str(value) for value in labels["paths"]]

    lengths = {len(matrix), len(species_ids), len(splits), len(paths)}
    if len(lengths) != 1:
        raise ValueError(
            f"Jumlah label tidak sama dengan jumlah baris fitur: "
            f"{len(matrix)}, {len(species_ids)}, {len(splits)}, {len(paths)}"
        )

    return FeatureStore(matrix, species_ids, splits, paths)


def cache_signature(images_dir: Path | str, rows: list[dict], augment: bool) -> str:
    """Hitung sidik jari cache dari isi citra dan aturan pipeline.

    Sidik jari memuat nama berkas, ukuran, dan waktu modifikasi setiap citra.
    Mengubah satu piksel pada citra sumber karena itu membatalkan cache.

    Args:
        images_dir: Folder dasar tempat path relatif dihitung.
        rows: Baris index.
        augment: Apakah augmentasi dilLewati saat ekstraksi.

    Returns:
        Sidik jari hex sepanjang 16 karakter.
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
            marker = "tidak-ada"
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
    """Ekstraksi vektor fitur untuk setiap baris index.

    Augmentasi hanya berlaku pada baris ber-split train, sesuai keputusan D3.
    Jumlah baris latih menjadi AUGMENT_VARIANTS kali lebih banyak: satu citra
    asli dan AUGMENT_VARIANTS - 1 varian augmentasi.

    Ekstraksi memakai preprocess_tensor, bukan preprocess, sehingga segmentasi
    tidak dijalankan. Segmentasi hanya untuk panel visualisasi dan tidak
    memengaruhi metrik model.

    Args:
        model: Model BacteriaNet. Backbonenya dibekukan oleh model itu sendiri.
        images_dir: Folder dasar untuk path relatif pada baris index.
        rows: Baris index dengan kunci path, species_id, dan split.
        augment: Bila True, baris train diberi varian augmentasi.
        batch_size: Jumlah citra per inferensi backbone.
        verbose: Bila True, cetak kemajuan setiap PROGRESS_EVERY citra.

    Returns:
        FeatureStore berisi matriks fitur dan labelnya.

    Raises:
        FileNotFoundError: Bila ada citra yang tidak ditemukan.
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
                f"Ekstraksi {position + 1}/{len(rows)} citra, "
                f"{len(tensors)} baris fitur",
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
    """Ambil fitur dari cache bila tersedia, ekstrak ulang bila tidak.

    Args:
        model: Model BacteriaNet.
        images_dir: Folder dasar untuk path relatif pada baris index.
        rows: Baris index.
        augment: Bila True, baris train diberi varian augmentasi.
        cache_root: Folder akar cache. Default-nya FEATURES_DIR dari config.
        verbose: Bila True, cetak kemajuan saat ekstraksi.

    Returns:
        Tiga elemen (store, folder_cache, dipakai_cache).
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
# Metrik
# =====================================================================


def f1_macro(truth: np.ndarray, prediction: np.ndarray, n_classes: int) -> float:
    """Hitung F1 makro tanpa dependensi sklearn.

    Args:
        truth: Label sebenarnya.
        prediction: Label hasil prediksi.
        n_classes: Jumlah kelas yang dinilai.

    Returns:
        Rata-rata F1 per kelas. Kelas yang tidak pernah muncul di sebenarnya
        maupun prediksi diberi skor nol agar kelas kosong tidak menaikkan
        rata-rata.
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
    """Hitung akurasi sebagai proporsi prediksi benar.

    Args:
        truth: Label sebenarnya.
        prediction: Label hasil prediksi.

    Returns:
        Nilai antara nol dan satu, nol bila arrays kosong.
    """
    if truth.size == 0:
        return 0.0
    return float(np.mean(truth == prediction))


def _head_predictions(
    model: BacteriaNet, features: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Ambil prediksi kedua head untuk sekumpulan fitur.

    Args:
        model: Model BacteriaNet.
        features: Tensor N x FEATURE_DIM.

    Returns:
        Pasangan (prediksi_bentuk, prediksi_gram) sebagai array integer.
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
    """Nilai satu split pada keempat metrik.

    Args:
        model: Model BacteriaNet.
        features: Matriks fitur seluruh data.
        shape_truth: Label bentuk seluruh data.
        gram_truth: Label Gram seluruh data.
        mask: Masker baris yang termasuk split ini.

    Returns:
        Empat nilai (f1_bentuk, f1_gram, akurasi_bentuk, akurasi_gram).
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
# Loop pelatihan
# =====================================================================


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameter pelatihan. Semua nilai bawaan berasal dari config.

    Attributes:
        epochs: Batas atas epoch.
        patience: Jumlah epoch tanpa perbaikan sebelum berhenti.
        learning_rate: Learning rate AdamW.
        weight_decay: Weight decay AdamW.
        min_delta: Perbaikan minimum agar epoch dihitung sebagai perbaikan.
        batch_size: Jumlah baris per langkah optimizer.
        seed: Seed untuk inisialisasi head dan pengocokan batch.
        checkpoint_dir: Folder tempat checkpoint ditulis.
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
    """Hasil pelatihan beserta metrik akhir.

    test_f1_shape dan test_f1_gram adalah daftar sepanjang satu karena data
    uji hanya dievaluasi sekali. Bentuk daftar membuat aturan itu terlihat di
    laporan, bukan sekadar tersirat.

    Attributes:
        epochs_run: Jumlah epoch yang benar-benar dijalankan.
        best_epoch: Epoch dengan skor validasi terbaik, satu basis.
        best_val_f1_shape: F1 makro bentuk pada epoch terbaik.
        best_val_f1_gram: F1 makro Gram pada epoch terbaik.
        test_f1_shape: Daftar F1 makro bentuk pada data uji, panjang satu.
        test_f1_gram: Daftar F1 makro Gram pada data uji, panjang satu.
        test_accuracy_shape: Akurasi bentuk pada data uji.
        test_accuracy_gram: Akurasi Gram pada data uji.
        history: Riwayat per epoch tanpa metrik data uji.
        checkpoint_path: Lokasi checkpoint bobot terbaik.
        n_train: Jumlah baris latih setelah augmentasi.
        n_val: Jumlah baris validasi.
        n_test: Jumlah baris uji.
        class_counts: Jumlah baris latih per kelas bentuk dan Gram.
        config: Konfigurasi pelatihan yang dipakai.
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
        """Ubah laporan menjadi dictionary yang bisa diserialisasi ke JSON.

        Returns:
            Dictionary tanpa objek Path dan tanpa array numpy.
        """
        payload = asdict(self)
        checkpoint = payload.get("checkpoint_path")
        if isinstance(checkpoint, Path):
            payload["checkpoint_path"] = str(checkpoint)
        payload["labels"] = {
            "shape": list(SHAPE_LABELS),
            "gram": list(GRAM_LABELS),
        }
        return payload


def _split_masks(splits: list[str]) -> dict[str, torch.Tensor]:
    """Buat masker baris untuk tiap split.

    Args:
        splits: Nama split tiap baris.

    Returns:
        Dictionary masker boolean untuk train, val, dan test.

    Raises:
        ValueError: Bila split latih atau validasi kosong.
    """
    masks = {
        name: torch.tensor([value == name for value in splits], dtype=torch.bool)
        for name in (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)
    }
    for name in REQUIRED_SPLITS:
        if not bool(masks[name].any()):
            raise ValueError(f"Split {name!r} kosong pada feature store.")
    return masks


def _safe_weights(targets: list[int], n_classes: int) -> torch.Tensor:
    """Hitung bobot kelas, fallback ke satu bila hanya ada satu kelas."""
    try:
        return class_weights(targets, n_classes)
    except ValueError:
        return torch.ones(n_classes, dtype=torch.float32)


def train(store: FeatureStore, config: TrainConfig | None = None) -> TrainingReport:
    """Latih kedua head pada fitur beku.

    Args:
        store: Feature store hasil ekstraksi.
        config: Hyperparameter pelatihan. Default-nya TrainConfig().

    Returns:
        TrainingReport berisi riwayat, metrik validasi terbaik, dan satu
        evaluasi data uji.

    Raises:
        ValueError: Bila dimensi fitur tidak cocok atau split wajib kosong.
        KeyError: Bila ada species_id yang tidak ada di lookup table.
    """
    settings = config or TrainConfig()

    matrix = np.ascontiguousarray(store.matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Matriks fitur harus berbentuk (N, {FEATURE_DIM}), bukan {matrix.shape}"
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

    # Data uji dievaluasi tepat satu kali, di titik ini.
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
            "checkpoint_dir": str(settings.checkpoint_dir),
        },
    )


def read_index_rows(index_path: Path | str = INDEX_PATH) -> list[dict]:
    """Baca data/index.csv menjadi daftar dict.

    Args:
        index_path: Lokasi berkas index.

    Returns:
        Daftar baris dengan kunci path, species, species_id, split, dan fold.

    Raises:
        FileNotFoundError: Bila berkas index tidak ada.
    """
    location = Path(index_path)
    if not location.is_file():
        raise FileNotFoundError(f"Berkas index tidak ditemukan: {location.name}")

    with location.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    """Jalankan feature caching lalu pelatihan dari baris perintah.

    Args:
        argv: Daftar argumen. Default-nya sys.argv.

    Returns:
        Kode keluar, nol bila pelatihan selesai.
    """
    parser = argparse.ArgumentParser(
        description="Latih dua head BacteriaCV pada fitur backbone beku."
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
        help="Lewati augmentasi dan pakai fitur apa adanya.",
    )
    args = parser.parse_args(argv)

    # TIFF DIBaS memakai tag 33560 yang tidak dikenal OpenCV. Peringatan ini
    # muncul sekali per citra dan tidak memengaruhi hasil baca.
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

    print(f"Baris index       : {len(rows)}")
    print(f"Baris fitur       : {store.matrix.shape[0]}")
    print(f"Cache             : {directory.name} (dipakai ulang: {reused})")

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

    print(f"Epoch dijalankan  : {report.epochs_run} (terbaik epoch {report.best_epoch})")
    print(f"Validasi bentuk   : F1 {report.best_val_f1_shape:.4f}")
    print(f"Validasi Gram     : F1 {report.best_val_f1_gram:.4f}")
    print(f"Uji bentuk        : F1 {report.test_f1_shape[0]:.4f} akurasi {report.test_accuracy_shape:.4f}")
    print(f"Uji Gram          : F1 {report.test_f1_gram[0]:.4f} akurasi {report.test_accuracy_gram:.4f}")
    print(f"Checkpoint        : {report.checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
