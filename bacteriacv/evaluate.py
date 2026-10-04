"""Evaluasi checkpoint pada data uji.

Modul ini membaca checkpoint head yang sama dengan yang dipakai inferensi,
lalu menilainya pada split test saja. Split train dan val sengaja tidak
disentuh supaya angka yang dilaporkan tetap berasal dari data yang belum pernah
dipakai untuk mengambil keputusan apa pun.

Feature ditukar dari cache yang sama dengan pelatihan. Kunci cache memuat
nama berkas, ukuran, dan waktu modifikasi setiap citra, sehingga evaluate.py
tidak pernah menjalankan backbone dua kali untuk citra yang sama.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import cv2.utils.logging
import numpy as np
import torch

from .config import (
    CHECKPOINT_DIR,
    FEATURES_DIR,
    GRAM_LABELS,
    N_GRAM_CLASSES,
    N_SHAPE_CLASSES,
    SHAPE_LABELS,
    SHAPE_LABELS_FULL,
    SHAPE_UNPOPULATED,
)
from .label_map import LOOKUP
from .model import build_model, load_checkpoint
from .paths import INDEX_PATH, PROJECT_ROOT
from .train import (
    FeatureStore,
    build_or_load_features,
    f1_macro,
    read_index_rows,
)
from .datasets.species_map import display_name

TEST_SPLIT = "test"

CHECKPOINT_NAME = "heads.pt"
REPORT_NAME = "evaluation.json"


@dataclass(frozen=True)
class HeadMetrics:
    """Metrik satu head pada split uji.

    Attributes:
        head: Nama head.
        labels: Nama kelas sesuai urutan indeks.
        f1_macro: F1 makro lintas kelas.
        accuracy: Proporsi prediksi benar.
        per_class: F1, presisi, dan recall tiap kelas.
        confusion: Matriks kebingungan dengan baris sebenarnya.
        support: Jumlah sampel per kelas sebenarnya.
        positive_support: Jumlah sampel kelas positif.
    """

    head: str
    labels: list[str]
    f1_macro: float
    accuracy: float
    per_class: list[dict[str, float]]
    confusion: list[list[int]]
    support: list[int]
    positive_support: int

    def to_dict(self) -> dict:
        """Ubah metrik menjadi dictionary untuk serialisasi JSON."""
        return asdict(self)


def confusion_matrix(truth: np.ndarray, prediction: np.ndarray, n_classes: int) -> list[list[int]]:
    """Bangun matriks kebingungan dengan baris kelas sebenarnya.

    Args:
        truth: Label sebenarnya.
        prediction: Label hasil prediksi.
        n_classes: Jumlah kelas.

    Returns:
        Matriks n_classes kali n_classes, elemen int.
    """
    matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for actual, predicted in zip(truth, prediction):
        matrix[int(actual)][int(predicted)] += 1
    return matrix


def per_class_metrics(
    truth: np.ndarray, prediction: np.ndarray, labels: list[str], n_classes: int
) -> tuple[list[dict[str, float]], list[int]]:
    """Hitung presisi, recall, dan F1 untuk tiap kelas.

    Args:
        truth: Label sebenarnya.
        prediction: Label hasil prediksi.
        labels: Nama kelas sesuai urutan indeks.
        n_classes: Jumlah kelas.

    Returns:
        Pasangan (daftar metrik per kelas, jumlah sampel per kelas).
    """
    rows: list[dict[str, float]] = []
    support: list[int] = []
    for index in range(n_classes):
        actual_positive = truth == index
        predicted_positive = prediction == index
        true_positive = int(np.sum(actual_positive & predicted_positive))
        false_positive = int(np.sum(~actual_positive & predicted_positive))
        false_negative = int(np.sum(actual_positive & ~predicted_positive))

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        f1_denominator = 2 * true_positive + false_positive + false_negative

        rows.append(
            {
                "label": labels[index],
                "precision": 0.0
                if precision_denominator == 0
                else true_positive / precision_denominator,
                "recall": 0.0
                if recall_denominator == 0
                else true_positive / recall_denominator,
                "f1": 0.0 if f1_denominator == 0 else 2 * true_positive / f1_denominator,
                "support": float(np.sum(actual_positive)),
            }
        )
        support.append(int(np.sum(actual_positive)))
    return rows, support


def evaluate_head(
    head: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    labels: list[str],
    n_classes: int,
) -> HeadMetrics:
    """Rangkum seluruh metrik satu head.

    Args:
        head: Nama head untuk laporan.
        truth: Label sebenarnya.
        prediction: Label hasil prediksi.
        labels: Nama kelas sesuai urutan indeks.
        n_classes: Jumlah kelas.

    Returns:
        HeadMetrics untuk head tersebut.
    """
    rows, support = per_class_metrics(truth, prediction, labels, n_classes)
    return HeadMetrics(
        head=head,
        labels=labels,
        f1_macro=f1_macro(truth, prediction, n_classes),
        accuracy=float(np.mean(truth == prediction)) if truth.size else 0.0,
        per_class=rows,
        confusion=confusion_matrix(truth, prediction, n_classes),
        support=support,
        positive_support=int(np.sum(truth == 1)),
    )


def select_split(store: FeatureStore, split: str) -> np.ndarray:
    """Kembalikan indeks baris untuk satu split.

    Args:
        store: Feature store.
        split: Nama split yang dicari.

    Returns:
        Array indeks baris yang terpilih, urut sesuai baris feature store.

    Raises:
        ValueError: Bila split tidak punya baris.
    """
    positions = np.array(
        [index for index, value in enumerate(store.splits) if value == split],
        dtype=np.int64,
    )
    if positions.size == 0:
        raise ValueError(f"Split {split!r} kosong pada feature store.")
    return positions


def species_breakdown(
    species_ids: list[str], correct: np.ndarray, predictions: np.ndarray
) -> list[dict]:
    """Ringkas akurasi per spesies untuk interpretasi hasil.

    Args:
        species_ids: species_id tiap citra uji.
        correct: Masker boolean ketepatan prediksi bentuk.
        predictions: Indeks kelas hasil prediksi bentuk.

    Returns:
        Daftar baris per spesies, terurut menurut species_id.
    """
    rows: list[dict] = []
    for species_id in sorted(set(species_ids)):
        mask = np.array([value == species_id for value in species_ids], dtype=bool)
        shape_label, gram_label = LOOKUP[species_id]
        rows.append(
            {
                "species_id": species_id,
                "display_name": display_name(species_id),
                "expected_shape": shape_label,
                "expected_gram": gram_label,
                "n": int(np.sum(mask)),
                "shape_accuracy": float(np.mean(correct[mask])),
                "predicted_shapes": sorted(
                    {
                        SHAPE_LABELS[int(value)]
                        for value in predictions[mask]
                    }
                ),
            }
        )
    return rows


def evaluate_checkpoint(
    store: FeatureStore, checkpoint_path: Path | str, split: str = TEST_SPLIT
) -> dict:
    """Nilai checkpoint pada satu split dan susun laporan JSON.

    Args:
        store: Feature store berisi fitur untuk seluruh data.
        checkpoint_path: Lokasi checkpoint head.
        split: Split yang dinilai.

    Returns:
        Dictionary laporan yang bisa diserialisasi ke JSON.

    Raises:
        FileNotFoundError: Bila checkpoint tidak ada.
        ValueError: Bila split kosong atau dimensi fitur tidak cocok.
    """
    model = build_model(pretrained=False)
    load_checkpoint(model, checkpoint_path)
    model.eval()

    positions = select_split(store, split)
    features = torch.from_numpy(
        np.ascontiguousarray(store.matrix[positions], dtype=np.float32)
    )

    with torch.no_grad():
        shape_probabilities = model.head_a(features).softmax(dim=-1)
        shape_prediction = shape_probabilities.argmax(dim=-1).numpy()
        gram_probability = model.head_b(features)[:, 1].sigmoid().numpy()
    shape_confidence = shape_probabilities.max(dim=-1).values.numpy()
    gram_prediction = (gram_probability > 0.5).astype(np.int64)

    from .label_map import to_targets

    species_ids = [store.species_ids[int(i)] for i in positions]
    shape_truth_list, gram_truth_list = to_targets(species_ids)
    shape_truth = np.array(shape_truth_list, dtype=np.int64)
    gram_truth = np.array(gram_truth_list, dtype=np.int64)

    shape_metrics = evaluate_head(
        "shape", shape_truth, shape_prediction, list(SHAPE_LABELS), N_SHAPE_CLASSES
    )
    gram_metrics = evaluate_head(
        "gram", gram_truth, gram_prediction, list(GRAM_LABELS), N_GRAM_CLASSES
    )

    # Confidence dilaporkan terpisah per head. Menjumlahkan confidence bentuk
    # dengan confidence Gram lalu membagi dua menghasilkan angka yang tidak
    # bermakna: kedua head memakai skala dan basiskalibrasi yang berbeda, dan
    # tidak ada kelas yang menyatukan keduanya. Angka gabungan semacam ini
    # pernah menyesatkan, jadi tidak ada lagi di laporan ini.
    return {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "n_images": int(positions.size),
        "n_species": len(set(species_ids)),
        "shape": shape_metrics.to_dict(),
        "gram": gram_metrics.to_dict(),
        "mean_confidence_shape": float(np.mean(shape_confidence)),
        "mean_confidence_gram": float(
            np.mean(np.where(gram_prediction == 1, gram_probability, 1.0 - gram_probability))
        ),
        "per_species": species_breakdown(
            species_ids, shape_prediction == shape_truth, shape_prediction
        ),
        "scope_note": (
            "Head A dilatih pada dua kelas karena DIBaS tidak memuat spesies "
            f"berbentuk {SHAPE_UNPOPULATED}. Kelas {SHAPE_UNPOPULATED} pada "
            f"{list(SHAPE_LABELS_FULL)[2:]} tidak terisi dan tidak masuk metrik."
        ),
        "gram_calibration_note": (
            "Head B dilatih sebagai klasifier biner satu logit dengan "
            "BCEWithLogitsLoss pada kolom keluaran kedua. sigmoid(logit) adalah "
            "probabilitas kelas positif, sehingga ambang 0,5 berlaku langsung "
            "dan confidence adalah probabilitas kelas yang dipilih. Kolom "
            "keluaran pertama tidak pernah masuk loss dan tidak membawa "
            "informasi yang dipelajari, jadi tidak boleh dipakai: softmax atas "
            "dua kolom akan mencampur logit yang dilatih dengan logit yang "
            "hanya mengalami weight decay."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Jalankan evaluasi dari baris perintah.

    Args:
        argv: Daftar argumen. Default-nya sys.argv.

    Returns:
        Kode keluar, nol bila evaluasi selesai.
    """
    parser = argparse.ArgumentParser(
        description="Nilai checkpoint BacteriaCV pada split data uji."
    )
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument(
        "--checkpoint", type=Path, default=CHECKPOINT_DIR / CHECKPOINT_NAME
    )
    parser.add_argument("--split", default=TEST_SPLIT)
    parser.add_argument("--out", type=Path, default=CHECKPOINT_DIR / REPORT_NAME)
    args = parser.parse_args(argv)

    # TIFF DIBaS memakai tag 33560 yang tidak dikenal OpenCV. Peringatan ini
    # muncul sekali per citra dan tidak memengaruhi hasil baca.
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

    rows = read_index_rows(args.index)
    store, _, _ = build_or_load_features(
        build_model(pretrained=True), args.root, rows, augment=True, cache_root=args.features_dir
    )
    report = evaluate_checkpoint(store, args.checkpoint, args.split)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    shape = report["shape"]
    gram = report["gram"]
    print(f"Split                  : {report['split']} ({report['n_images']} citra)")
    print(f"Head A bentuk F1       : {shape['f1_macro']:.4f} akurasi {shape['accuracy']:.4f}")
    print(f"Head A confidence rata : {report['mean_confidence_shape']:.4f}")
    print(f"Head B Gram F1         : {gram['f1_macro']:.4f} akurasi {gram['accuracy']:.4f}")
    print(f"Head B confidence rata : {report['mean_confidence_gram']:.4f}")
    print(f"Laporan             : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
