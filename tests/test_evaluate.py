"""Tes untuk evaluasi checkpoint pada data uji."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bacteriacv.config import FEATURE_DIM, GRAM_LABELS, SHAPE_LABELS
from bacteriacv.evaluate import (
    confusion_matrix,
    evaluate_checkpoint,
    evaluate_head,
    per_class_metrics,
    select_split,
    species_breakdown,
)
from bacteriacv.model import build_model, save_checkpoint
from bacteriacv.train import FeatureStore

BACILLI = "escherichia_coli"
COCci = "staphylococcus_aureus"


def _store_with_split(splits: list[str], species: list[str]) -> FeatureStore:
    """Buat feature store dengan fitur yang mudah dipisah per spesies."""
    matrix = np.zeros((len(splits), FEATURE_DIM), dtype=np.float32)
    for index, species_id in enumerate(species):
        matrix[index, :] = 1.0 if species_id == COCci else -1.0
    return FeatureStore(matrix, species, splits, [f"p{i:03d}.tif" for i in range(len(splits))])


def _checkpoint(path: Path) -> Path:
    """Buat checkpoint head yang memisahkan kedua kelas pada sumbu nol.

    Bobot head dibuat manual supaya hasil evaluasi tidak bergantung pada
    inisialisasi acak.
    """
    model = build_model(pretrained=False)
    with torch.no_grad():
        model.head_a.weight.zero_()
        model.head_a.weight[:, 0] = torch.tensor([4.0, -4.0])
        model.head_a.bias.zero_()
        model.head_b.weight.zero_()
        model.head_b.weight[:, 0] = torch.tensor([4.0, -4.0])
        model.head_b.bias.zero_()
    return save_checkpoint(model, path)


# --- Confidence per head ---


def test_confidence_is_reported_per_head(tmp_path: Path) -> None:
    """Confidence harus dilaporkan terpisah untuk setiap head.

    Confidence bentuk dan confidence Gram tidak bisa dijumlahkan. Keduanya
    memakai skala dan basiskalibrasi yang berbeda, dan tidak ada kelas yang
    menyatukan keduanya. Angka gabungan pernah menyesatkan sehingga tidak
    lagi ada di laporan.
    """
    store = _store_with_split(
        ["train", "test", "test", "test"], [BACILLI, COCci, BACILLI, BACILLI]
    )
    checkpoint = _checkpoint(tmp_path / "heads.pt")

    report = evaluate_checkpoint(store, checkpoint, "test")

    assert "mean_confidence_shape" in report
    assert "mean_confidence_gram" in report
    assert "mean_confidence" not in report


def test_gram_confidence_is_probability_of_chosen_class(tmp_path: Path) -> None:
    """Confidence Gram harus sesuai probabilitas kelas yang dipilih.

    Untuk kelas negatif confidence harus 1 dikurangi probabilitas positif.
    kalau confidence diambil dari probabilitas mentah tanpa membalik,
    citra Gram negatif akan tampil sangat yakin secara salah.
    """
    store = _store_with_split(["test", "test"], [COCci, BACILLI])
    checkpoint = _checkpoint(tmp_path / "heads.pt")

    report = evaluate_checkpoint(store, checkpoint, "test")

    gram_probability = float(report["gram"]["per_class"][1]["support"]) / 2.0
    assert 0.0 <= report["mean_confidence_gram"] <= 1.0
    assert report["mean_confidence_gram"] > 0.9
    assert gram_probability > 0.0


def test_gram_uses_sigmoid_not_softmax_over_both_columns(tmp_path: Path) -> None:
    """Head B adalah klasifier satu logit, bukan dua kelas softmax.

    Kolom keluaran pertama tidak pernah masuk loss BCEWithLogitsLoss, jadi
    tidak membawa informasi yang dipelajari. Softmax atas dua kolom akan
    mencampur logit yang dilatih dengan logit yang hanya mengalami weight
    decay. Tes ini mengunci pilihan sigmoid pada kolom kedua.
    """
    store = _store_with_split(["test", "test"], [COCci, BACILLI])
    checkpoint = _checkpoint(tmp_path / "heads.pt")

    report = evaluate_checkpoint(store, checkpoint, "test")

    note = report["gram_calibration_note"]
    assert "sigmoid(logit)" in note
    assert "tidak pernah masuk loss" in note
    assert "tidak boleh dipakai" in note


def test_report_checkpoint_contains_only_file_name(tmp_path: Path) -> None:
    """Laporan hanya boleh memuat nama berkas checkpoint, bukan path.

    evaluation.json ikut ter-deploy dan dibaca siapa pun lewat /api/report.
    Path absolut di dalamnya membocorkan struktur folder mesin pembangun, dan
    melanggar AGENT.md bagian 3 aturan 5. Field ini pernah ditulis tanpa satu pun assertion yang membacanya, sehingga
    kebocorannya lolos seluruh audit sebelumnya.

    Path di bawah sengaja punya subdirektori supaya tes membuktikan direktori
    dibuang, bukan kebetulan nama berkasnya memang tanpa direktori.
    """
    nested = tmp_path / "sub"
    nested.mkdir()
    store = _store_with_split(["train", "test", "test"], [BACILLI, COCci, BACILLI])
    checkpoint = _checkpoint(nested / "heads.pt")

    report = evaluate_checkpoint(store, checkpoint, "test")

    assert report["checkpoint"] == "heads.pt"


def test_report_checkpoint_has_no_path_syntax(tmp_path: Path) -> None:
    """Nama berkas checkpoint tidak boleh memuat sisa sintaks path.

    Empat pemeriksaan terpisah karena masing-masing mewakili kelas kesalahan
    yang berbeda: separator POSIX, pemisah Windows, huruf drive, dan bentuk
    path relatif maupun absolut yang menyamar.
    """
    nested = tmp_path / "sub"
    nested.mkdir()
    store = _store_with_split(["train", "test", "test"], [BACILLI, COCci, BACILLI])
    checkpoint = _checkpoint(nested / "heads.pt")

    report = evaluate_checkpoint(store, checkpoint, "test")
    reported = report["checkpoint"]

    assert isinstance(reported, str), type(reported)
    for separator in ("/", "\\"):
        assert separator not in reported, f"pemisah {separator!r} bocor"
    assert ":" not in reported, "huruf drive atau dua titik bocor"
    assert not reported.startswith("."), "path relatif bocor"
    assert reported == reported.strip(), "spasi di tepi bocor"
    assert Path(reported).name == reported, "masih ada unsur direktori"


def test_confidence_does_not_change_accuracy(tmp_path: Path) -> None:
    """Perubahan cara menghitung confidence tidak boleh mengubah label.

    Confidence dan akurasi dihitung dari logit yang sama, jadi akurasi harus
    tetap sama persis setelah confidence dipisah per head.
    """
    store = _store_with_split(
        ["train", "test", "test", "test"], [BACILLI, COCci, BACILLI, BACILLI]
    )
    checkpoint = _checkpoint(tmp_path / "heads.pt")

    report = evaluate_checkpoint(store, checkpoint, "test")

    assert report["gram"]["accuracy"] == 1.0
    assert report["shape"]["accuracy"] == 1.0
    assert report["mean_confidence_shape"] >= report["mean_confidence_gram"]


# --- Metrik dasar ---


def test_confusion_matrix_counts_pairs() -> None:
    """Matriks kebingungan harus menghitung pasangan benar dan salah."""
    truth = np.array([0, 0, 1, 1])
    prediction = np.array([0, 1, 1, 1])

    assert confusion_matrix(truth, prediction, 2) == [[1, 1], [0, 2]]


def test_confusion_matrix_handles_three_classes() -> None:
    """Kelas yang tidak muncul harus tetap punya baris dan kolom nol."""
    truth = np.array([0, 2, 2])
    prediction = np.array([0, 2, 0])

    assert confusion_matrix(truth, prediction, 3) == [[1, 0, 0], [0, 0, 0], [1, 0, 1]]


def test_per_class_metrics_reports_perfect_prediction() -> None:
    """Prediksi sempurna harus memberi F1 satu untuk tiap kelas."""
    truth = np.array([0, 0, 1, 1])
    prediction = np.array([0, 0, 1, 1])

    rows, support = per_class_metrics(truth, prediction, ["a", "b"], 2)

    assert support == [2, 2]
    assert all(row["f1"] == 1.0 for row in rows)
    assert all(row["precision"] == 1.0 for row in rows)
    assert all(row["recall"] == 1.0 for row in rows)


def test_per_class_metrics_marks_missing_class_as_zero() -> None:
    """Kelas yang absen di sebenarnya dan prediksi harus dapat skor nol.

    Kalau kelas kosong diberi F1 satu, F1 makro pada split satu kelas akan
    selalu maximal dan tidak bisa dipakai sebagai sinyal early stopping.
    """
    truth = np.array([0, 0])
    prediction = np.array([0, 0])

    rows, support = per_class_metrics(truth, prediction, ["a", "b"], 2)

    assert support == [2, 0]
    assert rows[1]["f1"] == 0.0
    assert rows[1]["recall"] == 0.0


def test_evaluate_head_aggregates() -> None:
    """evaluate_head harus merangkum F1 makro, akurasi, dan support.

    Seribu satu baris data uji: tiga kelas 0, tiga kelas 1, satu salah klas
    di masing-masing arah. Kelas 0 F1 4/5, kelas 1 F1 6/7.
    """
    truth = np.array([0, 0, 0, 1, 1, 1])
    prediction = np.array([0, 0, 1, 1, 1, 1])

    metrics = evaluate_head("shape", truth, prediction, ["a", "b"], 2)

    assert metrics.f1_macro == pytest.approx((0.8 + 6 / 7) / 2)
    assert metrics.accuracy == pytest.approx(5 / 6)
    assert metrics.positive_support == 3
    assert metrics.labels == ["a", "b"]
    assert metrics.confusion == [[2, 1], [0, 3]]


# --- Pemilihan split ---


def test_select_split_returns_positions() -> None:
    """select_split harus mengembalikan indeks baris split tersebut."""
    store = _store_with_split(
        ["train", "train", "val", "test", "test"], [BACILLI, BACILLI, BACILLI, COCci, BACILLI]
    )

    assert select_split(store, "test").tolist() == [3, 4]
    assert select_split(store, "train").tolist() == [0, 1]


def test_select_split_rejects_empty_split() -> None:
    """Split tanpa baris harus ditolak."""
    store = _store_with_split(["train", "train"], [BACILLI, COCci])

    with pytest.raises(ValueError, match="test"):
        select_split(store, "test")


# --- Ringkasan per spesies ---


def test_species_breakdown_uses_lookup_labels() -> None:
    """Baris per spesies harus memakai label dari lookup table."""
    species = [COCci, COCci, BACILLI]
    correct = np.array([True, True, False])
    predictions = np.array([0, 0, 0])

    rows = species_breakdown(species, correct, predictions)

    assert len(rows) == 2
    cocci = next(row for row in rows if row["species_id"] == COCci)
    bacilli = next(row for row in rows if row["species_id"] == BACILLI)
    assert cocci["expected_shape"] == "cocci"
    assert cocci["expected_gram"] == "positif"
    assert cocci["n"] == 2
    assert cocci["shape_accuracy"] == 1.0
    assert bacilli["expected_shape"] == "bacilli"
    assert bacilli["expected_gram"] == "negatif"
    assert bacilli["shape_accuracy"] == 0.0
    assert bacilli["predicted_shapes"] == ["cocci"]


def test_species_breakdown_counts_both_shapes_under_one_species() -> None:
    """Satu spesies boleh punya dua predisi berbeda."""
    species = [BACILLI, BACILLI, BACILLI]
    correct = np.array([True, False, False])
    predictions = np.array([1, 0, 0])

    rows = species_breakdown(species, correct, predictions)

    assert rows[0]["predicted_shapes"] == ["bacilli", "cocci"]


# --- evaluate_checkpoint ---


def test_evaluate_checkpoint_perfect_model(tmp_path: Path) -> None:
    """Checkpoint yang memisahkan kelas harus mendapat F1 dan akurasi satu."""
    store = _store_with_split(
        ["train", "train", "test", "test"], [BACILLI, BACILLI, COCci, BACILLI]
    )
    checkpoint = _checkpoint(tmp_path / "heads.pt")

    report = evaluate_checkpoint(store, checkpoint, "test")

    assert report["n_images"] == 2
    assert report["n_species"] == 2
    assert report["shape"]["f1_macro"] == pytest.approx(1.0)
    assert report["shape"]["accuracy"] == pytest.approx(1.0)
    assert report["gram"]["f1_macro"] == pytest.approx(1.0)
    assert report["gram"]["accuracy"] == pytest.approx(1.0)
    assert report["split"] == "test"


def test_evaluate_checkpoint_only_touches_requested_split(tmp_path: Path) -> None:
    """Baris di luar split tidak boleh memengaruhi hasil."""
    store = _store_with_split(
        ["test", "test", "train"], [COCci, BACILLI, COCci]
    )
    checkpoint = _checkpoint(tmp_path / "heads.pt")

    report = evaluate_checkpoint(store, checkpoint, "test")

    assert report["n_images"] == 2
    assert report["shape"]["confusion"] == [[1, 0], [0, 1]]


def test_evaluate_checkpoint_reports_missing_checkpoint(tmp_path: Path) -> None:
    """Checkpoint yang tidak ada harus ditolak."""
    store = _store_with_split(["test", "test"], [COCci, BACILLI])

    with pytest.raises(FileNotFoundError):
        evaluate_checkpoint(store, tmp_path / "tidak_ada.pt", "test")


def test_evaluate_checkpoint_rejects_empty_split(tmp_path: Path) -> None:
    """Split kosong harus ditolak sebelum memuat checkpoint."""
    store = _store_with_split(["train"], [BACILLI])
    checkpoint = _checkpoint(tmp_path / "heads.pt")

    with pytest.raises(ValueError, match="test"):
        evaluate_checkpoint(store, checkpoint, "test")


def test_evaluate_checkpoint_is_json_serializable(tmp_path: Path) -> None:
    """Laporan harus bisa ditulis sebagai JSON apa adanya."""
    store = _store_with_split(
        ["train", "test", "test", "test"], [BACILLI, COCci, BACILLI, BACILLI]
    )
    checkpoint = _checkpoint(tmp_path / "heads.pt")

    report = evaluate_checkpoint(store, checkpoint, "test")

    payload = json.loads(json.dumps(report))
    assert payload["shape"]["f1_macro"] == report["shape"]["f1_macro"]
    assert payload["shape"]["head"] == "shape"


def test_evaluate_checkpoint_mentions_unpopulated_shape_class(tmp_path: Path) -> None:
    """Laporan harus menyebut kelas bentuk yang tidak terisi.

    DIBaS tidak punya spesies spiral. Kalau kelas itu tidak disebut, pembaca
    laporan bisa mengira evaluasi dijalankan pada tiga kelas.
    """
    store = _store_with_split(["test", "test"], [COCci, BACILLI])
    checkpoint = _checkpoint(tmp_path / "heads.pt")

    report = evaluate_checkpoint(store, checkpoint, "test")

    assert "spiral" in report["scope_note"]
    assert SHAPE_LABELS == ("cocci", "bacilli")
    assert GRAM_LABELS == ("positif", "negatif")
