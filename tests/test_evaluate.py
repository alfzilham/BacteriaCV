"""Tests for checkpoint evaluation on the test data."""

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
    """Build a feature store with features that separate cleanly per species."""
    matrix = np.zeros((len(splits), FEATURE_DIM), dtype=np.float32)
    for index, species_id in enumerate(species):
        matrix[index, :] = 1.0 if species_id == COCci else -1.0
    return FeatureStore(matrix, species, splits, [f"p{i:03d}.tif" for i in range(len(splits))])


def _checkpoint(path: Path) -> Path:
    """Build a head checkpoint that separates both classes on the zero axis.

    The head weights are set by hand so the evaluation result does not depend on
    random initialisation.
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
    """Confidence must be reported separately for each head.

    The shape confidence and the Gram confidence cannot be added up. They use
    different scales and different calibrations, and no class unifies them. A
    combined figure has misled before, so it is gone from the report.
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
    """Gram confidence must match the probability of the chosen class.

    For the negative class the confidence must be 1 minus the positive probability.
    If the confidence were taken straight from the raw probability without
    inverting it, a Gram negative image would look wrongly certain.
    """
    store = _store_with_split(["test", "test"], [COCci, BACILLI])
    checkpoint = _checkpoint(tmp_path / "heads.pt")

    report = evaluate_checkpoint(store, checkpoint, "test")

    gram_probability = float(report["gram"]["per_class"][1]["support"]) / 2.0
    assert 0.0 <= report["mean_confidence_gram"] <= 1.0
    assert report["mean_confidence_gram"] > 0.9
    assert gram_probability > 0.0


def test_gram_uses_sigmoid_not_softmax_over_both_columns(tmp_path: Path) -> None:
    """Head B is a single logit classifier, not a two class softmax.

    The first output column never enters the BCEWithLogitsLoss, so it carries no
    learned information. A softmax over two columns would mix the trained logit
    with one that only experiences weight decay. This test locks the choice of
    sigmoid on the second column.
    """
    store = _store_with_split(["test", "test"], [COCci, BACILLI])
    checkpoint = _checkpoint(tmp_path / "heads.pt")

    report = evaluate_checkpoint(store, checkpoint, "test")

    note = report["gram_calibration_note"]
    assert "sigmoid(logit)" in note
    assert "tidak pernah masuk loss" in note
    assert "tidak boleh dipakai" in note


def test_report_checkpoint_contains_only_file_name(tmp_path: Path) -> None:
    """The report may only carry the checkpoint filename, not a path.

    evaluation.json is deployed and read by anyone through /api/report. An
    absolute path inside it leaks the folder structure of the build machine and
    breaks AGENT.md section 3 rule 5. This field was once written without a single
    assertion reading it, so the leak passed every earlier audit.

    The path below deliberately has a subdirectory so the test proves the
    directory is stripped, rather than the filename happening to have none.
    """
    nested = tmp_path / "sub"
    nested.mkdir()
    store = _store_with_split(["train", "test", "test"], [BACILLI, COCci, BACILLI])
    checkpoint = _checkpoint(nested / "heads.pt")

    report = evaluate_checkpoint(store, checkpoint, "test")

    assert report["checkpoint"] == "heads.pt"


def test_report_checkpoint_has_no_path_syntax(tmp_path: Path) -> None:
    """The checkpoint filename must not carry leftover path syntax.

    Four separate checks, because each covers a different error class: POSIX
    separators, Windows separators, a drive letter, and disguised relative or
    absolute path shapes.
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
    """Changing how the confidence is computed must not change the labels.

    The confidence and the accuracy come from the same logit, so the accuracy
    must stay exactly the same once the confidence is split per head.
    """
    store = _store_with_split(
        ["train", "test", "test", "test"], [BACILLI, COCci, BACILLI, BACILLI]
    )
    checkpoint = _checkpoint(tmp_path / "heads.pt")

    report = evaluate_checkpoint(store, checkpoint, "test")

    assert report["gram"]["accuracy"] == 1.0
    assert report["shape"]["accuracy"] == 1.0
    assert report["mean_confidence_shape"] >= report["mean_confidence_gram"]


# --- Basic metrics ---


def test_confusion_matrix_counts_pairs() -> None:
    """The confusion matrix must count the correct and incorrect pairs."""
    truth = np.array([0, 0, 1, 1])
    prediction = np.array([0, 1, 1, 1])

    assert confusion_matrix(truth, prediction, 2) == [[1, 1], [0, 2]]


def test_confusion_matrix_handles_three_classes() -> None:
    """A class that does not appear must still get a zero row and column."""
    truth = np.array([0, 2, 2])
    prediction = np.array([0, 2, 0])

    assert confusion_matrix(truth, prediction, 3) == [[1, 0, 0], [0, 0, 0], [1, 0, 1]]


def test_per_class_metrics_reports_perfect_prediction() -> None:
    """A perfect prediction must give an F1 of one for every class."""
    truth = np.array([0, 0, 1, 1])
    prediction = np.array([0, 0, 1, 1])

    rows, support = per_class_metrics(truth, prediction, ["a", "b"], 2)

    assert support == [2, 2]
    assert all(row["f1"] == 1.0 for row in rows)
    assert all(row["precision"] == 1.0 for row in rows)
    assert all(row["recall"] == 1.0 for row in rows)


def test_per_class_metrics_marks_missing_class_as_zero() -> None:
    """A class absent from both truth and prediction must be able to score zero.

    If the empty class were given an F1 of one, the macro F1 on a single class
    split would always be maximal and useless as an early stopping signal.
    """
    truth = np.array([0, 0])
    prediction = np.array([0, 0])

    rows, support = per_class_metrics(truth, prediction, ["a", "b"], 2)

    assert support == [2, 0]
    assert rows[1]["f1"] == 0.0
    assert rows[1]["recall"] == 0.0


def test_evaluate_head_aggregates() -> None:
    """evaluate_head must summarise macro F1, accuracy, and support.

    One thousand one row of test data: three class 0, three class 1, and one
    misclassification in each direction. Class 0 F1 4/5, class 1 F1 6/7.
    """
    truth = np.array([0, 0, 0, 1, 1, 1])
    prediction = np.array([0, 0, 1, 1, 1, 1])

    metrics = evaluate_head("shape", truth, prediction, ["a", "b"], 2)

    assert metrics.f1_macro == pytest.approx((0.8 + 6 / 7) / 2)
    assert metrics.accuracy == pytest.approx(5 / 6)
    assert metrics.positive_support == 3
    assert metrics.labels == ["a", "b"]
    assert metrics.confusion == [[2, 1], [0, 3]]


# --- Split selection ---


def test_select_split_returns_positions() -> None:
    """select_split must return the row indices of that split."""
    store = _store_with_split(
        ["train", "train", "val", "test", "test"], [BACILLI, BACILLI, BACILLI, COCci, BACILLI]
    )

    assert select_split(store, "test").tolist() == [3, 4]
    assert select_split(store, "train").tolist() == [0, 1]


def test_select_split_rejects_empty_split() -> None:
    """A split with no rows must be rejected."""
    store = _store_with_split(["train", "train"], [BACILLI, COCci])

    with pytest.raises(ValueError, match="test"):
        select_split(store, "test")


# --- Per species summary ---


def test_species_breakdown_uses_lookup_labels() -> None:
    """The per species rows must use the labels from the lookup table."""
    species = [COCci, COCci, BACILLI]
    correct = np.array([True, True, False])
    predictions = np.array([0, 0, 0])

    rows = species_breakdown(species, correct, predictions)

    assert len(rows) == 2
    cocci = next(row for row in rows if row["species_id"] == COCci)
    bacilli = next(row for row in rows if row["species_id"] == BACILLI)
    assert cocci["expected_shape"] == "cocci"
    assert cocci["expected_gram"] == "positive"
    assert cocci["n"] == 2
    assert cocci["shape_accuracy"] == 1.0
    assert bacilli["expected_shape"] == "bacilli"
    assert bacilli["expected_gram"] == "negative"
    assert bacilli["shape_accuracy"] == 0.0
    assert bacilli["predicted_shapes"] == ["cocci"]


def test_species_breakdown_counts_both_shapes_under_one_species() -> None:
    """One species may have two different predictions."""
    species = [BACILLI, BACILLI, BACILLI]
    correct = np.array([True, False, False])
    predictions = np.array([1, 0, 0])

    rows = species_breakdown(species, correct, predictions)

    assert rows[0]["predicted_shapes"] == ["bacilli", "cocci"]


# --- evaluate_checkpoint ---


def test_evaluate_checkpoint_perfect_model(tmp_path: Path) -> None:
    """A checkpoint that separates the classes must get an F1 and accuracy of one."""
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
    """Rows outside the split must not affect the result."""
    store = _store_with_split(
        ["test", "test", "train"], [COCci, BACILLI, COCci]
    )
    checkpoint = _checkpoint(tmp_path / "heads.pt")

    report = evaluate_checkpoint(store, checkpoint, "test")

    assert report["n_images"] == 2
    assert report["shape"]["confusion"] == [[1, 0], [0, 1]]


def test_evaluate_checkpoint_reports_missing_checkpoint(tmp_path: Path) -> None:
    """A checkpoint that does not exist must be rejected."""
    store = _store_with_split(["test", "test"], [COCci, BACILLI])

    with pytest.raises(FileNotFoundError):
        evaluate_checkpoint(store, tmp_path / "tidak_ada.pt", "test")


def test_evaluate_checkpoint_rejects_empty_split(tmp_path: Path) -> None:
    """An empty split must be rejected before the checkpoint is loaded."""
    store = _store_with_split(["train"], [BACILLI])
    checkpoint = _checkpoint(tmp_path / "heads.pt")

    with pytest.raises(ValueError, match="test"):
        evaluate_checkpoint(store, checkpoint, "test")


def test_evaluate_checkpoint_is_json_serializable(tmp_path: Path) -> None:
    """The report must be writable as JSON as it stands."""
    store = _store_with_split(
        ["train", "test", "test", "test"], [BACILLI, COCci, BACILLI, BACILLI]
    )
    checkpoint = _checkpoint(tmp_path / "heads.pt")

    report = evaluate_checkpoint(store, checkpoint, "test")

    payload = json.loads(json.dumps(report))
    assert payload["shape"]["f1_macro"] == report["shape"]["f1_macro"]
    assert payload["shape"]["head"] == "shape"


def test_evaluate_checkpoint_mentions_unpopulated_shape_class(tmp_path: Path) -> None:
    """The report must name the shape class that is unpopulated.

    DIBaS has no spiral species. If that class were not named, a reader of the
    report could assume the evaluation ran on three classes.
    """
    store = _store_with_split(["test", "test"], [COCci, BACILLI])
    checkpoint = _checkpoint(tmp_path / "heads.pt")

    report = evaluate_checkpoint(store, checkpoint, "test")

    assert "spiral" in report["scope_note"]
    assert SHAPE_LABELS == ("cocci", "bacilli")
    assert GRAM_LABELS == ("positive", "negative")
