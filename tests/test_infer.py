"""Tes untuk inferensi satu citra."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bacteriacv.config import ALLOWED_SUFFIXES, GRAM_LABELS, SHAPE_LABELS
from bacteriacv.infer import (
    Prediction,
    Predictor,
    check_suffix,
)
from bacteriacv.model import build_model, save_checkpoint
from bacteriacv.visualize import SEGMENTATION_LIMITATION

GOOD_IMAGE = "sehat.png"
BAD_IMAGE = "catatan.txt"


def _image(dark: bool = True) -> np.ndarray:
    """Buat citra uji RGB."""
    array = np.full((240, 320, 3), 215, dtype=np.uint8)
    if dark:
        array[80:180, 90:230] = (20, 20, 220)
    return array


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    """Checkpoint head yang selalu terklasifikasi sebagai bacilli negatif."""
    model = build_model(pretrained=False)
    with torch.no_grad():
        model.head_a.weight.zero_()
        model.head_a.weight[:, 0] = torch.tensor([-4.0, 4.0])
        model.head_a.bias.zero_()
        model.head_b.weight.zero_()
        model.head_b.weight[:, 0] = torch.tensor([-4.0, 4.0])
        model.head_b.bias.zero_()
    return save_checkpoint(model, tmp_path / "heads.pt")


@pytest.fixture()
def predictor(checkpoint: Path) -> Predictor:
    """Predictor dengan bobot head yang sudah ditentukan."""
    return Predictor(checkpoint)



# --- Pemeriksaan suffix ---


def test_check_suffix_accepts_allowed_suffixes() -> None:
    """Semua ekstensi yang diizinkan harus diterima."""
    for suffix in ALLOWED_SUFFIXES:
        check_suffix(f"citra{suffix}")


def test_check_suffix_is_case_insensitive() -> None:
    """Ekstensi huruf besar harus tetap diterima."""
    check_suffix("CITRA.PNG")


def test_check_suffix_rejects_other_files() -> None:
    """Berkas noncitra harus ditolak."""
    with pytest.raises(ValueError, match="not supported"):
        check_suffix(BAD_IMAGE)


def test_check_suffix_rejects_missing_suffix() -> None:
    """Berkas tanpa ekstensi harus ditolak dengan pesan jelas."""
    with pytest.raises(ValueError, match="without extension"):
        check_suffix("namaberkas")


# --- Prediksi ---


def test_predict_returns_labels_from_lookup(predictor: Predictor) -> None:
    """Label harus memakai nama dari config, bukan angka indeks."""
    prediction = predictor.predict(_image()).prediction

    assert prediction.shape_label in SHAPE_LABELS
    assert prediction.gram_label in GRAM_LABELS


def test_predict_reports_confidence_in_range(predictor: Predictor) -> None:
    """Confidence harus berada di rentang nol sampai satu."""
    prediction = predictor.predict(_image()).prediction

    assert 0.0 <= prediction.shape_confidence <= 1.0
    assert 0.0 <= prediction.gram_confidence <= 1.0


def test_predict_confidence_is_probability_of_chosen_class(
    predictor: Predictor,
) -> None:
    """Confidence Gram harus probabilitas kelas yang dipilih.

    Kalau confidence dihitung dari ambang 0,5 tanpa membalik untuk kelas negatif,
    nilai yang tampil ke pengguna bukan keyakinan model.
    """
    prediction = predictor.predict(_image()).prediction

    if prediction.gram_index == 1:
        assert prediction.gram_confidence >= 0.5
    else:
        assert prediction.gram_confidence >= 0.5


def test_predict_shape_index_matches_label(predictor: Predictor) -> None:
    """Indeks dan label bentuk harus konsisten."""
    prediction = predictor.predict(_image()).prediction

    assert SHAPE_LABELS[prediction.shape_index] == prediction.shape_label


def test_predict_marks_segmentation_as_not_validated(predictor: Predictor) -> None:
    """Segmentasi tidak tervalidasi dan itu harus tercatat di hasil."""
    prediction = predictor.predict(_image()).prediction

    assert prediction.segmentation_validated is False
    assert any("not yet validated" in note for note in prediction.notes)


def test_predict_keeps_result_when_segmentation_fails(predictor: Predictor) -> None:
    """Citra tanpa area gelap membuat segmentasi gagal, prediksi tetap ada."""
    prediction = predictor.predict(_image(dark=False)).prediction

    assert prediction.segmentation_ok is False
    assert prediction.object_count == 0
    assert prediction.shape_label in SHAPE_LABELS


def test_predict_attaches_five_panels(predictor: Predictor) -> None:
    """Setiap prediksi harus membawa lima panel."""
    result = predictor.predict(_image())

    assert len(result.visualization.panels) == 5


def test_predict_records_stage_status(predictor: Predictor) -> None:
    """Status tiap tahap harus ikut tersimpan."""
    prediction = predictor.predict(_image()).prediction

    assert prediction.stages_ok["load"] is True
    assert prediction.stages_ok["resize"] is True


def test_predict_accepts_path(predictor: Predictor, tmp_path: Path) -> None:
    """Praproses harus menerima path, bukan hanya array."""
    import cv2

    location = tmp_path / GOOD_IMAGE
    cv2.imwrite(str(location), cv2.cvtColor(_image(), cv2.COLOR_RGB2BGR))

    prediction = predictor.predict(location).prediction

    assert prediction.shape_label in SHAPE_LABELS


def test_predict_rejects_missing_path(predictor: Predictor, tmp_path: Path) -> None:
    """Path yang tidak ada harus ditolak dengan pesan jelas."""
    with pytest.raises(FileNotFoundError):
        predictor.predict(tmp_path / "tidak_ada.png")


def test_predict_many_returns_one_result_each(predictor: Predictor) -> None:
    """predict_many harus mengembalikan satu hasil per citra."""
    results = predictor.predict_many([_image(), _image(dark=False), _image()])

    assert len(results) == 3
    assert all(result.prediction.shape_label in SHAPE_LABELS for result in results)


def test_predict_is_deterministic(predictor: Predictor) -> None:
    """Citra sama harus menghasilkan prediksi sama."""
    first = predictor.predict(_image()).prediction
    second = predictor.predict(_image()).prediction

    assert first.shape_label == second.shape_label
    assert first.shape_confidence == second.shape_confidence


def test_predictor_requires_existing_checkpoint(tmp_path: Path) -> None:
    """Checkpoint yang tidak ada harus ditolak saat Predictor dibangun."""
    with pytest.raises(FileNotFoundError):
        Predictor(tmp_path / "tidak_ada.pt")


def test_prediction_is_json_serializable(predictor: Predictor) -> None:
    """Prediksi harus dapat dikirim sebagai JSON ke antarmuka."""
    payload = predictor.predict(_image()).prediction.to_dict()

    restored = json.loads(json.dumps(payload))
    assert restored["shape_label"] in SHAPE_LABELS
    assert isinstance(restored["notes"], list)
    assert SEGMENTATION_LIMITATION in restored["notes"]


def test_prediction_notes_never_claim_diagnosis(predictor: Predictor) -> None:
    """Hasil harus selalu menyebut sifat alat bantu, bukan diagnosis."""
    prediction = predictor.predict(_image()).prediction

    assert any("not a diagnosis" in note for note in prediction.notes)
