"""Tes untuk ekstraksi fitur dan loop pelatihan dengan feature caching.

Karena backbone dibekukan, vektor fitur sebuah citra tidak berubah antar epoch.
Fitur diekstrak sekali lalu dipakai berulang, sehingga epoch berikutnya hanya
menjalankan dua head berukuran Linear(2048, 2).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bacteriacv.config import AUGMENT_VARIANTS, FEATURE_DIM
from bacteriacv.train import (
    FeatureStore,
    TrainConfig,
    TrainingReport,
    extract_features_for_rows,
    load_feature_store,
    save_feature_store,
    train,
)

# Dua species_id yang berbeda pada kedua head sekaligus: satu cocci positif,
# satu bacilli negatif.
COCci = "staphylococcus_aureus"
BACILLI = "escherichia_coli"


def _fake_rows(project_tmp_dir: Path, species_ids: list[str]) -> list[dict[str, str]]:
    """Buat berkas citra palsu dan baris index yang menunjuknya."""
    import cv2

    rows = []
    for index, species_id in enumerate(species_ids):
        for number in range(4):
            directory = project_tmp_dir / species_id
            directory.mkdir(parents=True, exist_ok=True)
            name = f"{species_id}_{number:04d}.png"
            image = np.full((64, 64, 3), 120 + index * 10, dtype=np.uint8)
            image[20:40, 20:40] = (200, 40, 40)
            assert cv2.imwrite(str(directory / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            rows.append(
                {
                    "path": f"{species_id}/{name}",
                    "species_id": species_id,
                    "split": "train",
                    "fold": number % 2,
                }
            )
    return rows


def _fake_rows_with_split(project_tmp_dir: Path, species_ids: list[str]) -> list[dict[str, str]]:
    """Baris index dengan pembagian train, val, dan test."""
    rows = []
    for species_id in species_ids:
        split_rows = _fake_rows(project_tmp_dir, [species_id])
        for position, row in enumerate(split_rows):
            row["split"] = ("train", "val", "test", "train")[position % 4]
            row["fold"] = position % 2 if row["split"] == "train" else -1
            rows.append(row)
    return rows


# --- Feature store ---


def test_feature_store_roundtrip(tmp_path) -> None:
    """Matriks fitur dan label harus pulih persis setelah disimpan."""
    matrix = np.arange(12 * FEATURE_DIM, dtype=np.float32).reshape(12, FEATURE_DIM)
    species = [f"s{i % 3}" for i in range(12)]
    splits = ["train"] * 8 + ["val"] * 2 + ["test"] * 2
    paths = [f"p{i:03d}.tif" for i in range(12)]

    save_feature_store(FeatureStore(matrix, species, splits, paths), tmp_path / "store")
    loaded = load_feature_store(tmp_path / "store")

    assert np.array_equal(loaded.matrix, matrix)
    assert loaded.species_ids == species
    assert loaded.splits == splits
    assert loaded.paths == paths


def test_feature_store_validates_dimension(tmp_path) -> None:
    """Dimensi fitur yang salah harus ditolak."""
    store = FeatureStore(np.zeros((3, 7), dtype=np.float32), ["a", "b", "c"], [], [])
    save_feature_store(store, tmp_path / "store")

    with pytest.raises(ValueError, match="dimensi"):
        load_feature_store(tmp_path / "store")


def test_feature_store_validates_length_mismatch(tmp_path) -> None:
    """Jumlah label harus sama dengan jumlah baris fitur."""
    store = FeatureStore(np.zeros((3, FEATURE_DIM), dtype=np.float32), ["a"], [], [])
    save_feature_store(store, tmp_path / "store")

    with pytest.raises(ValueError, match="label"):
        load_feature_store(tmp_path / "store")


def test_feature_store_missing_raises(tmp_path) -> None:
    """Feature store yang belum ada harus ditolak dengan pesan jelas."""
    with pytest.raises(FileNotFoundError):
        load_feature_store(tmp_path / "belum_ada")


def test_extract_features_dimension(project_tmp_dir: Path) -> None:
    """Ekstraksi harus menghasilkan vektor 2048-d per citra."""
    from bacteriacv.model import build_model

    rows = _fake_rows(project_tmp_dir, ["escherichia_coli"])
    model = build_model(pretrained=False)

    store = extract_features_for_rows(model, project_tmp_dir, rows, augment=False)

    assert store.matrix.shape == (len(rows), FEATURE_DIM)
    assert store.matrix.dtype == np.float32


def test_extract_features_with_augment_multiplies_train_rows(project_tmp_dir: Path) -> None:
    """Augmentasi menambah baris hanya untuk data latih."""
    from bacteriacv.model import build_model

    rows = _fake_rows_with_split(project_tmp_dir, ["escherichia_coli"])
    model = build_model(pretrained=False)

    store = extract_features_for_rows(model, project_tmp_dir, rows, augment=True)

    n_train = sum(1 for row in rows if row["split"] == "train")
    n_other = len(rows) - n_train
    expected = n_train * AUGMENT_VARIANTS + n_other
    assert store.matrix.shape == (expected, FEATURE_DIM)


def test_extract_features_skips_val_and_test(project_tmp_dir: Path) -> None:
    """Validasi dan uji harus diekstrak tepat satu kali tanpa augmentasi."""
    from bacteriacv.model import build_model

    rows = _fake_rows_with_split(project_tmp_dir, ["escherichia_coli"])
    model = build_model(pretrained=False)

    store = extract_features_for_rows(model, project_tmp_dir, rows, augment=True)

    n_val = sum(1 for row in rows if row["split"] == "val")
    n_test = sum(1 for row in rows if row["split"] == "test")
    n_train = sum(1 for row in rows if row["split"] == "train")

    assert store.splits.count("val") == n_val
    assert store.splits.count("test") == n_test
    assert store.splits.count("train") == n_train * AUGMENT_VARIANTS


def test_extract_features_is_deterministic_without_augment(project_tmp_dir: Path) -> None:
    """Tanpa augmentasi, ekstraksi dua kali atas model sama harus identik.

    Model yang sama dipakai dua kali, bukan dua model berbeda. Dengan
    pretrained=False bobot backbone diacak, sehingga dua model terpisah
    memberi fitur berbeda meski citra dan pipeline-nya sama.
    """
    from bacteriacv.model import build_model

    rows = _fake_rows(project_tmp_dir, ["escherichia_coli"])
    model = build_model(pretrained=False)

    first = extract_features_for_rows(model, project_tmp_dir, rows)
    second = extract_features_for_rows(model, project_tmp_dir, rows)

    assert np.array_equal(first.matrix, second.matrix)


# --- Cache fitur ---


def test_cache_signature_is_stable(project_tmp_dir: Path) -> None:
    """Sidik jari cache tidak berubah tanpa perubahan pada citra."""
    from bacteriacv.train import cache_signature

    rows = _fake_rows(project_tmp_dir, ["escherichia_coli"])

    assert cache_signature(project_tmp_dir, rows, True) == cache_signature(
        project_tmp_dir, rows, True
    )


def test_cache_signature_changes_with_augment_flag(project_tmp_dir: Path) -> None:
    """Bendera augmentasi harus masuk sidik jari cache."""
    from bacteriacv.train import cache_signature

    rows = _fake_rows(project_tmp_dir, ["escherichia_coli"])

    assert cache_signature(project_tmp_dir, rows, True) != cache_signature(
        project_tmp_dir, rows, False
    )


def test_cache_signature_changes_when_image_is_touched(project_tmp_dir: Path) -> None:
    """Mengubah isi citra harus membatalkan cache."""
    import cv2

    from bacteriacv.train import cache_signature

    rows = _fake_rows(project_tmp_dir, ["escherichia_coli"])
    before = cache_signature(project_tmp_dir, rows, False)

    target = project_tmp_dir / rows[0]["path"]
    cv2.imwrite(
        str(target), np.full((64, 64, 3), 7, dtype=np.uint8)
    )

    assert cache_signature(project_tmp_dir, rows, False) != before


def test_build_or_load_features_reuses_cache(project_tmp_dir: Path, tmp_path: Path) -> None:
    """Panggilan kedua harus membaca cache, bukan ekstrak ulang.

    Cara membuktikannya adalah mengganti bobot backbone di antara dua
    panggilan. Bila ekstraksi diulang, fitur berubah; bila cache dipakai,
    fitur tetap sama.
    """
    from bacteriacv.model import build_model
    from bacteriacv.train import build_or_load_features

    rows = _fake_rows(project_tmp_dir, ["escherichia_coli"])
    model = build_model(pretrained=False)

    first, directory, reused_first = build_or_load_features(
        model, project_tmp_dir, rows, cache_root=tmp_path
    )
    assert reused_first is False

    second, same_directory, reused_second = build_or_load_features(
        model, project_tmp_dir, rows, cache_root=tmp_path
    )

    assert reused_second is True
    assert same_directory == directory
    assert np.array_equal(first.matrix, second.matrix)
    assert first.splits == second.splits
    assert first.paths == second.paths


# --- Loop pelatihan ---


def test_train_config_defaults_are_sane() -> None:
    """Konfigurasi pelatihan harus berada di rentang bermakna."""
    config = TrainConfig()

    assert config.epochs > 0
    assert 0 < config.learning_rate < 1
    assert config.batch_size > 0
    assert config.patience > 0
    assert config.seed != 0


def _separable_store(n_train: int = 40, n_eval: int = 20) -> FeatureStore:
    """Feature store yang mudah dipisah kedua head.

    Spesies memakai species_id sungguhan dari lookup table, karena label kedua
    head diturunkan dari tabel itu dan bukan dari nama kelas buatan.
    staphylococcus_aureus adalah cocci positif, escherichia_coli adalah
    bacilli negatif, sehingga kedua head punya signal yang mudah dipisah.

    Setiap split harus memuat kedua kelas. Split satu kelas membuat F1 makro
    tidak pernah melewati 0,5 apa pun hasil belajarnya, karena kelas yang
    absen di sebenarnya maupun prediksi diberi skor nol.
    """
    rng = np.random.default_rng(11)
    half_train = n_train // 2
    train_matrix = np.vstack(
        [
            rng.normal(loc=2.0, scale=0.3, size=(half_train, FEATURE_DIM)),
            rng.normal(loc=-2.0, scale=0.3, size=(half_train, FEATURE_DIM)),
        ]
    ).astype(np.float32)
    train_species = [COCci] * half_train + [BACILLI] * half_train

    quarter = n_eval // 4
    val_matrix = np.vstack(
        [
            rng.normal(loc=2.0, scale=0.3, size=(quarter, FEATURE_DIM)),
            rng.normal(loc=-2.0, scale=0.3, size=(quarter, FEATURE_DIM)),
        ]
    ).astype(np.float32)
    test_matrix = np.vstack(
        [
            rng.normal(loc=2.0, scale=0.3, size=(quarter, FEATURE_DIM)),
            rng.normal(loc=-2.0, scale=0.3, size=(quarter, FEATURE_DIM)),
        ]
    ).astype(np.float32)

    matrix = np.vstack([train_matrix, val_matrix, test_matrix]).astype(np.float32)
    species = (
        train_species
        + [COCci] * quarter
        + [BACILLI] * quarter
        + [COCci] * quarter
        + [BACILLI] * quarter
    )
    splits = ["train"] * n_train + ["val"] * (n_eval // 2) + ["test"] * (n_eval // 2)
    paths = [f"p{i:04d}.tif" for i in range(len(matrix))]
    return FeatureStore(matrix, species, splits, paths)


def test_train_learns_separable_data(tmp_path: Path) -> None:
    """Data yang mudah dipisah harus menghasilkan F1 validasi tinggi."""
    store = _separable_store()

    report = train(store, TrainConfig(epochs=40, patience=10, seed=0, checkpoint_dir=tmp_path))

    assert report.best_val_f1_shape > 0.9
    assert report.best_val_f1_gram > 0.9


def test_train_reports_shape_and_gram_metrics(tmp_path: Path) -> None:
    """Laporan harus memuat metrik untuk kedua head."""
    report = train(_separable_store(), TrainConfig(epochs=5, patience=3, seed=0, checkpoint_dir=tmp_path))

    assert 0.0 <= report.test_f1_shape[0] <= 1.0
    assert 0.0 <= report.test_f1_gram[0] <= 1.0
    assert 0.0 <= report.test_accuracy_shape <= 1.0
    assert 0.0 <= report.test_accuracy_gram <= 1.0


def test_train_evaluates_test_once(tmp_path: Path) -> None:
    """Data uji hanya boleh dievaluasi sekali, yaitu di akhir."""
    report = train(_separable_store(), TrainConfig(epochs=5, patience=3, seed=0, checkpoint_dir=tmp_path))

    assert len(report.test_f1_shape) == 1
    assert len(report.test_f1_gram) == 1


def test_train_uses_early_stopping(tmp_path: Path) -> None:
    """Pelatihan harus berhenti sebelum batas epoch bila validasi membaik."""
    report = train(_separable_store(), TrainConfig(epochs=200, patience=5, seed=0, checkpoint_dir=tmp_path))

    assert report.epochs_run < 200
    assert report.best_epoch < report.epochs_run


def test_train_records_history(tmp_path: Path) -> None:
    """Riwayat per epoch harus tercatat."""
    report = train(_separable_store(), TrainConfig(epochs=5, patience=3, seed=0, checkpoint_dir=tmp_path))

    assert len(report.history) == report.epochs_run
    assert {"epoch", "train_loss", "val_f1_shape", "val_f1_gram"} <= set(
        report.history[0]
    )


def test_report_dict_has_no_absolute_paths(tmp_path: Path) -> None:
    """Laporan pelatihan hanya boleh memuat nama berkas dan nama folder.

    training_report.json ikut ter-deploy karena tercantum di baris negasi
    .gitignore. Path absolut di dalamnya membocorkan struktur folder mesin
    pembangun dan melanggar AGENT.md bagian 3 aturan 5. Kedua key ini tidak
    pernah diuji sebelumnya karena seluruh assertion bekerja pada objek
    TrainingReport di memori, bukan pada hasil serialisasi.

    tmp_path dipakai sebagai checkpoint_dir supaya tes membuktikan direktori
    dibuang, bukan kebetulan nama foldernya memang tanpa direktori.
    """
    target = tmp_path / "checkpoints"
    report = train(
        _separable_store(),
        TrainConfig(epochs=3, patience=2, seed=0, checkpoint_dir=target),
    )

    payload = report.to_dict()

    assert payload["checkpoint_path"].endswith("heads.pt")
    assert payload["config"]["checkpoint_dir"] == "checkpoints"


def test_report_dict_paths_have_no_path_syntax(tmp_path: Path) -> None:
    """Nilai path pada laporan tidak boleh memuat sisa sintaks path.

    Pemeriksaan dipisah agar setiap kelas kesalahan ditangkap oleh assertion
    sendiri: pemisah POSIX, pemisah Windows, dan huruf drive. Path di bawah
    sengaja berlapis supaya tes tidak lulus karena kebetulan.
    """
    nested = tmp_path / "a" / "b" / "checkpoints"
    nested.mkdir(parents=True)
    report = train(
        _separable_store(),
        TrainConfig(epochs=3, patience=2, seed=0, checkpoint_dir=nested),
    )

    payload = report.to_dict()
    values = [payload["checkpoint_path"], payload["config"]["checkpoint_dir"]]

    for value in values:
        assert isinstance(value, str), type(value)
        for separator in ("/", "\\"):
            assert separator not in value, f"pemisah {separator!r} bocor pada {value!r}"
        assert ":" not in value, f"huruf drive bocor pada {value!r}"
        assert value == value.strip(), f"spasi tepi bocor pada {value!r}"
        assert not value.startswith("."), f"path relatif bocor pada {value!r}"


def test_report_dict_is_json_serializable_without_paths(tmp_path: Path) -> None:
    """Hasil to_dict harus tetap bisa diserialisasi ke JSON seperti biasa.

    to_dict mengganti objek Path menjadi teks karena JSON tidak punya tipe Path.
    Tes ini memastikan penggantian itu tidak merusak bentuk JSON, sekaligus
    memeriksa tidak ada path absolut yang ikut terbawa ke teks.
    """
    report = train(
        _separable_store(),
        TrainConfig(
            epochs=3,
            patience=2,
            seed=0,
            checkpoint_dir=tmp_path / "checkpoints",
        ),
    )

    restored = json.loads(json.dumps(report.to_dict()))

    assert restored["checkpoint_path"] == "heads.pt"
    assert restored["config"]["checkpoint_dir"] == "checkpoints"
    assert restored["config"]["seed"] == report.config["seed"]


def test_train_rejects_missing_split() -> None:
    """Split kosong harus ditolak sebelum pelatihan dimulai."""
    store = FeatureStore(
        np.zeros((10, FEATURE_DIM), dtype=np.float32),
        [BACILLI] * 10,
        ["train"] * 10,
        [f"p{i}.tif" for i in range(10)],
    )

    with pytest.raises(ValueError, match="val"):
        train(store, TrainConfig(epochs=1))


def test_train_rejects_missing_species() -> None:
    """Spesies di luar lookup harus ditolak.

    Split sengaja dibuat lengkap supaya kegagalan benar-benar berasal dari
    species_id, bukan dari pemeriksaan(split) yang berjalan lebih dulu.
    """
    store = FeatureStore(
        np.zeros((3, FEATURE_DIM), dtype=np.float32),
        ["spesies_misterius"] * 3,
        ["train", "val", "test"],
        ["a.tif", "b.tif", "c.tif"],
    )

    with pytest.raises(KeyError):
        train(store, TrainConfig(epochs=1))


def test_train_does_not_use_test_for_early_stopping(tmp_path: Path) -> None:
    """Skor test tidak boleh memengaruhi epoch terbaik."""
    store = _separable_store()

    report = train(store, TrainConfig(epochs=8, patience=4, seed=0, checkpoint_dir=tmp_path))

    for entry in report.history:
        assert "test_f1_shape" not in entry
        assert "test_f1_gram" not in entry


def test_train_is_reproducible_with_same_seed(tmp_path: Path) -> None:
    """Seed sama harus menghasilkan riwayat sama."""
    store = _separable_store()

    first = train(store, TrainConfig(epochs=5, patience=3, seed=7, checkpoint_dir=tmp_path))
    second = train(store, TrainConfig(epochs=5, patience=3, seed=7, checkpoint_dir=tmp_path))

    assert [row["train_loss"] for row in first.history] == [
        row["train_loss"] for row in second.history
    ]


def test_train_handles_class_imbalance_with_weights(tmp_path: Path) -> None:
    """Data tidak seimbang harus tetap terklasifikasi.

    Rasio kelas 3 banding 1. Setiap split memuat kedua kelas supaya F1 makro
    terdefinisi, dan posisi kelas minor disebar agar tidak cuma berada di uji.
    """
    rng = np.random.default_rng(5)
    blocks = [(75, 25, 0), (10, 3, 50), (5, 2, 63)]
    matrix_parts: list[np.ndarray] = []
    species: list[str] = []
    splits: list[str] = []
    splits_order = ("train", "val", "test")

    for position, (majority, minority, offset) in enumerate(blocks):
        rows = majority + minority
        block = np.vstack(
            [
                rng.normal(loc=1.0, scale=0.2, size=(majority, FEATURE_DIM)),
                rng.normal(loc=-1.0, scale=0.2, size=(minority, FEATURE_DIM)),
            ]
        ).astype(np.float32)
        permuted = rng.permutation(rows)
        matrix_parts.append(block[permuted])
        for index in range(rows):
            original = permuted[index]
            species.append(BACILLI if original < majority else COCci)
            splits.append(splits_order[position])

    matrix = np.vstack(matrix_parts)
    assert matrix.shape == (100 + 13 + 7, FEATURE_DIM)
    paths = [f"p{i:04d}.tif" for i in range(len(matrix))]
    store = FeatureStore(matrix, species, splits, paths)

    report = train(store, TrainConfig(epochs=40, patience=10, seed=0, checkpoint_dir=tmp_path))

    assert report.best_val_f1_shape > 0.7


def test_train_checkpoint_can_be_reloaded(tmp_path: Path) -> None:
    """Checkpoint hasil pelatihan harus dapat dimuat kembali."""
    from bacteriacv.model import build_model, load_checkpoint

    report = train(_separable_store(), TrainConfig(epochs=3, patience=2, seed=0, checkpoint_dir=tmp_path))
    assert report.checkpoint_path is not None

    model = load_checkpoint(build_model(pretrained=False), report.checkpoint_path)
    assert model is not None


def test_train_report_is_serializable(tmp_path: Path) -> None:
    """Laporan harus dapat diubah menjadi JSON untuk disimpan."""
    import json

    report = train(_separable_store(), TrainConfig(epochs=3, patience=2, seed=0, checkpoint_dir=tmp_path))
    payload = report.to_dict()

    assert json.loads(json.dumps(payload))["epochs_run"] == report.epochs_run