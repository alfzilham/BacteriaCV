"""Inferensi untuk satu citra dan perakitan hasil lengkap.

Jalur inferensi tidak pernah menyentuh augmentasi. preprocess tidak punya
parameter augmentasi dan tidak memanggil augment_train_variants, sehingga tidak
ada jalan untuk membocorkannya ke sini.

Segmentasi tidak memengaruhi metrik model. ARCHITECTURE bagian 3 menyatakan
segmentasi tidak dibutuhkan inferensi karena klasifikasi bentuk dan status Gram
berasal dari citra, bukan dari mask. Hasil segmentasi tetap dihitung karena
dipakai panel visualisasi, dan kegagalan segmentasi tidak menghentikan prediksi.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .config import ALLOWED_SUFFIXES, CHECKPOINT_DIR, LOW_CONFIDENCE_WARNING
from .model import build_model, gram_label, load_checkpoint, shape_label
from .paths import IMAGES_DIR
from .preprocess import PreprocessResult, preprocess
from .visualize import (
    SEGMENTATION_LIMITATION,
    VisualizationBundle,
    build_visualization,
    confidence_level,
)

CHECKPOINT_NAME = "heads.pt"

DEFAULT_CHECKPOINT = CHECKPOINT_DIR / CHECKPOINT_NAME


@dataclass(frozen=True)
class Prediction:
    """Hasil prediksi untuk satu citra.

    Attributes:
        shape_label: Nama kelas bentuk.
        shape_confidence: Confidence bentuk.
        gram_label: Nama kelas status Gram.
        gram_confidence: Confidence Gram.
        shape_index: Indeks kelas bentuk.
        gram_index: Indeks kelas status Gram.
        shape_level: Level verbal confidence bentuk.
        gram_level: Level verbal confidence Gram.
        object_count: Jumlah objek segmentasi, nol bila segmentasi gagal.
        segmentation_ok: Apakah segmentasi menghasilkan objek.
        segmentation_validated: Selalu salah, see SEGMENTATION_LIMITATION.
        stages_ok: Status keberhasilan tiap tahap pra-pemrosesan.
        notes: Catatan yang wajib ditampilkan pengguna.
    """

    shape_label: str
    shape_confidence: float
    gram_label: str
    gram_confidence: float
    shape_index: int
    gram_index: int
    shape_level: str
    gram_level: str
    object_count: int
    segmentation_ok: bool
    segmentation_validated: bool
    stages_ok: dict[str, bool]
    notes: tuple[str, ...]

    def to_dict(self) -> dict:
        """Ubah prediksi menjadi dictionary untuk serialisasi JSON."""
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True)
class InferenceResult:
    """Prediksi beserta panel visualisasi untuk satu citra.

    Attributes:
        prediction: Hasil prediksi kedua head.
        visualization: Lima panel berannotasi.
    """

    prediction: Prediction
    visualization: VisualizationBundle


class Predictor:
    """Pembungkus model untuk prediksi satu citra.

    Model dibangun sekali lalu dipakai berulang. Backbone dibekukan, sehingga
    tidak ada alasan membangun ulang di antara permintaan.
    """

    def __init__(self, checkpoint_path: Path | str = DEFAULT_CHECKPOINT) -> None:
        """Bangun model dan muat checkpoint head.

        Args:
            checkpoint_path: Lokasi checkpoint head.

        Raises:
            FileNotFoundError: Bila checkpoint tidak ada.
        """
        self.model = load_checkpoint(build_model(pretrained=True), checkpoint_path)
        self.model.eval()
        self.checkpoint_path = Path(checkpoint_path)

    def predict(self, image: np.ndarray | Path | str) -> InferenceResult:
        """Praproses satu citra lalu prediksi kedua head.

        Args:
            image: Array RGB, atau path berkas citra.

        Returns:
            InferenceResult berisi prediksi dan panel visualisasi.

        Raises:
            FileNotFoundError: Bila path citra tidak ada.
            ValueError: Bila citra tidak dapat dibaca.
        """
        prepared: PreprocessResult = preprocess(image)

        with torch.no_grad():
            batch = prepared.tensor.unsqueeze(0)
            features = self.model.extract_features(batch)
            shape_probabilities = self.model.head_a(features).softmax(dim=-1)[0]
            gram_probability = float(self.model.head_b(features)[0, 1].sigmoid())

        shape_index = int(shape_probabilities.argmax())
        shape_confidence = float(shape_probabilities[shape_index])
        gram_index = 1 if gram_probability > 0.5 else 0
        gram_confidence = gram_probability if gram_index == 1 else 1.0 - gram_probability

        prediction = Prediction(
            shape_label=shape_label(shape_index),
            shape_confidence=shape_confidence,
            gram_label=gram_label(gram_index),
            gram_confidence=gram_confidence,
            shape_index=shape_index,
            gram_index=gram_index,
            shape_level=confidence_level(shape_confidence),
            gram_level=confidence_level(gram_confidence),
            object_count=prepared.object_count,
            segmentation_ok=prepared.stage_ok.get("segment", False),
            segmentation_validated=False,
            stages_ok=dict(prepared.stage_ok),
            notes=_notes(prepared),
        )

        return InferenceResult(
            prediction=prediction,
            visualization=build_visualization(
                prepared,
                prediction.shape_label,
                prediction.shape_confidence,
                prediction.gram_label,
                prediction.gram_confidence,
            ),
        )

    def predict_many(self, images: list[np.ndarray | Path | str]) -> list[InferenceResult]:
        """Prediksi sekumpulan citra secara berurutan.

        Args:
            images: Daftar citra.

        Returns:
            Daftar InferenceResult sepanjang daftar masukan.
        """
        return [self.predict(image) for image in images]


def _notes(prepared: PreprocessResult) -> tuple[str, ...]:
    """Susun catatan yang wajib ditampilkan pengguna.

    Args:
        prepared: Hasil pra-pemrosesan.

    Returns:
        Catatan berurutan dari yang paling penting.
    """
    notes: list[str] = [SEGMENTATION_LIMITATION, LOW_CONFIDENCE_WARNING]
    if prepared.message:
        notes.insert(0, prepared.message)
    if prepared.failed_panels:
        notes.append(
            "Tahap gagal: "
            + ", ".join(prepared.failed_panels)
            + ". Panel terkait dikosongkan, klasifikasi tetap dihitung."
        )
    return tuple(notes)


def check_suffix(path: Path | str) -> None:
    """Pastikan ekstensi citra yang diizinkan.

    Args:
        path: Path berkas citra.

    Raises:
        ValueError: Bila ekstensi tidak ada di ALLOWED_SUFFIXES.
    """
    suffix = Path(path).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        allowed = ", ".join(ALLOWED_SUFFIXES)
        raise ValueError(f"Format {suffix or 'tanpa ekstensi'} tidak didukung. Gunakan: {allowed}.")


def main(argv: list[str] | None = None) -> int:
    """Jalankan inferensi satu citra dari baris perintah.

    Args:
        argv: Daftar argumen. Default-nya sys.argv.

    Returns:
        Kode keluar, nol bila prediksi selesai.
    """
    parser = argparse.ArgumentParser(
        description="Prediksi bentuk sel dan status Gram untuk satu citra."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    check_suffix(args.image)
    location = args.image
    if not location.is_absolute() and not location.is_file():
        location = args.images_dir / args.image

    predictor = Predictor(args.checkpoint)
    result = predictor.predict(location)
    payload = result.prediction.to_dict()

    print(json.dumps(payload, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
