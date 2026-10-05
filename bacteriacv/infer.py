"""Inference for a single image and assembly of the full result.

The inference path never touches augmentation. preprocess has no
augmentation parameter and never calls augment_train_variants, so there is
no way for it to leak in here.

Segmentation does not affect the model metrics. ARCHITECTURE section 3 states
that segmentation is not needed for inference because shape and Gram status
classification comes from the image, not the mask. The segmentation result is
still computed because the visualisation panel uses it, and a segmentation
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
    """Prediction result for one image.

    Attributes:
        shape_label: The shape class name.
        shape_confidence: The shape confidence.
        gram_label: The Gram status class name.
        gram_confidence: The Gram confidence.
        shape_index: The shape class index.
        gram_index: The Gram status class index.
        shape_level: The verbal confidence level for shape.
        gram_level: The verbal confidence level for Gram.
        object_count: The segmentation object count, zero when it failed.
        segmentation_ok: Whether segmentation produced objects.
        segmentation_validated: Always False, see SEGMENTATION_LIMITATION.
        stages_ok: Success status of each preprocessing stage.
        notes: Notes that must be shown to the user.
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
        """Turn the prediction into a dictionary for JSON serialisation."""
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True)
class InferenceResult:
    """A prediction together with its visualisation panels for one image.

    Attributes:
        prediction: The prediction result of both heads.
        visualization: Five annotated panels.
    """

    prediction: Prediction
    visualization: VisualizationBundle


class Predictor:
    """Model wrapper for single image prediction.

    The model is built once and reused. The backbone is frozen, so
    there is no reason to rebuild it between requests.
    """

    def __init__(self, checkpoint_path: Path | str = DEFAULT_CHECKPOINT) -> None:
        """Build the model and load the head checkpoint.

        Args:
            checkpoint_path: The head checkpoint location.

        Raises:
            FileNotFoundError: When the checkpoint does not exist.
        """
        self.model = load_checkpoint(build_model(pretrained=True), checkpoint_path)
        self.model.eval()
        self.checkpoint_path = Path(checkpoint_path)

    def predict(self, image: np.ndarray | Path | str) -> InferenceResult:
        """Preprocess one image then predict both heads.

        Args:
            image: An RGB array, or an image file path.

        Returns:
            An InferenceResult holding the prediction and visualisation panels.

        Raises:
            FileNotFoundError: When the image path does not exist.
            ValueError: When the image cannot be read.
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
        """Predict a batch of images in order.

        Args:
            images: The list of images.

        Returns:
            A list of InferenceResult, one per input image.
        """
        return [self.predict(image) for image in images]


def _notes(prepared: PreprocessResult) -> tuple[str, ...]:
    """Assemble the notes that must be shown to the user.

    Args:
        prepared: The preprocessing result.

    Returns:
        Notes ordered from most to least important.
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
    """Make sure the image extension is allowed.

    Args:
        path: The image file path.

    Raises:
        ValueError: When the extension is not in ALLOWED_SUFFIXES.
    """
    suffix = Path(path).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        allowed = ", ".join(ALLOWED_SUFFIXES)
        raise ValueError(f"Format {suffix or 'tanpa ekstensi'} tidak didukung. Gunakan: {allowed}.")


def main(argv: list[str] | None = None) -> int:
    """Run single image inference from the command line.

    Args:
        argv: The argument list. Defaults to sys.argv.

    Returns:
        The exit code, zero when the prediction finished.
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
