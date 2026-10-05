"""Builder of the visualisation panels for the web interface.

DESIGN section 3 establishes one panel with a toggle, not five panels
at once. This module therefore prepares the five panels as separate files
then burns the annotations into each one, so the toggle does not have to
rewrite the text every time the panel changes.

A rule that must not be broken: the panel must state that shape classification
is not yet validated from the segmented objects. The figures in that note come
from an experiment on DIBaS data and must not change without a new experiment.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np

from .config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    LOW_CONFIDENCE_WARNING,
    SHAPE_LABELS,
)
from .preprocess import STAGE_NAMES, PreprocessResult

STAGE_TITLES = {
    "original": "Original image",
    "resized": "Resize 224 x 224",
    "normalized": "ImageNet normalization",
    "segment": "Segmentation mask",
    "watershed": "Object boundary",
}

# One i18n key per stage. The client translates these; the server only knows
# the English titles above.
STAGE_KEYS = {
    "original": "stage_original",
    "resized": "stage_resized",
    "normalized": "stage_normalized",
    "segment": "stage_segment",
    "watershed": "stage_watershed",
}

# Suffix appended to a stage title whose panel could not be produced.
STAGE_FAILED_SUFFIX = " (failed)"
STAGE_FAILED_SUFFIX_KEY = "stage_failed_suffix"

# Evidence from the segmentation experiment on DIBaS. These figures are measured,
# not estimated. Elongation uses the major/minor axis metric.
SEGMENTATION_LIMITATION = (
    "Cell shape is not yet validated from segmentation: median elongation "
    "1.30-1.43 for cocci and 1.64-1.76 for bacilli, but the intraspecies "
    "range 1.03-4.19 exceeds the between-group difference of 0.068."
)

# Key for SEGMENTATION_LIMITATION, used by the client dictionary.
SEGMENTATION_LIMITATION_KEY = "shape_not_validated"

FONT_SCALE = 0.42
FONT_THICKNESS = 1
TEXT_COLOR = (255, 255, 255)
STRIP_COLOR = (0, 0, 0)
LINE_HEIGHT = 18


@dataclass(frozen=True)
class VisualizationBundle:
    """The five panels, already annotated.

    Attributes:
        panels: Five annotated RGB images, ordered per stage_names.
        stage_names: The stage keys for the toggle.
        titles: The English stage titles matching stage_names.
        title_keys: The i18n keys matching stage_names, index aligned with
            titles.
        failed_stages: The stages that failed, highlighted in the interface.
        object_count: The segmentation object count, zero when it failed.
        notes: The notes that must appear below the panel.
    """

    panels: tuple[np.ndarray, ...]
    stage_names: tuple[str, ...]
    titles: tuple[str, ...]
    title_keys: tuple[str, ...]
    failed_stages: tuple[str, ...]
    object_count: int
    notes: tuple[str, ...]


def confidence_level(value: float) -> str:
    """Turn a confidence into a verbal level.

    Args:
        value: A confidence between zero and one.

    Returns:
        "high", "medium", or "low".
    """
    if value >= CONFIDENCE_HIGH:
        return "high"
    if value >= CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


def annotate(
    panel: np.ndarray, lines: list[str], strip_alpha: float = 0.65
) -> np.ndarray:
    """Burn text onto a panel with a dark backdrop.

    Args:
        panel: A uint8 RGB array.
        lines: The text lines drawn at the top left.
        strip_alpha: The backdrop opacity, zero meaning transparent.

    Returns:
        A copy of the panel with the text burned in.

    Raises:
        ValueError: When lines is empty.
    """
    if not lines:
        raise ValueError("Setidaknya satu baris annotate wajib ada.")

    result = np.ascontiguousarray(panel.copy())
    height = LINE_HEIGHT * len(lines)
    strip = result[:height].astype(np.float32)
    result[:height] = (
        strip * (1.0 - strip_alpha) + np.array(STRIP_COLOR, dtype=np.float32) * strip_alpha
    ).astype(np.uint8)

    for index, line in enumerate(lines):
        cv2.putText(
            result,
            line,
            (6, LINE_HEIGHT * index + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )
    return result


def encode_png(panel: np.ndarray) -> bytes:
    """Convert an RGB panel into an in memory PNG.

    Args:
        panel: A uint8 RGB array.

    Returns:
        The PNG file content.
    """
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Panel tidak dapat dikodekan menjadi PNG.")
    return buffer.tobytes()


def encode_png_base64(panel: np.ndarray) -> str:
    """Convert an RGB panel into base64 PNG for embedding in HTML.

    Args:
        panel: A uint8 RGB array.

    Returns:
        Base64 PNG text without the data URI prefix.
    """
    return base64.b64encode(encode_png(panel)).decode("ascii")


def _headline(
    shape_label: str,
    shape_confidence: float,
    gram_label: str,
    gram_confidence: float,
) -> str:
    """Assemble the prediction summary lines for annotation."""
    return (
        f"Shape: {shape_label} ({shape_confidence:.0%}, "
        f"{confidence_level(shape_confidence)})   "
        f"Gram: {gram_label} ({gram_confidence:.0%}, "
        f"{confidence_level(gram_confidence)})"
    )


def build_visualization(
    result: PreprocessResult,
    shape_label: str,
    shape_confidence: float,
    gram_label: str,
    gram_confidence: float,
) -> VisualizationBundle:
    """Build the five annotated panels from the preprocessing result.

    Args:
        result: The preprocess result.
        shape_label: The predicted shape class name.
        shape_confidence: The shape prediction confidence.
        gram_label: The predicted Gram status class name.
        gram_confidence: The Gram prediction confidence.

    Returns:
        A VisualizationBundle ready to send to the interface.

    Raises:
        ValueError: When a confidence falls outside the range zero to one.
    """
    for name, value in (
        ("shape_confidence", shape_confidence),
        ("gram_confidence", gram_confidence),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")

    notes: list[str] = [LOW_CONFIDENCE_WARNING, SEGMENTATION_LIMITATION]
    if result.failed_panels:
        notes.insert(
            0,
            "Failed stages: " + ", ".join(result.failed_panels) + ". "
            "The related panels are empty, the classification is still "
            "computed.",
        )

    base = [_headline(shape_label, shape_confidence, gram_label, gram_confidence)]
    base.extend(notes)

    titles: list[str] = []
    title_keys: list[str] = []
    panels: list[np.ndarray] = []
    for stage, panel in zip(STAGE_NAMES, result.panels):
        title = STAGE_TITLES[stage]
        key = STAGE_KEYS[stage]
        if stage in result.failed_panels:
            title = f"{title}{STAGE_FAILED_SUFFIX}"
            key = f"{key}.{STAGE_FAILED_SUFFIX_KEY}"
        titles.append(title)
        title_keys.append(key)
        panels.append(annotate(panel, base[:2] + [title] + base[2:4]))

    return VisualizationBundle(
        panels=tuple(panels),
        stage_names=STAGE_NAMES,
        titles=tuple(titles),
        title_keys=tuple(title_keys),
        failed_stages=result.failed_panels,
        object_count=result.object_count,
        notes=tuple(notes),
    )


def panel_sizes(bundle: VisualizationBundle) -> list[tuple[int, int]]:
    """Return the size of each panel for interface checks.

    Args:
        bundle: The visualisation bundle.

    Returns:
        A list of (height, width) pairs, one per panel.
    """
    return [panel.shape[:2] for panel in bundle.panels]


def shape_label_text(index: int) -> str:
    """Turn a shape class index into a label, with a safe fallback."""
    if 0 <= index < len(SHAPE_LABELS):
        return SHAPE_LABELS[index]
    return "tidak_diketahui"
