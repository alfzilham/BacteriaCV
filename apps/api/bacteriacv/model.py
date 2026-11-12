"""Frozen ResNet-50 backbone with two classification heads.

Head A uses a two class softmax over cell shape, Head B uses a two class sigmoid
over Gram status. The backbone is frozen so only the heads are trained.

The checkpoint file stores only the heads. The backbone is reloaded from torchvision
each time the model is built, because its weights are large and never change.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from .config import (
    BACKBONE_NAME,
    BACKBONE_WEIGHTS,
    FEATURE_DIM,
    GRAM_LABELS,
    N_GRAM_CLASSES,
    N_SHAPE_CLASSES,
    SHAPE_LABELS,
)

HEAD_PREFIX = "head_"


class BacteriaNet(nn.Module):
    """Frozen ResNet-50 with two classification heads.

    Attributes:
        backbone: ResNet-50 with the top classifier removed, entirely frozen.
        head_a: Cell shape classification layer.
        head_b: Gram status classification layer.
    """

    def __init__(self, pretrained: bool = True) -> None:
        """Build the model.

        Args:
            pretrained: When True, load the ImageNet weights into the backbone.

        Raises:
            ValueError: When the backbone name in config is not supported.
        """
        super().__init__()

        if BACKBONE_NAME != "resnet50":
            raise ValueError(f"Backbone {BACKBONE_NAME} is not supported.")

        weights = ResNet50_Weights[BACKBONE_WEIGHTS] if pretrained else None
        network = resnet50(weights=weights)
        network.fc = nn.Identity()
        self.backbone = network

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self.head_a = nn.Linear(FEATURE_DIM, N_SHAPE_CLASSES)
        self.head_b = nn.Linear(FEATURE_DIM, N_GRAM_CLASSES)
        self.reset_heads()

    def reset_heads(self) -> None:
        """Reinitialise the head weights with the PyTorch defaults."""
        for head in (self.head_a, self.head_b):
            nn.init.kaiming_normal_(head.weight)
            nn.init.zeros_(head.bias)

    def train(self, mode: bool = True) -> "BacteriaNet":
        """Set train mode, but the backbone stays in eval because it is frozen.

        Args:
            mode: The mode requested for the heads.

        Returns:
            The module itself, so the caller can chain.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        """Extract the feature vectors from a batch of images.

        Args:
            batch: B x 3 x 224 x 224 tensor.

        Returns:
            B x 2048 tensor without gradients.
        """
        self.backbone.eval()
        with torch.inference_mode():
            return self.backbone(batch)

    def head_state_dict(self) -> dict[str, torch.Tensor]:
        """Extract only the head weights, to be saved as the checkpoint.

        Returns:
            A dictionary holding only head prefixed keys.
        """
        return {
            key: value.detach().clone()
            for key, value in self.state_dict().items()
            if key.startswith(HEAD_PREFIX)
        }


def build_model(pretrained: bool = True) -> BacteriaNet:
    """Build BacteriaNet.

    Args:
        pretrained: When True, load the ImageNet weights into the backbone.

    Returns:
        A ready to use BacteriaNet instance.
    """
    return BacteriaNet(pretrained=pretrained)


def load_state_dicts(model: BacteriaNet, state: dict[str, torch.Tensor]) -> None:
    """Load the head weights into the model.

    Backbone weights are deliberately ignored because the checkpoint does not
    store them. A mismatch on a head key is still an error, while a missing
    backbone key is treated as expected.

    Args:
        model: The target model.
        state: Dictionary of head weights.

    Raises:
        RuntimeError: When a head key is missing or does not match.
    """
    unexpected = [key for key in state if not key.startswith(HEAD_PREFIX)]
    if unexpected:
        raise RuntimeError(
            f"Checkpoint carries keys that are not head keys: {sorted(unexpected)}"
        )

    current = model.state_dict()
    missing = [
        key
        for key in current
        if key.startswith(HEAD_PREFIX) and key not in state
    ]
    if missing:
        raise RuntimeError(f"Checkpoint does not carry the head keys: {sorted(missing)}")

    model.load_state_dict(state, strict=False)


def save_checkpoint(model: BacteriaNet, path: Path | str) -> Path:
    """Save the head weights to the given path.

    Args:
        model: The model whose heads are saved.
        path: Target file location.

    Returns:
        The path of the file that was written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.head_state_dict(), target)
    return target


def load_checkpoint(model: BacteriaNet, path: Path | str) -> BacteriaNet:
    """Load the head weights from a checkpoint.

    Args:
        model: The target model.
        path: Checkpoint location.

    Returns:
        The loaded model.

    Raises:
        FileNotFoundError: When the checkpoint does not exist.
    """
    location = Path(path)
    if not location.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {location.name}")
    load_state_dicts(model, torch.load(location, map_location="cpu", weights_only=True))
    return model


def predict_batch(
    model: BacteriaNet, features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict shape and Gram status from a batch of features.

    Args:
        model: The BacteriaNet model.
        features: B x 2048 tensor.

    Returns:
        A tuple of four tensors (shape_label, shape_confidence, gram_label,
        gram_confidence). Gram confidence is the probability of the chosen
        class, not an absolute threshold value.
    """
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            shape_probabilities = model.head_a(features).softmax(dim=-1)
            gram_probabilities = model.head_b(features).sigmoid()

            shape_index = shape_probabilities.argmax(dim=-1)
            # The 0.5 threshold is applied only to the positive class probability.
            # Comparing the full matrix against the threshold would yield a
            # B x 2 tensor instead of B x 1.
            gram_index = (gram_probabilities[:, 1] > 0.5).to(torch.int64)
            shape_confidence = shape_probabilities.max(dim=-1).values
            gram_confidence = gram_probabilities.gather(
                1, gram_index.unsqueeze(1)
            ).reshape(-1)
    finally:
        if was_training:
            model.train()

    return shape_index, shape_confidence, gram_index, gram_confidence


def shape_label(index: int) -> str:
    """Turn a shape class index into a label.

    Args:
        index: The class index.

    Returns:
        The label name, or "tidak_diketahui" when out of range.
    """
    if 0 <= index < len(SHAPE_LABELS):
        return SHAPE_LABELS[index]
    return "tidak_diketahui"


def gram_label(index: int) -> str:
    """Turn a Gram status class index into a label.

    Args:
        index: The class index.

    Returns:
        The label name, or "tidak_diketahui" when out of range.
    """
    if 0 <= index < len(GRAM_LABELS):
        return GRAM_LABELS[index]
    return "tidak_diketahui"