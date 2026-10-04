"""Backbone ResNet-50 beku dengan dua classification head.

Head A memakai softmax dua kelas bentuk sel, Head B memakai sigmoid dua kelas
status Gram. Backbone dibekukan sehingga hanya head yang dilatih.

Berkas checkpoint hanya menyimpan head. Backbone diambil ulang dari torchvision
setiap kali model dibangun, karena bobotnya besar dan tidak pernah berubah.
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
    """ResNet-50 beku dengan dua head klasifikasi.

    Attributes:
        backbone: ResNet-50 dengan classifier atas dilepas, seluruhnya beku.
        head_a: Lapisan klasifikasi bentuk sel.
        head_b: Lapisan klasifikasi status Gram.
    """

    def __init__(self, pretrained: bool = True) -> None:
        """Bangun model.

        Args:
            pretrained: Bila True, muat bobot ImageNet pada backbone.

        Raises:
            ValueError: Bila nama backbone pada config tidak didukung.
        """
        super().__init__()

        if BACKBONE_NAME != "resnet50":
            raise ValueError(f"Backbone {BACKBONE_NAME} belum didukung.")

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
        """Inisialisasi ulang bobot head dengan standar PyTorch."""
        for head in (self.head_a, self.head_b):
            nn.init.kaiming_normal_(head.weight)
            nn.init.zeros_(head.bias)

    def train(self, mode: bool = True) -> "BacteriaNet":
        """Set mode(latihan), tetapi backbone tetap eval karena dibekukan.

        Args:
            mode: Mode yang diminta untuk head.

        Returns:
            Module itu sendiri, agar pemanggil bisa merangkai.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        """Ambil vektor fitur dari batch citra.

        Args:
            batch: Tensor B x 3 x 224 x 224.

        Returns:
            Tensor B x 2048 tanpa gradien.
        """
        self.backbone.eval()
        with torch.inference_mode():
            return self.backbone(batch)

    def head_state_dict(self) -> dict[str, torch.Tensor]:
        """Ambil bobot head saja untuk disimpan sebagai checkpoint.

        Returns:
            Dictionary yang hanya memuat kunci berawalan head.
        """
        return {
            key: value.detach().clone()
            for key, value in self.state_dict().items()
            if key.startswith(HEAD_PREFIX)
        }


def build_model(pretrained: bool = True) -> BacteriaNet:
    """Bangun BacteriaNet.

    Args:
        pretrained: Bila True, muat bobot ImageNet pada backbone.

    Returns:
        Instans BacteriaNet siap pakai.
    """
    return BacteriaNet(pretrained=pretrained)


def load_state_dicts(model: BacteriaNet, state: dict[str, torch.Tensor]) -> None:
    """Muat bobot head ke model.

    Bobot backbone sengaja diabaikan karena checkpoint tidak menyimpannya.
    Ketidakcocokan pada kunci head tetap menjadi error, sedangkan kunci
    backbone yang absen dianggap wajar.

    Args:
        model: Model tujuan.
        state: Dictionary bobot head.

    Raises:
        RuntimeError: Bila ada kunci head yang hilang atau tidak cocok.
    """
    unexpected = [key for key in state if not key.startswith(HEAD_PREFIX)]
    if unexpected:
        raise RuntimeError(f"Checkpoint memuat kunci bukan head: {sorted(unexpected)}")

    current = model.state_dict()
    missing = [
        key
        for key in current
        if key.startswith(HEAD_PREFIX) and key not in state
    ]
    if missing:
        raise RuntimeError(f"Checkpoint tidak memuat kunci head: {sorted(missing)}")

    model.load_state_dict(state, strict=False)


def save_checkpoint(model: BacteriaNet, path: Path | str) -> Path:
    """Simpan bobot head ke path yang diberikan.

    Args:
        model: Model yang head-nya disimpan.
        path: Lokasi berkas tujuan.

    Returns:
        Path berkas yang ditulis.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.head_state_dict(), target)
    return target


def load_checkpoint(model: BacteriaNet, path: Path | str) -> BacteriaNet:
    """Muat bobot head dari checkpoint.

    Args:
        model: Model tujuan.
        path: Lokasi checkpoint.

    Returns:
        Model yang sudah dimuat.

    Raises:
        FileNotFoundError: Bila checkpoint tidak ada.
    """
    location = Path(path)
    if not location.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {location.name}")
    load_state_dicts(model, torch.load(location, map_location="cpu", weights_only=True))
    return model


def predict_batch(
    model: BacteriaNet, features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prediksi bentuk dan status Gram dari sekumpulan fitur.

    Args:
        model: Model BacteriaNet.
        features: Tensor B x 2048.

    Returns:
        Pasangan empat tensor (label_bentuk, confidence_bentuk, label_gram,
        confidence_gram). Confidence Gram adalah probabilitas kelas yang
        dipilih, bukan nilai ambang mutlak.
    """
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            shape_probabilities = model.head_a(features).softmax(dim=-1)
            gram_probabilities = model.head_b(features).sigmoid()

            shape_index = shape_probabilities.argmax(dim=-1)
            # Ambang 0,5 diterapkan pada probabilitas kelas positif saja.
            # Membandingkan matriks penuh dengan ambang akan menghasilkan
            # tensor B x 2, bukan B x 1.
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
    """Ubah indeks kelas bentuk menjadi label.

    Args:
        index: Indeks kelas.

    Returns:
        Nama label, atau "tidak_diketahui" bila di luar rentang.
    """
    if 0 <= index < len(SHAPE_LABELS):
        return SHAPE_LABELS[index]
    return "tidak_diketahui"


def gram_label(index: int) -> str:
    """Ubah indeks kelas status Gram menjadi label.

    Args:
        index: Indeks kelas.

    Returns:
        Nama label, atau "tidak_diketahui" bila di luar rentang.
    """
    if 0 <= index < len(GRAM_LABELS):
        return GRAM_LABELS[index]
    return "tidak_diketahui"