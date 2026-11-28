"""Arquiteturas e pré-processamento de ``ianpan/mammoscreen`` e ``ianpan/mammo-crop``.

Reimplementação revisada do código remoto dos cartões (Apache-2.0; ver
``NOTICE``). Importa torch/timm/cv2 no topo: só é importado por
``load``/``predict`` dos adaptadores, nunca por ``import radiologyai``.

Fidelidade ao original
----------------------
* ``MammoCropNet``: ``mobilenetv3_small_100`` (1 canal) → GeM(p=3) → linear(4)
  → sigmoide; coordenadas ``xywh`` normalizadas reescaladas pela forma da
  imagem e truncadas para inteiro (``Tensor.int()``), como em ``forward``.
* ``MammoNet``: ``tf_efficientnetv2_s`` (1 canal) → média global → linear(5).
  Saída 0 = câncer (sigmoide); 1-4 = densidade (softmax), média sobre as
  incidências da MESMA mama; o ensemble faz a média das três redes.
* Redimensionamento: ``A.Resize`` = ``cv2.resize(..., INTER_LINEAR)``;
  ``A.LongestMaxSize`` arredonda as dimensões com ``round`` e usa a mesma
  interpolação; ``A.PadIfNeeded`` centraliza com constante 0 (padrão do
  albumentations 2.0.8, a versão que o código remoto executa hoje).
* Normalização: ``[0, 255] → [-1, 1]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F  # noqa: N812 - convenção do PyTorch
from timm import create_model
from torch import nn

__all__ = [
    "CROP_INPUT_SIZE",
    "ENSEMBLE_IMAGE_SIZES",
    "ENSEMBLE_PAD_TO_ASPECT",
    "GeM",
    "MammoCropNet",
    "MammoEnsemble",
    "MammoNet",
    "longest_max_size_pad",
    "normalize_u8",
    "pad_to_aspect_ratio",
    "resize_u8",
]

U8Array = npt.NDArray[np.uint8]

CROP_INPUT_SIZE = 256
#: (altura, largura) de cada rede, de ``config.json`` do mammoscreen
ENSEMBLE_IMAGE_SIZES: tuple[tuple[int, int], ...] = ((2048, 1024), (1920, 1280), (1536, 1536))
ENSEMBLE_PAD_TO_ASPECT: tuple[bool, ...] = (True, True, False)


def resize_u8(img: U8Array, height: int, width: int) -> U8Array:
    """``cv2.resize`` bilinear (o que ``A.Resize`` executa em uint8)."""
    out = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.asarray(out, dtype=np.uint8)


def pad_to_aspect_ratio(img: U8Array, aspect_ratio: float) -> U8Array:
    """Preenche a largura com 0 até ``altura/largura == aspect_ratio`` (se for maior).

    Lógica idêntica a ``modeling._pad_to_aspect_ratio`` do original.
    """
    h, w = img.shape[:2]
    if h / w > aspect_ratio:
        new_w = round(h / aspect_ratio)
        w_diff = new_w - w
        left = w_diff // 2
        padded: U8Array = np.pad(
            img, ((0, 0), (left, w_diff - left)), mode="constant", constant_values=0
        )
        return padded
    return img


def longest_max_size_pad(img: U8Array, height: int, width: int) -> U8Array:
    """``A.LongestMaxSize(height)`` seguido de ``A.PadIfNeeded(height, width)`` (constante 0)."""
    h, w = img.shape[:2]
    scale = height / float(max(h, w))
    if scale != 1.0:
        img = resize_u8(img, round(h * scale), round(w * scale))
    h, w = img.shape[:2]
    ph, pw = max(0, height - h), max(0, width - w)
    pads = ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2))
    out: U8Array = np.pad(img, pads, mode="constant", constant_values=0)
    return out


def normalize_u8(x: torch.Tensor) -> torch.Tensor:
    """``[0, 255] → [-1, 1]`` como ``normalize`` do original."""
    return (x / 255.0 - 0.5) * 2.0


class GeM(nn.Module):
    """Generalized mean pooling 2D (p aprendido), como no original."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), output_size=1)
        return pooled.pow(1.0 / self.p).flatten(1)


class MammoCropNet(nn.Module):
    """``ianpan/mammo-crop``: regressão de caixa ``xywh`` da mama."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone: Any = create_model(
            "mobilenetv3_small_100",
            pretrained=False,
            num_classes=0,
            global_pool="",
            features_only=False,
            in_chans=1,
        )
        self.pooling = GeM(p=3.0)
        self.linear = nn.Linear(1024, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Coordenadas normalizadas ``[N, 4]`` em ``[0, 1]`` (entrada em ``[0, 255]``)."""
        features = self.pooling(self.backbone(normalize_u8(x)))
        coords: torch.Tensor = self.linear(features).sigmoid()
        return coords

    @torch.inference_mode()
    def crop_box(self, img: U8Array) -> tuple[int, int, int, int]:
        """Caixa ``(x, y, w, h)`` em pixels da imagem original (truncada como ``.int()``)."""
        height, width = img.shape[:2]
        resized = resize_u8(img, CROP_INPUT_SIZE, CROP_INPUT_SIZE)
        device = next(self.parameters()).device
        x = torch.from_numpy(resized).to(device).float()[None, None]
        coords = self.forward(x)
        shape = torch.tensor([[height, width]], device=device)
        rescaled = coords.clone()
        rescaled[:, 0] = rescaled[:, 0] * shape[:, 1]
        rescaled[:, 1] = rescaled[:, 1] * shape[:, 0]
        rescaled[:, 2] = rescaled[:, 2] * shape[:, 1]
        rescaled[:, 3] = rescaled[:, 3] * shape[:, 0]
        x0, y0, w, h = (int(v) for v in rescaled.int()[0].cpu().tolist())
        return x0, y0, w, h


class MammoNet(nn.Module):
    """Uma rede do ensemble (EfficientNetV2-S, 1 canal, 5 saídas)."""

    def __init__(self, image_size: tuple[int, int], pad_to_aspect: bool) -> None:
        super().__init__()
        self.backbone: Any = create_model(
            "tf_efficientnetv2_s",
            pretrained=False,
            num_classes=0,
            global_pool="",
            features_only=False,
            in_chans=1,
        )
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.1)  # sem efeito em eval; mantido pela paridade de módulos
        self.linear = nn.Linear(1280, 5)
        self.image_size = image_size
        self.pad_to_aspect = pad_to_aspect

    def preprocess(self, img: U8Array) -> U8Array:
        """Pré-processamento por rede (pad de aspecto + resize, ou longest-max + pad)."""
        height, width = self.image_size
        if self.pad_to_aspect:
            return resize_u8(pad_to_aspect_ratio(img, height / width), height, width)
        return longest_max_size_pad(img, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits ``[N, 5]`` para lote ``[N, 1, H, W]`` em ``[0, 255]``."""
        features = self.pooling(self.backbone(normalize_u8(x)))
        logits: torch.Tensor = self.linear(features.flatten(1))
        return logits

    @torch.inference_mode()
    def breast(self, views: Sequence[U8Array]) -> tuple[torch.Tensor, torch.Tensor]:
        """(câncer, densidade[4]) de UMA mama: média sobre as incidências."""
        device = next(self.parameters()).device
        batch = torch.stack(
            [torch.from_numpy(self.preprocess(v)).to(device).float()[None] for v in views]
        )
        logits = self.forward(batch)
        cancer = logits[:, 0].sigmoid().mean(0)
        density = logits[:, 1:].softmax(dim=1).mean(0)
        return cancer, density


class MammoEnsemble(nn.Module):
    """Ensemble ``net0..net2``; média das redes (como ``MammoEnsemble.forward``)."""

    def __init__(self) -> None:
        super().__init__()
        self.net0 = MammoNet(ENSEMBLE_IMAGE_SIZES[0], ENSEMBLE_PAD_TO_ASPECT[0])
        self.net1 = MammoNet(ENSEMBLE_IMAGE_SIZES[1], ENSEMBLE_PAD_TO_ASPECT[1])
        self.net2 = MammoNet(ENSEMBLE_IMAGE_SIZES[2], ENSEMBLE_PAD_TO_ASPECT[2])

    @property
    def nets(self) -> tuple[MammoNet, MammoNet, MammoNet]:
        return self.net0, self.net1, self.net2

    @torch.inference_mode()
    def breast(self, views: Sequence[U8Array]) -> tuple[float, list[float], list[float]]:
        """(câncer, densidade[4], câncer por rede) para as incidências recortadas de uma mama."""
        if not views:
            raise ValueError("ao menos uma incidência é necessária")
        outs = [net.breast(views) for net in self.nets]
        cancer = float(torch.stack([o[0] for o in outs]).mean())
        density = [float(v) for v in torch.stack([o[1] for o in outs]).mean(0).tolist()]
        per_net = [float(o[0]) for o in outs]
        return cancer, density, per_net
