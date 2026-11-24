"""Utilidades compartilhadas pelos adaptadores de radiografia (RX).

Pré-processamento TorchXRayVision
---------------------------------
Os modelos do TorchXRayVision (xrv) foram treinados com imagens em escala
``[-1024, 1024]`` obtidas de ``xrv.utils.normalize(img, maxval)``, isto é,
``(2 * img / maxval - 1) * 1024`` sobre o raster exportado, recorte central
quadrado e redimensionamento ``skimage`` para o lado nativo (224 ou 512).

Aqui a mesma fórmula é aplicada a :class:`~radiologyai.analysis.types.ImageData`
usando a FAIXA REPRESENTÁVEL do formato (``value_range``, derivada de
BitsStored), nunca o máximo dos pixels. ``ImageData`` já está em MONOCHROME2
(o carregador inverte MONOCHROME1 pela faixa representável). A janela VOI do
arquivo NÃO é aplicada aos pixels (fica registrada como limitação).

Pesos
-----
:func:`read_verified` lê os bytes do arquivo UMA vez, confere o sha256 contra o
manifesto e devolve esses mesmos bytes para ``torch.load`` — não há janela entre
a verificação e a desserialização.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

from radiologyai.analysis.types import ImageData, ImageKind
from radiologyai.errors import (
    BackendUnavailableError,
    IntegrityCheckError,
    UnsupportedInputError,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from radiologyai.analysis.base import ResolvedWeights

__all__ = [
    "XRV_SCALE",
    "center_crop_box",
    "crop_and_resize",
    "load_torch_verified",
    "read_verified",
    "require_backend",
    "resolve_torch_device",
    "to_gray",
    "xrv_input",
    "xrv_normalize",
]

XRV_SCALE = 1024.0
# Tolerância relativa para pixels fora da faixa representável (erro de arredondamento
# da Modality LUT). Acima disso o carregador está errado e recusamos a imagem.
_RANGE_TOLERANCE = 1e-3


def require_backend(*modules: str, extra: str = "xray") -> None:
    """Confere dependências opcionais sem importá-las.

    Raises:
        BackendUnavailableError: módulo ausente; a mensagem nomeia o extra.
    """
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        raise BackendUnavailableError(
            f"dependência(s) ausente(s): {', '.join(missing)}; "
            f"instale o extra: pip install 'radiologyai[{extra}]'"
        )


def resolve_torch_device(device: str) -> Any:
    """Converte o texto do dispositivo em ``torch.device``."""
    import torch

    return torch.device(device)


def read_verified(weights: ResolvedWeights, key: str) -> bytes:
    """Bytes de um peso, conferidos contra o sha256 fixado no manifesto.

    Raises:
        IntegrityCheckError: hash divergente (arquivo trocado após o registro).
    """
    path = weights.path(key)
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    expected = weights.sha256[key]
    if digest != expected:
        raise IntegrityCheckError(
            str(path), f"peso {key!r}: sha256 {digest} difere do fixado {expected}"
        )
    return data


def load_torch_verified(weights: ResolvedWeights, key: str, *, allow_pickle: bool) -> Any:
    """``torch.load`` sobre os bytes verificados (nunca sobre o caminho).

    ``allow_pickle`` só pode ser verdadeiro se o manifesto declarar
    ``unsafe_pickle: true`` para o peso (checkpoint que é um objeto Python
    serializado; aceito apenas porque o sha256 foi conferido).

    Raises:
        IntegrityCheckError: hash divergente ou pickle não autorizado pelo manifesto.
    """
    import torch

    if allow_pickle and not weights.unsafe_pickle.get(key, False):
        raise IntegrityCheckError(
            key, "desserialização completa exige unsafe_pickle: true no manifesto"
        )
    data = read_verified(weights, key)
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=not allow_pickle)


def to_gray(image: ImageData) -> npt.NDArray[np.float32]:
    """Plano 2D float32 de uma projeção (média dos canais se colorida, como no xrv)."""
    import numpy as np

    if image.kind is not ImageKind.PROJECTION:
        raise UnsupportedInputError(
            f"adaptador de RX exige projeção 2D; recebido {image.kind.value}"
        )
    pixels = image.pixels
    if image.color:
        pixels = pixels.mean(axis=2)
    return np.asarray(pixels, dtype=np.float32)


def xrv_normalize(image: ImageData) -> npt.NDArray[np.float32]:
    """Projeção → plano em ``[-1024, 1024]`` pela faixa representável do formato.

    ``(2 * (p - lo) / (hi - lo) - 1) * 1024`` com ``(lo, hi) = value_range``.
    Equivale a ``xrv.utils.normalize(img, 2**bits - 1)`` para dados sem sinal.

    Raises:
        UnsupportedInputError: pixels fora da faixa representável (carregador
            inconsistente) ou valores não finitos. Nada é recortado em silêncio.
    """
    import numpy as np

    plane = to_gray(image)
    lo, hi = image.value_range
    if not np.isfinite(plane).all():
        raise UnsupportedInputError("imagem com pixels não finitos")
    span = hi - lo
    tol = _RANGE_TOLERANCE * span
    pmin, pmax = float(plane.min()), float(plane.max())
    if pmin < lo - tol or pmax > hi + tol:
        raise UnsupportedInputError(
            f"pixels [{pmin:g}, {pmax:g}] fora da faixa representável [{lo:g}, {hi:g}]; "
            "o carregador deve declarar value_range coerente com os dados"
        )
    out = (2.0 * ((plane - np.float32(lo)) / np.float32(span)) - 1.0) * np.float32(XRV_SCALE)
    result: npt.NDArray[np.float32] = np.clip(out, -XRV_SCALE, XRV_SCALE).astype(np.float32)
    return result


def center_crop_box(height: int, width: int) -> tuple[int, int, int]:
    """``(y0, x0, lado)`` do recorte central quadrado (mesma aritmética do xrv)."""
    side = min(height, width)
    return height // 2 - side // 2, width // 2 - side // 2, side


def crop_and_resize(plane: npt.NDArray[np.float32], size: int) -> npt.NDArray[np.float32]:
    """Recorte central quadrado + ``skimage.transform.resize`` (= ``XRayResizer``)."""
    import warnings

    import numpy as np
    import skimage.transform

    resize: Any = skimage.transform.resize
    y0, x0, side = center_crop_box(*plane.shape)
    crop = plane[y0 : y0 + side, x0 : x0 + side]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = resize(crop[None], (1, size, size), mode="constant", preserve_range=True)
    return np.asarray(out[0], dtype=np.float32)


def xrv_input(image: ImageData, size: int) -> npt.NDArray[np.float32]:
    """Projeção → plano ``(size, size)`` pronto para os modelos xrv."""
    require_backend("skimage", extra="xray")
    return crop_and_resize(xrv_normalize(image), size)
