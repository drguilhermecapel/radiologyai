"""Utilitários compartilhados pelos adaptadores de ultrassom e ecocardiografia.

* conversão de um quadro (US/eco) para 8 bits pela faixa representável do
  formato — nunca pelo mínimo/máximo do conteúdo;
* leitura de caixas (``Primitive`` bbox) desenhadas pelo usuário;
* limpeza de máscara (maior componente + preenchimento de buracos);
* morfometria de lesão: área e diâmetros (maior diâmetro de Feret e o maior
  diâmetro perpendicular a ele), em mm SOMENTE quando há espaçamento de pixel.

Sem torch: só numpy (e scipy.ndimage para componentes conexos).
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from radiologyai.analysis.types import ImageData, ImageKind, Primitive
from radiologyai.errors import BackendUnavailableError, UnsupportedInputError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = [
    "LesionMorphometry",
    "bbox_array",
    "box_prompts",
    "clean_mask",
    "frame_uint8",
    "lesion_morphometry",
    "mask_bbox",
    "pad_box",
    "sector_bbox",
]


def _require_scipy() -> None:
    if importlib.util.find_spec("scipy") is None:
        raise BackendUnavailableError(
            "scipy não instalado (componentes conexos); instale o extra: "
            "pip install 'radiologyai[eval]'"
        )


def frame_uint8(image: ImageData, frame: int = 0) -> npt.NDArray[np.uint8]:
    """Um quadro em tons de cinza 8 bits ``(H, W)``, escalado pela faixa representável.

    Cor (US Doppler/RGB) vira luminância ITU-R 601.

    Raises:
        UnsupportedInputError: quadro fora do intervalo ou imagem volumétrica.
    """
    import numpy as np

    if image.kind is ImageKind.VOLUME:
        raise UnsupportedInputError("US/eco: volume 3D não suportado")
    px = image.pixels
    if image.kind is ImageKind.CINE:
        n = int(px.shape[0])
        if not 0 <= frame < n:
            raise UnsupportedInputError(f"quadro {frame} fora do clipe de {n} quadros")
        px = px[frame]
    elif frame != 0:
        raise UnsupportedInputError("imagem estática só tem o quadro 0")
    arr = np.asarray(px, dtype=np.float32)
    if image.color:
        arr = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
    lo, hi = image.value_range
    scaled = (arr - np.float32(lo)) * np.float32(255.0 / (hi - lo))
    out: npt.NDArray[np.uint8] = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
    return out


def box_prompts(
    prompts: tuple[Primitive, ...], *, label_prefix: str | None = None
) -> list[Primitive]:
    """Caixas (bbox) entre os prompts, opcionalmente filtradas por prefixo de rótulo."""
    out: list[Primitive] = []
    for p in prompts:
        if p.type != "bbox":
            continue
        if label_prefix is not None and not (p.label or "").lower().startswith(label_prefix):
            continue
        out.append(p)
    return out


def bbox_array(prompt: Primitive, shape: tuple[int, int]) -> npt.NDArray[np.float64]:
    """Caixa ``[x0, y0, x1, y1]`` ordenada e recortada ao plano ``(H, W)``.

    Raises:
        UnsupportedInputError: caixa degenerada ou fora da imagem.
    """
    import numpy as np

    (xa, ya), (xb, yb) = prompt.xy
    h, w = shape
    x0, x1 = sorted((xa, xb))
    y0, y1 = sorted((ya, yb))
    x0, x1 = max(0.0, x0), min(float(w - 1), x1)
    y0, y1 = max(0.0, y0), min(float(h - 1), y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise UnsupportedInputError(f"caixa degenerada ou fora da imagem: {prompt.xy}")
    return np.array([x0, y0, x1, y1], dtype=np.float64)


def mask_bbox(mask: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Caixa ``[x0, y0, x1, y1]`` (inclusiva) de uma máscara não vazia.

    Raises:
        UnsupportedInputError: máscara vazia.
    """
    import numpy as np

    ys, xs = np.nonzero(np.asarray(mask))
    if ys.size == 0:
        raise UnsupportedInputError("máscara vazia: sem caixa")
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)


def pad_box(
    box: npt.NDArray[np.float64], shape: tuple[int, int], margin: float
) -> npt.NDArray[np.float64]:
    """Expande a caixa em ``margin`` pixels, recortada à imagem."""
    import numpy as np

    h, w = shape
    return np.array(
        [
            max(box[0] - margin, 0.0),
            max(box[1] - margin, 0.0),
            min(box[2] + margin, float(w - 1)),
            min(box[3] + margin, float(h - 1)),
        ],
        dtype=np.float64,
    )


def sector_bbox(frame_u8: npt.NDArray[np.uint8], threshold: int = 0) -> npt.NDArray[np.float64]:
    """Caixa do setor de imagem (pixels > ``threshold``).

    Raises:
        UnsupportedInputError: quadro totalmente escuro.
    """
    return mask_bbox(frame_u8 > threshold)


def clean_mask(mask: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
    """Maior componente conexo (8-conexo) com buracos preenchidos; vazio continua vazio."""
    _require_scipy()
    import numpy as np
    from scipy import ndimage

    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return m
    labels, n = ndimage.label(m, structure=np.ones((3, 3), dtype=bool))
    if n > 1:
        sizes = np.bincount(labels.ravel())[1:]
        m = labels == (1 + int(np.argmax(sizes)))
    filled: npt.NDArray[np.bool_] = ndimage.binary_fill_holes(m).astype(bool)
    return filled


@dataclass(frozen=True)
class LesionMorphometry:
    """Área e diâmetros de uma máscara. Valores em mm só com espaçamento."""

    area_px: int
    area_mm2: float | None
    max_diameter_mm: float | None
    perpendicular_diameter_mm: float | None
    max_diameter_line: tuple[tuple[float, float], tuple[float, float]]
    perpendicular_line: tuple[tuple[float, float], tuple[float, float]] | None


def _boundary_points(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
    """Cantos de pixel da borda da máscara, ``(N, 2)`` em ``(linha, coluna)``.

    Usa os 4 cantos de cada pixel de borda: o diâmetro de Feret de um objeto
    pixelizado é medido de borda a borda, não de centro a centro.
    """
    import numpy as np

    pad = np.pad(mask, 1)
    inner = pad[1:-1, 1:-1]
    eroded = inner & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    rows, cols = np.nonzero(inner & ~eroded)
    corners = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    pts = np.stack([rows, cols], axis=1)[:, None, :].astype(np.float64) + corners[None]
    out: npt.NDArray[np.float64] = np.unique(pts.reshape(-1, 2), axis=0)
    return out


def _hull(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Fecho convexo (cadeia monótona de Andrew)."""
    import numpy as np

    pts = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(pts) <= 2:
        return np.asarray(pts, dtype=np.float64)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def lesion_morphometry(
    mask: npt.NDArray[Any], spacing_mm: tuple[float, ...] | None
) -> LesionMorphometry:
    """Área, maior diâmetro (Feret) e maior diâmetro perpendicular de uma máscara 2D.

    Sem ``spacing_mm`` as medidas em mm ficam ``None`` (o adaptador se abstém);
    as linhas de geometria são sempre devolvidas em pixels ``(x, y)``.

    Raises:
        UnsupportedInputError: máscara vazia ou não 2D.
    """
    import numpy as np

    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not m.any():
        raise UnsupportedInputError("morfometria exige máscara 2D não vazia")
    spacing: tuple[float, float] | None = None
    if spacing_mm is not None:
        if len(spacing_mm) != 2 or not all(math.isfinite(s) and s > 0 for s in spacing_mm):
            raise UnsupportedInputError(f"espaçamento inválido: {spacing_mm}")
        spacing = (float(spacing_mm[0]), float(spacing_mm[1]))
    scale = np.asarray(spacing if spacing else (1.0, 1.0), dtype=np.float64)
    hull = _hull(_boundary_points(m)) * scale[None, :]
    diff = hull[:, None, :] - hull[None, :, :]
    d2 = (diff**2).sum(axis=-1)
    a, b = np.unravel_index(int(np.argmax(d2)), d2.shape)
    p, q = hull[a], hull[b]
    major = float(math.sqrt(float(d2[a, b])))
    unit = (q - p) / major if major > 0 else np.array([1.0, 0.0])
    normal = np.array([-unit[1], unit[0]])
    # maior extensão perpendicular, medida sobre a máscara inteira (todas as cordas
    # perpendiculares ao eixo maior), não só a largura no ponto médio
    pts = _boundary_points(m) * scale[None, :]
    along = (pts - p) @ unit
    across = (pts - p) @ normal
    bins = np.round(along / (float(scale.min()) / 2.0)).astype(np.int64)
    best, best_line = 0.0, None
    for key in np.unique(bins):
        sel = bins == key
        lo, hi = float(across[sel].min()), float(across[sel].max())
        if hi - lo > best:
            t = float(along[sel].mean())
            best = hi - lo
            best_line = (p + unit * t + normal * lo, p + unit * t + normal * hi)

    def px(v: npt.NDArray[np.float64]) -> tuple[float, float]:
        return float(v[1] / scale[1]), float(v[0] / scale[0])

    area_px = int(m.sum())
    return LesionMorphometry(
        area_px=area_px,
        area_mm2=area_px * spacing[0] * spacing[1] if spacing else None,
        max_diameter_mm=major if spacing else None,
        perpendicular_diameter_mm=best if spacing else None,
        max_diameter_line=(px(p), px(q)),
        perpendicular_line=None if best_line is None else (px(best_line[0]), px(best_line[1])),
    )
