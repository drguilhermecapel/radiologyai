"""Geometria de máscaras em volumes ``(Z, Y, X)`` com espaçamento e afim LPS.

Convenções (as de :class:`~radiologyai.analysis.types.ImageData`):

* ``spacing_zyx``: espaçamento em mm por eixo do array ``(z, y, x)``;
* ``affine_lps``: 4x4 que leva o índice ``(x=coluna, y=linha, z=corte, 1)`` a mm LPS.
  O eixo 0 do array corresponde à coluna 2 da afim, o eixo 1 à coluna 1 e o eixo 2 à coluna 0.

numpy e scipy são importados sob demanda; scipy ausente levanta
:class:`~radiologyai.errors.BackendUnavailableError` nomeando o extra.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from radiologyai.errors import BackendUnavailableError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    BoolArray = npt.NDArray[np.bool_]
    FloatArray = npt.NDArray[np.float64]

__all__ = [
    "AxisInfo",
    "CrossSection",
    "axis_info",
    "component_volumes_ml",
    "cross_section_diameters",
    "erode_mm",
    "index_to_lps",
    "largest_component_2d",
    "lps_to_index",
    "require_scipy",
    "touches_border",
    "voxel_volume_ml",
]


def require_scipy() -> None:
    """Levanta se scipy (dependência do TotalSegmentator) não estiver instalado."""
    if importlib.util.find_spec("scipy") is None:
        raise BackendUnavailableError(
            "scipy não instalado; instale o extra: pip install 'radiologyai[volumetric]'"
        )


def voxel_volume_ml(spacing_zyx: tuple[float, ...]) -> float:
    """Volume de um voxel em mL (1 mL = 1000 mm³)."""
    if len(spacing_zyx) != 3:
        raise ValueError(f"espaçamento 3D esperado; recebido {spacing_zyx}")
    return float(spacing_zyx[0] * spacing_zyx[1] * spacing_zyx[2]) / 1000.0


def erode_mm(mask: BoolArray, spacing_zyx: tuple[float, ...], mm: float) -> BoolArray:
    """Erosão anisotrópica: mantém voxels a mais de ``mm`` da borda da máscara."""
    require_scipy()
    import numpy as np
    from scipy import ndimage

    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    dist = ndimage.distance_transform_edt(mask, sampling=spacing_zyx)
    out: BoolArray = np.asarray(dist > mm, dtype=bool)
    return out


def touches_border(mask: BoolArray, axes: tuple[int, ...] = (0, 1, 2)) -> bool:
    """A máscara encosta na primeira ou na última fatia de algum dos ``axes``?"""
    for axis in axes:
        n = mask.shape[axis]
        first = mask.take(0, axis=axis)
        last = mask.take(n - 1, axis=axis)
        if bool(first.any()) or bool(last.any()):
            return True
    return False


def component_volumes_ml(mask: BoolArray, spacing_zyx: tuple[float, ...]) -> list[float]:
    """Volumes (mL) dos componentes conexos 3D (conectividade 26), em ordem decrescente."""
    require_scipy()
    import numpy as np
    from scipy import ndimage

    if not mask.any():
        return []
    labels, n = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    counts = np.bincount(labels.ravel())[1:]
    vox = voxel_volume_ml(spacing_zyx)
    return sorted((float(c) * vox for c in counts[:n]), reverse=True)


def largest_component_2d(mask2d: BoolArray) -> BoolArray:
    """Maior componente conexo de uma máscara 2D (conectividade 8)."""
    require_scipy()
    import numpy as np
    from scipy import ndimage

    if not mask2d.any():
        return mask2d.copy()
    labels, _ = ndimage.label(mask2d, structure=np.ones((3, 3), dtype=bool))
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    out: BoolArray = labels == int(np.argmax(counts))
    return out


# --------------------------------------------------------------------------- #
# Afim
# --------------------------------------------------------------------------- #
def _zyx_matrix(affine_lps: FloatArray) -> FloatArray:
    """Matriz 3x4 que leva ``(z, y, x, 1)`` (ordem do array) a mm LPS."""
    import numpy as np

    m: FloatArray = np.zeros((3, 4), dtype=np.float64)
    m[:, 0] = affine_lps[:3, 2]
    m[:, 1] = affine_lps[:3, 1]
    m[:, 2] = affine_lps[:3, 0]
    m[:, 3] = affine_lps[:3, 3]
    return m


def index_to_lps(affine_lps: FloatArray, zyx: FloatArray) -> FloatArray:
    """Índices ``(N, 3)`` na ordem ``(z, y, x)`` → pontos ``(N, 3)`` em mm LPS."""
    import numpy as np

    m = _zyx_matrix(affine_lps)
    pts = np.atleast_2d(np.asarray(zyx, dtype=np.float64))
    out: FloatArray = pts @ m[:, :3].T + m[:, 3]
    return out


def lps_to_index(affine_lps: FloatArray, lps: FloatArray) -> FloatArray:
    """Pontos ``(N, 3)`` em mm LPS → índices contínuos ``(N, 3)`` na ordem ``(z, y, x)``."""
    import numpy as np

    m = _zyx_matrix(affine_lps)
    inv = np.linalg.inv(m[:, :3])
    pts = np.atleast_2d(np.asarray(lps, dtype=np.float64))
    out: FloatArray = (pts - m[:, 3]) @ inv.T
    return out


@dataclass(frozen=True, slots=True)
class AxisInfo:
    """Que eixo do array é craniocaudal, laterolateral e anteroposterior."""

    superior_axis: int  # eixo do array mais alinhado com S (+z LPS)
    superior_sign: int  # +1 se índice crescente vai para cranial
    superior_cos: float  # |cos| entre o eixo e S (1 = corte axial puro)
    lr_axis: int  # eixo do array mais alinhado com L/R
    ap_axis: int  # eixo do array mais alinhado com A/P
    anterior_sign: int  # +1 se índice crescente vai para anterior


def axis_info(affine_lps: FloatArray) -> AxisInfo:
    """Orientação dos eixos do array a partir da afim LPS."""
    import numpy as np

    m = _zyx_matrix(affine_lps)[:, :3]
    dirs = m / np.linalg.norm(m, axis=0, keepdims=True)  # colunas: eixos z, y, x do array
    s_comp = dirs[2, :]
    sup = int(np.argmax(np.abs(s_comp)))
    remaining = [a for a in range(3) if a != sup]
    l_comp = dirs[0, remaining]
    lr = remaining[int(np.argmax(np.abs(l_comp)))]
    ap = next(a for a in remaining if a != lr)
    return AxisInfo(
        superior_axis=sup,
        superior_sign=1 if s_comp[sup] > 0 else -1,
        superior_cos=float(abs(s_comp[sup])),
        lr_axis=lr,
        ap_axis=ap,
        anterior_sign=1 if dirs[1, ap] < 0 else -1,  # LPS: +y = posterior
    )


# --------------------------------------------------------------------------- #
# Secção transversal perpendicular à linha central
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class CrossSection:
    """Diâmetros de uma secção transversal (mm) e onde foi medida."""

    max_diameter_mm: float  # maior calibre (Feret) no plano perpendicular
    perpendicular_diameter_mm: float  # calibre ortogonal ao maior
    min_diameter_mm: float  # menor calibre no plano
    area_mm2: float
    center_lps: tuple[float, float, float]
    endpoints_lps: tuple[tuple[float, float, float], tuple[float, float, float]]


def _orthonormal_basis(t: FloatArray) -> tuple[FloatArray, FloatArray]:
    import numpy as np

    t = t / np.linalg.norm(t)
    helper = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(t, helper)
    u /= np.linalg.norm(u)
    v = np.cross(t, u)
    return u, v


def cross_section_diameters(
    mask: BoolArray,
    affine_lps: FloatArray,
    center_lps: FloatArray,
    tangent_lps: FloatArray,
    *,
    radius_mm: float = 50.0,
    step_mm: float = 0.5,
    angle_step_deg: float = 2.0,
) -> CrossSection | None:
    """Mede a secção da máscara no plano perpendicular a ``tangent`` passando por ``center``.

    Amostra a máscara (interpolação trilinear, limiar 0,5) numa grade de ``step_mm``
    no plano, fica com o componente conexo que contém o centro (ou o mais próximo
    dele) e mede os calibres de Feret por projeção a cada ``angle_step_deg`` graus.
    Devolve ``None`` quando o plano não corta a máscara perto do centro, ou
    quando o componente encosta na borda da janela de amostragem (estrutura
    fundida a outra ou maior que a janela: medida não confiável).
    """
    require_scipy()
    import numpy as np
    from scipy import ndimage

    u, v = _orthonormal_basis(np.asarray(tangent_lps, dtype=np.float64))
    offsets = np.arange(-radius_mm, radius_mm + step_mm / 2, step_mm)
    n = len(offsets)
    gu, gv = np.meshgrid(offsets, offsets, indexing="ij")
    pts = center_lps[None, :] + gu.reshape(-1, 1) * u[None, :] + gv.reshape(-1, 1) * v[None, :]
    idx = lps_to_index(affine_lps, pts).T  # (3, N) na ordem z, y, x
    sampled = ndimage.map_coordinates(
        mask.astype(np.float32), idx, order=1, mode="constant", cval=0.0
    ).reshape(n, n)
    plane = sampled >= 0.5
    if not plane.any():
        return None
    labels, count = ndimage.label(plane, structure=np.ones((3, 3), dtype=bool))
    c = n // 2
    chosen = int(labels[c, c])
    if chosen == 0:
        coords = np.argwhere(labels > 0)
        d2 = ((coords - c) ** 2).sum(axis=1)
        nearest = coords[int(np.argmin(d2))]
        if math.sqrt(float(d2.min())) * step_mm > 5.0:
            return None
        chosen = int(labels[nearest[0], nearest[1]])
    comp = labels == chosen
    if comp[0, :].any() or comp[-1, :].any() or comp[:, 0].any() or comp[:, -1].any():
        return None
    ij = np.argwhere(comp).astype(np.float64) * step_mm - radius_mm  # (M, 2) em mm (u, v)
    angles = np.deg2rad(np.arange(0.0, 180.0, angle_step_deg))
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (A, 2)
    proj = ij @ dirs.T  # (M, A)
    widths = proj.max(axis=0) - proj.min(axis=0) + step_mm
    k = int(np.argmax(widths))
    kp = int(np.argmin(np.abs(np.abs(np.cos(angles - angles[k])))))  # ângulo mais ortogonal
    lo, hi = proj[:, k].min(), proj[:, k].max()
    mid_perp = float(np.median(ij @ np.array([-dirs[k, 1], dirs[k, 0]])))
    d = dirs[k]
    nrm = np.array([-d[1], d[0]])
    p0 = center_lps + (lo * d[0] + mid_perp * nrm[0]) * u + (lo * d[1] + mid_perp * nrm[1]) * v
    p1 = center_lps + (hi * d[0] + mid_perp * nrm[0]) * u + (hi * d[1] + mid_perp * nrm[1]) * v
    return CrossSection(
        max_diameter_mm=float(widths[k]),
        perpendicular_diameter_mm=float(widths[kp]),
        min_diameter_mm=float(widths.min()),
        area_mm2=float(comp.sum()) * step_mm * step_mm,
        center_lps=(float(center_lps[0]), float(center_lps[1]), float(center_lps[2])),
        endpoints_lps=(
            (float(p0[0]), float(p0[1]), float(p0[2])),
            (float(p1[0]), float(p1[1]), float(p1[2])),
        ),
    )


def smooth_polyline(points: FloatArray, window: int) -> FloatArray:
    """Média móvel (janela ímpar, bordas encurtadas) de uma polilinha ``(N, 3)``."""
    import numpy as np

    n = len(points)
    half = max(0, window // 2)
    out = np.empty_like(points, dtype=np.float64)
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        out[i] = points[a:b].mean(axis=0)
    return out


def as_float_array(value: Any) -> FloatArray:
    import numpy as np

    out: FloatArray = np.asarray(value, dtype=np.float64)
    return out
