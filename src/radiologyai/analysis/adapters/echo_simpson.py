"""Volumes do ventrículo esquerdo pelo método dos discos (Simpson), mono e biplanar.

Algoritmo determinístico sobre máscaras binárias da cavidade do VE em cortes
apicais (4 e 2 câmaras), com espaçamento de pixel em mm. Implementação própria
(numpy), seguindo a descrição clássica do método:

1. contorno da máscara (rastreamento de Moore, 8-conexo) convertido para mm
   com o espaçamento ``(linha, coluna)`` — espaçamento anisotrópico é tratado
   na geometria, sem reamostrar a imagem;
2. plano da valva mitral = a MAIOR corda do contorno cujo arco intermediário
   fica a até ``base_tolerance_mm`` da corda (trecho reto) e cuja inclinação é
   de até 45° em relação à horizontal da imagem (ápice para cima ou para baixo);
3. ápice = ponto do contorno mais distante do ponto médio da base; o eixo longo
   ``L`` vai do ponto médio da base ao ápice;
4. ``n_discs`` discos perpendiculares ao eixo, centrados em ``(k + 0,5)/n`` do
   eixo (regra do ponto médio); o diâmetro de cada disco é a largura da máscara
   ao longo da perpendicular, medida a partir do eixo;
5. monoplanar: ``V = π/4 · Σ d_k² · L/n``; biplanar: ``V = π/4 · Σ a_k·b_k · L/n``
   com ``L`` = o maior dos dois eixos longos do mesmo instante (convenção
   ``per_instant``, a das diretrizes de quantificação de câmaras). A convenção
   ``shared_max`` (usada pelo código de referência do CAMUS para calcular a FE
   do próprio dataset) usa o MAIOR eixo longo entre todos os cortes e instantes
   para ED e ES; existe só para comparar com aquela referência.

Nada aqui decide normalidade: devolve volumes (mL), FE (%) e a geometria usada,
para que o usuário veja exatamente o que foi medido.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from radiologyai.errors import RadiologyAIError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = [
    "DiscGeometry",
    "SimpsonError",
    "VolumeResult",
    "biplane_volume",
    "disc_geometry",
    "ejection_fraction",
    "single_plane_volume",
    "trace_contour",
]

DEFAULT_DISCS = 20


class SimpsonError(RadiologyAIError):
    """A máscara não permite aplicar o método dos discos (vazia, sem base reta, degenerada)."""


@dataclass(frozen=True)
class DiscGeometry:
    """Geometria de um corte: base, ápice, eixo longo e diâmetros dos discos.

    Coordenadas de pixel ``(x = coluna, y = linha)`` da imagem de origem.
    """

    base: tuple[tuple[float, float], tuple[float, float]]
    apex: tuple[float, float]
    long_axis_mm: float
    diameters_mm: tuple[float, ...]
    disc_segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    spacing_mm: tuple[float, float]

    @property
    def n_discs(self) -> int:
        return len(self.diameters_mm)


@dataclass(frozen=True)
class VolumeResult:
    """Volume (mL) e a geometria de cada corte usada no cálculo."""

    volume_ml: float
    method: str
    long_axis_mm: float
    views: tuple[DiscGeometry, ...]


def _as_mask(mask: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
    import numpy as np

    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise SimpsonError(f"máscara deve ser 2D; recebido shape {arr.shape}")
    out: npt.NDArray[np.bool_] = arr.astype(bool)
    if not out.any():
        raise SimpsonError("máscara vazia: não há cavidade para medir")
    return out


def _check_spacing(spacing_mm: tuple[float, float]) -> tuple[float, float]:
    if len(spacing_mm) != 2:
        raise SimpsonError(f"espaçamento deve ser (linha, coluna); recebido {spacing_mm}")
    row, col = float(spacing_mm[0]), float(spacing_mm[1])
    if not (math.isfinite(row) and math.isfinite(col) and row > 0 and col > 0):
        raise SimpsonError(f"espaçamento inválido: {spacing_mm}")
    return row, col


# Vizinhança de Moore em sentido horário começando a oeste: (dy, dx).
_MOORE: tuple[tuple[int, int], ...] = (
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
)


def trace_contour(mask: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Contorno externo ordenado (centros de pixel, ``(N, 2)`` em ``(linha, coluna)``).

    Rastreia o componente que contém o primeiro pixel em ordem de varredura;
    a máscara deve ter um único componente (o adaptador mantém o maior).

    Raises:
        SimpsonError: máscara vazia ou com menos de 3 pixels de borda.
    """
    import numpy as np

    m = np.pad(_as_mask(mask), 1)
    rows, cols = np.nonzero(m)
    start = (int(rows[0]), int(cols[0]))  # primeiro na varredura: vizinho oeste é fundo
    contour = [start]
    current = start
    backtrack_dir = 0  # veio de oeste
    max_steps = 4 * int(m.sum()) + 16
    for _ in range(max_steps):
        found = False
        for k in range(8):
            d = (backtrack_dir + k) % 8
            ny, nx = current[0] + _MOORE[d][0], current[1] + _MOORE[d][1]
            if m[ny, nx]:
                # próxima busca começa no vizinho anterior ao encontrado (sentido horário)
                backtrack_dir = (d + 6) % 8
                current = (ny, nx)
                found = True
                break
        if not found:  # pixel isolado
            break
        if current == start and len(contour) > 1:
            break
        contour.append(current)
    if len(contour) < 3:
        raise SimpsonError("contorno com menos de 3 pontos: máscara degenerada")
    out: npt.NDArray[np.float64] = np.asarray(contour, dtype=np.float64) - 1.0
    return out


def _resample_closed(points: npt.NDArray[np.float64], step: float) -> npt.NDArray[np.float64]:
    """Reamostra um polígono fechado em passos de comprimento de arco ``step``."""
    import numpy as np

    closed = np.vstack([points, points[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 0:
        raise SimpsonError("contorno de comprimento nulo")
    n = max(int(round(total / step)), 8)
    s = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(s, cum, closed[:, 0])
    y = np.interp(s, cum, closed[:, 1])
    out: npt.NDArray[np.float64] = np.stack([x, y], axis=1)
    return out


def _find_base(
    pts: npt.NDArray[np.float64], step: float, tolerance: float, max_chord: float
) -> tuple[int, int]:
    """Índices (i, j) da maior corda quase reta e quase horizontal do contorno (em mm)."""
    import numpy as np

    n = len(pts)
    mmax = min(n - 2, int(math.ceil(max_chord * 1.5 / step)))
    if mmax < 3:
        raise SimpsonError("contorno curto demais para localizar a base")
    best_len, best = 0.0, (-1, -1)
    ks = np.arange(mmax + 1)
    lower = ks[None, :] < ks[:, None]  # k < m
    for i in range(n):
        seq = pts[(i + ks) % n] - pts[i]  # (mmax+1, 2) relativo ao início; coluna 0 = linha
        v = seq  # corda até cada m
        norms = np.linalg.norm(v, axis=1)
        cross = np.abs(v[:, None, 0] * seq[None, :, 1] - v[:, None, 1] * seq[None, :, 0])
        with np.errstate(divide="ignore", invalid="ignore"):
            dist = cross / norms[:, None]
        dist = np.where(lower, dist, 0.0)
        dist[:, 0] = 0.0
        max_dist = dist.max(axis=1)
        # ângulo com a horizontal: |Δlinha| <= |Δcoluna|
        horizontal = np.abs(v[:, 0]) <= np.abs(v[:, 1])
        ok = (max_dist <= tolerance) & horizontal & (norms <= max_chord)
        ok[:3] = False
        if ok.any():
            cand = np.where(ok, norms, -1.0)
            m = int(np.argmax(cand))
            if cand[m] > best_len:
                best_len, best = float(cand[m]), (i, (i + m) % n)
    if best_len <= 0:
        raise SimpsonError("não foi encontrado trecho reto compatível com o plano mitral")
    return best


def _trim_base(
    pts: npt.NDArray[np.float64], i: int, j: int, trim_tolerance: float
) -> tuple[int, int]:
    """Afasta as pontas da corda que subiram pelas paredes (cantos da base).

    Ajusta uma reta (mínimos quadrados totais) ao terço central do arco da base
    e descarta, nas duas pontas, os pontos a mais de ``trim_tolerance`` dela.
    """
    import numpy as np

    n = len(pts)
    m = (j - i) % n
    idx = (i + np.arange(m + 1)) % n
    arc = pts[idx]
    lo, hi = int(m * 0.3), max(int(m * 0.7), int(m * 0.3) + 2)
    central = arc[lo : hi + 1]
    center = central.mean(axis=0)
    _, _, vt = np.linalg.svd(central - center, full_matrices=False)
    direction = vt[0]
    rel = arc - center
    dist = np.abs(rel[:, 0] * direction[1] - rel[:, 1] * direction[0])
    close = np.nonzero(dist <= trim_tolerance)[0]
    if close.size < 2:
        return i, j
    return int(idx[close[0]]), int(idx[close[-1]])


def _width_along(
    mask: npt.NDArray[np.bool_],
    spacing: tuple[float, float],
    origin_mm: npt.NDArray[np.float64],
    direction: npt.NDArray[np.float64],
    max_mm: float,
    step_mm: float,
) -> float:
    """Distância (mm) do ponto ``origin`` até sair da máscara ao longo de ``direction``."""
    import numpy as np

    t = np.arange(0.0, max_mm + step_mm, step_mm)
    pos = origin_mm[None, :] + t[:, None] * direction[None, :]
    r = np.rint(pos[:, 0] / spacing[0]).astype(np.int64)
    c = np.rint(pos[:, 1] / spacing[1]).astype(np.int64)
    inside = (r >= 0) & (r < mask.shape[0]) & (c >= 0) & (c < mask.shape[1])
    inside[inside] = mask[r[inside], c[inside]]
    outside = np.nonzero(~inside)[0]
    if outside.size == 0:
        return float(max_mm)
    first = int(outside[0])
    if first == 0:
        return 0.0
    return float(t[first] - step_mm / 2.0)


def disc_geometry(
    mask: npt.NDArray[Any],
    spacing_mm: tuple[float, float],
    *,
    n_discs: int = DEFAULT_DISCS,
    base_tolerance_mm: float = 2.5,
    sampling_mm: float = 0.5,
) -> DiscGeometry:
    """Base, ápice, eixo longo e diâmetros dos discos de uma máscara de cavidade do VE.

    Raises:
        SimpsonError: máscara vazia/degenerada, espaçamento inválido ou base não encontrada.
    """
    import numpy as np

    if n_discs < 1:
        raise SimpsonError("n_discs deve ser >= 1")
    m = _as_mask(mask)
    spacing = _check_spacing(spacing_mm)
    scale = np.asarray(spacing, dtype=np.float64)
    contour_mm = trace_contour(m) * scale[None, :]
    pts = _resample_closed(contour_mm, sampling_mm)
    extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    i, j = _find_base(pts, sampling_mm, base_tolerance_mm, max_chord=extent)
    i, j = _trim_base(pts, i, j, trim_tolerance=max(0.75, 1.5 * max(spacing)))
    b0, b1 = pts[i], pts[j]
    mid = (b0 + b1) / 2.0
    dist = np.linalg.norm(pts - mid[None, :], axis=1)
    apex = pts[int(np.argmax(dist))]
    axis = apex - mid
    length = float(np.linalg.norm(axis))
    if length <= 0:
        raise SimpsonError("eixo longo de comprimento nulo")
    unit = axis / length
    normal = np.array([-unit[1], unit[0]])
    step = min(spacing) / 4.0
    diameters: list[float] = []
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for k in range(n_discs):
        center = mid + unit * length * (k + 0.5) / n_discs
        d1 = _width_along(m, spacing, center, normal, extent, step)
        d2 = _width_along(m, spacing, center, -normal, extent, step)
        diameters.append(d1 + d2)
        p1, p2 = center + normal * d1, center - normal * d2
        segments.append(
            (
                (float(p1[1] / spacing[1]), float(p1[0] / spacing[0])),
                (float(p2[1] / spacing[1]), float(p2[0] / spacing[0])),
            )
        )

    def px(p: npt.NDArray[np.float64]) -> tuple[float, float]:
        return float(p[1] / spacing[1]), float(p[0] / spacing[0])

    return DiscGeometry(
        base=(px(b0), px(b1)),
        apex=px(apex),
        long_axis_mm=length,
        diameters_mm=tuple(diameters),
        disc_segments=tuple(segments),
        spacing_mm=spacing,
    )


def single_plane_volume(geom: DiscGeometry, *, length_mm: float | None = None) -> VolumeResult:
    """Volume monoplanar (discos circulares) em mL.

    ``length_mm`` substitui o eixo longo do corte (convenção ``shared_max``).
    """
    length = geom.long_axis_mm if length_mm is None else length_mm
    h = length / geom.n_discs
    vol_mm3 = sum(math.pi / 4.0 * d * d * h for d in geom.diameters_mm)
    return VolumeResult(
        volume_ml=vol_mm3 / 1000.0,
        method=f"Simpson monoplanar, {geom.n_discs} discos",
        long_axis_mm=length,
        views=(geom,),
    )


def biplane_volume(
    a4c: DiscGeometry, a2c: DiscGeometry, *, length_mm: float | None = None
) -> VolumeResult:
    """Volume biplanar (discos elípticos, eixos dos dois cortes ortogonais) em mL.

    Raises:
        SimpsonError: número de discos diferente entre os cortes.
    """
    if a4c.n_discs != a2c.n_discs:
        raise SimpsonError("os dois cortes devem ter o mesmo número de discos")
    length = max(a4c.long_axis_mm, a2c.long_axis_mm) if length_mm is None else length_mm
    h = length / a4c.n_discs
    vol_mm3 = sum(
        math.pi / 4.0 * a * b * h for a, b in zip(a4c.diameters_mm, a2c.diameters_mm, strict=True)
    )
    return VolumeResult(
        volume_ml=vol_mm3 / 1000.0,
        method=f"Simpson biplanar (A4C + A2C), {a4c.n_discs} discos, L = maior eixo",
        long_axis_mm=length,
        views=(a4c, a2c),
    )


def ejection_fraction(edv_ml: float, esv_ml: float) -> float:
    """FE (%) = 100·(VDF − VSF)/VDF.

    Raises:
        SimpsonError: VDF não positivo ou VSF negativo.
    """
    if not (math.isfinite(edv_ml) and edv_ml > 0):
        raise SimpsonError(f"VDF inválido: {edv_ml}")
    if not (math.isfinite(esv_ml) and esv_ml >= 0):
        raise SimpsonError(f"VSF inválido: {esv_ml}")
    return 100.0 * (edv_ml - esv_ml) / edv_ml
