"""Biomarcadores de TC a partir das máscaras do TotalSegmentator e das HU reais.

Cada função devolve :class:`~radiologyai.analysis.base.RawMeasurement` com valor
OU abstenção explícita. Os limiares de decisão ficam nos manifestos (pontos de
operação com origem); aqui só se mede e se conferem pré-condições.

Fontes das pré-condições: ``/root/medai-cache/research/evidence-ct-mr.verified.md``
(seções 2-6 e 8). Onde a regra é de engenharia do projeto, o texto diz.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from radiologyai.analysis.base import RawMeasurement
from radiologyai.analysis.biomarkers import acquisition as acq
from radiologyai.analysis.biomarkers.geometry import (
    axis_info,
    component_volumes_ml,
    cross_section_diameters,
    erode_mm,
    index_to_lps,
    largest_component_2d,
    lps_to_index,
    require_scipy,
    smooth_polyline,
    touches_border,
    voxel_volume_ml,
)
from radiologyai.analysis.types import AbstentionReason, ImageMeta, Primitive

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    BoolArray = npt.NDArray[np.bool_]
    FloatArray = npt.NDArray[np.float64]

__all__ = [
    "HU",
    "LUNG_LOBES",
    "CTContext",
    "aortic_diameters",
    "evans_index",
    "l1_trabecular_hu",
    "laa950_percent",
    "liver_attenuation",
    "liver_spleen_difference",
    "organ_volume",
]

HU = "[hnsf'U]"
LUNG_LOBES = (
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
)
_AXIAL_MIN_COS = 0.9  # corte axial: normal a <= ~25 graus do eixo craniocaudal (engenharia)


@dataclass(frozen=True, slots=True)
class CTContext:
    """Volume real + mapa de rótulos do modelo a montante, na mesma grade."""

    hu: npt.NDArray[np.float32]  # (Z, Y, X)
    spacing_zyx: tuple[float, float, float]
    affine_lps: FloatArray
    meta: ImageMeta
    labelmap: npt.NDArray[np.integer[Any]]
    names: dict[str, int]  # chave de rótulo -> índice no mapa
    source: str  # nome do LabelMap (para ``derived_from``)

    def __post_init__(self) -> None:
        if self.hu.shape != self.labelmap.shape:
            raise ValueError(
                f"grade do mapa de rótulos {self.labelmap.shape} difere da imagem {self.hu.shape}"
            )

    def has(self, name: str) -> bool:
        return name in self.names

    def mask(self, *names: str) -> BoolArray:
        """União das máscaras nomeadas (rótulo inexistente no modelo levanta)."""
        import numpy as np

        missing = [n for n in names if n not in self.names]
        if missing:
            raise KeyError(f"rótulo(s) inexistente(s) no mapa {self.source!r}: {missing}")
        out: BoolArray = np.isin(self.labelmap, [self.names[n] for n in names])
        return out

    @property
    def voxel_ml(self) -> float:
        return voxel_volume_ml(self.spacing_zyx)


def _abstain(
    key: str,
    unit: str,
    method: str,
    reason: AbstentionReason,
    note: str,
    ctx: CTContext,
    preconditions: dict[str, str] | None = None,
) -> RawMeasurement:
    return RawMeasurement(
        key=key,
        value=None,
        unit=unit,
        method=method,
        derived_from=[ctx.source],
        preconditions=preconditions or acq.acquisition_facts(ctx.meta),
        abstention=reason,
        note=note,
    )


def _gate_fail(
    key: str, unit: str, method: str, gate: acq.Gate, ctx: CTContext
) -> RawMeasurement | None:
    if gate.ok:
        return None
    assert gate.reason is not None
    assert gate.note is not None
    return _abstain(key, unit, method, gate.reason, gate.note, ctx)


def _axial(ctx: CTContext) -> acq.Gate:
    info = axis_info(ctx.affine_lps)
    if info.superior_axis != 0 or info.superior_cos < _AXIAL_MIN_COS:
        return acq.Gate.failed(
            AbstentionReason.GEOMETRY,
            "série não axial (eixo de cortes não alinhado ao craniocaudal); medida exige axial",
        )
    return acq.Gate.passed()


# --------------------------------------------------------------------------- #
# Volumes de órgão
# --------------------------------------------------------------------------- #
def organ_volume(ctx: CTContext, key: str, labels: Iterable[str]) -> RawMeasurement:
    """Volume (mL) da união de ``labels``; abstém-se se ausente ou cortada pelo campo de visão."""
    names = tuple(labels)
    method = f"contagem de voxels da máscara {'+'.join(names)} x volume do voxel ({ctx.source})"
    present = [n for n in names if ctx.has(n)]
    if not present:
        return _abstain(key, "mL", method, AbstentionReason.INPUT, "rótulo ausente", ctx)
    mask = ctx.mask(*present)
    if not mask.any():
        return _abstain(
            key,
            "mL",
            method,
            AbstentionReason.ACQUISITION,
            "estrutura ausente do campo de visão",
            ctx,
        )
    if touches_border(mask):
        return _abstain(
            key,
            "mL",
            method,
            AbstentionReason.ACQUISITION,
            "estrutura cortada pelo limite do campo de visão: o volume seria um limite inferior",
            ctx,
        )
    return RawMeasurement(
        key=key,
        value=round(float(mask.sum()) * ctx.voxel_ml, 3),
        unit="mL",
        method=method,
        derived_from=[ctx.source],
        preconditions={"complete_in_fov": "sim"},
    )


# --------------------------------------------------------------------------- #
# Osteoporose oportunista: atenuação trabecular de L1
# --------------------------------------------------------------------------- #
L1_KEY = "l1_trabecular_hu"
L1_METHOD = (
    "média de HU no núcleo trabecular de L1: máscara vertebrae_L1 erodida 4 mm, terço médio "
    "craniocaudal, maior componente por corte axial (aproxima a ROI oval manual de Pickhardt "
    "2013, doi:10.7326/0003-4819-158-8-201304160-00003)"
)


def l1_trabecular_hu(ctx: CTContext) -> RawMeasurement:
    """Atenuação trabecular de L1 (HU). Exige 120 kV e adulto; contraste é permitido."""
    import numpy as np

    key, unit = L1_KEY, HU
    for gate in (acq.require_kvp(ctx.meta, 120.0), acq.require_adult(ctx.meta), _axial(ctx)):
        if (fail := _gate_fail(key, unit, L1_METHOD, gate, ctx)) is not None:
            return fail
    if not ctx.has("vertebrae_L1"):
        return _abstain(key, unit, L1_METHOD, AbstentionReason.INPUT, "rótulo L1 ausente", ctx)
    l1 = ctx.mask("vertebrae_L1")
    if not l1.any():
        return _abstain(
            key, unit, L1_METHOD, AbstentionReason.ACQUISITION, "L1 fora do campo de visão", ctx
        )
    if touches_border(l1, axes=(0,)):
        return _abstain(
            key,
            unit,
            L1_METHOD,
            AbstentionReason.ACQUISITION,
            "L1 cortada pelo limite craniocaudal do campo de visão",
            ctx,
        )
    core = erode_mm(l1, ctx.spacing_zyx, 4.0)
    zs = np.flatnonzero(l1.any(axis=(1, 2)))
    z0, z1 = int(zs.min()), int(zs.max())
    n = z1 - z0 + 1
    lo, hi = z0 + n // 3, z0 + (2 * n) // 3
    values: list[npt.NDArray[np.float32]] = []
    for z in range(lo, max(lo + 1, hi + 1)):
        if not core[z].any():
            continue
        body = largest_component_2d(core[z])
        values.append(ctx.hu[z][body])
    if not values:
        return _abstain(
            key, unit, L1_METHOD, AbstentionReason.QUALITY, "núcleo trabecular vazio", ctx
        )
    hu = np.concatenate(values)
    if hu.size * ctx.voxel_ml < 0.5:
        return _abstain(
            key,
            unit,
            L1_METHOD,
            AbstentionReason.QUALITY,
            f"núcleo trabecular com {hu.size * ctx.voxel_ml:.2f} mL (< 0,5 mL; engenharia)",
            ctx,
        )
    return RawMeasurement(
        key=key,
        value=round(float(hu.mean()), 1),
        unit=unit,
        method=L1_METHOD,
        derived_from=[ctx.source],
        preconditions={
            **acq.acquisition_facts(ctx.meta),
            "core_ml": f"{hu.size * ctx.voxel_ml:.2f}",
        },
        note=(
            "Método automatizado; a ROI validada é manual. Jang 2019 relata que o automatizado "
            "deles leu em média 21 HU acima do manual; o viés DESTE método não foi medido. "
            "Fratura, cimento, lesão focal em L1 não são detectados e invalidam a medida."
        ),
    )


# --------------------------------------------------------------------------- #
# Esteatose hepática
# --------------------------------------------------------------------------- #
_LIVER_METHOD = (
    "média de HU no fígado erodido 5 mm (afasta cápsula e vasos hilares) na máscara 'liver'"
)
_LIVER_CORE_MIN_ML = 10.0  # regra de engenharia: núcleo pequeno demais não representa o órgão
_SPLEEN_CORE_MIN_ML = 5.0


def _core_mean(ctx: CTContext, name: str, erosion_mm: float) -> tuple[float, float] | None:
    if not ctx.has(name):
        return None
    mask = ctx.mask(name)
    if not mask.any():
        return None
    core = erode_mm(mask, ctx.spacing_zyx, erosion_mm)
    if not core.any():
        return None
    return float(ctx.hu[core].mean()), float(core.sum()) * ctx.voxel_ml


def liver_attenuation(ctx: CTContext, phase: Literal["none", "portal_venous"]) -> RawMeasurement:
    """Atenuação hepática (HU) em fase confirmada; sem contraste exige também 120 kV."""
    key = (
        "liver_attenuation_unenhanced_hu"
        if phase == "none"
        else "liver_attenuation_portal_venous_hu"
    )
    gates = [acq.require_phase(ctx.meta, phase)]
    if phase == "none":
        gates.append(acq.require_kvp(ctx.meta, 120.0))
    for gate in gates:
        if (fail := _gate_fail(key, HU, _LIVER_METHOD, gate, ctx)) is not None:
            return fail
    core = _core_mean(ctx, "liver", 5.0)
    if core is None:
        return _abstain(
            key,
            HU,
            _LIVER_METHOD,
            AbstentionReason.ACQUISITION,
            "fígado ausente ou sem núcleo",
            ctx,
        )
    mean, volume = core
    if volume < _LIVER_CORE_MIN_ML:
        return _abstain(
            key,
            HU,
            _LIVER_METHOD,
            AbstentionReason.QUALITY,
            f"núcleo hepático com {volume:.1f} mL (< {_LIVER_CORE_MIN_ML:g} mL; engenharia)",
            ctx,
        )
    note = (
        "Sobrecarga de ferro, lesões extensas e artefato de endurecimento de feixe alteram a "
        "atenuação e não são detectados."
    )
    if phase == "portal_venous":
        note += " Evidência de fase portal é de um centro e um protocolo (Pickhardt 2024)."
    return RawMeasurement(
        key=key,
        value=round(mean, 1),
        unit=HU,
        method=_LIVER_METHOD,
        derived_from=[ctx.source],
        preconditions={**acq.acquisition_facts(ctx.meta), "liver_core_ml": f"{volume:.1f}"},
        note=note,
    )


def liver_spleen_difference(ctx: CTContext) -> RawMeasurement:
    """CT(L-S) sem contraste: média hepática menos média esplênica (núcleos erodidos 5 mm)."""
    key = "liver_spleen_difference_unenhanced_hu"
    method = "HU médio do fígado erodido 5 mm menos HU médio do baço erodido 5 mm (Park 2006)"
    for gate in (acq.require_phase(ctx.meta, "none"), acq.require_kvp(ctx.meta, 120.0)):
        if (fail := _gate_fail(key, HU, method, gate, ctx)) is not None:
            return fail
    liver = _core_mean(ctx, "liver", 5.0)
    spleen = _core_mean(ctx, "spleen", 5.0)
    if liver is None or spleen is None:
        return _abstain(
            key, HU, method, AbstentionReason.ACQUISITION, "fígado ou baço ausente/sem núcleo", ctx
        )
    if liver[1] < _LIVER_CORE_MIN_ML or spleen[1] < _SPLEEN_CORE_MIN_ML:
        return _abstain(
            key,
            HU,
            method,
            AbstentionReason.QUALITY,
            f"núcleos pequenos (fígado {liver[1]:.1f} mL, baço {spleen[1]:.1f} mL; engenharia)",
            ctx,
        )
    return RawMeasurement(
        key=key,
        value=round(liver[0] - spleen[0], 1),
        unit=HU,
        method=method,
        derived_from=[ctx.source],
        preconditions={
            **acq.acquisition_facts(ctx.meta),
            "liver_mean_hu": f"{liver[0]:.1f}",
            "spleen_mean_hu": f"{spleen[0]:.1f}",
        },
        note="Baço infartado ou ausente invalida o índice; não é detectado automaticamente.",
    )


# --------------------------------------------------------------------------- #
# Enfisema: %LAA-950
# --------------------------------------------------------------------------- #
LAA_MAX_THICKNESS_MM = 1.5  # regra de engenharia; a evidência pede ~1 mm e não fixa limite
_LAA_METHOD = (
    "percentual de voxels pulmonares (5 lobos do TotalSegmentator; traqueia/brônquios "
    "principais excluídos) com HU <= -950"
)


def laa950_percent(ctx: CTContext) -> RawMeasurement:
    """%LAA-950: exige sem contraste confirmado, corte fino, kernel suave e pulmões completos."""
    import numpy as np

    key = "lung_laa950_percent"
    gates = [
        acq.require_phase(ctx.meta, "none"),
        acq.require_thin_slices(ctx.meta, LAA_MAX_THICKNESS_MM),
        acq.require_smooth_kernel(ctx.meta),
        _axial(ctx),
    ]
    desc = (ctx.meta.series_description or "").upper()
    if "EXP" in desc:
        gates.append(
            acq.Gate.failed(
                AbstentionReason.ACQUISITION, f"série possivelmente expiratória ({desc!r})"
            )
        )
    for gate in gates:
        if (fail := _gate_fail(key, "%", _LAA_METHOD, gate, ctx)) is not None:
            return fail
    present = [n for n in LUNG_LOBES if ctx.has(n)]
    if len(present) != len(LUNG_LOBES):
        return _abstain(key, "%", _LAA_METHOD, AbstentionReason.INPUT, "lobos ausentes", ctx)
    lung = ctx.mask(*present)
    if not lung.any():
        return _abstain(
            key, "%", _LAA_METHOD, AbstentionReason.ACQUISITION, "pulmões fora do campo", ctx
        )
    if touches_border(lung, axes=(0,)):
        return _abstain(
            key,
            "%",
            _LAA_METHOD,
            AbstentionReason.ACQUISITION,
            "pulmões cortados no limite craniocaudal (ápices ou bases fora do campo)",
            ctx,
        )
    values = ctx.hu[lung]
    pct = 100.0 * float(np.count_nonzero(values <= -950.0)) / float(values.size)
    return RawMeasurement(
        key=key,
        value=round(pct, 3),
        unit="%",
        method=_LAA_METHOD,
        derived_from=[ctx.source],
        preconditions={
            **acq.acquisition_facts(ctx.meta),
            "lung_volume_ml": f"{values.size * ctx.voxel_ml:.0f}",
            "inspiration": "assumida (não verificável pela imagem)",
        },
        note=(
            "Inspiração máxima não é verificável pela imagem; consolidação, pneumotórax, "
            "movimento e reconstrução iterativa/DL não caracterizada alteram o índice."
        ),
    )


# --------------------------------------------------------------------------- #
# Aorta: diâmetro máximo perpendicular à linha central, por região
# --------------------------------------------------------------------------- #
AORTA_METHOD = (
    "diâmetro máximo (Feret) da secção da máscara 'aorta' no plano perpendicular à linha "
    "central (centroides por corte axial, média móvel ~15 mm, tangente por diferenças "
    "centrais), amostrada a 0,5 mm; borda externa da máscara (ACC/AHA 2022, "
    "doi:10.1161/CIR.0000000000001106: medir perpendicular ao eixo longo)"
)
_MIN_COMPONENT_MM2 = 100.0  # secção axial >= 1 cm² (engenharia)
_ROOT_EXCLUSION_MM = 20.0  # raiz/seios sem sincronização ECG: fora (engenharia)
_BIFURCATION_EXCLUSION_MM = 10.0  # alargamento da bifurcação: fora (engenharia)
AortaRegion = Literal["abdominal", "ascending", "descending_thoracic"]
AORTA_KEYS: dict[AortaRegion, str] = {
    "abdominal": "aorta_abdominal_max_diameter_mm",
    "ascending": "aorta_ascending_max_diameter_mm",
    "descending_thoracic": "aorta_descending_thoracic_max_diameter_mm",
}


def _slice_s(ctx: CTContext, z: int) -> float:
    import numpy as np

    ny, nx = ctx.hu.shape[1], ctx.hu.shape[2]
    return float(index_to_lps(ctx.affine_lps, np.array([[z, ny / 2, nx / 2]]))[0, 2])


def _crop(
    mask: BoolArray, affine: FloatArray, spacing: tuple[float, float, float], margin_mm: float
) -> tuple[BoolArray, FloatArray]:
    """Recorta a máscara ao seu envelope + margem; devolve o recorte e a afim ajustada."""
    import numpy as np

    idx = np.argwhere(mask)
    lo = idx.min(axis=0)
    hi = idx.max(axis=0) + 1
    pad = np.ceil(margin_mm / np.asarray(spacing)).astype(int)
    lo = np.maximum(lo - pad, 0)
    hi = np.minimum(hi + pad, mask.shape)
    sub = mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    origin = index_to_lps(affine, lo[None, :].astype(np.float64))[0]
    new = affine.copy()
    new[:3, 3] = origin
    return sub, new


def aortic_diameters(ctx: CTContext) -> list[RawMeasurement]:
    """Diâmetro máximo da aorta abdominal, ascendente e descendente torácica (mm).

    A raiz (sem ECG) e o arco (secções axiais não separam o arco) não são medidos.
    Cada região abstém-se com motivo quando incompleta no campo de visão, curta
    demais, fragmentada, ou quando o marco vertebral que a define está ausente.
    """
    require_scipy()
    import numpy as np
    from scipy import ndimage

    unit = "mm"
    keys = AORTA_KEYS

    def all_abstain(reason: AbstentionReason, note: str) -> list[RawMeasurement]:
        return [_abstain(k, unit, AORTA_METHOD, reason, note, ctx) for k in keys.values()]

    gate = _axial(ctx)
    if not gate.ok:
        assert gate.reason is not None
        assert gate.note is not None
        return all_abstain(gate.reason, gate.note)
    if not ctx.has("aorta"):
        return all_abstain(AbstentionReason.INPUT, "rótulo 'aorta' ausente")
    aorta = ctx.mask("aorta")
    if not aorta.any():
        return all_abstain(AbstentionReason.ACQUISITION, "aorta fora do campo de visão")
    comps = component_volumes_ml(aorta, ctx.spacing_zyx)
    if len(comps) > 1 and comps[1] >= 1.0:
        return all_abstain(
            AbstentionReason.QUALITY,
            f"máscara da aorta fragmentada ({len(comps)} componentes; 2º com {comps[1]:.1f} mL)",
        )

    sy, sx = ctx.spacing_zyx[1], ctx.spacing_zyx[2]
    zs = [int(z) for z in np.flatnonzero(aorta.any(axis=(1, 2)))]
    s_of = {z: _slice_s(ctx, z) for z in zs}
    # Marco toracoabdominal: limite caudal de T12 (hiato aórtico ~T12); senão cranial de L1.
    boundary: float | None = None
    landmark = "nenhum"
    for name, pick in (("vertebrae_T12", min), ("vertebrae_L1", max)):
        if ctx.has(name):
            vz = np.flatnonzero(ctx.mask(name).any(axis=(1, 2)))
            if vz.size:
                boundary = pick(_slice_s(ctx, int(z)) for z in vz)
                landmark = f"{name} ({'caudal' if pick is min else 'cranial'})"
                break

    per_slice: dict[int, list[tuple[float, FloatArray]]] = {}
    for z in zs:
        labels, n = ndimage.label(aorta[z], structure=np.ones((3, 3), dtype=bool))
        items: list[tuple[float, FloatArray]] = []
        for lab in range(1, n + 1):
            pix = np.argwhere(labels == lab)
            area = len(pix) * sy * sx
            if area < _MIN_COMPONENT_MM2:
                continue
            cy, cx = pix.mean(axis=0)
            items.append((area, np.array([z, cy, cx], dtype=np.float64)))
        if items:
            per_slice[z] = items
    two = [s_of[z] for z, it in per_slice.items() if len(it) >= 2]
    top_two = max(two) if two else None

    regions: dict[AortaRegion, list[FloatArray]] = {r: [] for r in keys}
    for z, items in per_slice.items():
        s = s_of[z]
        if boundary is not None and s <= boundary:
            regions["abdominal"].append(max(items, key=lambda it: it[0])[1])
        elif len(items) >= 2:
            # anterior = menor coordenada P (LPS +y = posterior)
            ordered = sorted(items, key=lambda it: index_to_lps(ctx.affine_lps, it[1][None])[0, 1])
            regions["ascending"].append(ordered[0][1])
            regions["descending_thoracic"].append(ordered[-1][1])
        elif top_two is not None and s > top_two:
            continue  # arco: não medido
        elif boundary is not None or top_two is not None:
            regions["descending_thoracic"].append(items[0][1])

    crop, crop_affine = _crop(aorta, ctx.affine_lps, ctx.spacing_zyx, 60.0)
    first_slice_s = _slice_s(ctx, 0)
    last_slice_s = _slice_s(ctx, ctx.hu.shape[0] - 1)
    caudal_border_z = 0 if first_slice_s < last_slice_s else ctx.hu.shape[0] - 1
    cranial_border_z = ctx.hu.shape[0] - 1 - caudal_border_z
    caudal_cut = caudal_border_z in s_of
    cranial_cut = cranial_border_z in s_of

    out: list[RawMeasurement] = []
    for region, key in keys.items():
        pts = regions[region]
        facts = {
            "region_landmark": landmark,
            "n_axial_sections": str(len(pts)),
            "ecg_gated": "não verificado (raiz não medida)",
        }
        if region == "abdominal" and boundary is None:
            out.append(
                _abstain(
                    key,
                    unit,
                    AORTA_METHOD,
                    AbstentionReason.ACQUISITION,
                    "marco vertebral T12/L1 ausente: região abdominal indefinida",
                    ctx,
                    facts,
                )
            )
            continue
        if region == "abdominal" and caudal_cut:
            out.append(
                _abstain(
                    key,
                    unit,
                    AORTA_METHOD,
                    AbstentionReason.ACQUISITION,
                    "aorta cortada no limite caudal: o máximo pode estar fora do campo",
                    ctx,
                    facts,
                )
            )
            continue
        if region != "abdominal" and cranial_cut:
            out.append(
                _abstain(
                    key,
                    unit,
                    AORTA_METHOD,
                    AbstentionReason.ACQUISITION,
                    "aorta cortada no limite cranial: segmento torácico incompleto",
                    ctx,
                    facts,
                )
            )
            continue
        if len(pts) < 5:
            out.append(
                _abstain(
                    key,
                    unit,
                    AORTA_METHOD,
                    AbstentionReason.ACQUISITION,
                    f"segmento {region} ausente ou curto ({len(pts)} cortes)",
                    ctx,
                    facts,
                )
            )
            continue
        zyx = np.array(
            sorted(pts, key=lambda p: -float(index_to_lps(ctx.affine_lps, p[None])[0, 2]))
        )
        lps = index_to_lps(ctx.affine_lps, zyx)
        window = max(3, round(15.0 / ctx.spacing_zyx[0]))
        window += 1 - window % 2
        smooth = smooth_polyline(lps, window)
        tangents = np.gradient(smooth, axis=0)
        s_vals = smooth[:, 2]
        keep = np.ones(len(smooth), dtype=bool)
        keep[[0, -1]] = False
        if region == "ascending":
            keep &= s_vals > s_vals.min() + _ROOT_EXCLUSION_MM
        if region == "abdominal":
            keep &= s_vals > s_vals.min() + _BIFURCATION_EXCLUSION_MM
        sections = []
        for center, tangent in zip(smooth[keep], tangents[keep], strict=True):
            if not np.isfinite(tangent).all() or np.linalg.norm(tangent) == 0:
                continue
            sec = cross_section_diameters(crop, crop_affine, center, tangent)
            if sec is not None:
                sections.append(sec)
        if not sections:
            out.append(
                _abstain(
                    key,
                    unit,
                    AORTA_METHOD,
                    AbstentionReason.QUALITY,
                    "nenhuma secção perpendicular mensurável",
                    ctx,
                    facts,
                )
            )
            continue
        best = max(sections, key=lambda s: s.max_diameter_mm)
        ends = lps_to_index(ctx.affine_lps, np.array(best.endpoints_lps))
        frame = int(round(float(ends[:, 0].mean())))
        geometry = [
            Primitive(
                type="line",
                xy=((float(ends[0, 2]), float(ends[0, 1])), (float(ends[1, 2]), float(ends[1, 1]))),
                frame=min(max(frame, 0), ctx.hu.shape[0] - 1),
                label=f"{region}: {best.max_diameter_mm:.1f} mm (projeção no corte axial)",
            )
        ]
        out.append(
            RawMeasurement(
                key=key,
                value=round(best.max_diameter_mm, 1),
                unit=unit,
                method=AORTA_METHOD,
                geometry=geometry,
                derived_from=[ctx.source],
                preconditions={**facts, "n_perpendicular_sections": str(len(sections))},
                note=(
                    f"diâmetro perpendicular ao máximo {best.perpendicular_diameter_mm:.1f} mm; "
                    f"área {best.area_mm2:.0f} mm²; parede externa (inclui trombo). Enxerto, "
                    "stent e dissecção não são detectados e invalidam a medida."
                ),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Índice de Evans
# --------------------------------------------------------------------------- #
EVANS_KEY = "evans_index"
EVANS_METHOD = (
    "maior largura laterolateral conjunta dos cornos frontais (ventricle_parts) dividida pela "
    "maior largura laterolateral da máscara 'brain' (cavidade craniana, TotalSegmentator "
    "total) no MESMO corte axial (diretriz japonesa de HPN 2021, doi:10.2176/nmc.st.2020-0292)"
)


def _max_lr_width(
    mask2d: BoolArray, lr_axis2d: int, spacing_lr: float
) -> tuple[float, int, int, int]:
    """Maior extensão LR (mm) por linha; devolve (largura, linha, início, fim)."""
    import numpy as np

    arr = mask2d if lr_axis2d == 1 else mask2d.T
    best = (0.0, -1, -1, -1)
    for row in np.flatnonzero(arr.any(axis=1)):
        cols = np.flatnonzero(arr[row])
        width = float(cols[-1] - cols[0] + 1) * spacing_lr
        if width > best[0]:
            best = (width, int(row), int(cols[0]), int(cols[-1]))
    return best


def evans_index(
    hu: npt.NDArray[np.float32],
    spacing_zyx: tuple[float, float, float],
    affine_lps: FloatArray,
    meta: ImageMeta,
    frontal_horns: BoolArray,
    cranial_cavity: BoolArray,
    source: str,
) -> RawMeasurement:
    """Índice de Evans; abstém-se sem cornos frontais, sem crânio inteiro ou fora do axial."""
    ctx_like = CTContext(
        hu=hu,
        spacing_zyx=spacing_zyx,
        affine_lps=affine_lps,
        meta=meta,
        labelmap=frontal_horns.astype("uint8"),
        names={},
        source=source,
    )
    key, unit = EVANS_KEY, "1"
    for gate in (_axial(ctx_like), acq.require_adult(meta)):
        if (fail := _gate_fail(key, unit, EVANS_METHOD, gate, ctx_like)) is not None:
            return fail
    if not frontal_horns.any():
        return _abstain(
            key,
            unit,
            EVANS_METHOD,
            AbstentionReason.QUALITY,
            "cornos frontais não segmentados",
            ctx_like,
        )
    if not cranial_cavity.any():
        return _abstain(
            key,
            unit,
            EVANS_METHOD,
            AbstentionReason.ACQUISITION,
            "encéfalo fora do campo",
            ctx_like,
        )
    if touches_border(cranial_cavity, axes=(1, 2)):
        return _abstain(
            key,
            unit,
            EVANS_METHOD,
            AbstentionReason.ACQUISITION,
            "crânio cortado no plano axial: largura intracraniana não mensurável",
            ctx_like,
        )
    info = axis_info(affine_lps)
    lr2d = info.lr_axis - 1  # eixo no corte (0 = linhas/y, 1 = colunas/x)
    spacing_lr = spacing_zyx[info.lr_axis]
    best: tuple[float, int, tuple[float, int, int, int]] | None = None
    for z in range(frontal_horns.shape[0]):
        if not frontal_horns[z].any():
            continue
        horns = _max_lr_width(frontal_horns[z], lr2d, spacing_lr)
        if best is None or horns[0] > best[0]:
            best = (horns[0], z, horns)
    assert best is not None
    _, z, horns = best
    skull = _max_lr_width(cranial_cavity[z], lr2d, spacing_lr)
    if skull[0] <= 0:
        return _abstain(
            key,
            unit,
            EVANS_METHOD,
            AbstentionReason.QUALITY,
            "cavidade craniana vazia no corte dos cornos frontais",
            ctx_like,
        )
    ratio = horns[0] / skull[0]

    def line(w: tuple[float, int, int, int], label: str) -> Primitive:
        _, row, a, b = w
        if lr2d == 1:
            pts = ((float(a), float(row)), (float(b), float(row)))
        else:
            pts = ((float(row), float(a)), (float(row), float(b)))
        return Primitive(type="line", xy=pts, frame=z, label=label)

    return RawMeasurement(
        key=key,
        value=round(ratio, 4),
        unit=unit,
        method=EVANS_METHOD,
        geometry=[
            line(horns, f"cornos frontais {horns[0]:.1f} mm"),
            line(skull, f"largura intracraniana {skull[0]:.1f} mm"),
        ],
        derived_from=[source],
        preconditions={
            **acq.acquisition_facts(meta),
            "frontal_horn_width_mm": f"{horns[0]:.1f}",
            "intracranial_width_mm": f"{skull[0]:.1f}",
            "plane": "axial da aquisição (alinhamento AC-PC não verificado)",
        },
        note=(
            "Ventriculomegalia pelo índice de Evans, não 'hidrocefalia' nem 'HPN' (atrofia dá o "
            "mesmo valor). Craniectomia, derivação, efeito de massa e desvio de linha média "
            "não são detectados e invalidam a medida. Largura intracraniana aproximada pela "
            "máscara 'brain'."
        ),
    )
