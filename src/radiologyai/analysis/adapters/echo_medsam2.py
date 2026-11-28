"""MedSAM2 (``MedSAM2_US_Heart.pt``): traçado ASSISTIDO da cavidade do VE + Simpson.

O que faz, exatamente
---------------------
1. Recebe um ou dois clipes apicais (papéis ``a4c`` obrigatório, ``a2c``
   opcional) e caixas desenhadas pelo usuário sobre a cavidade do VE nos
   quadros de fim de diástole (ED) e fim de sístole (ES). Convenção dos
   prompts: ``Primitive(type="bbox", frame=<índice do quadro>, label="a4c:ed")``
   (``a4c:es``, ``a2c:ed``, ``a2c:es``).
2. Para cada caixa, o MedSAM2 (SAM 2.1 Hiera-T a 512 px, ajustado em eco) roda
   no modo IMAGEM sobre aquele quadro — sem propagação temporal: na avaliação
   preliminar a propagação a partir do ED superestimou a cavidade em ES.
3. Máscara limpa (maior componente + buracos) → método dos discos
   (:mod:`.echo_simpson`): biplanar com A4C + A2C, senão monoplanar A4C.
4. Devolve as máscaras (mapa de rótulos por clipe, só nos quadros ED/ES), a
   área da cavidade por corte/instante e VDF, VSF e FE com a geometria usada
   (base, ápice, discos) para conferência visual.

Não decide disfunção: nenhum limiar de FE com origem verificada foi registrado
para este projeto, então o adaptador não declara achado derivado — a FE é uma
MEDIDA geométrica dos contornos exibidos. Sem espaçamento físico, volumes e FE
se abstêm (geometria ausente); nada é estimado em pixels.

A variante sem prompt (caixa por atlas, :func:`atlas_box`) existe só para a
avaliação (``scripts/predict_echo_camus.py``): ela depende de corte, quadros
ED/ES e setor conhecidos, e não é oferecida ao usuário.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from radiologyai.analysis.adapters.echo_simpson import (
    DiscGeometry,
    SimpsonError,
    biplane_volume,
    disc_geometry,
    ejection_fraction,
    single_plane_volume,
)
from radiologyai.analysis.adapters.us_common import bbox_array, clean_mask, frame_uint8
from radiologyai.analysis.base import (
    AdapterOutput,
    LabelMap,
    ModelAdapter,
    ModelInputs,
    RawMeasurement,
    ResolvedWeights,
)
from radiologyai.analysis.types import (
    AbstentionReason,
    ImageData,
    ImageKind,
    Primitive,
    QACheck,
    QAReport,
)
from radiologyai.errors import BackendUnavailableError, RadiologyAIError, UnsupportedInputError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = [
    "INSTANTS",
    "VIEWS",
    "MedSAM2EchoLVAdapter",
    "MedSAM2Segmenter",
    "atlas_box",
    "lv_measurements",
]

VIEWS = ("a4c", "a2c")
INSTANTS = ("ed", "es")
_FEAT_SIZES_512 = [(128, 128), (64, 64), (32, 32)]


class EchoPromptError(RadiologyAIError):
    """Prompts de eco ausentes, duplicados ou fora do clipe."""


def _require_torch() -> None:
    if importlib.util.find_spec("torch") is None:
        raise BackendUnavailableError(
            "torch não instalado; instale os extras: pip install 'radiologyai[ml,cpu]'"
        )


class MedSAM2Segmenter:
    """Preditor de imagem do MedSAM2 (uma caixa → uma máscara).

    Reutilizado pelo script de avaliação.
    """

    def __init__(self, checkpoint: Any, device: str = "cpu") -> None:
        _require_torch()
        from radiologyai.analysis.vendor.sam2.build import build_sam2_hiera_t512
        from radiologyai.analysis.vendor.sam2.sam2_image_predictor import SAM2ImagePredictor

        self.model = build_sam2_hiera_t512(checkpoint, device=device)
        predictor = SAM2ImagePredictor(self.model)
        # O preditor fixa tamanhos de 1024 px; o checkpoint do MedSAM2 usa 512 px.
        predictor._bb_feat_sizes = list(_FEAT_SIZES_512)
        self.predictor = predictor

    def segment(
        self, frame_u8: npt.NDArray[np.uint8], box: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.bool_], float]:
        """Máscara binária limpa e o IoU previsto pelo próprio modelo."""
        import numpy as np
        import torch

        rgb = np.repeat(frame_u8[..., None], 3, axis=2)
        with torch.inference_mode():
            self.predictor.set_image(rgb)
            masks, iou, _ = self.predictor.predict(
                box=np.asarray(box, dtype=np.float32), multimask_output=False
            )
        return clean_mask(np.asarray(masks[0]) > 0), float(np.asarray(iou).ravel()[0])


def atlas_box(
    frame_u8: npt.NDArray[np.uint8], normalized_box: tuple[float, float, float, float]
) -> npt.NDArray[np.float64]:
    """Caixa do VE por atlas: caixa normalizada ao retângulo do setor (pixels > 0).

    Só para avaliação: exige corte e instante conhecidos para escolher o atlas.
    """
    import numpy as np

    ys, xs = np.nonzero(frame_u8 > 0)
    if ys.size == 0:
        raise UnsupportedInputError("quadro sem setor de imagem")
    x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    w, h = x1 - x0, y1 - y0
    a, b, c, d = normalized_box
    return np.array([x0 + a * w, y0 + b * h, x0 + c * w, y0 + d * h], dtype=np.float64)


@dataclass(frozen=True)
class _Traced:
    view: str
    instant: str
    image_idx: int
    frame: int
    mask: npt.NDArray[np.bool_]
    iou: float
    spacing: tuple[float, float] | None


def _parse_prompts(inputs: ModelInputs) -> dict[tuple[str, str], Primitive]:
    found: dict[tuple[str, str], Primitive] = {}
    for p in inputs.prompts:
        if p.type != "bbox" or not p.label:
            continue
        parts = p.label.strip().lower().split(":")
        if len(parts) != 2 or parts[0] not in VIEWS or parts[1] not in INSTANTS:
            continue
        key = (parts[0], parts[1])
        if key in found:
            raise EchoPromptError(f"prompt duplicado para {parts[0]}:{parts[1]}")
        found[key] = p
    return found


def _geometry_primitives(geom: DiscGeometry, frame: int, tag: str) -> list[Primitive]:
    prims = [
        Primitive(type="line", xy=geom.base, frame=frame, label=f"{tag}:base_mitral"),
        Primitive(type="point", xy=(geom.apex,), frame=frame, label=f"{tag}:apex"),
    ]
    prims.extend(
        Primitive(type="line", xy=seg, frame=frame, label=f"{tag}:disc{k + 1}")
        for k, seg in enumerate(geom.disc_segments)
    )
    return prims


# Trava de plausibilidade (não é limiar diagnóstico): volumes de VE adulto fora desta
# faixa indicam contorno errado ou espaçamento físico errado, nunca um achado. Os limites
# são deliberadamente largos (bem além dos extremos fisiológicos relatados), para só
# barrar o que é fisicamente impossível; abaixo deles o adaptador se abstém.
EDV_PLAUSIBLE_ML = (20.0, 600.0)


def _implausible_volumes(edv_ml: float, esv_ml: float) -> str | None:
    """Motivo legível se VDF/VSF são fisicamente implausíveis; ``None`` se plausíveis."""
    lo, hi = EDV_PLAUSIBLE_ML
    if not (lo <= edv_ml <= hi):
        return (
            f"VDF {edv_ml:.1f} mL fora da faixa de plausibilidade [{lo:.0f}, {hi:.0f}] mL: "
            "contorno ou espaçamento inconsistente; revise as caixas/contornos"
        )
    if esv_ml > hi:
        return f"VSF {esv_ml:.1f} mL acima de {hi:.0f} mL: contorno inconsistente"
    return None


def lv_measurements(
    traced: dict[tuple[str, str], tuple[npt.NDArray[np.bool_], tuple[float, float] | None, int]],
    *,
    image_idx: dict[str, int] | None = None,
    length_convention: Literal["per_instant", "shared_max"] = "per_instant",
) -> list[RawMeasurement]:
    """Medidas do VE a partir de máscaras ED/ES por corte.

    ``traced[(view, instant)] = (máscara, espaçamento (linha, coluna) ou None, quadro)``.
    Exige ``a4c`` ED e ES; ``a2c`` ED e ES (ambos) tornam o cálculo biplanar.
    Abstém-se por geometria (sem espaçamento) ou qualidade (máscara vazia /
    sem base reta), sempre com o motivo.
    """
    idx = image_idx or {"a4c": 0, "a2c": 1}
    out: list[RawMeasurement] = []
    geoms: dict[tuple[str, str], DiscGeometry] = {}
    problems: list[str] = []
    for (view, inst), (mask, spacing, frame) in sorted(traced.items()):
        key = f"lv_cavity_area_{view}_{inst}"
        method = f"MedSAM2 (caixa do usuário) {view.upper()} {inst.upper()}, quadro {frame}"
        if spacing is None:
            out.append(
                RawMeasurement(
                    key=key,
                    value=None,
                    unit="mm2",
                    method=method,
                    abstention=AbstentionReason.GEOMETRY,
                    note="imagem sem espaçamento físico",
                    image_idx=idx[view],
                )
            )
            problems.append(f"{view}:{inst} sem espaçamento")
            continue
        area = float(mask.sum()) * spacing[0] * spacing[1]
        if area <= 0:
            out.append(
                RawMeasurement(
                    key=key,
                    value=None,
                    unit="mm2",
                    method=method,
                    abstention=AbstentionReason.QUALITY,
                    note="o modelo não segmentou cavidade na caixa",
                    image_idx=idx[view],
                )
            )
            problems.append(f"{view}:{inst} máscara vazia")
            continue
        try:
            geom = disc_geometry(mask, spacing)
        except SimpsonError as exc:
            out.append(
                RawMeasurement(
                    key=key,
                    value=area,
                    unit="mm2",
                    method=method,
                    note=f"método dos discos inaplicável: {exc}",
                    image_idx=idx[view],
                )
            )
            problems.append(f"{view}:{inst} {exc}")
            continue
        geoms[(view, inst)] = geom
        out.append(
            RawMeasurement(
                key=key,
                value=area,
                unit="mm2",
                method=method,
                geometry=_geometry_primitives(geom, frame, f"{view}:{inst}"),
                preconditions={"spacing_mm": f"{spacing[0]:.4f}x{spacing[1]:.4f}"},
                image_idx=idx[view],
            )
        )

    biplane = all((v, i) in geoms for v in VIEWS for i in INSTANTS)
    a4c_ok = all(("a4c", i) in geoms for i in INSTANTS)
    volumes: dict[str, float] = {}
    method = ""
    shared: float | None = None
    if length_convention == "shared_max" and (biplane or a4c_ok):
        views_used = VIEWS if biplane else ("a4c",)
        shared = max(geoms[(v, i)].long_axis_mm for v in views_used for i in INSTANTS)
    if biplane:
        for inst in INSTANTS:
            res = biplane_volume(geoms[("a4c", inst)], geoms[("a2c", inst)], length_mm=shared)
            volumes[inst] = res.volume_ml
            method = res.method
    elif a4c_ok:
        for inst in INSTANTS:
            res = single_plane_volume(geoms[("a4c", inst)], length_mm=shared)
            volumes[inst] = res.volume_ml
            method = res.method + " (A4C)"
    if shared is not None:
        method += "; L comum = maior eixo entre cortes e instantes (convenção CAMUS)"
    reason = (
        AbstentionReason.GEOMETRY
        if any("espaçamento" in p for p in problems)
        else AbstentionReason.QUALITY
    )
    note = "; ".join(problems) if problems else None
    derived = [f"{v}:{i}" for v in VIEWS for i in INSTANTS if (v, i) in geoms]
    pre = {"views": "A4C+A2C" if biplane else "A4C", "prompt": "caixa do usuário por quadro"}
    for inst, key in (("ed", "lv_edv"), ("es", "lv_esv")):
        value = volumes.get(inst)
        out.append(
            RawMeasurement(
                key=key,
                value=value,
                unit="mL",
                method=method or "Simpson (método dos discos)",
                abstention=None if value is not None else reason,
                note=None if value is not None else note,
                preconditions=pre,
                derived_from=["lv_cavity"],
                image_idx=idx["a4c"],
            )
        )
    ef: float | None = None
    ef_note = note
    if "ed" in volumes and "es" in volumes:
        implausible = _implausible_volumes(volumes["ed"], volumes["es"])
        if implausible is not None:
            reason = AbstentionReason.QUALITY
            ef_note = implausible
        elif volumes["es"] >= volumes["ed"]:
            # Fisicamente incoerente (FE <= 0): contorno errado em algum quadro.
            reason = AbstentionReason.QUALITY
            ef_note = (
                f"VSF ({volumes['es']:.1f} mL) >= VDF ({volumes['ed']:.1f} mL): "
                "contornos inconsistentes; revise as caixas/contornos"
            )
        else:
            try:
                ef = ejection_fraction(volumes["ed"], volumes["es"])
            except SimpsonError as exc:
                ef_note = str(exc)
    out.append(
        RawMeasurement(
            key="lv_ef",
            value=ef,
            unit="%",
            method=(method or "Simpson (método dos discos)") + "; FE = (VDF − VSF)/VDF",
            abstention=None if ef is not None else reason,
            note=None if ef is not None else ef_note,
            preconditions={**pre, "traced": ",".join(derived)},
            derived_from=["lv_cavity"],
            image_idx=idx["a4c"],
        )
    )
    return out


class MedSAM2EchoLVAdapter(ModelAdapter):
    """Traçado assistido do VE (MedSAM2) + Simpson. Rota interativa, modo pesquisa."""

    segmenter: MedSAM2Segmenter | None = None

    def load(self, weights: ResolvedWeights, device: str) -> None:
        self.segmenter = MedSAM2Segmenter(weights.path("checkpoint"), device=device)
        self.device = device
        self.loaded = True

    def unload(self) -> None:
        self.segmenter = None
        self.loaded = False

    def applicability(self, inputs: ModelInputs) -> QAReport:
        base = super().applicability(inputs)
        checks = list(base.checks)
        reasons = list(base.reasons)
        try:
            prompts = _parse_prompts(inputs)
        except EchoPromptError as exc:
            checks.append(QACheck(name="echo_prompts", passed=False, detail=str(exc)))
            reasons.append(AbstentionReason.INPUT)
            prompts = {}
        missing = [f"a4c:{i}" for i in INSTANTS if ("a4c", i) not in prompts]
        checks.append(
            QACheck(
                name="a4c_ed_es_boxes",
                passed=not missing,
                value=",".join(missing) or None,
                detail="caixas do VE em ED e ES do corte A4C são obrigatórias",
            )
        )
        if missing:
            reasons.append(AbstentionReason.MISSING_INPUT)
        for (view, inst), p in prompts.items():
            if not inputs.has_role(view):
                continue
            image = inputs.by_role(view)
            n = int(image.pixels.shape[0]) if image.kind is ImageKind.CINE else 1
            ok = 0 <= p.frame < n
            checks.append(
                QACheck(name=f"frame[{view}:{inst}]", passed=ok, value=float(p.frame), threshold=n)
            )
            if not ok:
                reasons.append(AbstentionReason.INPUT)
        if reasons:
            return QAReport.reject(*dict.fromkeys(reasons), checks=checks)
        return QAReport.ok(checks)

    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        import numpy as np

        if self.segmenter is None:
            raise RuntimeError("MedSAM2EchoLVAdapter.predict antes de load()")
        prompts = _parse_prompts(inputs)
        traced: dict[
            tuple[str, str], tuple[npt.NDArray[np.bool_], tuple[float, float] | None, int]
        ] = {}
        labelmaps: dict[str, npt.NDArray[np.uint8]] = {}
        image_idx: dict[str, int] = {}
        warnings: list[str] = [
            "FE e volumes são medidas geométricas dos contornos exibidos, traçados a partir "
            "das caixas do usuário; confira e corrija os contornos antes de usar."
        ]
        for view in VIEWS:
            pair = [prompts.get((view, i)) for i in INSTANTS]
            if not inputs.has_role(view) or any(p is None for p in pair):
                if view == "a2c" and inputs.has_role("a2c"):
                    warnings.append("A2C sem caixas em ED e ES: cálculo monoplanar (A4C)")
                continue
            image: ImageData = inputs.by_role(view)
            idx = inputs.roles[view]
            image_idx[view] = idx
            spacing = _spacing2(image)
            n = int(image.pixels.shape[0]) if image.kind is ImageKind.CINE else 1
            lm = np.zeros((n, *image.plane_shape), dtype=np.uint8)
            for inst, prompt in zip(INSTANTS, pair, strict=True):
                assert prompt is not None
                frame = frame_uint8(image, prompt.frame)
                mask, iou = self.segmenter.segment(frame, bbox_array(prompt, frame.shape))
                traced[(view, inst)] = (mask, spacing, prompt.frame)
                lm[prompt.frame][mask] = 1
                warnings.append(f"{view}:{inst} IoU previsto pelo modelo {iou:.3f}")
            labelmaps[view] = lm
        measurements = lv_measurements(
            traced, image_idx={"a4c": image_idx.get("a4c", 0), "a2c": image_idx.get("a2c", 0)}
        )
        out_maps = [
            LabelMap(
                name=f"lv_cavity_{view}",
                array=(arr if inputs.images[image_idx[view]].kind is ImageKind.CINE else arr[0]),
                labels={1: "lv_cavity"},
                image_idx=image_idx[view],
            )
            for view, arr in labelmaps.items()
        ]
        for m in measurements:
            if m.derived_from == ["lv_cavity"]:
                m.derived_from = [lm.name for lm in out_maps]
        return AdapterOutput(labelmaps=out_maps, measurements=measurements, warnings=warnings)


def _spacing2(image: ImageData) -> tuple[float, float] | None:
    if image.spacing_mm is None or len(image.spacing_mm) < 2:
        return None
    return float(image.spacing_mm[-2]), float(image.spacing_mm[-1])
