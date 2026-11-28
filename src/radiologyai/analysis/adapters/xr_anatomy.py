"""Anatomia do tórax (PSPNet ChestX-Det, via TorchXRayVision) e índice cardiotorácico.

O segmentador devolve 14 estruturas (sigmoide por estrutura, podem se
sobrepor). Este adaptador publica as máscaras de coração e pulmões na geometria
da imagem de origem e deriva delas o índice cardiotorácico (ICT):

* largura do coração = maior extensão horizontal da máscara do coração;
* largura do tórax = maior distância horizontal, numa mesma linha, entre a borda
  externa do pulmão direito e a do pulmão esquerdo (diâmetro torácico interno);
* ICT = coração / tórax, com as duas linhas devolvidas como geometria.

Abstenções (nunca número inventado):

* máscara vazia ou pequena demais (``low_image_quality``);
* pulmão tocando a borda lateral do recorte — tórax cortado (``missing_geometry``);
* pulmões em posição incoerente para incidência frontal, ou nenhuma linha com os
  dois pulmões (``missing_geometry``);
* incidência não frontal (portões estático e de conteúdo).

O limiar de diretriz (ICT > 0,5) só vale em PA: a medida ``cardiothoracic_ratio``
é publicada em PA e AP, mas ``cardiothoracic_ratio_pa`` — a que o rótulo derivado
limiariza — se abstém (``not_applicable_view``) fora de PA.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from radiologyai.analysis.adapters.xr_common import (
    center_crop_box,
    load_torch_verified,
    require_backend,
    xrv_input,
)
from radiologyai.analysis.adapters.xr_xrv import gate_checks
from radiologyai.analysis.base import (
    AdapterOutput,
    LabelMap,
    ModelAdapter,
    ModelInputs,
    RawMeasurement,
    ResolvedWeights,
)
from radiologyai.analysis.types import AbstentionReason, Primitive, QAReport
from radiologyai.errors import InferenceError, ManifestError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from radiologyai.analysis.manifest import ModelManifest

    BoolArray = npt.NDArray[np.bool_]

__all__ = [
    "PSPNET_TARGETS",
    "ChestXDetAnatomyAdapter",
    "CtrResult",
    "build_pspnet",
    "cardiothoracic_ratio",
]

PSPNET_TARGETS = (
    "Left Clavicle",
    "Right Clavicle",
    "Left Scapula",
    "Right Scapula",
    "Left Lung",
    "Right Lung",
    "Left Hilus Pulmonis",
    "Right Hilus Pulmonis",
    "Heart",
    "Aorta",
    "Facies Diaphragmatica",
    "Mediastinum",
    "Weasand",
    "Spine",
)
_RES = 512
_MIN_HEART_FRAC = 0.01
_MIN_LUNG_FRAC = 0.03
_BORDER_PX = 2  # pulmão a <= 2 px da borda lateral do recorte => tórax cortado


@dataclass(frozen=True, slots=True)
class CtrResult:
    """ICT na grade da máscara; ``value=None`` exige ``reason``."""

    value: float | None
    reason: AbstentionReason | None
    detail: str
    heart_line: tuple[tuple[float, float], tuple[float, float]] | None = None
    thorax_line: tuple[tuple[float, float], tuple[float, float]] | None = None
    heart_width_px: int | None = None
    thorax_width_px: int | None = None


def cardiothoracic_ratio(
    heart: BoolArray, left_lung: BoolArray, right_lung: BoolArray
) -> CtrResult:
    """ICT a partir de máscaras binárias na mesma grade (incidência frontal, PA/AP).

    Convenção radiológica: o pulmão DIREITO do paciente aparece à ESQUERDA da
    imagem. Linhas em coordenadas ``(x=coluna, y=linha)`` da grade das máscaras.
    """
    import numpy as np

    h, w = heart.shape
    area = float(h * w)
    if heart.sum() < _MIN_HEART_FRAC * area:
        return CtrResult(None, AbstentionReason.QUALITY, "máscara do coração vazia ou pequena")
    if left_lung.sum() < _MIN_LUNG_FRAC * area or right_lung.sum() < _MIN_LUNG_FRAC * area:
        return CtrResult(None, AbstentionReason.QUALITY, "máscara de pulmão vazia ou pequena")
    lungs = left_lung | right_lung
    cols = np.flatnonzero(lungs.any(axis=0))
    if cols[0] <= _BORDER_PX or cols[-1] >= w - 1 - _BORDER_PX:
        return CtrResult(
            None, AbstentionReason.GEOMETRY, "pulmão toca a borda lateral: tórax cortado"
        )
    r_cx = float(np.flatnonzero(right_lung.any(axis=0)).mean())
    l_cx = float(np.flatnonzero(left_lung.any(axis=0)).mean())
    if not r_cx < l_cx:
        return CtrResult(
            None,
            AbstentionReason.GEOMETRY,
            "pulmões em posição incoerente com incidência frontal (direito deve ficar à "
            "esquerda da imagem)",
        )

    best_row, best_width, best_x0, best_x1 = -1, 0, 0, 0
    for y in range(h):
        r_cols = np.flatnonzero(right_lung[y])
        l_cols = np.flatnonzero(left_lung[y])
        if r_cols.size == 0 or l_cols.size == 0:
            continue
        x0, x1 = int(r_cols[0]), int(l_cols[-1])
        if x1 - x0 + 1 > best_width:
            best_row, best_width, best_x0, best_x1 = y, x1 - x0 + 1, x0, x1
    if best_row < 0:
        return CtrResult(None, AbstentionReason.GEOMETRY, "nenhuma linha contém os dois pulmões")

    heart_cols = np.flatnonzero(heart.any(axis=0))
    hx0, hx1 = int(heart_cols[0]), int(heart_cols[-1])
    heart_width = hx1 - hx0 + 1
    heart_rows = np.flatnonzero(heart.any(axis=1))
    hy = float(heart_rows.mean())
    value = heart_width / best_width
    return CtrResult(
        value=float(value),
        reason=None,
        detail="ok",
        heart_line=((float(hx0), hy), (float(hx1), hy)),
        thorax_line=((float(best_x0), float(best_row)), (float(best_x1), float(best_row))),
        heart_width_px=heart_width,
        thorax_width_px=best_width,
    )


def build_pspnet(state: Any) -> torch.nn.Module:
    """PSPNet ChestX-Det do xrv sem o ``__init__`` da biblioteca (que baixa pesos)."""
    import torch
    import torchvision
    from torchxrayvision.baseline_models.chestx_det import PSPNet
    from torchxrayvision.baseline_models.chestx_det.ptsemseg.pspnet import pspnet

    model = PSPNet.__new__(PSPNet)
    torch.nn.Module.__init__(model)
    model.transform = torchvision.transforms.Compose(
        [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    net = pspnet(len(PSPNET_TARGETS))
    converted = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state.items())
    net.load_state_dict(converted)
    net.eval()
    model.model = net
    module: torch.nn.Module = model
    module.eval()
    return module


def _to_source(
    mask: BoolArray, height: int, width: int, y0: int, x0: int, side: int
) -> npt.NDArray[np.bool_]:
    """Máscara da grade 512 do recorte → geometria completa da imagem (vizinho mais próximo)."""
    import numpy as np

    idx = np.minimum(((np.arange(side) + 0.5) * mask.shape[0] / side).astype(np.int64), _RES - 1)
    out = np.zeros((height, width), dtype=bool)
    out[y0 : y0 + side, x0 : x0 + side] = mask[np.ix_(idx, idx)]
    return out


class ChestXDetAnatomyAdapter(ModelAdapter):
    """Segmentação anatômica + ICT, com abstenção explícita."""

    weight_key = "model"

    def __init__(self, manifest: ModelManifest) -> None:
        super().__init__(manifest)
        self._model: Any = None
        self.gate_ids: tuple[str, ...] = tuple(manifest.config.get("gates", ()))
        missing = [g for g in self.gate_ids if g not in manifest.depends_on]
        if missing:
            raise ManifestError(f"{manifest.id}: portões {missing} não estão em depends_on")

    def load(self, weights: ResolvedWeights, device: str) -> None:
        require_backend("torch", "torchvision", "torchxrayvision", "skimage", extra="ml")
        import torch

        state = load_torch_verified(weights, self.weight_key, allow_pickle=False)
        self._model = build_pspnet(state).to(torch.device(device))

    def unload(self) -> None:
        self._model = None

    def applicability(self, inputs: ModelInputs) -> QAReport:
        base = super().applicability(inputs)
        if not base.applicable or not self.gate_ids:
            return base
        checks, reasons = gate_checks(inputs, self.gate_ids)
        all_checks = (*base.checks, *checks)
        if reasons:
            return QAReport.reject(*dict.fromkeys(reasons), checks=all_checks)
        return QAReport.ok(all_checks)

    def segment(self, plane: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        """Máscaras ``[14, 512, 512]`` (sigmoide > 0,5, regra do exemplo da biblioteca)."""
        import torch

        if self._model is None:
            raise InferenceError(f"{self.manifest.id}: modelo não carregado")
        device = next(self._model.parameters()).device
        with torch.inference_mode():
            logits = self._model(torch.from_numpy(plane)[None, None].to(device))[0]
        masks: npt.NDArray[np.bool_] = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        return masks

    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        import numpy as np

        image = inputs.single
        masks = self.segment(xrv_input(image, _RES))
        heart = masks[PSPNET_TARGETS.index("Heart")]
        left = masks[PSPNET_TARGETS.index("Left Lung")]
        right = masks[PSPNET_TARGETS.index("Right Lung")]
        height, width = image.plane_shape
        y0, x0, side = center_crop_box(height, width)
        scale = side / _RES

        def src(pt: tuple[float, float]) -> tuple[float, float]:
            return (x0 + (pt[0] + 0.5) * scale - 0.5, y0 + (pt[1] + 0.5) * scale - 0.5)

        heart_map = _to_source(heart, height, width, y0, x0, side).astype(np.uint8)
        lungs_map = np.zeros((height, width), dtype=np.uint8)
        lungs_map[_to_source(left, height, width, y0, x0, side)] = 1
        lungs_map[_to_source(right, height, width, y0, x0, side)] = 2
        labelmaps = [
            LabelMap(name="heart", array=heart_map, labels={1: "heart"}),
            LabelMap(name="lungs", array=lungs_map, labels={1: "left_lung", 2: "right_lung"}),
        ]

        ctr = cardiothoracic_ratio(heart, left, right)
        view = (image.meta.view_position or "").upper()
        method = "PSPNet ChestX-Det: largura máxima do coração / maior diâmetro torácico interno"
        geometry: list[Primitive] = []
        if ctr.heart_line is not None and ctr.thorax_line is not None:
            geometry = [
                Primitive(type="line", xy=tuple(src(p) for p in ctr.heart_line), label="heart"),
                Primitive(type="line", xy=tuple(src(p) for p in ctr.thorax_line), label="thorax"),
            ]
        pre = {"view_position": view or "desconhecida"}
        note = None
        if ctr.value is not None:
            note = (
                f"larguras na grade {_RES}px: coração {ctr.heart_width_px}, "
                f"tórax {ctr.thorax_width_px}"
            )
        measurements = [
            RawMeasurement(
                key="cardiothoracic_ratio",
                value=ctr.value,
                unit="1",
                method=method,
                geometry=geometry,
                derived_from=["heart", "lungs"],
                preconditions=pre,
                abstention=ctr.reason,
                note=note if ctr.value is not None else ctr.detail,
            )
        ]
        if ctr.value is not None and view == "PA":
            measurements.append(
                RawMeasurement(
                    key="cardiothoracic_ratio_pa",
                    value=ctr.value,
                    unit="1",
                    method=method + " (somente PA)",
                    geometry=geometry,
                    derived_from=["heart", "lungs"],
                    preconditions=pre,
                    note=note,
                )
            )
        else:
            measurements.append(
                RawMeasurement(
                    key="cardiothoracic_ratio_pa",
                    value=None,
                    unit="1",
                    method=method + " (somente PA)",
                    derived_from=["heart", "lungs"],
                    preconditions=pre,
                    abstention=ctr.reason if ctr.value is None else AbstentionReason.VIEW,
                    note=ctr.detail
                    if ctr.value is None
                    else "ICT só é interpretável contra 0,5 em PA; incidência "
                    f"{view or 'desconhecida'}",
                )
            )
        return AdapterOutput(
            labelmaps=labelmaps,
            measurements=measurements,
            extras={"image_sha256": image.sha256, "ctr_detail": ctr.detail},
        )
