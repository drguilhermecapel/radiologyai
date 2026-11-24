"""``ianpan/mammo-crop`` + ``ianpan/mammoscreen``: escore de câncer e densidade por mama.

Dois adaptadores, dois manifestos:

* :class:`MammoCropAdapter` (``mammo-crop``, rota ``dependency``): para cada
  imagem com papel ``L-CC``/``L-MLO``/``R-CC``/``R-MLO``, a caixa ``xywh`` da
  mama prevista pela MobileNetV3 do original. Publica as caixas em
  ``extras["crop_boxes"]`` (chave = sha256 do conteúdo da imagem). Não emite
  achado.
* :class:`MammoscreenAdapter` (``mammoscreen``): recorta cada incidência com a
  caixa a montante e roda o ensemble de 3 EfficientNetV2-S. Por mama presente
  emite ``malignancy_left``/``malignancy_right`` (sigmoide, média sobre as
  incidências e depois sobre as redes — exatamente como o original) e
  ``density_left``/``density_right`` (softmax de 4 classes, ordem A/B/C/D da
  codificação 0/1/2/3 do repositório de treino).

Nenhum limiar é aplicado aqui. Os limiares do cartão do modelo foram escolhidos
no hold-out interno do RSNA e NÃO entram como ponto de operação; a decisão fica
``score_only`` até haver medição própria (ver manifesto).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from radiologyai.analysis.adapters.mg_common import (
    breast_views,
    load_safetensors_verified,
    mg_applicability,
    require_mg_backend,
    role_inputs,
    to_uint8,
)
from radiologyai.analysis.base import (
    AdapterOutput,
    ModelAdapter,
    ModelInputs,
    ResolvedWeights,
    Score,
)
from radiologyai.analysis.types import AbstentionReason, QACheck, QAReport
from radiologyai.errors import InferenceError, ManifestError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from radiologyai.analysis.manifest import ModelManifest

__all__ = [
    "CROP_MODEL_ID",
    "DENSITY_CLASSES",
    "MammoCropAdapter",
    "MammoscreenAdapter",
    "apply_crop",
]

CROP_MODEL_ID = "mammo-crop"
#: codificação do repositório de treino: "encodes density A/B/C/D as 0/1/2/3"
DENSITY_CLASSES: tuple[str, ...] = ("A", "B", "C", "D")
_SIDE = {"L": "left", "R": "right"}


def apply_crop(
    img: npt.NDArray[np.uint8], box: tuple[int, int, int, int]
) -> npt.NDArray[np.uint8]:
    """``img[y:y+h, x:x+w]`` como no cartão do modelo.

    Raises:
        InferenceError: caixa vazia (o recorte não encontrou a mama).
    """
    x, y, w, h = box
    crop = img[max(y, 0) : y + h, max(x, 0) : x + w]
    if crop.shape[0] < 2 or crop.shape[1] < 2:
        raise InferenceError(f"recorte vazio {box} para imagem {img.shape}: mama não localizada")
    return crop


class MammoCropAdapter(ModelAdapter):
    """Caixa da mama (``ianpan/mammo-crop``) para cada incidência com papel."""

    weight_key = "model"

    def __init__(self, manifest: ModelManifest) -> None:
        super().__init__(manifest)
        self._model: Any = None

    def load(self, weights: ResolvedWeights, device: str) -> None:
        require_mg_backend()
        import torch

        from radiologyai.analysis.vendor.mammoscreen.model import MammoCropNet

        model = MammoCropNet()
        model.load_state_dict(load_safetensors_verified(weights, self.weight_key), strict=True)
        self._model = model.eval().to(torch.device(device))
        self.device = device
        self.loaded = True

    def unload(self) -> None:
        self._model = None
        self.loaded = False

    def applicability(self, inputs: ModelInputs) -> QAReport:
        return mg_applicability(self.manifest, inputs)

    def boxes(self, inputs: ModelInputs) -> dict[str, tuple[int, int, int, int]]:
        """``{papel: (x, y, w, h)}`` das imagens com papel."""
        if self._model is None:
            raise InferenceError(f"{self.manifest.id}: modelo não carregado")
        selected = role_inputs(inputs)
        return {
            role: self._model.crop_box(to_uint8(selected.images[idx]))
            for role, idx in selected.roles.items()
        }

    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        selected = role_inputs(inputs)
        boxes = self.boxes(selected)
        by_sha = {
            selected.images[selected.roles[role]].sha256: list(box) for role, box in boxes.items()
        }
        return AdapterOutput(
            extras={
                "crop_boxes": by_sha,
                "crop_boxes_by_role": {role: list(box) for role, box in boxes.items()},
            }
        )


class MammoscreenAdapter(ModelAdapter):
    """Ensemble ``ianpan/mammoscreen`` por mama, sobre incidências recortadas."""

    weight_key = "model"

    def __init__(self, manifest: ModelManifest) -> None:
        super().__init__(manifest)
        self._model: Any = None
        if CROP_MODEL_ID not in manifest.depends_on:
            raise ManifestError(f"{manifest.id}: depends_on deve incluir {CROP_MODEL_ID!r}")
        declared = {spec.key for spec in manifest.labels}
        needed = {f"{kind}_{side}" for kind in ("malignancy", "density") for side in _SIDE.values()}
        if declared != needed:
            raise ManifestError(
                f"{manifest.id}: rótulos declarados {sorted(declared)} != {sorted(needed)}"
            )
        for side in _SIDE.values():
            classes = manifest.label(f"density_{side}").classes
            if tuple(classes or ()) != DENSITY_CLASSES:
                raise ManifestError(f"{manifest.id}: density_{side} deve declarar A, B, C, D")

    def load(self, weights: ResolvedWeights, device: str) -> None:
        require_mg_backend()
        import torch

        from radiologyai.analysis.vendor.mammoscreen.model import MammoEnsemble

        model = MammoEnsemble()
        model.load_state_dict(load_safetensors_verified(weights, self.weight_key), strict=True)
        self._model = model.eval().to(torch.device(device))
        self.device = device
        self.loaded = True

    def unload(self) -> None:
        self._model = None
        self.loaded = False

    @staticmethod
    def _crop_boxes(inputs: ModelInputs) -> Mapping[str, Any]:
        output = inputs.upstream.get(CROP_MODEL_ID)
        boxes = None if output is None else output.extras.get("crop_boxes")
        return boxes if isinstance(boxes, Mapping) else {}

    def applicability(self, inputs: ModelInputs) -> QAReport:
        base = mg_applicability(self.manifest, inputs)
        if not base.applicable:
            return base
        selected = role_inputs(inputs)
        boxes = self._crop_boxes(selected)
        missing = [
            role for role, idx in selected.roles.items() if selected.images[idx].sha256 not in boxes
        ]
        check = QACheck(
            name=f"upstream:{CROP_MODEL_ID}",
            passed=not missing,
            value=",".join(missing) or None,
            detail="incidências sem caixa de recorte a montante" if missing else None,
        )
        if missing:
            return QAReport.reject(AbstentionReason.UPSTREAM, checks=(*base.checks, check))
        return QAReport.ok((*base.checks, check))

    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        if self._model is None:
            raise InferenceError(f"{self.manifest.id}: modelo não carregado")
        selected = role_inputs(inputs)
        boxes = self._crop_boxes(selected)
        output = AdapterOutput()
        per_net: dict[str, list[float]] = {}
        used: dict[str, list[str]] = {}
        for lat, views in sorted(breast_views(selected).items()):
            crops = []
            for _view, idx in views:
                image = selected.images[idx]
                box = boxes.get(image.sha256)
                if box is None:
                    raise InferenceError(f"{self.manifest.id}: caixa de recorte ausente ({lat})")
                x, y, w, h = (int(v) for v in box)
                crops.append(apply_crop(to_uint8(image), (x, y, w, h)))
            cancer, density, nets = self._model.breast(crops)
            side = _SIDE[lat]
            output.scores.append(Score(label=f"malignancy_{side}", value=cancer, kind="sigmoid"))
            output.class_scores[f"density_{side}"] = dict(
                zip(DENSITY_CLASSES, density, strict=True)
            )
            per_net[side] = nets
            used[side] = [view for view, _ in views]
            if len(views) == 1:
                output.warnings.append(
                    f"mama {side}: só a incidência {views[0][0]}; o cartão do modelo informa "
                    "desempenho reduzido com incidência única"
                )
        output.extras.update({"per_net_cancer": per_net, "views": used})
        return output
