"""Bundle MONAI ``breast_density_classification`` (InceptionV3, 4 classes A-D).

Segunda opinião de densidade, por imagem, só para o modo pesquisa (o próprio
bundle declara "This is an example, not to be used for diagnostic purposes").

Fidelidade ao bundle (``configs/inference.json``)
-------------------------------------------------
* Rede: ``TorchVisionFCModel("inception_v3", num_classes=4, pool=None,
  use_conv=False, bias=True, pretrained=True)``. Com ``pretrained=True`` o
  torchvision força ``transform_input=True``; aqui a mesma arquitetura é
  montada direto no torchvision (sem MONAI em tempo de execução) e o
  ``state_dict`` do ``model.pt`` entra com ``strict=True`` (prefixo
  ``features.`` removido).
* Pré-processamento: ``LoadImaged`` (PIL, ``reverse_indexing=True`` ⇒ imagem
  TRANSPOSTA ``(W, H, 3)``), ``EnsureChannelFirstd``, ``ScaleIntensityd``
  (min-max da imagem inteira para ``[0, 1]``), ``Resized`` 299x299 (modo
  ``area``).
* Pós-processamento: ``Activationsd(sigmoid=True)`` — sigmoide POR CLASSE
  (não softmax); a classe é o argmax.

A entrada do bundle é JPEG 8 bits RGB. Para DICOM, os 8 bits vêm da mesma
conversão VOI + min-max do mammoscreen (:func:`mg_common.to_uint8`); o bundle
não especifica a conversão de DICOM — registrado como limitação no manifesto.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from radiologyai.analysis.adapters.mg_common import (
    MG_EXTRA,
    mg_applicability,
    role_inputs,
    to_uint8,
)
from radiologyai.analysis.adapters.xr_common import read_verified, require_backend
from radiologyai.analysis.base import AdapterOutput, ModelAdapter, ModelInputs, ResolvedWeights
from radiologyai.analysis.types import QAReport
from radiologyai.errors import InferenceError, ManifestError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from radiologyai.analysis.manifest import ModelManifest

__all__ = ["DENSITY_CLASSES", "MonaiBreastDensityAdapter", "build_inception", "monai_input"]

DENSITY_CLASSES: tuple[str, ...] = ("A", "B", "C", "D")  # metadata.json do bundle
INPUT_SIZE = 299


def build_inception(state: dict[str, Any]) -> torch.nn.Module:
    """InceptionV3 do torchvision com o ``state_dict`` do bundle (``strict=True``)."""
    import torch
    import torchvision

    net = torchvision.models.inception_v3(
        weights=None, aux_logits=True, transform_input=True, init_weights=False
    )
    net.fc = torch.nn.Linear(2048, len(DENSITY_CLASSES))
    renamed = {
        (k[len("features.") :] if k.startswith("features.") else k): v for k, v in state.items()
    }
    net.load_state_dict(renamed, strict=True)
    module: torch.nn.Module = net.eval()
    return module


def monai_input(img8: npt.NDArray[np.uint8]) -> torch.Tensor:
    """``[1, 3, 299, 299]`` como as transformações do bundle (ver módulo)."""
    import numpy as np
    import torch
    import torch.nn.functional as F  # noqa: N812 - convenção do PyTorch

    rgb = np.repeat(img8.astype(np.float32)[:, :, None], 3, axis=2)
    rgb = np.ascontiguousarray(np.moveaxis(rgb, 0, 1))  # PILReader reverse_indexing=True
    t = torch.from_numpy(rgb).permute(2, 0, 1)
    lo, hi = t.min(), t.max()
    if float(hi - lo) <= 0.0:
        raise InferenceError("imagem constante: nada a analisar")
    t = (t - lo) / (hi - lo)
    out: torch.Tensor = F.interpolate(t[None], size=(INPUT_SIZE, INPUT_SIZE), mode="area")
    return out


class MonaiBreastDensityAdapter(ModelAdapter):
    """Densidade A-D por imagem (sigmoide por classe, como o bundle)."""

    weight_key = "model"
    label_key = "density"

    def __init__(self, manifest: ModelManifest) -> None:
        super().__init__(manifest)
        self._model: Any = None
        spec = manifest.label(self.label_key)
        if tuple(spec.classes or ()) != DENSITY_CLASSES or spec.score_kind != "sigmoid":
            raise ManifestError(f"{manifest.id}: 'density' deve ser A-D com score_kind sigmoid")

    def load(self, weights: ResolvedWeights, device: str) -> None:
        require_backend("torch", "torchvision", extra=MG_EXTRA)
        import io

        import torch

        state = torch.load(
            io.BytesIO(read_verified(weights, self.weight_key)),
            map_location="cpu",
            weights_only=True,
        )
        self._model = build_inception(state).to(torch.device(device))
        self.device = device
        self.loaded = True

    def unload(self) -> None:
        self._model = None
        self.loaded = False

    def applicability(self, inputs: ModelInputs) -> QAReport:
        return mg_applicability(self.manifest, inputs)

    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        import torch

        if self._model is None:
            raise InferenceError(f"{self.manifest.id}: modelo não carregado")
        selected = role_inputs(inputs)
        if len(selected.images) != 1:
            raise InferenceError(f"{self.manifest.id}: espera exatamente uma incidência")
        x = monai_input(to_uint8(selected.images[0]))
        device = next(self._model.parameters()).device
        with torch.inference_mode():
            logits = self._model(x.to(device))[0]
        probs = [float(v) for v in logits.sigmoid().cpu().tolist()]
        return AdapterOutput(
            class_scores={self.label_key: dict(zip(DENSITY_CLASSES, probs, strict=True))},
            extras={"role": next(iter(selected.roles))},
        )
