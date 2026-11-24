"""Portões de conteúdo para RX de tórax: incidência (frontal/lateral) e fora de distribuição.

Os portões NÃO produzem achados. Cada um roda um modelo real sobre a imagem e
publica a decisão em ``AdapterOutput.extras["gate"]``; os classificadores que os
declaram em ``depends_on`` (``config.gates``) leem essa decisão na sua
aplicabilidade e se abstêm com o motivo correspondente.

* :class:`XrvViewGateAdapter` — ResNet50 ``xinario/chestViewSplit`` distribuído
  pelo TorchXRayVision. Reprova quando o modelo classifica a imagem como lateral
  (``argmax`` das duas saídas, a regra de decisão do próprio modelo), mesmo que o
  cabeçalho diga PA/AP: metadado errado não passa em silêncio.
* :class:`XrvAutoencoderGateAdapter` — autoencoder ResNet101 ``101-elastic`` do
  TorchXRayVision. Métrica: erro absoluto médio de reconstrução como fração da
  faixa ``[-1024, 1024]``. O limiar vem do manifesto (``config.ood_mae_max``) e
  foi escolhido a partir de medição em imagens de tórax e não tórax registrada
  em ``config.ood_rule_evidence``; sem limiar o adaptador recusa carregar.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from radiologyai.analysis.adapters.xr_common import (
    XRV_SCALE,
    load_torch_verified,
    require_backend,
    xrv_input,
)
from radiologyai.analysis.base import AdapterOutput, ModelAdapter, ModelInputs, ResolvedWeights
from radiologyai.analysis.types import AbstentionReason
from radiologyai.errors import InferenceError, ManifestError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from radiologyai.analysis.manifest import ModelManifest

__all__ = [
    "XrvAutoencoderGateAdapter",
    "XrvViewGateAdapter",
    "ae_reconstruction_mae",
    "build_autoencoder",
    "build_view_model",
    "view_probabilities",
]

_AE_RESOLUTION = 224
_VIEW_RESOLUTION = 224


def build_view_model(state: Any) -> torch.nn.Module:
    """``xinario`` ViewModel sem o ``__init__`` da biblioteca (que lê/baixa em ``~``)."""
    import torch
    import torchvision
    from torchxrayvision.baseline_models.xinario import ViewModel

    model = ViewModel.__new__(ViewModel)
    torch.nn.Module.__init__(model)
    model.model = torchvision.models.resnet.resnet50()
    model.model.load_state_dict(state)
    model.norm = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    module: torch.nn.Module = model
    module.eval()
    return module


def build_autoencoder(state: Any) -> torch.nn.Module:
    """``ResNetAE101`` com os pesos ``101-elastic`` já desserializados."""
    import torchxrayvision as xrv

    model = xrv.autoencoders.ResNetAE101()
    model.load_state_dict(state)
    model.weights = "101-elastic"
    model.weights_metadata = xrv.autoencoders.model_urls["101-elastic"]
    module: torch.nn.Module = model
    module.eval()
    return module


def view_probabilities(model: torch.nn.Module, batch: torch.Tensor) -> npt.NDArray[np.float64]:
    """Softmax ``[p_frontal, p_lateral]`` por imagem do lote ``[N, 1, 224, 224]``."""
    import torch

    with torch.inference_mode():
        logits = model(batch)
        probs = torch.softmax(logits.double(), dim=1)
    out: npt.NDArray[np.float64] = probs.cpu().numpy()
    return out


def ae_reconstruction_mae(model: torch.nn.Module, batch: torch.Tensor) -> npt.NDArray[np.float64]:
    """Erro absoluto médio de reconstrução / 2048 por imagem do lote ``[N, 1, 224, 224]``."""
    import torch

    with torch.inference_mode():
        out = model(batch)["out"]
        err = (out - batch).abs().mean(dim=(1, 2, 3)) / (2.0 * XRV_SCALE)
    values: npt.NDArray[np.float64] = err.double().cpu().numpy()
    return values


class _GateBase(ModelAdapter):
    weight_key = "model"

    def __init__(self, manifest: ModelManifest) -> None:
        super().__init__(manifest)
        self._model: Any = None

    def unload(self) -> None:
        self._model = None

    def _batch(self, inputs: ModelInputs, size: int) -> torch.Tensor:
        import torch

        if self._model is None:
            raise InferenceError(f"{self.manifest.id}: modelo não carregado")
        plane = xrv_input(inputs.single, size)
        device = next(self._model.parameters()).device
        return torch.from_numpy(plane)[None, None].to(device)


class XrvViewGateAdapter(_GateBase):
    """Classificador de incidência frontal × lateral (portão a montante)."""

    def load(self, weights: ResolvedWeights, device: str) -> None:
        require_backend("torch", "torchvision", "torchxrayvision", "skimage", extra="ml")
        import torch

        state = load_torch_verified(weights, self.weight_key, allow_pickle=False)
        self._model = build_view_model(state).to(torch.device(device))

    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        p_frontal, p_lateral = view_probabilities(
            self._model, self._batch(inputs, _VIEW_RESOLUTION)
        )[0]
        passed = bool(p_frontal >= p_lateral)
        return AdapterOutput(
            extras={
                "gate": {
                    "image_sha256": inputs.single.sha256,
                    "name": "view_frontal",
                    "passed": passed,
                    "reason": AbstentionReason.VIEW.value,
                    "value": round(float(p_frontal), 6),
                    "threshold": "argmax(frontal, lateral)",
                    "detail": None
                    if passed
                    else "o modelo de incidência classificou a imagem como lateral",
                },
                "p_frontal": float(p_frontal),
                "p_lateral": float(p_lateral),
            }
        )


class XrvAutoencoderGateAdapter(_GateBase):
    """Erro de reconstrução do autoencoder do xrv como portão fora de distribuição."""

    def __init__(self, manifest: ModelManifest) -> None:
        super().__init__(manifest)
        raw = manifest.config.get("ood_mae_max")
        if raw is None or not math.isfinite(float(raw)) or float(raw) <= 0:
            raise ManifestError(
                f"{manifest.id}: config.ood_mae_max ausente/ inválido; o limiar do portão "
                "precisa vir de medição registrada"
            )
        self.threshold = float(raw)

    def load(self, weights: ResolvedWeights, device: str) -> None:
        require_backend("torch", "torchvision", "torchxrayvision", "skimage", extra="ml")
        import torch

        state = load_torch_verified(weights, self.weight_key, allow_pickle=False)
        self._model = build_autoencoder(state).to(torch.device(device))

    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        mae = float(ae_reconstruction_mae(self._model, self._batch(inputs, _AE_RESOLUTION))[0])
        passed = mae <= self.threshold
        return AdapterOutput(
            extras={
                "gate": {
                    "image_sha256": inputs.single.sha256,
                    "name": "ae_reconstruction_mae",
                    "passed": passed,
                    "reason": AbstentionReason.OOD.value,
                    "value": round(mae, 6),
                    "threshold": self.threshold,
                    "detail": None
                    if passed
                    else "erro de reconstrução acima do limiar: imagem fora da distribuição "
                    "de RX de tórax",
                },
                "ae_mae": mae,
            }
        )
