"""Classificadores multirrótulo de RX de tórax do TorchXRayVision (DenseNet121 / ResNet50).

O que este adaptador faz
------------------------
* Constrói a arquitetura do xrv SEM deixar a biblioteca baixar nada: os pesos
  vêm do arquivo verificado pelo registro (sha256 conferido sobre os mesmos
  bytes que são desserializados).
* Devolve LOGITS BRUTOS (``op_threshs=None``, sem sigmoide) com
  ``score_kind: logit``. Os pontos de operação publicados pela biblioteca
  (``op_threshs``) entram no MANIFESTO como pontos de operação, com origem;
  nunca são aplicados aqui.
* Emite somente rótulos declarados no manifesto; cabeças não treinadas do
  conjunto de pesos (rótulo vazio / limiar NaN na biblioteca) são recusadas na
  carga — nunca reportadas.
* Aplicabilidade: portão estático (RX de tórax CR/DX frontal PA/AP adulto,
  tamanho mínimo) + portões de conteúdo vindos dos modelos a montante
  (incidência e fora de distribuição; ver :mod:`xr_gates`).
* Explicação: CAM (DenseNet: ReLU → média global → linear, logo o CAM é exato).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from radiologyai.analysis.adapters.xr_common import (
    center_crop_box,
    load_torch_verified,
    require_backend,
    xrv_input,
)
from radiologyai.analysis.base import (
    AdapterOutput,
    ExplanationMap,
    ModelAdapter,
    ModelInputs,
    ResolvedWeights,
    Score,
)
from radiologyai.analysis.types import AbstentionReason, Primitive, QACheck, QAReport
from radiologyai.errors import InferenceError, ManifestError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from radiologyai.analysis.manifest import ModelManifest

__all__ = [
    "XrvClassifierAdapter",
    "build_xrv_classifier",
    "gate_checks",
    "trained_heads",
]


def trained_heads(library_key: str) -> dict[str, tuple[int, float]]:
    """``{rótulo da biblioteca: (índice da cabeça, op_thresh)}`` das cabeças treinadas.

    Cabeça treinada = rótulo não vazio e ``op_thresh`` finito em
    ``torchxrayvision.models.model_urls``.
    """
    import torchxrayvision as xrv

    info = xrv.models.model_urls[library_key]
    labels: list[str] = list(info["labels"])
    thresholds: list[Any] = list(info.get("op_threshs") or [None] * len(labels))
    heads: dict[str, tuple[int, float]] = {}
    for idx, (label, thr) in enumerate(zip(labels, thresholds, strict=True)):
        if not label or thr is None or not math.isfinite(float(thr)):
            continue
        heads[label] = (idx, float(thr))
    return heads


def build_xrv_classifier(arch: str, library_key: str, checkpoint: Any) -> torch.nn.Module:
    """Arquitetura xrv carregada a partir de um checkpoint já desserializado.

    Args:
        arch: ``"densenet"`` ou ``"resnet"``.
        library_key: chave de ``xrv.models.model_urls`` (ex.: ``densenet121-res224-pc``).
        checkpoint: objeto devolvido por ``torch.load`` (módulo serializado para
            DenseNet, ``state_dict`` para ResNet).

    Raises:
        ManifestError: arquitetura desconhecida.
    """
    import torch
    import torchvision
    import torchxrayvision as xrv

    info = xrv.models.model_urls[library_key]
    n_out = len(info["labels"])
    if arch == "densenet":
        model = xrv.models.DenseNet(weights=None, num_classes=n_out, op_threshs=None)
        state = checkpoint.state_dict() if isinstance(checkpoint, torch.nn.Module) else checkpoint
        model.load_state_dict(state)
        model.targets = list(info["labels"])
        model.pathologies = model.targets
        model.op_threshs = None
        model.apply_sigmoid = False
        model.input_resolution = int(info["input_resolution"])
    elif arch == "resnet":
        model = xrv.models.ResNet.__new__(xrv.models.ResNet)
        torch.nn.Module.__init__(model)
        model.weights = library_key
        model.apply_sigmoid = False
        model.weights_dict = info
        model.targets = list(info["labels"])
        model.pathologies = model.targets
        base = torchvision.models.resnet50(num_classes=n_out, weights=None)
        base.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        state = checkpoint.state_dict() if isinstance(checkpoint, torch.nn.Module) else checkpoint
        base.load_state_dict(state)
        model.model = base
        model.input_resolution = int(info["input_resolution"])
    else:
        raise ManifestError(f"arquitetura xrv desconhecida {arch!r}")
    module: torch.nn.Module = model
    module.eval()
    return module


def gate_checks(
    inputs: ModelInputs, gate_ids: Sequence[str]
) -> tuple[list[QACheck], list[AbstentionReason]]:
    """Lê as decisões dos portões a montante (incidência / OOD) para a imagem desta entrada.

    Cada portão publica em ``extras["gate"]`` um dicionário com ``image_sha256``,
    ``passed``, ``reason`` (valor de :class:`AbstentionReason`), ``name``,
    ``value`` e ``threshold``. Portão de outra imagem conta como ausente.
    """
    checks: list[QACheck] = []
    reasons: list[AbstentionReason] = []
    sha = inputs.single.sha256
    for gate_id in gate_ids:
        output = inputs.upstream.get(gate_id)
        gate = None if output is None else output.extras.get("gate")
        if not isinstance(gate, dict) or gate.get("image_sha256") != sha:
            checks.append(
                QACheck(
                    name=f"gate:{gate_id}",
                    passed=False,
                    detail="portão a montante sem decisão para esta imagem",
                )
            )
            reasons.append(AbstentionReason.UPSTREAM)
            continue
        passed = bool(gate["passed"])
        checks.append(
            QACheck(
                name=f"gate:{gate_id}:{gate['name']}",
                passed=passed,
                value=gate.get("value"),
                threshold=gate.get("threshold"),
                detail=gate.get("detail"),
            )
        )
        if not passed:
            reasons.append(AbstentionReason(gate["reason"]))
    return checks, reasons


class XrvClassifierAdapter(ModelAdapter):
    """DenseNet121/ResNet50 do TorchXRayVision; logits brutos por rótulo declarado.

    ``config`` do manifesto:

    * ``library_weights``: chave em ``xrv.models.model_urls``.
    * ``arch``: ``densenet`` | ``resnet``.
    * ``gates``: ids dos portões a montante consultados na aplicabilidade.
    """

    weight_key = "model"

    def __init__(self, manifest: ModelManifest) -> None:
        super().__init__(manifest)
        self._model: Any = None
        self._heads: dict[str, int] = {}
        self._size = 224
        cfg = manifest.config
        self.library_key = str(cfg.get("library_weights", ""))
        self.arch = str(cfg.get("arch", "densenet"))
        self.gate_ids: tuple[str, ...] = tuple(cfg.get("gates", ()))
        missing = [g for g in self.gate_ids if g not in manifest.depends_on]
        if missing:
            raise ManifestError(f"{manifest.id}: portões {missing} não estão em depends_on")

    # -- ciclo de vida ---------------------------------------------------- #
    def load(self, weights: ResolvedWeights, device: str) -> None:
        require_backend("torch", "torchvision", "torchxrayvision", "skimage", extra="ml")
        import torch
        import torchxrayvision as xrv

        if self.library_key not in xrv.models.model_urls:
            raise ManifestError(
                f"{self.manifest.id}: library_weights {self.library_key!r} desconhecido"
            )
        heads = trained_heads(self.library_key)
        mapping: dict[str, int] = {}
        for spec in self.manifest.labels:
            if spec.source_label not in heads:
                raise ManifestError(
                    f"{self.manifest.id}: rótulo {spec.key!r} ({spec.source_label!r}) não é "
                    f"cabeça treinada de {self.library_key}"
                )
            mapping[spec.key] = heads[spec.source_label][0]
        pickled = self.manifest.weights[0].unsafe_pickle
        checkpoint = load_torch_verified(weights, self.weight_key, allow_pickle=pickled)
        model = build_xrv_classifier(self.arch, self.library_key, checkpoint)
        self._model = model.to(torch.device(device))
        self._heads = mapping
        self._size = int(xrv.models.model_urls[self.library_key]["input_resolution"])

    def unload(self) -> None:
        self._model = None

    # -- aplicabilidade --------------------------------------------------- #
    def applicability(self, inputs: ModelInputs) -> QAReport:
        base = super().applicability(inputs)
        if not base.applicable or not self.gate_ids:
            return base
        checks, reasons = gate_checks(inputs, self.gate_ids)
        all_checks = (*base.checks, *checks)
        if reasons:
            return QAReport.reject(*dict.fromkeys(reasons), checks=all_checks)
        return QAReport.ok(all_checks)

    # -- inferência ------------------------------------------------------- #
    def _tensor(self, inputs: ModelInputs) -> torch.Tensor:
        import torch

        plane = xrv_input(inputs.single, self._size)
        return torch.from_numpy(plane)[None, None].to(next(self._model.parameters()).device)

    def logits(self, batch: torch.Tensor) -> torch.Tensor:
        """Logits de todas as cabeças para um lote ``[N, 1, S, S]`` já pré-processado."""
        import torch

        if self._model is None:
            raise InferenceError(f"{self.manifest.id}: modelo não carregado")
        with torch.inference_mode():
            out: torch.Tensor = self._model(batch)
        return out

    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        logits = self.logits(self._tensor(inputs))[0].cpu().numpy()
        scores = [
            Score(label=key, value=float(logits[idx]), kind="logit")
            for key, idx in self._heads.items()
        ]
        return AdapterOutput(
            scores=scores,
            extras={"image_sha256": inputs.single.sha256, "input_size": self._size},
        )

    # -- explicação ------------------------------------------------------- #
    def cam(self, batch: torch.Tensor, head: int) -> npt.NDArray[np.float32]:
        """CAM exato da DenseNet para uma cabeça: ``sum_k w_k * relu(F_k)`` (grade 7x7)."""
        import torch
        import torch.nn.functional as F  # noqa: N812

        if self.arch != "densenet":
            raise InferenceError("CAM implementado só para a DenseNet (GAP + linear)")
        with torch.inference_mode():
            feats = F.relu(self._model.features(batch))[0]  # [K, h, w]
            weight = self._model.classifier.weight[head]  # [K]
            cam = torch.einsum("k,khw->hw", weight, feats)
        out: npt.NDArray[np.float32] = cam.cpu().numpy().astype("float32")
        return out

    def explain(self, inputs: ModelInputs, labels: Sequence[str]) -> list[ExplanationMap]:
        import numpy as np

        if self.arch != "densenet":
            return []
        batch = self._tensor(inputs)
        height, width = inputs.single.plane_shape
        y0, x0, side = center_crop_box(height, width)
        placement = Primitive(
            type="bbox", xy=((float(x0), float(y0)), (float(x0 + side), float(y0 + side)))
        )
        maps: list[ExplanationMap] = []
        for label in labels:
            if label not in self._heads:
                continue
            cam = self.cam(batch, self._heads[label])
            cam = np.maximum(cam, 0.0)
            peak = float(cam.max())
            heat = cam / peak if peak > 0 else np.zeros_like(cam)
            maps.append(
                ExplanationMap(
                    label=label, heatmap=heat.astype(np.float32), placement=placement, method="cam"
                )
            )
        return maps
