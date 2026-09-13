"""Backend torchxrayvision — o único caminho para pesos de radiografia publicados.

Migrado de ``legacy/src/torchxray_integration.py``, com a diferença essencial:
o ``pathology_mapping`` do legado foi **eliminado**, não portado. Ele colapsava
as 18 saídas do modelo em 5 categorias, reportando pneumotórax como pneumonia e
cardiomegalia como "normal" (HONEST_STATUS.md §4). Aqui as 18 saídas são
expostas nativamente, com os nomes do próprio modelo.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

from radiologyai.errors import BackendUnavailableError, ModelNotFoundError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from radiologyai.models.card import ModelCard

from radiologyai.models.backends.base import InferenceBackend

# Mapa card_id -> nome de pesos do torchxrayvision.
XRV_WEIGHTS: dict[str, str] = {
    "xrv-densenet121-pc": "densenet121-res224-pc",
    "xrv-densenet121-nih": "densenet121-res224-nih",
    "xrv-densenet121-all": "densenet121-res224-all",
}

# Datasets de treino de cada conjunto de pesos. Usado para detectar vazamento
# quando o dataset de avaliação coincide com o de treino.
XRV_TRAINED_ON: dict[str, tuple[str, ...]] = {
    "densenet121-res224-pc": ("PadChest",),
    "densenet121-res224-nih": ("NIH ChestX-ray14",),
    "densenet121-res224-all": (
        "NIH ChestX-ray14",
        "PadChest",
        "CheXpert",
        "MIMIC-CXR",
        "Google",
        "OpenI",
        "RSNA Kaggle",
    ),
}

INPUT_SIZE = 224
XRV_MAXVAL = 255.0

UNTRAINED_PREFIX = "__untrained_"
"""Prefixo dos rótulos cuja cabeça de saída não foi treinada.

O torchxrayvision devolve string vazia em ``model.pathologies`` para as posições
que o dataset de treino não continha. Nos pesos ``-pc`` (PadChest) são os
índices 14, 16 e 17 — 'Lung Lesion', 'Lung Opacity' e 'Enlarged
Cardiomediastinum'. Essas saídas existem no tensor mas **não significam nada**.

Usar a lista ``default_pathologies`` como se fosse a do modelo faria o sistema
reportar AUROC para uma cabeça não treinada: um número que parece medição e não
é. Aqui elas recebem nome sentinela e são declaradas explicitamente.
"""


class TorchXRayVisionBackend(InferenceBackend):
    """DenseNet-121 pré-treinado do torchxrayvision.

    Args:
        weights_name: nome dos pesos, ex.: ``densenet121-res224-pc``.
        weights_path: caminho local dos pesos, já verificado por sha256.
    """

    def __init__(self, weights_name: str, weights_path: str | Path | None = None) -> None:
        # Verificação antes do import, em vez de `except ImportError`: a regra
        # REQ-001 proíbe capturar ImportError em src/, porque foi assim que o
        # sistema legado desativou 37 subsistemas em silêncio. Aqui a ausência
        # da dependência é detectada e reportada, nunca absorvida.
        missing = [
            name for name in ("torch", "torchxrayvision") if importlib.util.find_spec(name) is None
        ]
        if missing:
            raise BackendUnavailableError(
                f"backend torchxrayvision indisponível: faltam {missing}. "
                "Instale com: pip install 'radiologyai[ml]'. "
                "Nenhum caminho alternativo será usado."
            )

        import torch
        import torchxrayvision as xrv

        self._torch = torch
        self._xrv = xrv
        self._weights_name = weights_name
        self._weights_path = Path(weights_path) if weights_path else None

        self._model = xrv.models.DenseNet(weights=weights_name)
        self._model.eval()

        # model.pathologies traz "" onde a cabeça não foi treinada.
        raw = tuple(self._model.pathologies)
        self._labels: tuple[str, ...] = tuple(
            name if name else f"{UNTRAINED_PREFIX}{i}" for i, name in enumerate(raw)
        )
        self._untrained: tuple[int, ...] = tuple(i for i, name in enumerate(raw) if not name)

        self._crop = xrv.datasets.XRayCenterCrop()
        self._resize = xrv.datasets.XRayResizer(INPUT_SIZE)

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def device(self) -> Any:
        """Dispositivo onde o modelo está — lido dos parâmetros, nunca de um atributo.

        Um atributo separado pode dessincronizar do modelo; foi exatamente o
        que aconteceu na primeira execução em GPU: ``to('cuda')`` moveu os
        pesos, mas a entrada continuou na CPU. Lendo dos parâmetros, a entrada
        vai por construção para onde o modelo estiver.
        """
        return next(self._model.parameters()).device

    @property
    def trained_on(self) -> tuple[str, ...]:
        return XRV_TRAINED_ON.get(self._weights_name, ())

    @property
    def untrained_indices(self) -> tuple[int, ...]:
        """Índices de saída sem treino nestes pesos. Nunca devem ser reportados."""
        return self._untrained

    @property
    def trained_labels(self) -> tuple[str, ...]:
        """Somente os rótulos com cabeça efetivamente treinada."""
        return tuple(name for name in self._labels if not name.startswith(UNTRAINED_PREFIX))

    def predict(self, image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Escores por patologia para uma radiografia 2D em [0, 1]."""
        import numpy as np

        if image.ndim != 2:
            raise ValueError(f"esperado array 2D, recebido shape {image.shape}")

        # torchxrayvision espera a faixa [-1024, 1024].
        arr = self._xrv.datasets.normalize(image * XRV_MAXVAL, XRV_MAXVAL)
        arr = self._resize(self._crop(arr[None, ...]))
        tensor = self._torch.from_numpy(arr)[None, ...].float().to(self.device)

        with self._torch.no_grad():
            output = self._model(tensor)
        return np.asarray(output[0].cpu().numpy(), dtype=np.float32)

    def predict_batch(self, images: list[npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        """Executa um lote de uma vez — significativamente mais rápido em GPU."""
        import numpy as np

        batch = []
        for image in images:
            arr = self._xrv.datasets.normalize(image * XRV_MAXVAL, XRV_MAXVAL)
            batch.append(self._resize(self._crop(arr[None, ...])))
        tensor = self._torch.from_numpy(np.stack(batch)).float().to(self.device)

        with self._torch.no_grad():
            output = self._model(tensor)
        return np.asarray(output.cpu().numpy(), dtype=np.float32)

    def gradcam(self, image: npt.NDArray[np.float32], target_label: str) -> Any:
        """Grad-CAM real para um rótulo, usando os gradientes deste modelo.

        Raises:
            ValueError: rótulo desconhecido ou cabeça não treinada. Não se
                produz saliência para uma saída que não significa nada.
        """
        import numpy as np

        from radiologyai.explain.gradcam import GradCAM

        if target_label not in self._labels:
            raise ValueError(
                f"rótulo {target_label!r} não é saída deste modelo; "
                f"disponíveis: {list(self.trained_labels)}"
            )
        if target_label.startswith(UNTRAINED_PREFIX):
            raise ValueError(
                f"{target_label!r} é uma cabeça não treinada nestes pesos; "
                "explicar uma saída sem significado produziria um mapa enganoso"
            )

        index = self._labels.index(target_label)
        arr = self._xrv.datasets.normalize(image * XRV_MAXVAL, XRV_MAXVAL)
        arr = self._resize(self._crop(arr[None, ...]))
        tensor = self._torch.from_numpy(np.ascontiguousarray(arr))[None, ...].float()
        tensor = tensor.to(self.device).requires_grad_(True)

        # Última camada convolucional da DenseNet-121, antes do pooling global.
        target_layer = self._model.features.denseblock4
        cam = GradCAM(self._model, target_layer, "features.denseblock4")
        return cam.explain(tensor, index, target_label)

    def to(self, device: str) -> TorchXRayVisionBackend:
        """Move o modelo para um dispositivo ('cuda' ou 'cpu')."""
        self._model = self._model.to(device)
        return self

    def describe(self) -> dict[str, Any]:
        import torch

        return {
            "backend": "torchxrayvision",
            "weights_name": self._weights_name,
            "torch_version": torch.__version__,
            "torchxrayvision_version": self._xrv.__version__,
            "n_labels": len(self._labels),
            "n_trained_labels": len(self.trained_labels),
            "untrained_output_indices": list(self._untrained),
            "input_size": INPUT_SIZE,
            "trained_on": list(self.trained_on),
        }


def weights_cache_path(weights_name: str) -> Path:
    """Caminho do arquivo de pesos no cache do torchxrayvision.

    Raises:
        ModelNotFoundError: pesos não presentes no cache.
    """
    cache = Path.home() / ".torchxrayvision" / "models_data"
    prefix = weights_name.replace("densenet121-res224-", "")
    if cache.is_dir():
        for candidate in sorted(cache.glob("*.pt")):
            if candidate.name.startswith(f"{prefix}-densenet121"):
                return candidate
    raise ModelNotFoundError(
        f"pesos {weights_name!r} não encontrados em {cache}. "
        "Carregue o modelo uma vez para que o torchxrayvision faça o download."
    )


def build_backend(card: ModelCard) -> TorchXRayVisionBackend:
    """Constrói o backend a partir de um model card, verificando integridade."""
    if card.card_id not in XRV_WEIGHTS:
        raise ModelNotFoundError(
            f"card {card.card_id!r} não é um modelo torchxrayvision conhecido; "
            f"conhecidos: {sorted(XRV_WEIGHTS)}"
        )
    weights_name = XRV_WEIGHTS[card.card_id]

    # Garante o download antes de verificar o hash.
    backend = TorchXRayVisionBackend(weights_name)
    card.verify_weights(weights_cache_path(weights_name))
    return backend
