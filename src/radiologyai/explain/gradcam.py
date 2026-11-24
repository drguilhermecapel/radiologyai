"""Grad-CAM real — derivado dos gradientes do modelo efetivamente usado.

Substitui ``legacy/src/medai_explainability.py``, que **nunca consultava o
modelo**. O "Grad-CAM" servido pelo endpoint ``/api/v1/explain`` do v1 era:

    edges = cv2.Canny(gray, 50, 150)
    heatmap = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0)
    heatmap += np.random.normal(0, 0.1, heatmap.shape)

Detecção de bordas, borrada, com ruído somado. Um mapa de saliência fabricado
sobre a radiografia real de um paciente é pior que um número de acurácia
fabricado: manufatura a *aparência* de um modelo raciocinando sobre anatomia,
e um clínico atribui significado a um contorno de Canny.

Aqui o mapa vem de ``register_full_backward_hook`` na última camada
convolucional. A propriedade que torna isso verificável: **um modelo cujos
gradientes são nulos produz um mapa nulo.** O código do v1 reprovaria esse teste.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from radiologyai.errors import BackendUnavailableError, InferenceError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@dataclass(frozen=True)
class Explanation:
    """Mapa de saliência para um alvo específico.

    Args:
        heatmap: mapa em [0, 1], na resolução do mapa de ativação.
        target_label: rótulo para o qual o gradiente foi calculado.
        target_score: escore do modelo para esse rótulo.
        layer_name: camada de onde as ativações vieram, registrada para auditoria.
        method: identificação do método.
    """

    heatmap: npt.NDArray[np.float32]
    target_label: str
    target_score: float
    layer_name: str
    method: str = "grad-cam"

    @property
    def is_null(self) -> bool:
        """True quando o mapa é uniformemente zero (gradiente nulo)."""
        return bool(self.heatmap.max() == 0.0)

    def describe(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "target_label": self.target_label,
            "target_score": round(self.target_score, 6),
            "layer_name": self.layer_name,
            "shape": list(self.heatmap.shape),
            "is_null": self.is_null,
        }


class GradCAM:
    """Grad-CAM sobre um ``torch.nn.Module``.

    Args:
        model: modelo em modo eval.
        target_layer: camada convolucional cujas ativações serão ponderadas.
            Tipicamente a última antes do pooling global.
        layer_name: nome da camada, gravado na explicação para auditoria.
    """

    def __init__(self, model: Any, target_layer: Any, layer_name: str) -> None:
        if importlib.util.find_spec("torch") is None:
            raise BackendUnavailableError(
                "Grad-CAM exige torch. Instale com: pip install 'radiologyai[ml]'. "
                "Nenhuma saliência simulada será produzida como substituto."
            )
        import torch

        self._torch = torch
        self._model = model
        self._layer = target_layer
        self._layer_name = layer_name
        self._activations: Any = None
        self._gradients: Any = None

    def _forward_hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        self._activations = output.detach()

    def _backward_hook(self, _module: Any, _grad_in: Any, grad_out: Any) -> None:
        self._gradients = grad_out[0].detach()

    def explain(self, input_tensor: Any, target_index: int, target_label: str) -> Explanation:
        """Calcula o mapa para um índice de saída.

        Raises:
            InferenceError: os hooks não capturaram ativação ou gradiente, o que
                indica camada-alvo errada. Falha alto — nunca devolve um mapa
                inventado no lugar.
        """
        import numpy as np

        torch = self._torch
        handles = [
            self._layer.register_forward_hook(self._forward_hook),
            self._layer.register_full_backward_hook(self._backward_hook),
        ]
        try:
            self._activations = None
            self._gradients = None

            self._model.zero_grad(set_to_none=True)
            output = self._model(input_tensor)
            score = output[0, target_index]
            score.backward()

            if self._activations is None or self._gradients is None:
                raise InferenceError(
                    f"hooks não capturaram dados em {self._layer_name!r}. "
                    "A camada-alvo provavelmente não participa do forward. "
                    "Nenhum mapa substituto será gerado."
                )

            # Peso de cada canal = média espacial do gradiente (Selvaraju et al., 2017)
            weights = self._gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * self._activations).sum(dim=1, keepdim=True))
            heatmap = cam[0, 0].cpu().numpy().astype(np.float32)
        finally:
            for handle in handles:
                handle.remove()

        peak = float(heatmap.max())
        if peak > 0:
            heatmap = heatmap / peak

        return Explanation(
            heatmap=heatmap,
            target_label=target_label,
            target_score=float(score.detach().cpu()),
            layer_name=self._layer_name,
        )


def overlay_heatmap(
    image: npt.NDArray[np.float32],
    heatmap: npt.NDArray[np.float32],
    *,
    alpha: float = 0.4,
) -> npt.NDArray[np.float32]:
    """Sobrepõe um mapa em [0,1] a uma imagem em [0,1], devolvendo RGB em [0,1].

    Único trecho aproveitado do módulo de explicabilidade do legado — é
    apresentação, não inferência, e estava correto.
    """
    import numpy as np

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha deve estar em [0, 1], recebido {alpha}")
    if image.ndim != 2:
        raise ValueError(f"esperada imagem 2D, recebido shape {image.shape}")

    # Redimensiona o mapa por repetição de vizinho (sem dependência de cv2).
    if heatmap.shape != image.shape:
        ys = (np.arange(image.shape[0]) * heatmap.shape[0] // image.shape[0]).clip(
            0, heatmap.shape[0] - 1
        )
        xs = (np.arange(image.shape[1]) * heatmap.shape[1] // image.shape[1]).clip(
            0, heatmap.shape[1] - 1
        )
        heatmap = heatmap[np.ix_(ys, xs)]

    base = np.stack([image] * 3, axis=-1)
    # Mapa quente simples: vermelho cresce, azul decresce.
    colored = np.stack([heatmap, np.zeros_like(heatmap), 1.0 - heatmap], axis=-1)
    blended: npt.NDArray[np.float32] = np.clip(
        (1 - alpha) * base + alpha * colored, 0.0, 1.0
    ).astype(np.float32)
    return blended
