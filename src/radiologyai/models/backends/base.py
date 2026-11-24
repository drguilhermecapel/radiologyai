"""Contrato de backend de inferência."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    import numpy as np
    import numpy.typing as npt


class InferenceBackend(ABC):
    """Executa um modelo carregado sobre imagens pré-processadas.

    O backend é responsável pela transformação específica do modelo (resolução,
    normalização, layout do tensor). O plugin de modalidade entrega a imagem na
    escala física correta em [0, 1]; o backend a adapta ao que o modelo espera.
    """

    @property
    @abstractmethod
    def labels(self) -> tuple[str, ...]:
        """Rótulos de saída, na ordem em que o modelo os produz."""

    @abstractmethod
    def predict(self, image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Executa o modelo numa imagem 2D em [0, 1]. Devolve um escore por rótulo."""

    def predict_batch(self, images: list[npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        """Executa em lote. A implementação default itera; sobrescreva se puder vetorizar."""
        import numpy as np

        return np.stack([self.predict(img) for img in images])

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Identificação do backend, gravada no artefato de avaliação."""
