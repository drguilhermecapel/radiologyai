"""Calibração por temperatura (temperature scaling).

Um escore de 0,85 saído de uma rede não significa 85% de probabilidade de
doença. Redes modernas são sistematicamente superconfiantes. Sem calibração,
qualquer limiar clínico definido sobre o escore bruto é arbitrário.

Temperature scaling (Guo et al., ICML 2017) é o método mais simples que
funciona: um único parâmetro T por rótulo, ajustado num conjunto de validação
retido, dividindo o logit antes da sigmoide. Não altera a ordenação — portanto
**não altera o AUROC** — apenas realinha os escores à frequência observada.

Migrado da parte numpy/scipy de ``legacy/src/medai_confidence_calibration.py``.
O método que dependia de ``tf.keras.Model`` foi descartado junto com o framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from radiologyai.errors import EvaluationError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

EPS = 1e-7
MIN_SAMPLES = 20


def _to_logit(p: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
    """Inversa da sigmoide, com recorte para evitar infinito."""
    import numpy as np

    clipped = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(z: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
    import numpy as np

    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def _nll(
    temperature: float,
    logits: npt.NDArray[np.float64],
    y_true: npt.NDArray[np.integer[Any]],
) -> float:
    """Log-verossimilhança negativa binária para uma dada temperatura."""
    import numpy as np

    p = np.clip(_sigmoid(logits / temperature), EPS, 1.0 - EPS)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1.0 - p)))


@dataclass(frozen=True)
class TemperatureCalibrator:
    """Calibrador de um rótulo. ``temperature > 1`` reduz a confiança.

    Args:
        temperature: parâmetro ajustado. 1.0 é identidade.
        label: rótulo ao qual se aplica.
        n_fit_samples: tamanho do conjunto de ajuste, gravado para auditoria.
        nll_before: NLL antes da calibração.
        nll_after: NLL depois.
    """

    temperature: float
    label: str
    n_fit_samples: int
    nll_before: float
    nll_after: float

    def __post_init__(self) -> None:
        if not self.temperature > 0:
            raise ValueError(f"temperatura deve ser > 0, recebido {self.temperature}")

    @property
    def improved(self) -> bool:
        """True quando a calibração reduziu a NLL no conjunto de ajuste."""
        return self.nll_after <= self.nll_before

    def apply(self, scores: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
        """Aplica a calibração. Preserva a ordenação, logo preserva o AUROC."""
        return _sigmoid(_to_logit(scores) / self.temperature)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "temperature": round(self.temperature, 6),
            "n_fit_samples": self.n_fit_samples,
            "nll_before": round(self.nll_before, 6),
            "nll_after": round(self.nll_after, 6),
            "improved": self.improved,
        }


def fit_temperature(
    y_true: npt.NDArray[np.integer[Any]],
    y_score: npt.NDArray[np.floating[Any]],
    *,
    label: str,
    bounds: tuple[float, float] = (0.05, 20.0),
) -> TemperatureCalibrator:
    """Ajusta a temperatura de um rótulo minimizando a NLL.

    O ajuste DEVE ser feito num conjunto retido, nunca no de teste: calibrar no
    conjunto de teste e reportar o ECE resultante é uma forma de vazamento.

    Raises:
        EvaluationError: amostras insuficientes ou classe única.
    """
    import numpy as np
    from scipy.optimize import minimize_scalar

    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)

    if truth.shape != scores.shape:
        raise EvaluationError(f"shapes incompatíveis: {truth.shape} vs {scores.shape}")
    if truth.size < MIN_SAMPLES:
        raise EvaluationError(
            f"{truth.size} amostras é pouco para calibrar {label!r} "
            f"(mínimo {MIN_SAMPLES}). Uma temperatura ajustada em poucos pontos "
            "é ruído com aparência de calibração."
        )
    n_pos = int((truth == 1).sum())
    if n_pos == 0 or n_pos == truth.size:
        raise EvaluationError(
            f"calibração de {label!r} exige as duas classes; "
            f"encontrado {n_pos} positivos em {truth.size}"
        )

    logits = _to_logit(scores)
    before = _nll(1.0, logits, truth)

    result = minimize_scalar(_nll, bounds=bounds, args=(logits, truth), method="bounded")
    temperature = float(result.x)

    return TemperatureCalibrator(
        temperature=temperature,
        label=label,
        n_fit_samples=int(truth.size),
        nll_before=before,
        nll_after=_nll(temperature, logits, truth),
    )
