"""Calibração de Platt (2 parâmetros: slope * logit(p) + intercept)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

EPS = 1e-7
_LOGIT_BOUND = 500.0
_PARAM_BOUNDS = ((-50.0, 50.0), (-50.0, 50.0))


def _sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    import numpy as np

    clipped = np.clip(x, -_LOGIT_BOUND, _LOGIT_BOUND)
    raw = np.where(
        clipped >= 0.0,
        1.0 / (1.0 + np.exp(-clipped)),
        np.exp(clipped) / (1.0 + np.exp(clipped)),
    )
    out: npt.NDArray[np.float64] = np.clip(raw, EPS, 1.0 - EPS).astype(np.float64)
    return out


def _binary_nll(
    probs: npt.NDArray[np.float64],
    labels: npt.NDArray[np.float64],
) -> float:
    import numpy as np

    p = np.clip(probs, EPS, 1.0 - EPS)
    return float(-np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))


def _validate_probabilities(probabilities: npt.ArrayLike) -> npt.NDArray[np.float64]:
    import numpy as np

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.size == 0:
        raise ValueError("probabilities must not be empty.")
    if probs.ndim != 1:
        raise ValueError("probabilities must be a 1D array.")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must contain only finite values.")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probabilities must be in [0, 1].")
    return probs


@dataclass(frozen=True)
class PlattScaler:
    """Calibrador de Platt (regressão logística no espaço logit)."""

    slope: float
    intercept: float
    n_samples: int
    pre_nll: float
    post_nll: float

    def transform(self, probabilities: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Aplica calibração de Platt às probabilidades fornecidas."""
        import numpy as np

        probs = _validate_probabilities(probabilities)
        p_clipped = np.clip(probs, EPS, 1.0 - EPS)
        logits = np.log(p_clipped / (1.0 - p_clipped))
        return _sigmoid(self.slope * logits + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        """Serializa os parâmetros e métricas do calibrador."""
        return {
            "slope": float(self.slope),
            "intercept": float(self.intercept),
            "n_samples": int(self.n_samples),
            "pre_nll": float(self.pre_nll),
            "post_nll": float(self.post_nll),
        }


def fit_platt_scaler(
    probabilities: npt.ArrayLike,
    labels: npt.ArrayLike,
) -> PlattScaler:
    """Ajusta PlattScaler por minimização determinística da NLL."""
    import numpy as np
    from scipy.optimize import minimize

    probs = _validate_probabilities(probabilities)
    y = np.asarray(labels, dtype=np.float64)
    if y.size == 0:
        raise ValueError("labels must not be empty.")
    if y.ndim != 1:
        raise ValueError("labels must be a 1D array.")
    if probs.shape != y.shape:
        raise ValueError("probabilities and labels must have the same shape.")
    if not np.all(np.isfinite(y)):
        raise ValueError("labels must contain only finite values.")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("labels must be binary in {0, 1}.")

    n_pos = int(np.sum(y == 1.0))
    n_neg = int(np.sum(y == 0.0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("labels must contain at least one positive and one negative sample.")

    p_clipped = np.clip(probs, EPS, 1.0 - EPS)
    logits = np.log(p_clipped / (1.0 - p_clipped))
    pre_nll = _binary_nll(p_clipped, y)

    def _objective(
        params: npt.NDArray[np.float64],
    ) -> tuple[float, npt.NDArray[np.float64]]:
        a = float(params[0])
        b = float(params[1])
        z = a * logits + b
        loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
        preds = _sigmoid(z)
        diff = preds - y
        grad_a = float(np.mean(diff * logits))
        grad_b = float(np.mean(diff))
        grad = np.array([grad_a, grad_b], dtype=np.float64)
        return loss, grad

    result = minimize(
        _objective,
        x0=np.array([1.0, 0.0], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=_PARAM_BOUNDS,
    )
    slope = float(result.x[0])
    intercept = float(result.x[1])
    post_probs = _sigmoid(slope * logits + intercept)
    post_nll = _binary_nll(post_probs, y)

    return PlattScaler(
        slope=slope,
        intercept=intercept,
        n_samples=int(probs.size),
        pre_nll=pre_nll,
        post_nll=post_nll,
    )
