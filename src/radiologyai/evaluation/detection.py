"""Desfecho de detecção numa regra fixa e pré-especificada.

Exemplo: "volume de hemorragia segmentado > 0 mL" é positivo. A regra é escrita
**antes** de ver o resultado e não é ajustada nos dados; por isso não há metade
de calibração: o conjunto inteiro é o de teste. O protocolo julga pelo limite
inferior do IC da sensibilidade e da especificidade (``sens_spec_ci_low_min``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from radiologyai.errors import EvaluationError
from radiologyai.evaluation.metrics import (
    ClusterBootstrap,
    bootstrap_method,
    percentile_ci,
    weighted_rates_at,
)
from radiologyai.evaluation.protocol_eval import Counts, Estimate, counts_for

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    IntArray = npt.NDArray[np.integer[Any]]


class DetectionReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: str  # descrição legível da regra pré-especificada
    counts: Counts
    sensitivity: Estimate
    specificity: Estimate
    ppv: Estimate | None
    npv: Estimate | None
    ci_method: str


def evaluate_detection(
    *,
    rule: str,
    patient_ids: Sequence[str],
    y_true: IntArray,
    y_pred: IntArray,
    seed: int,
    n_bootstrap: int,
    ci_level: float = 0.95,
) -> DetectionReport:
    """Sensibilidade/especificidade de uma decisão binária já tomada pela regra."""
    import math

    import numpy as np

    if not rule.strip():
        raise EvaluationError("detecção exige a regra pré-especificada declarada")
    y = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if y.shape != pred.shape or y.ndim != 1 or y.size != len(patient_ids):
        raise EvaluationError("y_true, y_pred e patient_ids devem ter o mesmo comprimento")
    for name, arr in (("y_true", y), ("y_pred", pred)):
        if not set(np.unique(arr).tolist()) <= {0, 1}:
            raise EvaluationError(f"{name} deve ser 0/1")
    if y.min() == y.max():
        raise EvaluationError("detecção exige positivos e negativos de referência")
    pids = [str(p) for p in patient_ids]
    score = pred.astype(np.float64)
    point = weighted_rates_at(y, score, 0.5, np.ones(y.size))
    resampler = ClusterBootstrap(pids, seed=seed, strata=y)
    blocks = [weighted_rates_at(y, score, 0.5, w) for w in resampler.weights(n_bootstrap)]

    def est(key: str) -> Estimate | None:
        value = float(point[key][0])
        if not math.isfinite(value):
            return None
        lo, hi = percentile_ci(np.concatenate([b[key] for b in blocks]), ci_level)
        return Estimate(value=value, ci_low=lo, ci_high=hi)

    sens, spec = est("sensitivity"), est("specificity")
    if sens is None or spec is None:
        raise EvaluationError("sensibilidade/especificidade indefinidas")
    return DetectionReport(
        rule=rule,
        counts=counts_for(pids, y),
        sensitivity=sens,
        specificity=spec,
        ppv=est("ppv"),
        npv=est("npv"),
        ci_method=bootstrap_method(n_bootstrap, seed, ci_level),
    )
