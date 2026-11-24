"""Comparação de dois modelos sobre as **mesmas** imagens.

Dois métodos, com papéis distintos:

* **DeLong** (DeLong, DeLong & Clarke-Pearson 1988; algoritmo rápido de Sun &
  Xu 2014, DOI 10.1109/LSP.2014.2337313): variância analítica das AUROCs
  pareadas. Supõe imagens **independentes** — com várias imagens por paciente,
  o erro-padrão sai subestimado. Reportado como referência.
* **Bootstrap pareado por paciente**: as duas AUROCs são calculadas sobre a
  mesma réplica (mesmos pesos), preservando a correlação entre modelos e entre
  imagens do mesmo paciente. É o método primário sob o protocolo.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from radiologyai.errors import EvaluationError
from radiologyai.evaluation.metrics import (
    ClusterBootstrap,
    _validate_binary,
    bootstrap_method,
    percentile_ci,
    weighted_auroc,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    FloatArray = npt.NDArray[np.floating[Any]]
    IntArray = npt.NDArray[np.integer[Any]]


class DeLongResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    auroc_a: float
    auroc_b: float
    difference: float  # a - b
    se_difference: float
    z: float
    p_value: float  # bilateral
    var_a: float
    var_b: float
    covariance: float
    assumption: str = "imagens independentes (sem correção por paciente)"


class PairedBootstrapResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    auroc_a: float
    auroc_b: float
    difference: float
    ci_low: float
    ci_high: float
    p_value: float  # bilateral, pela proporção de réplicas do lado oposto de zero
    ci_method: str


def _midrank(x: FloatArray) -> FloatArray:
    from scipy.stats import rankdata

    out: FloatArray = rankdata(x, method="average")
    return out


def _placements(y: IntArray, s: FloatArray) -> tuple[FloatArray, FloatArray, float]:
    """Componentes estruturais V10 (positivos) e V01 (negativos) e a AUROC."""
    import numpy as np

    pos = s[y == 1]
    neg = s[y == 0]
    m, n = pos.size, neg.size
    tz = _midrank(np.concatenate([pos, neg]))
    tx = _midrank(pos)
    ty = _midrank(neg)
    auc = float((tz[:m].sum() - m * (m + 1) / 2.0) / (m * n))
    v10 = (tz[:m] - tx) / n
    v01 = 1.0 - (tz[m:] - ty) / m
    return v10, v01, auc


def delong_variance(y_true: IntArray, y_score: FloatArray) -> tuple[float, float]:
    """``(AUROC, variância de DeLong)`` de um único modelo."""
    import numpy as np

    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(y_score, dtype=np.float64)
    _validate_binary(y, s)
    v10, v01, auc = _placements(y, s)
    if v10.size < 2 or v01.size < 2:
        raise EvaluationError("DeLong exige ao menos 2 positivos e 2 negativos")
    var = float(np.var(v10, ddof=1) / v10.size + np.var(v01, ddof=1) / v01.size)
    return auc, var


def delong_test(y_true: IntArray, score_a: FloatArray, score_b: FloatArray) -> DeLongResult:
    """Teste de DeLong para AUROCs correlacionadas (mesmas imagens)."""
    import numpy as np

    y = np.asarray(y_true, dtype=np.int64)
    a = np.asarray(score_a, dtype=np.float64)
    b = np.asarray(score_b, dtype=np.float64)
    _validate_binary(y, a)
    _validate_binary(y, b)
    v10a, v01a, auc_a = _placements(y, a)
    v10b, v01b, auc_b = _placements(y, b)
    m, n = v10a.size, v01a.size
    if m < 2 or n < 2:
        raise EvaluationError("DeLong exige ao menos 2 positivos e 2 negativos")
    s10 = np.cov(np.stack([v10a, v10b]), ddof=1)
    s01 = np.cov(np.stack([v01a, v01b]), ddof=1)
    cov = s10 / m + s01 / n
    var_diff = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    diff = auc_a - auc_b
    if var_diff <= 0:
        if diff != 0:
            raise EvaluationError(
                "variância de DeLong nula com diferença não nula: teste indefinido"
            )
        se, z, p = 0.0, 0.0, 1.0
    else:
        se = math.sqrt(var_diff)
        z = diff / se
        p = math.erfc(abs(z) / math.sqrt(2.0))
    return DeLongResult(
        auroc_a=auc_a,
        auroc_b=auc_b,
        difference=diff,
        se_difference=se,
        z=z,
        p_value=p,
        var_a=float(cov[0, 0]),
        var_b=float(cov[1, 1]),
        covariance=float(cov[0, 1]),
    )


def paired_cluster_bootstrap(
    *,
    patient_ids: Sequence[str],
    y_true: IntArray,
    score_a: FloatArray,
    score_b: FloatArray,
    seed: int,
    n_bootstrap: int,
    ci_level: float = 0.95,
) -> PairedBootstrapResult:
    """Diferença de AUROC (a − b) com IC por bootstrap pareado por paciente."""
    import numpy as np

    y = np.asarray(y_true, dtype=np.int64)
    a = np.asarray(score_a, dtype=np.float64)
    b = np.asarray(score_b, dtype=np.float64)
    _validate_binary(y, a)
    _validate_binary(y, b)
    if len(patient_ids) != y.size:
        raise EvaluationError("patient_ids com tamanho diferente de y_true")
    ones = np.ones(y.size)
    auc_a = float(weighted_auroc(y, a, ones)[0])
    auc_b = float(weighted_auroc(y, b, ones)[0])
    resampler = ClusterBootstrap([str(p) for p in patient_ids], seed=seed, strata=y)
    diffs = np.concatenate(
        [weighted_auroc(y, a, w) - weighted_auroc(y, b, w) for w in resampler.weights(n_bootstrap)]
    )
    diffs = diffs[np.isfinite(diffs)]
    lo, hi = percentile_ci(diffs, ci_level)
    p = float(min(1.0, 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
    return PairedBootstrapResult(
        auroc_a=auc_a,
        auroc_b=auc_b,
        difference=auc_a - auc_b,
        ci_low=lo,
        ci_high=hi,
        p_value=p,
        ci_method=bootstrap_method(n_bootstrap, seed, ci_level) + " (pareado)",
    )
