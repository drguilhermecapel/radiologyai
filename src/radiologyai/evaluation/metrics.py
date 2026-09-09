"""Métricas de desempenho — medidas, nunca geradas.

O legado tinha 164 chamadas a ``np.random`` em ``src/``, incluindo
``cv_scores = np.random.normal(base_accuracy, 0.02, n_folds)`` em
``medai_advanced_clinical_validation.py:357``: cross-validation **sintetizada**.

Aqui não há nenhuma fonte de aleatoriedade exceto o reamostrador de bootstrap,
que recebe seed explícita e é usado para estimar incerteza sobre dados reais.
Um verificador de CI (``scripts/check_honesty.py``) impede a reintrodução de
``np.random`` fora desse uso.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from radiologyai.errors import EvaluationError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    FloatArray = npt.NDArray[np.floating[Any]]
    IntArray = npt.NDArray[np.integer[Any]]


class AUROCResult(NamedTuple):
    """AUROC com intervalo de confiança e contagens de suporte."""

    auroc: float
    ci95_low: float
    ci95_high: float
    n_pos: int
    n_neg: int
    ci_method: str


class OperatingPoint(NamedTuple):
    """Ponto de operação num alvo de sensibilidade."""

    target_sensitivity: float
    threshold: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float


def _validate_binary(y_true: IntArray, y_score: FloatArray) -> tuple[int, int]:
    import numpy as np

    if y_true.shape != y_score.shape:
        raise EvaluationError(
            f"shapes incompatíveis: y_true {y_true.shape} vs y_score {y_score.shape}"
        )
    if y_true.size == 0:
        raise EvaluationError("conjunto vazio")

    unique = set(np.unique(y_true).tolist())
    if not unique <= {0, 1}:
        raise EvaluationError(f"y_true deve ser binário; encontrado {sorted(unique)}")

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise EvaluationError(
            f"AUROC indefinido: {n_pos} positivos, {n_neg} negativos. "
            "Reportar AUROC aqui produziria NaN — foi o que o legado fez "
            "(mean_auc: NaN sobre test_samples: 1)."
        )
    return n_pos, n_neg


def auroc(y_true: IntArray, y_score: FloatArray) -> float:
    """AUROC pela estatística de Mann-Whitney U, com tratamento de empates."""
    import numpy as np

    _validate_binary(y_true, y_score)
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    sorted_scores = y_score[order]

    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1

    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = len(y_true) - n_pos
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auroc_with_ci(
    y_true: IntArray,
    y_score: FloatArray,
    *,
    n_bootstrap: int = 2000,
    seed: int = 20260101,
) -> AUROCResult:
    """AUROC com IC95% por bootstrap estratificado.

    Estratificado por classe, para que cada reamostra preserve a prevalência e
    nunca degenere numa amostra de classe única.

    Args:
        seed: obrigatória e explícita. Duas execuções com a mesma seed produzem
            resultado idêntico — requisito de reprodutibilidade para V&V.
    """
    import numpy as np

    n_pos, n_neg = _validate_binary(y_true, y_score)
    point = auroc(y_true, y_score)

    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    samples = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = np.concatenate(
            [
                rng.choice(pos_idx, size=n_pos, replace=True),
                rng.choice(neg_idx, size=n_neg, replace=True),
            ]
        )
        samples[b] = auroc(y_true[idx], y_score[idx])

    low, high = np.percentile(samples, [2.5, 97.5])
    return AUROCResult(
        auroc=point,
        ci95_low=float(low),
        ci95_high=float(high),
        n_pos=n_pos,
        n_neg=n_neg,
        ci_method=f"bootstrap estratificado (n={n_bootstrap}, seed={seed})",
    )


def operating_point_at_sensitivity(
    y_true: IntArray, y_score: FloatArray, target_sensitivity: float = 0.90
) -> OperatingPoint:
    """Menor threshold que atinge a sensibilidade-alvo.

    Sensibilidade é priorizada porque, em triagem radiológica, o custo de um
    falso negativo em achado crítico domina (perigo H-01).
    """
    import numpy as np

    _validate_binary(y_true, y_score)
    if not 0.0 < target_sensitivity <= 1.0:
        raise EvaluationError(
            f"target_sensitivity deve estar em (0, 1]; recebido {target_sensitivity}"
        )

    thresholds = np.unique(y_score)[::-1]
    best: OperatingPoint | None = None
    for t in thresholds:
        pred = y_score >= t
        tp = int((pred & (y_true == 1)).sum())
        fn = int((~pred & (y_true == 1)).sum())
        tn = int((~pred & (y_true == 0)).sum())
        fp = int((pred & (y_true == 0)).sum())

        sens = tp / (tp + fn) if (tp + fn) else 0.0
        if sens < target_sensitivity:
            continue
        best = OperatingPoint(
            target_sensitivity=target_sensitivity,
            threshold=float(t),
            sensitivity=sens,
            specificity=tn / (tn + fp) if (tn + fp) else 0.0,
            ppv=tp / (tp + fp) if (tp + fp) else 0.0,
            npv=tn / (tn + fn) if (tn + fn) else 0.0,
        )
        break

    if best is None:
        raise EvaluationError(f"nenhum threshold atinge sensibilidade {target_sensitivity}")
    return best


def expected_calibration_error(y_true: IntArray, y_score: FloatArray, *, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) com binning de largura fixa.

    Um ECE alto significa que o escore **não** é probabilidade de doença — o que
    deve ser declarado na IFU (perigo H-10).
    """
    import numpy as np

    _validate_binary(y_true, y_score)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_score > lo) & (y_score <= hi) if i > 0 else (y_score >= lo) & (y_score <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(y_score[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += (count / n) * abs(accuracy - confidence)
    return ece
