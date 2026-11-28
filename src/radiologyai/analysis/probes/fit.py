"""Ajuste determinístico de sondas lineares (regressão logística sobre *embeddings*).

Receita (igual para toda sonda; pré-especificada):

1. padronização (média/variância) ajustada só no conjunto de treino;
2. ``LogisticRegression`` L2, ``lbfgs``, ``class_weight="balanced"``;
3. ``C`` escolhido na grade :data:`C_GRID` pela AUROC média numa validação
   cruzada interna de 5 dobras ESTRATIFICADA E AGRUPADA POR PACIENTE
   (``StratifiedGroupKFold``, semente fixa); empate → menor ``C``;
4. reajuste no treino inteiro com o ``C`` escolhido;
5. pontos de operação calculados nas predições FORA-DA-DOBRA do treino, nunca
   no conjunto de teste: exclusão (sensibilidade ≥ alvo do protocolo) e inclusão
   (especificidade ≥ alvo do protocolo), na escala logit da sonda.

Este módulo exige scikit-learn (extra ``eval``) e só é usado pelos scripts de
ajuste; aplicar uma sonda pronta não precisa dele (:mod:`.artifact`).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from radiologyai.errors import BackendUnavailableError, EvaluationError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = ["C_GRID", "FitResult", "fit_probe", "oof_operating_points"]

C_GRID: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


@dataclass(frozen=True)
class FitResult:
    """Sonda ajustada + diagnósticos da validação interna (não vão para o artefato)."""

    C: float
    mean: npt.NDArray[np.float64]
    scale: npt.NDArray[np.float64]
    coef: npt.NDArray[np.float64]
    intercept: float
    oof_logit: npt.NDArray[np.float64]
    inner_cv_auroc: dict[float, float]
    n_folds: int
    seed: int


def _require_sklearn() -> None:
    if importlib.util.find_spec("sklearn") is None:
        raise BackendUnavailableError(
            "scikit-learn não instalado; instale o extra: pip install 'radiologyai[eval]'"
        )


def _model(c: float) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", solver="lbfgs", max_iter=5000),
    )


def fit_probe(
    x: npt.NDArray[Any],
    y: npt.NDArray[Any],
    groups: npt.NDArray[Any],
    *,
    c_grid: tuple[float, ...] = C_GRID,
    n_folds: int = 5,
    seed: int = 20261124,
) -> FitResult:
    """Escolhe ``C`` por CV interna agrupada e reajusta no treino inteiro.

    Raises:
        EvaluationError: rótulos não binários, uma classe só, ou entradas não finitas.
    """
    _require_sklearn()
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(groups):
        raise EvaluationError("x, y e groups devem ter o mesmo número de linhas")
    if set(np.unique(y)) != {0, 1}:
        raise EvaluationError("a sonda exige rótulos binários com as duas classes presentes")
    if not np.isfinite(x).all():
        raise EvaluationError("embeddings não finitos")
    folds = list(
        StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(x, y, groups)
    )
    oof = {c: np.zeros(len(y), dtype=np.float64) for c in c_grid}
    aucs: dict[float, list[float]] = {c: [] for c in c_grid}
    for train_idx, val_idx in folds:
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
            raise EvaluationError("dobra interna sem as duas classes; reduza n_folds")
        for c in c_grid:
            model = _model(c).fit(x[train_idx], y[train_idx])
            score = model.decision_function(x[val_idx])
            oof[c][val_idx] = score
            aucs[c].append(float(roc_auc_score(y[val_idx], score)))
    mean_auc = {c: float(np.mean(v)) for c, v in aucs.items()}
    best = max(c_grid, key=lambda c: (round(mean_auc[c], 12), -c))
    final = _model(best).fit(x, y)
    scaler, clf = final.steps[0][1], final.steps[1][1]
    return FitResult(
        C=float(best),
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        coef=np.asarray(clf.coef_[0], dtype=np.float64),
        intercept=float(clf.intercept_[0]),
        oof_logit=oof[best],
        inner_cv_auroc=mean_auc,
        n_folds=n_folds,
        seed=seed,
    )


def oof_operating_points(
    oof_logit: npt.NDArray[Any],
    y: npt.NDArray[Any],
    *,
    sensitivity_target: float,
    specificity_target: float,
    source: str,
) -> list[dict[str, Any]]:
    """Pontos de exclusão e inclusão (escala logit) a partir das predições fora-da-dobra.

    * exclusão: o MAIOR limiar ``t`` com ``P(score >= t | y=1) >= sensitivity_target``;
    * inclusão: o MENOR limiar ``t`` com ``P(score < t | y=0) >= specificity_target``.

    Devolve dicionários no formato de ``OperatingPoint`` (sem métricas).
    """
    import numpy as np

    s = np.asarray(oof_logit, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    pos = np.sort(s[y == 1])
    neg = np.sort(s[y == 0])
    if pos.size == 0 or neg.size == 0:
        raise EvaluationError("pontos de operação exigem positivos e negativos")
    # sensibilidade >= alvo: no máximo floor((1-alvo)*n_pos) positivos abaixo do limiar
    k = int(np.floor((1.0 - sensitivity_target) * pos.size + 1e-9))
    rule_out = float(pos[k])
    # especificidade >= alvo: ao menos ceil(alvo*n_neg) negativos estritamente abaixo
    m = int(np.ceil(specificity_target * neg.size - 1e-9))
    rule_in = float(np.nextafter(neg[m - 1], np.inf)) if m > 0 else float(neg[0])
    return [
        {
            "name": "probe_rule_out",
            "role": "rule_out",
            "threshold": rule_out,
            "on": "logit",
            "direction": "ge",
            "source": f"{source}; sensibilidade-alvo {sensitivity_target} nas predições "
            "fora-da-dobra do treino",
        },
        {
            "name": "probe_rule_in",
            "role": "rule_in",
            "threshold": rule_in,
            "on": "logit",
            "direction": "ge",
            "source": f"{source}; especificidade-alvo {specificity_target} nas predições "
            "fora-da-dobra do treino",
        },
    ]
