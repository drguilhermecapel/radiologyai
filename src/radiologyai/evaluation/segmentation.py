"""Segmentação contra máscara de especialista: Dice por caso e média com IC.

Caso em que referência e predição estão **ambas vazias** não tem Dice definido
(0/0). Não é contado como acerto perfeito nem como erro: é excluído da média e
declarado em ``n_both_empty``. Atribuir 1,0 a esses casos inflaria a média com
um número que nenhuma sobreposição produziu.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from radiologyai.errors import EvaluationError
from radiologyai.evaluation.metrics import ClusterBootstrap, bootstrap_method, percentile_ci

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    FloatArray = npt.NDArray[np.floating[Any]]


class SegmentationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_cases: int  # casos com Dice definido
    n_patients: int
    n_both_empty: int  # excluídos: referência e predição vazias
    dice_mean: float
    dice_ci_low: float
    dice_ci_high: float
    dice_median: float
    ci_method: str


def dice_coefficient(prediction: npt.NDArray[Any], reference: npt.NDArray[Any]) -> float:
    """Dice ``2|A∩B| / (|A|+|B|)`` entre máscaras binárias; NaN se ambas vazias."""
    import numpy as np

    a = np.asarray(prediction).astype(bool)
    b = np.asarray(reference).astype(bool)
    if a.shape != b.shape:
        raise EvaluationError(f"máscaras com formas diferentes: {a.shape} vs {b.shape}")
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return float("nan")
    return 2.0 * int((a & b).sum()) / denom


def evaluate_dice(
    *,
    patient_ids: Sequence[str],
    dice_per_case: FloatArray,
    seed: int,
    n_bootstrap: int,
    ci_level: float = 0.95,
) -> SegmentationReport:
    """Média de Dice por caso com IC por bootstrap de paciente."""
    import numpy as np

    d = np.asarray(dice_per_case, dtype=np.float64)
    if d.ndim != 1 or d.size != len(patient_ids):
        raise EvaluationError("dice_per_case e patient_ids com tamanhos diferentes")
    defined = np.isfinite(d)
    if np.any((d[defined] < 0) | (d[defined] > 1)):
        raise EvaluationError("Dice fora de [0, 1]")
    pids = np.asarray([str(p) for p in patient_ids])[defined]
    values = d[defined]
    if values.size < 2:
        raise EvaluationError("menos de 2 casos com Dice definido")
    resampler = ClusterBootstrap(pids.tolist(), seed=seed)
    samples = np.concatenate(
        [(block @ values) / block.sum(axis=1) for block in resampler.weights(n_bootstrap)]
    )
    lo, hi = percentile_ci(samples, ci_level)
    return SegmentationReport(
        n_cases=int(values.size),
        n_patients=len(set(pids.tolist())),
        n_both_empty=int((~defined).sum()),
        dice_mean=float(values.mean()),
        dice_ci_low=lo,
        dice_ci_high=hi,
        dice_median=float(np.median(values)),
        ci_method=bootstrap_method(n_bootstrap, seed, ci_level).replace(
            ", estratificado por presença de positivo", ""
        ),
    )
