"""Concordância de medidas contínuas: ICC, Bland-Altman e erro absoluto médio.

Usado quando o modelo produz uma **medida** (fração de ejeção, diâmetro,
volume) comparada com uma medida de referência clínica. O protocolo
(``icc_min``) julga pelo ICC; Bland-Altman e MAE são reportados porque um ICC
alto pode conviver com viés sistemático clinicamente relevante.

Forma do ICC: **ICC(2,1)** de Shrout & Fleiss (1979) — dois fatores, efeitos
aleatórios, concordância absoluta, medida única; é o ``ICC(A,1)`` de McGraw &
Wong (1996). É a forma indicada para concordância entre métodos de medida
(Koo & Li 2016, DOI 10.1016/j.jcm.2016.02.012): penaliza diferença sistemática
entre modelo e referência, o que a forma de consistência (3,1) não faz. As
demais formas estão disponíveis em :func:`icc` para conferência.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from radiologyai.errors import EvaluationError
from radiologyai.evaluation.metrics import ClusterBootstrap, bootstrap_interval, bootstrap_method

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    FloatArray = npt.NDArray[np.floating[Any]]

ICCForm = Literal["1,1", "2,1", "3,1", "1,k", "2,k", "3,k"]
LOA_Z = 1.96


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Interval(_Frozen):
    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    # réplicas de bootstrap com estatística definida / total (None: sem bootstrap)
    n_defined_replicates: int | None = None
    n_replicates: int | None = None


class AgreementReport(_Frozen):
    n_cases: int  # casos com medida do modelo E referência (entram na concordância)
    n_patients: int  # pacientes desses casos
    # Abstenções: casos com referência e SEM medida do modelo. Contam no denominador
    # (n_cases_with_reference) e na elegibilidade; nunca somem do relato.
    n_abstained: int = 0
    n_cases_with_reference: int | None = None  # None = relatório antigo, sem a contagem
    n_patients_with_reference: int | None = None
    icc_form: str
    icc: Interval
    bias: Interval  # média de (modelo - referência)
    loa_low: Interval  # limites de concordância de 95% de Bland-Altman
    loa_high: Interval
    mae: Interval
    ci_method: str


def icc(ratings: FloatArray, form: ICCForm = "2,1") -> float:
    """Coeficiente de correlação intraclasse (Shrout & Fleiss 1979).

    Args:
        ratings: matriz ``n_alvos x k_avaliadores`` sem valores ausentes.

    Raises:
        EvaluationError: menos de 2 alvos ou 2 avaliadores, ou valores não finitos.
    """
    import numpy as np

    x = np.asarray(ratings, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        raise EvaluationError("ICC exige matriz n x k com n >= 2 e k >= 2")
    if not np.all(np.isfinite(x)):
        raise EvaluationError("ICC com valores ausentes/não finitos não é calculado")
    n, k = x.shape
    grand = x.mean()
    ss_rows = k * ((x.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((x.mean(axis=0) - grand) ** 2).sum()
    ss_total = ((x - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    msw = (ss_cols + ss_err) / (n * (k - 1))
    formulas = {
        "1,1": (msr - msw, msr + (k - 1) * msw),
        "2,1": (msr - mse, msr + (k - 1) * mse + k * (msc - mse) / n),
        "3,1": (msr - mse, msr + (k - 1) * mse),
        "1,k": (msr - msw, msr),
        "2,k": (msr - mse, msr + (msc - mse) / n),
        "3,k": (msr - mse, msr),
    }
    num, den = formulas[form]
    if den == 0:
        raise EvaluationError("ICC indefinido: variância nula")
    return float(num / den)


def bland_altman(measured: FloatArray, reference: FloatArray) -> tuple[float, float, float]:
    """``(viés, LoA inferior, LoA superior)`` com DP amostral (ddof=1)."""
    import numpy as np

    d = np.asarray(measured, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    if d.size < 2:
        raise EvaluationError("Bland-Altman exige ao menos 2 pares")
    bias = float(d.mean())
    sd = float(d.std(ddof=1))
    return bias, bias - LOA_Z * sd, bias + LOA_Z * sd


def mean_absolute_error(measured: FloatArray, reference: FloatArray) -> float:
    import numpy as np

    return float(
        np.abs(
            np.asarray(measured, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
        ).mean()
    )


def _stats(m: FloatArray, r: FloatArray, form: ICCForm) -> tuple[float, float, float, float, float]:
    import numpy as np

    value = icc(np.stack([m, r], axis=1), form)
    bias, lo, hi = bland_altman(m, r)
    return value, bias, lo, hi, mean_absolute_error(m, r)


def evaluate_agreement(
    *,
    patient_ids: Sequence[str],
    measured: FloatArray,
    reference: FloatArray,
    seed: int,
    n_bootstrap: int,
    ci_level: float = 0.95,
    form: ICCForm = "2,1",
    abstained_patient_ids: Sequence[str] = (),
) -> AgreementReport:
    """ICC, Bland-Altman e MAE com IC por bootstrap de paciente.

    ``abstained_patient_ids``: um id de paciente por caso com referência e sem medida
    do modelo. Não entram no ICC (não há par), mas são contados no relatório para que
    a elegibilidade e o limite de abstenção os vejam.
    """
    import numpy as np

    m = np.asarray(measured, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if m.shape != r.shape or m.ndim != 1 or m.size != len(patient_ids):
        raise EvaluationError("measured, reference e patient_ids devem ter o mesmo comprimento")
    if not (np.all(np.isfinite(m)) and np.all(np.isfinite(r))):
        raise EvaluationError("medidas não finitas; remova os casos sem medida antes")
    point = _stats(m, r, form)
    resampler = ClusterBootstrap([str(p) for p in patient_ids], seed=seed)
    samples: list[tuple[float, ...]] = []
    for idx in resampler.indices(n_bootstrap):
        try:
            samples.append(_stats(m[idx], r[idx], form))
        except EvaluationError:
            samples.append((float("nan"),) * 5)
    arr = np.asarray(samples, dtype=np.float64)

    def interval(i: int) -> Interval:
        ci = bootstrap_interval(arr[:, i], ci_level)
        return Interval(
            value=point[i],
            ci_low=ci.low,
            ci_high=ci.high,
            n_defined_replicates=ci.n_defined,
            n_replicates=ci.n_replicates,
        )

    kept_patients = {str(p) for p in patient_ids}
    abstained = [str(p) for p in abstained_patient_ids]
    return AgreementReport(
        n_cases=int(m.size),
        n_patients=len(kept_patients),
        n_abstained=len(abstained),
        n_cases_with_reference=int(m.size) + len(abstained),
        n_patients_with_reference=len(kept_patients | set(abstained)),
        icc_form=f"ICC({form})",
        icc=interval(0),
        bias=interval(1),
        loa_low=interval(2),
        loa_high=interval(3),
        mae=interval(4),
        ci_method=bootstrap_method(n_bootstrap, seed, ci_level).replace(
            ", estratificado por presença de positivo", ""
        ),
    )
