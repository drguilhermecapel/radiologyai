"""Avaliação binária pelo desenho pré-registrado (``protocol.yaml``).

Desenho (``docs/protocolo-evidencia.md`` §4, §5 e §7):

1. o conjunto de avaliação é dividido **por paciente**, de forma determinística
   e com semente, em metade de **calibração** e metade de **teste**
   (:func:`patient_split`), estratificada pela presença de positivo no paciente;
2. na metade de calibração ajustam-se a temperatura
   (:func:`radiologyai.calibration.temperature.fit_temperature`), o ponto de
   exclusão (*rule-out*: sensibilidade-alvo) e o ponto de inclusão (*rule-in*:
   especificidade-alvo);
3. na metade de teste reportam-se AUROC e AUPRC, sensibilidade e
   especificidade **nos limiares fixados na calibração**, ECE antes e depois da
   calibração — todos com IC por bootstrap de paciente;
4. subgrupos (sexo, faixa etária com ``<18`` separada, incidência) na metade de
   teste, com contagens; abaixo do mínimo do protocolo o subgrupo é reportado
   sem julgamento.

Nada aqui decide aprovação: :mod:`radiologyai.evidence.measure` aplica os
limiares do protocolo sobre este relatório.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from radiologyai.errors import EvaluationError
from radiologyai.evaluation.dataset import AGE_BANDS, UNKNOWN_BAND, age_band
from radiologyai.evaluation.metrics import (
    ClusterBootstrap,
    bootstrap_method,
    cluster_bootstrap_samples,
    percentile_ci,
    threshold_for_sensitivity,
    threshold_for_specificity,
    weighted_auroc,
    weighted_average_precision,
    weighted_ece,
    weighted_rates_at,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from radiologyai.catalog.schema import Protocol

    FloatArray = npt.NDArray[np.floating[Any]]
    IntArray = npt.NDArray[np.integer[Any]]

ScoreScale = Literal["probability", "logit"]

MIN_SUBGROUP_SUPPORT = 10
"""Mínimo de imagens positivas e negativas para *calcular* métricas num subgrupo.

Não é critério de aprovação (subgrupo nunca aprova nada); evita apenas reportar
AUROC de 3 casos. O critério de elegibilidade é o do protocolo (§2).
"""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Estimate(_Frozen):
    """Estimativa pontual com IC (``None`` quando indefinido)."""

    value: float
    ci_low: float | None = None
    ci_high: float | None = None


class Counts(_Frozen):
    """Contagens por imagem e por paciente."""

    n_images: int
    n_patients: int
    n_pos: int
    n_neg: int
    n_pos_patients: int  # pacientes com ao menos uma imagem positiva
    n_neg_patients: int  # pacientes com ao menos uma imagem negativa


class FixedPointResult(_Frozen):
    """Ponto de operação fixado na calibração e medido no teste."""

    role: Literal["rule_out", "rule_in"]
    target_metric: Literal["sensitivity", "specificity"]
    target: float
    threshold: float  # na escala bruta do escore de entrada
    score_scale: ScoreScale
    calibration_sensitivity: float
    calibration_specificity: float
    sensitivity: Estimate  # metade de teste
    specificity: Estimate
    ppv: Estimate | None = None
    npv: Estimate | None = None


class CalibrationResult(_Frozen):
    temperature: float
    n_fit_samples: int
    nll_before: float
    nll_after: float


class SubgroupResult(_Frozen):
    attribute: str
    value: str
    counts: Counts
    eligible: bool  # contagens atingem os mínimos do protocolo
    auroc: Estimate | None = None
    rule_out_sensitivity: Estimate | None = None
    rule_out_specificity: Estimate | None = None
    rule_in_sensitivity: Estimate | None = None
    rule_in_specificity: Estimate | None = None
    note: str | None = None


class SplitSummary(_Frozen):
    method: str
    seed: int
    calibration_fraction: float
    calibration: Counts
    test: Counts


class BinaryProtocolReport(_Frozen):
    """Relatório completo de um rótulo binário sob o protocolo."""

    protocol_id: str
    label: str
    score_scale: ScoreScale
    ci_method: str
    ci_level: float
    split: SplitSummary
    counts: Counts  # metade de teste (onde as métricas são reportadas)
    auroc: Estimate
    auprc: Estimate
    prevalence: float
    calibration: CalibrationResult | None
    calibration_error: str | None
    ece_before: Estimate | None
    ece_after: Estimate | None
    rule_out: FixedPointResult | None
    rule_in: FixedPointResult | None
    operating_point_errors: tuple[str, ...]
    subgroups: tuple[SubgroupResult, ...]
    subgroups_unavailable: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Split determinístico por paciente
# --------------------------------------------------------------------------- #
def _patient_key(seed: int, patient_id: str) -> str:
    return hashlib.sha256(f"{seed}:{patient_id}".encode()).hexdigest()


def patient_split(
    patient_ids: Sequence[str],
    y_true: IntArray,
    *,
    seed: int,
    calibration_fraction: float,
) -> npt.NDArray[np.bool_]:
    """Máscara por imagem: ``True`` = metade de calibração.

    Determinística e independente da ordem das linhas: cada paciente recebe a
    chave ``sha256(f"{seed}:{patient_id}")``; dentro de cada estrato (tem algum
    positivo / só negativos) os pacientes são ordenados pela chave e os
    primeiros ``round(fração x n)`` vão para a calibração. Nenhum paciente
    aparece nas duas metades.
    """
    import numpy as np

    if not 0.0 < calibration_fraction < 1.0:
        raise EvaluationError(f"calibration_fraction deve estar em (0, 1); {calibration_fraction}")
    pids = [str(p) for p in patient_ids]
    if len(pids) != len(y_true):
        raise EvaluationError("patient_ids e y_true com tamanhos diferentes")
    positive_patients = {
        p for p, y in zip(pids, np.asarray(y_true).tolist(), strict=True) if y == 1
    }
    calibration: set[str] = set()
    for stratum in (
        sorted(positive_patients),
        sorted(set(pids) - positive_patients),
    ):
        ordered = sorted(stratum, key=lambda p: _patient_key(seed, p))
        k = round(calibration_fraction * len(ordered))
        calibration.update(ordered[:k])
    return np.array([p in calibration for p in pids], dtype=bool)


# --------------------------------------------------------------------------- #
def counts_for(patient_ids: Sequence[str], y_true: IntArray) -> Counts:
    import numpy as np

    y = np.asarray(y_true)
    pids = np.asarray([str(p) for p in patient_ids])
    return Counts(
        n_images=int(y.size),
        n_patients=len(set(pids.tolist())),
        n_pos=int((y == 1).sum()),
        n_neg=int((y == 0).sum()),
        n_pos_patients=len(set(pids[y == 1].tolist())),
        n_neg_patients=len(set(pids[y == 0].tolist())),
    )


def _estimate(point: float, samples: FloatArray, ci_level: float) -> Estimate:
    import math

    if not math.isfinite(point):
        raise EvaluationError("estimativa pontual indefinida")
    lo, hi = percentile_ci(samples, ci_level)
    return Estimate(value=float(point), ci_low=lo, ci_high=hi)


def _to_probability(scores: FloatArray, scale: ScoreScale) -> FloatArray:
    import numpy as np

    arr = np.asarray(scores, dtype=np.float64)
    if scale == "logit":
        out: FloatArray = 1.0 / (1.0 + np.exp(-arr))
        return out
    if arr.min() < 0.0 or arr.max() > 1.0:
        raise EvaluationError(
            "score_scale='probability' exige escores em [0, 1]; "
            f"observado [{arr.min():.4g}, {arr.max():.4g}]. Use score_scale='logit'."
        )
    return arr


class _Bootstrapper:
    """Encapsula reamostrador + parâmetros para estimativas com IC."""

    def __init__(
        self, patient_ids: Sequence[str], y: IntArray, *, seed: int, n_boot: int, ci: float
    ) -> None:
        self.resampler = ClusterBootstrap(list(patient_ids), seed=seed, strata=y)
        self.n_boot = n_boot
        self.ci = ci

    def auroc(self, y: IntArray, s: FloatArray) -> Estimate:
        import numpy as np

        point = float(weighted_auroc(y, s, np.ones(len(y)))[0])
        boot = cluster_bootstrap_samples(
            self.resampler, lambda w: weighted_auroc(y, s, w), self.n_boot
        )
        return _estimate(point, boot, self.ci)

    def auprc(self, y: IntArray, s: FloatArray) -> Estimate:
        import numpy as np

        point = float(weighted_average_precision(y, s, np.ones(len(y)))[0])
        boot = cluster_bootstrap_samples(
            self.resampler, lambda w: weighted_average_precision(y, s, w), self.n_boot
        )
        return _estimate(point, boot, self.ci)

    def ece(self, y: IntArray, p: FloatArray) -> Estimate:
        import numpy as np

        point = float(weighted_ece(y, p, np.ones(len(y)))[0])
        boot = cluster_bootstrap_samples(
            self.resampler, lambda w: weighted_ece(y, p, w), self.n_boot
        )
        return _estimate(point, boot, self.ci)

    def rates(self, y: IntArray, s: FloatArray, threshold: float) -> dict[str, Estimate | None]:
        import math

        import numpy as np

        point = weighted_rates_at(y, s, threshold, np.ones(len(y)))
        keys = ("sensitivity", "specificity", "ppv", "npv")
        samples: dict[str, list[FloatArray]] = {k: [] for k in keys}
        for block in self.resampler.weights(self.n_boot):
            r = weighted_rates_at(y, s, threshold, block)
            for k in keys:
                samples[k].append(r[k])
        out: dict[str, Estimate | None] = {}
        for k in keys:
            value = float(point[k][0])
            if not math.isfinite(value):
                out[k] = None
                continue
            out[k] = _estimate(value, np.concatenate(samples[k]), self.ci)
        return out


def _fixed_point(
    role: Literal["rule_out", "rule_in"],
    target: float,
    cal_y: IntArray,
    cal_s: FloatArray,
    test_boot: _Bootstrapper,
    test_y: IntArray,
    test_s: FloatArray,
    scale: ScoreScale,
) -> FixedPointResult:
    import numpy as np

    if role == "rule_out":
        threshold = threshold_for_sensitivity(cal_y, cal_s, target)
        metric: Literal["sensitivity", "specificity"] = "sensitivity"
    else:
        threshold = threshold_for_specificity(cal_y, cal_s, target)
        metric = "specificity"
    cal_rates = weighted_rates_at(cal_y, cal_s, threshold, np.ones(len(cal_y)))
    test_rates = test_boot.rates(test_y, test_s, threshold)
    sens, spec = test_rates["sensitivity"], test_rates["specificity"]
    if sens is None or spec is None:
        raise EvaluationError("sensibilidade/especificidade indefinidas na metade de teste")
    return FixedPointResult(
        role=role,
        target_metric=metric,
        target=target,
        threshold=threshold,
        score_scale=scale,
        calibration_sensitivity=float(cal_rates["sensitivity"][0]),
        calibration_specificity=float(cal_rates["specificity"][0]),
        sensitivity=sens,
        specificity=spec,
        ppv=test_rates["ppv"],
        npv=test_rates["npv"],
    )


def _subgroup_values(attribute: str, raw: Sequence[Any]) -> tuple[list[str], list[str]] | None:
    """Valor de subgrupo por imagem e a lista ordenada de valores; None se sem metadado."""
    if attribute == "age_band":
        values = [age_band(None if v is None else float(v)) for v in raw]
        if all(v == UNKNOWN_BAND for v in values):
            return None
        order = [b for b in (*AGE_BANDS, UNKNOWN_BAND) if b in set(values)]
        return values, order
    values = ["desconhecido" if v is None or str(v).strip() == "" else str(v) for v in raw]
    if all(v == "desconhecido" for v in values):
        return None
    return values, sorted(set(values))


def evaluate_binary(
    *,
    label: str,
    patient_ids: Sequence[str],
    y_true: IntArray,
    y_score: FloatArray,
    protocol: Protocol,
    score_scale: ScoreScale = "probability",
    subgroup_columns: Mapping[str, Sequence[Any]] | None = None,
    n_bootstrap: int | None = None,
) -> BinaryProtocolReport:
    """Avalia um rótulo binário pelo desenho pré-registrado.

    Args:
        patient_ids: id do paciente por imagem (cluster do bootstrap e do split).
        y_true: 0/1 por imagem. Imagens sem rótulo devem ser removidas antes.
        y_score: escore bruto do modelo, na escala ``score_scale``.
        subgroup_columns: ``{"sex": [...], "age": [...], "view": [...]}`` por
            imagem; a chave ``age`` (anos) gera o subgrupo ``age_band``.
        n_bootstrap: sobrescreve ``protocol.bootstrap_iterations`` (só testes).

    Raises:
        EvaluationError: dados degenerados em alguma das metades.
    """
    import numpy as np

    from radiologyai.calibration.temperature import fit_temperature

    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(y_score, dtype=np.float64)
    pids = [str(p) for p in patient_ids]
    if not (y.shape == s.shape and y.ndim == 1 and len(pids) == y.size):
        raise EvaluationError("patient_ids, y_true e y_score devem ter o mesmo comprimento")
    if not np.all(np.isfinite(s)):
        raise EvaluationError("y_score contém NaN/inf; nenhuma métrica é calculada")
    if not set(np.unique(y).tolist()) <= {0, 1}:
        raise EvaluationError("y_true deve ser 0/1 (remova as imagens sem rótulo antes)")
    p_all = _to_probability(s, score_scale)

    seed = protocol.random_seed
    n_boot = n_bootstrap or protocol.bootstrap_iterations
    ci = protocol.ci_level

    cal_mask = patient_split(pids, y, seed=seed, calibration_fraction=protocol.calibration_fraction)
    test_mask = ~cal_mask
    cal_pids = [p for p, m in zip(pids, cal_mask, strict=True) if m]
    test_pids = [p for p, m in zip(pids, test_mask, strict=True) if m]
    cal_y, cal_s, cal_p = y[cal_mask], s[cal_mask], p_all[cal_mask]
    test_y, test_s, test_p = y[test_mask], s[test_mask], p_all[test_mask]
    for name, part in (("calibração", cal_y), ("teste", test_y)):
        if part.size == 0 or part.min() == part.max():
            raise EvaluationError(
                f"metade de {name} sem as duas classes ({int(part.sum())} positivos em "
                f"{part.size}); o desenho pré-registrado não pode ser aplicado"
            )

    test_counts = counts_for(test_pids, test_y)
    split = SplitSummary(
        method="sha256(seed:patient_id), estratificado por paciente com positivo",
        seed=seed,
        calibration_fraction=protocol.calibration_fraction,
        calibration=counts_for(cal_pids, cal_y),
        test=test_counts,
    )

    boot = _Bootstrapper(test_pids, test_y, seed=seed, n_boot=n_boot, ci=ci)
    auroc_est = boot.auroc(test_y, test_s)
    auprc_est = boot.auprc(test_y, test_s)

    calibration: CalibrationResult | None = None
    calibration_error: str | None = None
    ece_before = boot.ece(test_y, test_p)
    ece_after: Estimate | None = None
    try:
        calibrator = fit_temperature(cal_y, cal_p, label=label)
        calibration = CalibrationResult(
            temperature=calibrator.temperature,
            n_fit_samples=calibrator.n_fit_samples,
            nll_before=calibrator.nll_before,
            nll_after=calibrator.nll_after,
        )
        ece_after = boot.ece(test_y, calibrator.apply(test_p))
    except EvaluationError as exc:
        calibration_error = str(exc)

    op_errors: list[str] = []
    rule_out: FixedPointResult | None = None
    rule_in: FixedPointResult | None = None
    try:
        rule_out = _fixed_point(
            "rule_out",
            protocol.rule_out_sensitivity_target,
            cal_y,
            cal_s,
            boot,
            test_y,
            test_s,
            score_scale,
        )
    except EvaluationError as exc:
        op_errors.append(f"rule_out: {exc}")
    try:
        rule_in = _fixed_point(
            "rule_in",
            protocol.rule_in_specificity_target,
            cal_y,
            cal_s,
            boot,
            test_y,
            test_s,
            score_scale,
        )
    except EvaluationError as exc:
        op_errors.append(f"rule_in: {exc}")

    subgroups: list[SubgroupResult] = []
    unavailable: list[str] = []
    columns = dict(subgroup_columns or {})
    attr_source = {"sex": "sex", "age_band": "age", "view": "view"}
    for attribute in protocol.subgroups:
        source = attr_source.get(attribute, attribute)
        raw = columns.get(source)
        if raw is None:
            unavailable.append(f"{attribute}: subgrupo não avaliável: metadado ausente")
            continue
        if len(raw) != y.size:
            raise EvaluationError(f"coluna de subgrupo {source!r} com tamanho diferente")
        test_raw = [v for v, m in zip(raw, test_mask, strict=True) if m]
        parsed = _subgroup_values(attribute, test_raw)
        if parsed is None:
            unavailable.append(f"{attribute}: subgrupo não avaliável: metadado ausente")
            continue
        values, order = parsed
        values_arr = np.asarray(values)
        for value in order:
            m = values_arr == value
            sub_pids = [p for p, keep in zip(test_pids, m, strict=True) if keep]
            sub_y, sub_s = test_y[m], test_s[m]
            c = counts_for(sub_pids, sub_y)
            eligible = (
                c.n_pos_patients >= protocol.min_positive
                and c.n_neg_patients >= protocol.min_negative
            )
            kwargs: dict[str, Any] = {}
            note = None if eligible else "abaixo do mínimo do protocolo: reportado sem julgamento"
            if c.n_pos >= MIN_SUBGROUP_SUPPORT and c.n_neg >= MIN_SUBGROUP_SUPPORT:
                sb = _Bootstrapper(sub_pids, sub_y, seed=seed, n_boot=n_boot, ci=ci)
                kwargs["auroc"] = sb.auroc(sub_y, sub_s)
                if rule_out is not None:
                    r = sb.rates(sub_y, sub_s, rule_out.threshold)
                    kwargs["rule_out_sensitivity"] = r["sensitivity"]
                    kwargs["rule_out_specificity"] = r["specificity"]
                if rule_in is not None:
                    r = sb.rates(sub_y, sub_s, rule_in.threshold)
                    kwargs["rule_in_sensitivity"] = r["sensitivity"]
                    kwargs["rule_in_specificity"] = r["specificity"]
            else:
                note = (
                    f"menos de {MIN_SUBGROUP_SUPPORT} imagens positivas ou negativas: "
                    "apenas contagens"
                )
            subgroups.append(
                SubgroupResult(
                    attribute=attribute,
                    value=value,
                    counts=c,
                    eligible=eligible,
                    note=note,
                    **kwargs,
                )
            )

    return BinaryProtocolReport(
        protocol_id=protocol.protocol_id,
        label=label,
        score_scale=score_scale,
        ci_method=bootstrap_method(n_boot, seed, ci),
        ci_level=ci,
        split=split,
        counts=test_counts,
        auroc=auroc_est,
        auprc=auprc_est,
        prevalence=test_counts.n_pos / test_counts.n_images,
        calibration=calibration,
        calibration_error=calibration_error,
        ece_before=ece_before,
        ece_after=ece_after,
        rule_out=rule_out,
        rule_in=rule_in,
        operating_point_errors=tuple(op_errors),
        subgroups=tuple(subgroups),
        subgroups_unavailable=tuple(unavailable),
    )
