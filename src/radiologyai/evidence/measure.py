"""Medição genérica -> :class:`~radiologyai.catalog.schema.EvidenceRecord`.

O avaliador recebe dados reais (rótulos de referência e saídas do modelo) e
devolve um registro com proveniência completa. Aprovação (``passes_protocol``)
e motivos de reprovação (``failure_reasons``) são decididos **somente** pelos
limiares do protocolo carregado — não há parâmetro para forçá-los.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, field_validator

from radiologyai.calibration.decision import OperatingPoint
from radiologyai.catalog.schema import (
    Endpoint,
    EvidenceRecord,
    MeasurementProvenance,
    Metric,
    Protocol,
    ReferenceStandard,
)
from radiologyai.errors import EvidenceError
from radiologyai.evaluation.agreement import AgreementReport, evaluate_agreement
from radiologyai.evaluation.detection import DetectionReport, evaluate_detection
from radiologyai.evaluation.protocol_eval import (
    BinaryProtocolReport,
    Estimate,
    ScoreScale,
    evaluate_binary,
)
from radiologyai.evaluation.segmentation import SegmentationReport, evaluate_dice
from radiologyai.evidence.leakage import assess_leakage

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    FloatArray = npt.NDArray[np.floating[Any]]
    IntArray = npt.NDArray[np.integer[Any]]

FORMAT = "radiologyai.evidence/measured-v1"

BASE_CAVEATS: tuple[str, ...] = (
    "Medição retrospectiva de desempenho de algoritmo isolado; NÃO é validação clínica.",
    "Achado sugerido para confirmação médica; nunca diagnóstico.",
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class MeasurementContext(_Frozen):
    """O que identifica uma medição e permite reproduzi-la."""

    model_id: str
    label: str
    condition_id: str
    dataset_id: str  # chave em radiologyai.evidence.leakage.DATASETS
    training_data: tuple[str, ...]  # chaves de dataset de treino do modelo
    internal_holdout: bool = False  # hold-out por paciente do próprio dataset de treino
    reference_standard: ReferenceStandard | None = None
    population: str | None = None
    weights_sha256: dict[str, str]
    git_sha: str
    git_dirty: bool = False
    command: str
    dataset_sha256: str  # hash do manifesto (todas as colunas exceto o escore)
    created_at: datetime
    caveats: tuple[str, ...] = ()

    @field_validator("command")
    @classmethod
    def _command(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("comando exato da medição é obrigatório")
        return v


class MeasuredEvidence(_Frozen):
    """Registro + relatório completo + pontos de operação derivados."""

    format: str = FORMAT
    record: EvidenceRecord
    report: dict[str, Any]
    operating_points: tuple[OperatingPoint, ...] = ()
    inputs: dict[str, Any] = {}


_PT = {"sensitivity": "sensibilidade", "specificity": "especificidade"}


def _metric(e: Estimate | None) -> Metric | None:
    if e is None:
        return None
    return Metric(value=e.value, ci_low=e.ci_low, ci_high=e.ci_high)


def _fmt(x: float) -> str:
    return f"{x:.4f}"


# --------------------------------------------------------------------------- #
# Julgamento — exclusivamente pelos limiares do protocolo
# --------------------------------------------------------------------------- #
def _eligibility(n_pos: int, n_neg: int, protocol: Protocol, unit: str) -> list[str]:
    reasons: list[str] = []
    if n_pos < protocol.min_positive:
        reasons.append(f"{unit} positivos {n_pos} < min_positive {protocol.min_positive}")
    if n_neg < protocol.min_negative:
        reasons.append(f"{unit} negativos {n_neg} < min_negative {protocol.min_negative}")
    return reasons


def judge_discrimination(report: BinaryProtocolReport, protocol: Protocol) -> list[str]:
    """Motivos de reprovação; lista vazia = aprovado.

    Elegibilidade conta **pacientes** (com ao menos uma imagem positiva / negativa)
    na metade de teste, porque o desfecho é por imagem e há várias por paciente
    (protocolo §2).
    """
    c = report.counts
    reasons = _eligibility(c.n_pos_patients, c.n_neg_patients, protocol, "pacientes")
    a = report.auroc
    if a.value < protocol.auroc_min:
        reasons.append(f"AUROC {_fmt(a.value)} < auroc_min {protocol.auroc_min}")
    if a.ci_low is None or a.ci_low < protocol.auroc_ci_low_min:
        low = "indefinido" if a.ci_low is None else _fmt(a.ci_low)
        reasons.append(
            f"limite inferior do IC da AUROC {low} < auroc_ci_low_min {protocol.auroc_ci_low_min}"
        )
    return reasons


def judge_detection(report: DetectionReport, protocol: Protocol) -> list[str]:
    c = report.counts
    reasons = _eligibility(c.n_pos_patients, c.n_neg_patients, protocol, "pacientes")
    bar = protocol.sens_spec_ci_low_min
    for name, est in (
        ("sensibilidade", report.sensitivity),
        ("especificidade", report.specificity),
    ):
        if est.ci_low is None or est.ci_low < bar:
            low = "indefinido" if est.ci_low is None else _fmt(est.ci_low)
            reasons.append(f"limite inferior do IC da {name} {low} < sens_spec_ci_low_min {bar}")
    return reasons


def judge_agreement(report: AgreementReport, protocol: Protocol) -> list[str]:
    """ICC pontual >= ``icc_min``; elegibilidade: pacientes >= ``min_positive``.

    O protocolo ``2026-11-v1`` não fixa mínimo específico para concordância; usa-se
    ``min_positive`` (casos com medida de referência), declarado no registro.
    """
    reasons: list[str] = []
    if report.n_patients < protocol.min_positive:
        reasons.append(f"pacientes {report.n_patients} < min_positive {protocol.min_positive}")
    if report.icc.value < protocol.icc_min:
        reasons.append(f"{report.icc_form} {_fmt(report.icc.value)} < icc_min {protocol.icc_min}")
    return reasons


def judge_segmentation(report: SegmentationReport, protocol: Protocol) -> list[str]:
    reasons: list[str] = []
    if report.n_patients < protocol.min_positive:
        reasons.append(f"pacientes {report.n_patients} < min_positive {protocol.min_positive}")
    if report.dice_mean < protocol.segmentation_dice_min:
        reasons.append(
            f"Dice médio {_fmt(report.dice_mean)} < segmentation_dice_min "
            f"{protocol.segmentation_dice_min}"
        )
    return reasons


# --------------------------------------------------------------------------- #
def record_id(ctx: MeasurementContext, endpoint: Endpoint) -> str:
    return f"measured:{ctx.dataset_id}:{ctx.model_id}:{ctx.label}:{endpoint}"


def _build_record(
    ctx: MeasurementContext,
    protocol: Protocol,
    *,
    endpoint: Endpoint,
    metrics: dict[str, Metric | None],
    n_total: int,
    n_positive: int,
    reasons: list[str],
    extra_caveats: Sequence[str] = (),
) -> EvidenceRecord:
    leakage, note = assess_leakage(ctx.dataset_id, ctx.training_data)
    caveats = [*BASE_CAVEATS, *extra_caveats, *ctx.caveats]
    if ctx.git_dirty:
        caveats.append(
            "ATENÇÃO: medido com modificações não commitadas sobre o git_sha; "
            "não reproduzível a partir do sha. Não deve ser commitado como evidência."
        )
    if ctx.reference_standard == "report_nlp":
        caveats.append("Padrão de referência minerado de laudos por NLP (rótulo ruidoso).")
    if ctx.internal_holdout and leakage == "none":
        raise EvidenceError(
            "internal_holdout exige que o dataset de avaliação esteja na linhagem do treino; "
            f"{ctx.dataset_id!r} não está em {ctx.training_data}"
        )
    return EvidenceRecord(
        id=record_id(ctx, endpoint),
        condition_id=ctx.condition_id,
        model_id=ctx.model_id,
        label=ctx.label,
        kind="measured_internal" if ctx.internal_holdout else "measured_external",
        endpoint=endpoint,
        dataset=ctx.dataset_id,
        population=ctx.population,
        reference_standard=ctx.reference_standard,
        n_total=n_total,
        n_positive=n_positive,
        metrics={k: v for k, v in metrics.items() if v is not None},
        leakage=leakage,
        leakage_note=note,
        passes_protocol=not reasons,
        failure_reasons=tuple(reasons),
        provenance=MeasurementProvenance(
            git_sha=ctx.git_sha,
            protocol_id=protocol.protocol_id,
            dataset_id=ctx.dataset_id,
            dataset_sha256=ctx.dataset_sha256,
            weights_sha256=dict(ctx.weights_sha256),
            command=ctx.command,
            created_at=ctx.created_at,
        ),
        caveats=tuple(caveats),
    )


def measure_binary(
    ctx: MeasurementContext,
    protocol: Protocol,
    *,
    patient_ids: Sequence[str],
    y_true: IntArray,
    y_score: FloatArray,
    score_scale: ScoreScale = "probability",
    subgroup_columns: Mapping[str, Sequence[Any]] | None = None,
    n_bootstrap: int | None = None,
    inputs: dict[str, Any] | None = None,
    evidence_ref: str | None = None,
) -> MeasuredEvidence:
    """Desfecho de discriminação pelo desenho pré-registrado (calibração/teste)."""
    report = evaluate_binary(
        label=ctx.label,
        patient_ids=patient_ids,
        y_true=y_true,
        y_score=y_score,
        protocol=protocol,
        score_scale=score_scale,
        subgroup_columns=subgroup_columns,
        n_bootstrap=n_bootstrap,
    )
    reasons = judge_discrimination(report, protocol)
    metrics: dict[str, Metric | None] = {
        "auroc": _metric(report.auroc),
        "auprc": _metric(report.auprc),
        "ece_before_calibration": _metric(report.ece_before),
        "ece_after_calibration": _metric(report.ece_after),
    }
    points: list[OperatingPoint] = []
    on = "sigmoid" if score_scale == "probability" else "logit"
    for fp in (report.rule_out, report.rule_in):
        if fp is None:
            continue
        metrics[f"{fp.role}_sensitivity"] = _metric(fp.sensitivity)
        metrics[f"{fp.role}_specificity"] = _metric(fp.specificity)
        points.append(
            OperatingPoint(
                name=f"{ctx.label}:{fp.role}",
                role=fp.role,
                threshold=fp.threshold,
                on=on,  # type: ignore[arg-type]
                source=(
                    f"protocolo {protocol.protocol_id}: "
                    f"{_PT[fp.target_metric]} alvo {fp.target} "
                    f"ajustado na metade de calibração de {ctx.dataset_id}"
                ),
                evidence_ref=evidence_ref,
            )
        )
    split = report.split
    caveats = [
        f"Limiares e temperatura ajustados na metade de calibração "
        f"({split.calibration.n_patients} pacientes) e métricas reportadas na metade de "
        f"teste ({split.test.n_patients} pacientes), split determinístico por paciente.",
        f"IC: {report.ci_method}.",
    ]
    if report.calibration_error:
        caveats.append(f"Calibração não ajustada: {report.calibration_error}")
    caveats.extend(report.operating_point_errors)
    record = _build_record(
        ctx,
        protocol,
        endpoint="discrimination",
        metrics=metrics,
        n_total=report.counts.n_images,
        n_positive=report.counts.n_pos,
        reasons=reasons,
        extra_caveats=caveats,
    )
    return MeasuredEvidence(
        record=record,
        report=report.model_dump(mode="json"),
        operating_points=tuple(points),
        inputs=dict(inputs or {}),
    )


def measure_detection(
    ctx: MeasurementContext,
    protocol: Protocol,
    *,
    rule: str,
    patient_ids: Sequence[str],
    y_true: IntArray,
    y_pred: IntArray,
    inputs: dict[str, Any] | None = None,
) -> MeasuredEvidence:
    """Desfecho de detecção numa regra fixa pré-especificada (sem ajuste nos dados)."""
    report = evaluate_detection(
        rule=rule,
        patient_ids=patient_ids,
        y_true=y_true,
        y_pred=y_pred,
        seed=protocol.random_seed,
        n_bootstrap=protocol.bootstrap_iterations,
        ci_level=protocol.ci_level,
    )
    record = _build_record(
        ctx,
        protocol,
        endpoint="detection",
        metrics={
            "sensitivity": _metric(report.sensitivity),
            "specificity": _metric(report.specificity),
            "ppv": _metric(report.ppv),
            "npv": _metric(report.npv),
        },
        n_total=report.counts.n_images,
        n_positive=report.counts.n_pos,
        reasons=judge_detection(report, protocol),
        extra_caveats=[f"Regra pré-especificada: {rule}", f"IC: {report.ci_method}."],
    )
    return MeasuredEvidence(
        record=record, report=report.model_dump(mode="json"), inputs=dict(inputs or {})
    )


def measure_agreement(
    ctx: MeasurementContext,
    protocol: Protocol,
    *,
    patient_ids: Sequence[str],
    measured: FloatArray,
    reference: FloatArray,
    inputs: dict[str, Any] | None = None,
) -> MeasuredEvidence:
    """Concordância de medida contínua (ICC(2,1), Bland-Altman, MAE)."""
    report = evaluate_agreement(
        patient_ids=patient_ids,
        measured=measured,
        reference=reference,
        seed=protocol.random_seed,
        n_bootstrap=protocol.bootstrap_iterations,
        ci_level=protocol.ci_level,
    )

    def m(i: Any) -> Metric:
        return Metric(value=i.value, ci_low=i.ci_low, ci_high=i.ci_high)

    record = _build_record(
        ctx,
        protocol,
        endpoint="agreement",
        metrics={
            "icc": m(report.icc),
            "bland_altman_bias": m(report.bias),
            "bland_altman_loa_low": m(report.loa_low),
            "bland_altman_loa_high": m(report.loa_high),
            "mae": m(report.mae),
        },
        n_total=report.n_cases,
        n_positive=report.n_cases,
        reasons=judge_agreement(report, protocol),
        extra_caveats=[
            f"{report.icc_form}: dois fatores, efeitos aleatórios, concordância absoluta.",
            "n_positive = casos com medida de referência (desfecho contínuo).",
            f"IC: {report.ci_method}.",
        ],
    )
    return MeasuredEvidence(
        record=record, report=report.model_dump(mode="json"), inputs=dict(inputs or {})
    )


def measure_segmentation(
    ctx: MeasurementContext,
    protocol: Protocol,
    *,
    patient_ids: Sequence[str],
    dice_per_case: FloatArray,
    inputs: dict[str, Any] | None = None,
) -> MeasuredEvidence:
    """Segmentação contra máscara de especialista (Dice médio por caso)."""
    report = evaluate_dice(
        patient_ids=patient_ids,
        dice_per_case=dice_per_case,
        seed=protocol.random_seed,
        n_bootstrap=protocol.bootstrap_iterations,
        ci_level=protocol.ci_level,
    )
    record = _build_record(
        ctx,
        protocol,
        endpoint="segmentation",
        metrics={
            "dice_mean": Metric(
                value=report.dice_mean, ci_low=report.dice_ci_low, ci_high=report.dice_ci_high
            ),
            "dice_median": Metric(value=report.dice_median),
        },
        n_total=report.n_cases + report.n_both_empty,
        n_positive=report.n_cases,
        reasons=judge_segmentation(report, protocol),
        extra_caveats=[
            f"{report.n_both_empty} casos com referência e predição vazias excluídos da média.",
            f"IC: {report.ci_method}.",
        ],
    )
    return MeasuredEvidence(
        record=record, report=report.model_dump(mode="json"), inputs=dict(inputs or {})
    )
