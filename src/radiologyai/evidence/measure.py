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
from radiologyai.evaluation.metrics import MIN_DEFINED_REPLICATE_FRACTION
from radiologyai.evaluation.protocol_eval import (
    BinaryProtocolReport,
    Estimate,
    ScoreScale,
    evaluate_binary,
)
from radiologyai.evaluation.segmentation import SegmentationReport, evaluate_dice
from radiologyai.evidence.leakage import (
    assess_leakage,
    assess_training_leakage,
    selection_overlap,
)

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
DIRTY_CAVEAT = (
    "ATENÇÃO: medido com modificações não commitadas sobre o git_sha; "
    "não reproduzível a partir do sha. Não deve ser commitado como evidência."
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
    # chaves dos datasets em que o modelo/codificador/hiperparâmetro foi ESCOLHIDO
    selection_data: tuple[str, ...] = ()
    internal_holdout: bool = False  # hold-out por paciente do próprio dataset de treino
    # entrada do modelo derivada do padrão de referência (descrição); None = independente
    reference_derived_input: str | None = None
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


class MeasurementDeclaration(_Frozen):
    """O que foi DECLARADO antes de medir e decide vazamento e tipo do registro.

    Gravado no envelope para que :func:`rejudge` (e ``levels.json``) recalcule
    ``leakage``/``kind`` em vez de confiar no que o JSON diz.
    """

    training_data: tuple[str, ...]
    selection_data: tuple[str, ...] = ()
    internal_holdout: bool = False
    reference_derived_input: str | None = None
    git_dirty: bool = False

    @classmethod
    def from_context(cls, ctx: MeasurementContext) -> MeasurementDeclaration:
        return cls(
            training_data=ctx.training_data,
            selection_data=ctx.selection_data,
            internal_holdout=ctx.internal_holdout,
            reference_derived_input=ctx.reference_derived_input,
            git_dirty=ctx.git_dirty,
        )


class MeasuredEvidence(_Frozen):
    """Registro + relatório completo + pontos de operação derivados."""

    format: str = FORMAT
    record: EvidenceRecord
    report: dict[str, Any]
    operating_points: tuple[OperatingPoint, ...] = ()
    inputs: dict[str, Any] = {}
    # None = envelope anterior às declarações: não pode ser re-julgado (re-execute)
    declared: MeasurementDeclaration | None = None


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
ELIGIBILITY_PREFIX = "não elegível: "


def _eligibility(n_pos: int, n_neg: int, protocol: Protocol, unit: str) -> list[str]:
    reasons: list[str] = []
    if n_pos < protocol.min_positive:
        reasons.append(
            f"{ELIGIBILITY_PREFIX}{unit} positivos {n_pos} < min_positive {protocol.min_positive}"
        )
    if n_neg < protocol.min_negative:
        reasons.append(
            f"{ELIGIBILITY_PREFIX}{unit} negativos {n_neg} < min_negative {protocol.min_negative}"
        )
    return reasons


def verdict(reasons: Sequence[str]) -> bool | None:
    """``passes_protocol`` a partir dos motivos: ``None`` se o registro não é elegível.

    Protocolo §2: abaixo dos mínimos de casos o registro "não é elegível para A nem
    C, qualquer que seja a estimativa pontual". Também não pode **refutar** a
    condição: amostra insuficiente não é evidência contra o modelo. Por isso o
    veredito é indefinido (``None``), e os motivos continuam declarados.
    """
    if any(r.startswith(ELIGIBILITY_PREFIX) for r in reasons):
        return None
    return not reasons


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
    ``min_positive`` contado sobre **todos** os pacientes com medida de referência,
    inclusive os casos em que o modelo se absteve. Abstenção acima de
    ``protocol.max_abstention_fraction`` (zero no 2026-11-v1, que não pré-registrou
    tolerância) é motivo de reprovação: um modelo que se abstém nos casos difíceis e
    concorda nos fáceis não pode aprovar pelo ICC dos que sobraram.
    """
    reasons: list[str] = []
    n_ref_patients = (
        report.n_patients_with_reference
        if report.n_patients_with_reference is not None
        else report.n_patients
    )
    n_ref_cases = (
        report.n_cases_with_reference
        if report.n_cases_with_reference is not None
        else report.n_cases + report.n_abstained
    )
    if n_ref_patients < protocol.min_positive:
        reasons.append(
            f"{ELIGIBILITY_PREFIX}pacientes com referência {n_ref_patients} < min_positive "
            f"{protocol.min_positive}"
        )
    if report.n_abstained:
        fraction = report.n_abstained / n_ref_cases if n_ref_cases else 1.0
        if fraction > protocol.max_abstention_fraction:
            reasons.append(
                f"abstenção em {report.n_abstained} de {n_ref_cases} casos com referência "
                f"({fraction:.1%}) > max_abstention_fraction "
                f"{protocol.max_abstention_fraction} do protocolo {protocol.protocol_id}"
            )
    if report.icc.value < protocol.icc_min:
        reasons.append(f"{report.icc_form} {_fmt(report.icc.value)} < icc_min {protocol.icc_min}")
    return reasons


def judge_segmentation(report: SegmentationReport, protocol: Protocol) -> list[str]:
    reasons: list[str] = []
    if report.n_patients < protocol.min_positive:
        reasons.append(
            f"{ELIGIBILITY_PREFIX}pacientes {report.n_patients} < min_positive "
            f"{protocol.min_positive}"
        )
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
    leakage, note = assess_leakage(ctx.dataset_id, ctx.training_data, ctx.selection_data)
    training_leakage, _ = assess_training_leakage(ctx.dataset_id, ctx.training_data)
    reused = selection_overlap(ctx.dataset_id, ctx.selection_data)
    caveats = [*BASE_CAVEATS, *extra_caveats, *ctx.caveats]
    passes = verdict(reasons)
    if passes is None:
        caveats.append(
            "Registro NÃO ELEGÍVEL (casos abaixo dos mínimos do protocolo): não aprova "
            "nem refuta a condição."
        )
    if reused:
        caveats.append(
            "Dataset de avaliação reutilizado na SELEÇÃO do modelo/codificador "
            f"({', '.join(reused)}): estimativa otimista; não é validação externa."
        )
    if ctx.reference_derived_input:
        caveats.append(
            "Entrada do modelo derivada do padrão de referência "
            f"({ctx.reference_derived_input}): parte da referência entra no modelo; "
            "o registro não sustenta nível A."
        )
    if ctx.git_dirty:
        caveats.append(DIRTY_CAVEAT)
    if ctx.reference_standard == "report_nlp":
        caveats.append("Padrão de referência minerado de laudos por NLP (rótulo ruidoso).")
    if ctx.internal_holdout and training_leakage == "none":
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
        training_leakage=training_leakage,
        selection_overlap=reused,
        reference_derived_input=ctx.reference_derived_input,
        passes_protocol=passes,
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


# --------------------------------------------------------------------------- #
# Métricas do registro: função pura do relatório (re-julgamento usa as mesmas)
# --------------------------------------------------------------------------- #
def binary_metrics(report: BinaryProtocolReport) -> dict[str, Metric | None]:
    metrics: dict[str, Metric | None] = {
        "auroc": _metric(report.auroc),
        "auprc": _metric(report.auprc),
        "ece_before_calibration": _metric(report.ece_before),
        "ece_after_calibration": _metric(report.ece_after),
    }
    for fp in (report.rule_out, report.rule_in):
        if fp is None:
            continue
        metrics[f"{fp.role}_sensitivity"] = _metric(fp.sensitivity)
        metrics[f"{fp.role}_specificity"] = _metric(fp.specificity)
    return metrics


def detection_metrics(report: DetectionReport) -> dict[str, Metric | None]:
    return {
        "sensitivity": _metric(report.sensitivity),
        "specificity": _metric(report.specificity),
        "ppv": _metric(report.ppv),
        "npv": _metric(report.npv),
    }


def agreement_metrics(report: AgreementReport) -> dict[str, Metric | None]:
    def m(i: Any) -> Metric:
        return Metric(value=i.value, ci_low=i.ci_low, ci_high=i.ci_high)

    cases = report.n_cases_with_reference or (report.n_cases + report.n_abstained)
    return {
        "icc": m(report.icc),
        "bland_altman_bias": m(report.bias),
        "bland_altman_loa_low": m(report.loa_low),
        "bland_altman_loa_high": m(report.loa_high),
        "mae": m(report.mae),
        "abstention_fraction": Metric(value=report.n_abstained / cases if cases else 0.0),
    }


def segmentation_metrics(report: SegmentationReport) -> dict[str, Metric | None]:
    return {
        "dice_mean": Metric(
            value=report.dice_mean, ci_low=report.dice_ci_low, ci_high=report.dice_ci_high
        ),
        "dice_median": Metric(value=report.dice_median),
    }


def _undefined_replicates(report: dict[str, Any]) -> list[str]:
    """Caminhos do relatório cujas réplicas de bootstrap tiveram estatística indefinida."""
    found: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            n_def, n_rep = node.get("n_defined_replicates"), node.get("n_replicates")
            if isinstance(n_def, int) and isinstance(n_rep, int) and n_def < n_rep:
                ci = "" if node.get("ci_low") is not None else ", IC indefinido"
                found.append(f"{path} {n_rep - n_def}/{n_rep}{ci}")
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list | tuple):
            for idx, value in enumerate(node):
                walk(value, f"{path}[{idx}]")

    walk(report, "")
    return found


def _replicate_caveat(report: dict[str, Any]) -> list[str]:
    found = _undefined_replicates(report)
    if not found:
        return []
    shown = "; ".join(found[:6]) + (f"; e mais {len(found) - 6}" if len(found) > 6 else "")
    return [
        "Réplicas de bootstrap com estatística indefinida (contadas, fora do percentil; "
        f"abaixo de {MIN_DEFINED_REPLICATE_FRACTION:.0%} definidas o IC fica indefinido): "
        f"{shown}."
    ]


def _envelope(
    ctx: MeasurementContext,
    record: EvidenceRecord,
    report: BaseModel,
    *,
    operating_points: Sequence[OperatingPoint] = (),
    inputs: dict[str, Any] | None = None,
) -> MeasuredEvidence:
    return MeasuredEvidence(
        record=record,
        report=report.model_dump(mode="json"),
        operating_points=tuple(operating_points),
        inputs=dict(inputs or {}),
        declared=MeasurementDeclaration.from_context(ctx),
    )


# --------------------------------------------------------------------------- #
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
    points: list[OperatingPoint] = []
    on = "sigmoid" if score_scale == "probability" else "logit"
    for fp in (report.rule_out, report.rule_in):
        if fp is None:
            continue
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
    caveats.extend(_replicate_caveat(report.model_dump(mode="json")))
    record = _build_record(
        ctx,
        protocol,
        endpoint="discrimination",
        metrics=binary_metrics(report),
        n_total=report.counts.n_images,
        n_positive=report.counts.n_pos,
        reasons=reasons,
        extra_caveats=caveats,
    )
    return _envelope(ctx, record, report, operating_points=points, inputs=inputs)


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
        metrics=detection_metrics(report),
        n_total=report.counts.n_images,
        n_positive=report.counts.n_pos,
        reasons=judge_detection(report, protocol),
        extra_caveats=[
            f"Regra pré-especificada: {rule}",
            f"IC: {report.ci_method}.",
            *_replicate_caveat(report.model_dump(mode="json")),
        ],
    )
    return _envelope(ctx, record, report, inputs=inputs)


def measure_agreement(
    ctx: MeasurementContext,
    protocol: Protocol,
    *,
    patient_ids: Sequence[str],
    measured: FloatArray,
    reference: FloatArray,
    abstained_patient_ids: Sequence[str] = (),
    inputs: dict[str, Any] | None = None,
) -> MeasuredEvidence:
    """Concordância de medida contínua (ICC(2,1), Bland-Altman, MAE).

    ``abstained_patient_ids``: um id por caso com referência em que o modelo não
    mediu. Contam no denominador (``n_total``), na elegibilidade e no limite de
    abstenção do protocolo; nunca são imputados nem descartados em silêncio.
    """
    report = evaluate_agreement(
        patient_ids=patient_ids,
        measured=measured,
        reference=reference,
        seed=protocol.random_seed,
        n_bootstrap=protocol.bootstrap_iterations,
        ci_level=protocol.ci_level,
        abstained_patient_ids=abstained_patient_ids,
    )
    n_ref = report.n_cases_with_reference or report.n_cases
    record = _build_record(
        ctx,
        protocol,
        endpoint="agreement",
        metrics=agreement_metrics(report),
        n_total=n_ref,
        n_positive=n_ref,
        reasons=judge_agreement(report, protocol),
        extra_caveats=[
            f"{report.icc_form}: dois fatores, efeitos aleatórios, concordância absoluta.",
            "n_total = n_positive = casos com medida de referência (desfecho contínuo), "
            f"inclusive {report.n_abstained} em que o modelo se absteve (fora do ICC).",
            f"IC: {report.ci_method}.",
            *_replicate_caveat(report.model_dump(mode="json")),
        ],
    )
    return _envelope(ctx, record, report, inputs=inputs)


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
        metrics=segmentation_metrics(report),
        n_total=report.n_cases + report.n_both_empty,
        n_positive=report.n_cases,
        reasons=judge_segmentation(report, protocol),
        extra_caveats=[
            f"{report.n_both_empty} casos com referência e predição vazias excluídos da média.",
            f"IC: {report.ci_method}.",
        ],
    )
    return _envelope(ctx, record, report, inputs=inputs)


# --------------------------------------------------------------------------- #
# Re-julgamento: o JSON nunca é aceito pelo que diz de si mesmo
# --------------------------------------------------------------------------- #
def _judged(
    endpoint: Endpoint, report: dict[str, Any], protocol: Protocol
) -> tuple[list[str], dict[str, Metric | None], int, int]:
    """``(motivos, métricas, n_total, n_positive)`` recalculados do relatório."""
    from pydantic import ValidationError

    try:
        if endpoint == "discrimination":
            b = BinaryProtocolReport.model_validate(report)
            return (
                judge_discrimination(b, protocol),
                binary_metrics(b),
                b.counts.n_images,
                b.counts.n_pos,
            )
        if endpoint == "detection":
            d = DetectionReport.model_validate(report)
            return (
                judge_detection(d, protocol),
                detection_metrics(d),
                d.counts.n_images,
                d.counts.n_pos,
            )
        if endpoint == "agreement":
            a = AgreementReport.model_validate(report)
            n_ref = a.n_cases_with_reference or a.n_cases
            return judge_agreement(a, protocol), agreement_metrics(a), n_ref, n_ref
        if endpoint == "segmentation":
            g = SegmentationReport.model_validate(report)
            return (
                judge_segmentation(g, protocol),
                segmentation_metrics(g),
                g.n_cases + g.n_both_empty,
                g.n_cases,
            )
    except ValidationError as exc:
        raise EvidenceError(f"relatório do envelope inválido para {endpoint}: {exc}") from exc
    raise EvidenceError(f"desfecho {endpoint!r} sem regra de re-julgamento")


def rejudge(evidence: MeasuredEvidence, protocol: Protocol) -> list[str]:
    """Divergências entre o registro gravado e o que o protocolo e as declarações dão.

    Recalcula, a partir do relatório completo do envelope e do protocolo vigente,
    os motivos de reprovação, o veredito, as métricas e as contagens; e, a partir
    das declarações (treino, seleção, hold-out interno), o vazamento e o tipo.
    Lista vazia = o registro é exatamente o que o avaliador produziria.

    Raises:
        EvidenceError: envelope sem declarações (anterior a elas) ou relatório que
            não valida para o desfecho do registro.
    """
    rec = evidence.record
    declared = evidence.declared
    if declared is None:
        raise EvidenceError(
            f"registro {rec.id!r} sem declarações de treino/seleção no envelope: não pode "
            "ser re-julgado; re-execute a medição (scripts/evidence_run.py --plan)"
        )
    if rec.dataset is None:
        raise EvidenceError(f"registro medido {rec.id!r} sem dataset")
    out: list[str] = []
    reasons, metrics, n_total, n_positive = _judged(rec.endpoint, evidence.report, protocol)
    expected_metrics = {k: v for k, v in metrics.items() if v is not None}
    if tuple(reasons) != rec.failure_reasons:
        out.append(
            f"failure_reasons gravados {list(rec.failure_reasons)} != recalculados {reasons}"
        )
    if verdict(reasons) is not rec.passes_protocol:
        out.append(
            f"passes_protocol gravado {rec.passes_protocol} != recalculado {verdict(reasons)}"
        )
    if expected_metrics != dict(rec.metrics):
        out.append("métricas gravadas diferem das do relatório")
    if (n_total, n_positive) != (rec.n_total, rec.n_positive):
        out.append(
            f"contagens gravadas {(rec.n_total, rec.n_positive)} != do relatório "
            f"{(n_total, n_positive)}"
        )
    leakage, _ = assess_leakage(rec.dataset, declared.training_data, declared.selection_data)
    training_leakage, _ = assess_training_leakage(rec.dataset, declared.training_data)
    reused = selection_overlap(rec.dataset, declared.selection_data)
    kind = "measured_internal" if declared.internal_holdout else "measured_external"
    for name, got, want in (
        ("leakage", rec.leakage, leakage),
        ("training_leakage", rec.training_leakage, training_leakage),
        ("selection_overlap", rec.selection_overlap, reused),
        ("kind", rec.kind, kind),
        ("reference_derived_input", rec.reference_derived_input, declared.reference_derived_input),
    ):
        if got != want:
            out.append(f"{name} gravado {got!r} != recalculado {want!r}")
    if declared.git_dirty and DIRTY_CAVEAT not in rec.caveats:
        out.append("medição com árvore suja sem o aviso correspondente")
    return out
